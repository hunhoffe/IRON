# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import dataclasses
from typing import Any

import ml_dtypes
import numpy as np
from aie.iron import ObjectFifo, Worker
from aie.iron.controlflow import range_
from aie.iron.kernels import activation
from aie.utils.verify import Tolerance

from iron.common import (
    In,
    Incompatible,
    Operator,
    Out,
    Value,
    auto,
    param,
)
from iron.common.testing import Case, Testing, device_columns


def _columns_channels(total_cores):
    """The (columns, channels) split for a core count: 2x2 from four cores up
    (a 4x4 has placement issues on Phoenix), 1x2 for two, 1x1 for one.
    """
    return {1: (1, 1), 2: (1, 2)}.get(total_cores, (2, 2))


def _cases(cls):
    out = []
    for size, cols in [(32768, 1024), (32768, 512), (32768, 2048)]:
        columns, channels = _columns_channels(size // cols)
        if columns > device_columns():
            continue
        out.append(
            Case(
                dict(
                    rows=size // cols,
                    cols=cols,
                    num_aie_columns=columns,
                    num_channels=channels,
                )
            )
        )
    return out


class Softmax(Operator):
    """AIE-accelerated Softmax operation: one core per (column, channel), one
    row per tile.

    Each row is masked to ``vector_size`` valid elements before the softmax:
    the whole row, unless a graph binds a per-call value to it (``Softmax(x,
    vector_size=n)``, llama's decode mask), which the core then reads per
    call.
    """

    test = Testing(_cases, tolerance=Tolerance.relative(0.04, 1e-6))

    rows: int = param()
    cols: int = param()
    # None: every column the device's shim budget allows.
    num_aie_columns: int = auto()
    num_channels: int = auto(1)

    x = In(rows, cols, tile=(cols,), per=(num_aie_columns, num_channels))
    y = Out(rows, cols, tile=(cols,), per=(num_aie_columns, num_channels))
    count = Value(np.int32, derive=lambda op: op.rows // op.cores)  # rows per core
    vector_size = Value(np.int32, derive=lambda op: op.cols)  # valid elements per row

    def validate(self) -> None:
        if self.rows % 16 != 0:
            raise ValueError(f"rows ({self.rows}) must be a multiple of 16")
        if self.cols % 16 != 0:
            raise ValueError(f"cols ({self.cols}) must be a multiple of 16")

    def resolve(self, dev):
        """Columns default to the most the device's shim budget allows that
        leave every core a whole number of rows.
        """
        cols = self.resolve_columns(
            dev,
            self.num_aie_columns,
            self.num_channels,
            fits=lambda c: self.rows % (c * self.num_channels) == 0,
        )
        return dataclasses.replace(self, num_aie_columns=cols)

    @property
    def cores(self) -> int:
        return self.num_aie_columns * self.num_channels

    def compatible(self) -> None:
        if self.rows % self.num_aie_columns:
            raise Incompatible(
                f"rows ({self.rows}) must be a multiple of num_aie_columns ({self.num_aie_columns})"
            )
        if self.rows % self.cores:
            raise Incompatible(
                f"rows ({self.rows}) must be a multiple of the {self.cores} cores"
            )

    def _kernels(self, tile_ty):
        softmax_k = activation.softmax(self.cols)
        # mask_bf16 is exported by the same softmax.cc translation unit.
        mask_k = softmax_k.object_file.bind("mask_bf16", [tile_ty, np.int32, np.int32])
        return softmax_k, mask_k

    def array(self, target) -> list:

        tile_ty = self.x.tile
        cols, chans = self.num_aie_columns, self.num_channels
        n_cores = cols * chans
        softmax_k, mask_k = self._kernels(tile_ty)
        of_ins = [
            ObjectFifo(tile_ty, name=f"in1_{i}_{j}")
            for i in range(cols)
            for j in range(chans)
        ]
        of_outs = [
            ObjectFifo(tile_ty, name=f"out_{i}_{j}")
            for i in range(cols)
            for j in range(chans)
        ]
        # [count, vector_size] per core, or [count] when vector_size is a
        # per-call value the core reads from the scratchpad. On an image
        # without a scratchpad the per-call value is written into [1] by the
        # sequence instead.
        dynamic = self.uses_value("vector_size") and target.image == "elf"
        rtp_ty = np.ndarray[(1 if dynamic else 2,), np.dtype[np.int32]]
        rtps = [target.rtp(rtp_ty, name=f"rtp_{k}") for k in range(n_cores)]
        barriers = [target.barrier() for _ in range(n_cores)]
        per_tile = self.cols
        param = self.vector_size.param if dynamic else None

        def core_body(
            of_in,
            of_out,
            softmax_kernel,
            mask_kernel,
            rtp,
            barrier,
            vector_size_src: Any = None,
        ):
            barrier.wait_for_value(1)
            n = rtp[0]
            # `dynamic` is a compile-time constant, so only one of these is
            # emitted: a scratchpad parameter read or a write-RTP buffer load.
            vector_size = vector_size_src.read() if dynamic else rtp[1]
            for _ in range_(n):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                mask_kernel(elem_in, vector_size, per_tile)
                softmax_kernel(elem_in, elem_out, per_tile)
                of_in.release(1)
                of_out.release(1)

        workers = [
            Worker(
                core_body,
                [
                    of_ins[k].cons(),
                    of_outs[k].prod(),
                    softmax_k,
                    mask_k,
                    rtps[k],
                    barriers[k],
                ]
                + ([param] if dynamic else []),
            )
            for k in range(n_cores)
        ]
        for k in range(n_cores):
            self.x.lane(k).bind(of_ins[k].prod())
            self.y.lane(k).bind(of_outs[k].cons())
        self.count.bind(rtps, 0)
        if not dynamic:
            self.vector_size.bind(rtps, 1)
        return workers

    def reference(self, x, vector_size=None):
        """CPU reference: row-wise softmax over the first ``vector_size`` of ``cols``.

        The kernel fills ``[vector_size, cols)`` with the lowest bf16 before
        the softmax, so the masked tail comes out as exact zeros. Without a
        per-call value the whole row is valid.
        """
        if vector_size is None:
            vector_size = self.cols
        return reference(x.reshape(self.rows, self.cols), int(vector_size))


# --------------------------------------------------------------------------
# The CPU reference this operator is checked against.
# --------------------------------------------------------------------------


def reference(x, vector_size=None):
    """CPU reference: row-wise softmax over the last dim (ground truth).

    ``vector_size`` masks every column from there on to the lowest value of
    the dtype first, as the device kernel does, so those come out as zeros.
    """
    if vector_size is not None and vector_size < x.shape[-1]:
        x = x.copy()
        x[..., vector_size:] = ml_dtypes.finfo(x.dtype).min
    # In float32 and rounded once, as torch does internally for a bf16 input.
    f = x.astype(np.float32)
    e = np.exp(f - f.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)
