# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, ObjectFifo, Out, Worker
from aie.iron.controlflow import range_
from aie.iron.kernels import quant
from aie.utils.compile.jit.params import param, tunable, widths
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from iron.common import Built, Operator
from iron.common.testing import Case, Testing
from iron.operators import quantized

from . import layout
from .layout import BLOCK_BYTES, K_TILE, M_TILE


def _draw(op):
    """Weights quantized from normal draws, and a normal x for a GEMV."""
    rng = np.random.default_rng(0)
    w = layout.quantize(rng.standard_normal((op.N, op.K)) / np.sqrt(op.K))
    given = {op.w: w.reshape(-1)}
    if isinstance(op, TernaryGEMV):
        given[op.x] = rng.standard_normal(op.K).astype(bfloat16)
    return dict(given=given)


def _check(op) -> None:
    if op.K % K_TILE or op.N % M_TILE:
        raise ValueError(
            f"{type(op).__name__}: K ({op.K}) and N ({op.N}) must be multiples "
            f"of {K_TILE} and {M_TILE}, the ternary block"
        )


def _w_tap(op, t: int) -> TensorAccessPattern:
    """Tile ``t``'s blocks, k-tile by k-tile, one contiguous run; a block is
    split 17 x 128 to keep the innermost size inside the BD's field.
    """
    k_tiles = op.K // K_TILE
    return TensorAccessPattern(
        (op.w_bytes,),
        t * k_tiles * BLOCK_BYTES,
        [k_tiles, BLOCK_BYTES // 128, 128],
        [BLOCK_BYTES, 128, 1],
    )


class TernaryGEMV(Operator):
    """``y = w @ x`` for one row ``x`` of ``K`` values and ``N`` out-features
    of ternary weights ``w``, laid out as ``layout.quantize`` lays them out.

    Each column's core takes 32 out-features a round, streaming their
    blocks past ``x``: it adds and subtracts ``x`` and multiplies only by
    each group's scale. See README.md.
    """

    test = Testing(
        [
            Case(dict(K=512, N=256)),
            # More out-feature tiles than a round takes, unevenly.
            Case(dict(K=1024, N=288), bench=True),
            # Qwen3.8-27B's MLP down projection.
            Case(dict(K=17408, N=5120), extensive=True),
        ],
        draw=_draw,
    )

    K: int = param(array=True)
    N: int = param(array=True)
    w_bytes: int = param(derive=lambda op: op.N * op.K * 17 // 64, repr=False)
    cols: int = tunable(repr=False, domain=widths)

    y = Out(N, dtype=bfloat16, obj_shape=(M_TILE,), per=(cols,))
    w = In(w_bytes, dtype=np.uint8, obj_shape=(BLOCK_BYTES,), per=(cols,))
    x = In(K, dtype=bfloat16, obj_shape=(K,))

    def validate(self) -> None:
        _check(self)

    def resolve(self, dev):
        return self.replace(cols=quantized.cols(self, dev))

    lane_tiles = quantized.lane_tiles
    w_tap = _w_tap
    sequence = quantized.gemv_sequence

    def ops(self) -> int:
        return 2 * self.N * self.K

    def build(self) -> Built:
        kernel = quant.ternary_gemv(dim=self.K, m_tile=M_TILE, k_tile=K_TILE)
        return quantized.gemv_build(
            self, K_TILE, kernel.ternary_gemv_zero, kernel, kernel.ternary_gemv_store
        )

    def tolerance(self) -> Tolerance:
        def bound(w, x):
            weights = layout.dequantize(np.reshape(w, (self.N, -1)))
            x = np.asarray(x, np.float64).reshape(-1)
            # A float32 sum of K terms, narrowed to bf16 by floor against
            # the reference's nearest.
            return 2**-6 * np.abs(weights @ x) + self.K * 2**-24 * (
                np.abs(weights) @ np.abs(x)
            )

        return Tolerance.bounded(
            bound, note="float32 sums narrowed to bf16 by floor; derived, not measured"
        )

    def reference(self, w, x):
        """``w @ x`` in float64, rounded to bf16."""
        weights = layout.dequantize(np.reshape(w, (self.N, -1)))
        return (weights @ np.asarray(x, np.float64).reshape(-1)).astype(bfloat16)


class TernaryDequant(Operator):
    """Ternary weights ``w`` of ``N`` out-features by ``K`` in-features, laid
    out as ``layout.quantize`` lays them out, as their bf16 values ``(N, K)``
    row-major: a prompt's weight for ``flm.ProjectionGEMM``. Exact, since
    each value is a signed copy of its bf16 scale.

    Each column's core takes 32 out-features a round, as ``TernaryGEMV``'s.
    """

    test = Testing(
        [
            Case(dict(K=512, N=64)),
            Case(dict(K=1024, N=288), bench=True),
        ],
        draw=_draw,
    )

    K: int = param(array=True)
    N: int = param(array=True)
    w_bytes: int = param(derive=lambda op: op.N * op.K * 17 // 64, repr=False)
    size: int = param(derive=lambda op: op.N * op.K, repr=False)
    cols: int = tunable(repr=False, domain=widths)

    out = Out(size, dtype=bfloat16, obj_shape=(M_TILE * K_TILE,), per=(cols,))
    w = In(w_bytes, dtype=np.uint8, obj_shape=(BLOCK_BYTES,), per=(cols,))

    def validate(self) -> None:
        _check(self)

    def resolve(self, dev):
        return self.replace(cols=quantized.cols(self, dev))

    lane_tiles = quantized.lane_tiles
    w_tap = _w_tap

    def build(self) -> Built:
        kernel = quant.ternary_dequant(m_tile=M_TILE, k_tile=K_TILE)
        w_ty, out_ty = kernel.arg_types()

        def core(blocks):
            def core_fn(w_in, out_of, k):
                for _ in range_(blocks):
                    w = w_in.acquire(1)
                    out = out_of.acquire(1)
                    k(w, out)
                    w_in.release(1)
                    out_of.release(1)

            return core_fn

        workers, ws, outs = [], [], []
        for c in range(self.cols):
            of_w, of_out = ObjectFifo(w_ty), ObjectFifo(out_ty)
            ws.append(of_w.prod())
            outs.append(of_out.cons())
            blocks = len(self.lane_tiles(c)) * self.K // K_TILE
            workers.append(Worker(core(blocks), [of_w.cons(), of_out.prod(), kernel]))
        self.w.bind(ws)
        self.out.bind(outs)
        return Built(workers)

    def sequence(self, rt):
        K = self.K
        for lanes, tg in quantized.rounds(self):
            for col, t in lanes:
                # Block k lands at rows t * 32.., columns k * 256...
                self.out.drain(
                    TensorAccessPattern(
                        (self.N * K,),
                        t * M_TILE * K,
                        [K // K_TILE, M_TILE, K_TILE],
                        [K_TILE, K, 1],
                    ),
                    lane=col,
                    wait=True,
                    group=tg,
                )
                self.w.fill(self.w_tap(t), lane=col, group=tg)

    def tolerance(self) -> Tolerance:
        return Tolerance.exact()

    def reference(self, w):
        return layout.dequantize(np.reshape(w, (self.N, -1))).astype(bfloat16).ravel()
