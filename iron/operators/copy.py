# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A copy between two views: ``Copy(k, keys[i][:, pos])``.

Each side is a walk over its buffer (offset, sizes, strides), which is what a
DMA is given; the shim channels split the walk on its highest non-unit axis
and each share is legalized for the shim. A per-call value indexing a view
reaches the copy as ``in_offset``/``out_offset``, an element offset.
"""

import dataclasses
from dataclasses import field
from typing import Any

import numpy as np
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from iron.common.declare import In, Incompatible, Operator, Out, Scratchpad, auto, param
from iron.common.testing import Case, Testing
from iron.common.tiling import Walk, legalize

# Llama's KV-cache write, shrunk: the cache is (n_kv_groups, seq, head_dim)
# and one token's keys land in slot t of every group. SEQ is 128 rather than
# the real 2048 to keep the output buffer at 128 KB; the full-size arm is
# extensive.
_N_KV, _HEAD_DIM, _SEQ = 8, 64, 128


def _kv_slot(seq, slot, num_aie_channels=1) -> dict[str, Any]:
    """Kwargs writing one (N_KV, HEAD_DIM) token into cache slot ``slot``."""
    return dict(
        src=Walk.of((_N_KV, _HEAD_DIM)),
        dst=Walk.slice((_N_KV, seq, _HEAD_DIM), (slice(None), slot)),
        input_buffer_size=_N_KV * _HEAD_DIM,
        output_buffer_size=_N_KV * seq * _HEAD_DIM,
        num_aie_channels=num_aie_channels,
    )


def _flat(size, num_aie_channels=1, transfer_size=None) -> dict[str, Any]:
    """Kwargs for a contiguous copy of ``size`` elements."""
    return dict(
        input_buffer_size=size,
        output_buffer_size=size,
        num_aie_channels=num_aie_channels,
        transfer_size=transfer_size,
    )


def _pad4(sizes, strides):
    """Pad to 4-D: dropping leading dimensions leaves BD registers uninitialised."""
    sizes, strides = list(sizes), list(strides)
    return [1] * (4 - len(sizes)) + sizes, [0] * (4 - len(strides)) + strides


class Copy(Operator):
    """AIE-accelerated copy between two views of two buffers.

    Gathers by ``src`` and scatters by ``dst``, split across
    ``num_aie_channels`` memtile pass-throughs (no cores) on the highest
    non-unit axis. In a graph the walks come from the operands: ``Copy(k,
    keys[i][:, pos])``, ``Copy(x.transpose(1, 0, 2), y[:, :n])``; a per-call
    index on a view binds ``in_offset`` or ``out_offset``. Standalone,
    ``src``/``dst`` are given, or default to the whole of each buffer.

    Each channel's descriptor carries 1/num_aie_channels of the walk, so the
    fifo object is sized against the per-channel share (``transfer_size``). A
    descriptor shorter than the object starves the memtile's S2MM: it never
    completes an object, never releases the lock, and the drain never returns
    (ERT_CMD_STATE_TIMEOUT). An integer multiple is fine; it cycles the buffer.
    """

    # The params that take an operand's view, and the value its per-call
    # index binds, in operand order.
    accept_views = (("src", "in_offset"), ("dst", "out_offset"))

    # Copy moves data and computes nothing, so the gate is exact.
    test = Testing(
        [
            Case(_flat(1024), id="contiguous"),
            Case(_flat(1024, num_aie_channels=2), id="two_channels"),
            Case(_flat(1024, num_aie_channels=4), id="four_channels"),
            Case(
                _flat(1024, num_aie_channels=2, transfer_size=256),
                id="two_channels_chunked",
            ),
            Case(_flat(1024, transfer_size=256), id="chunked_transfer"),
            Case(_kv_slot(_SEQ, 0), id="kv_slot0"),
            Case(_kv_slot(_SEQ, 5), id="kv_slot5"),
            Case(_kv_slot(_SEQ, _SEQ - 1), id="kv_slot_last"),
            # The KV-cache write is what num_aie_channels exists to widen, so
            # it carries the strided arms too: the flat cases split a
            # stride-1 run, these split head_dim.
            Case(_kv_slot(_SEQ, 5, num_aie_channels=2), id="kv_slot5_two_channels"),
            Case(_kv_slot(_SEQ, 5, num_aie_channels=4), id="kv_slot5_four_channels"),
            Case(_kv_slot(2048, 1000), id="kv_llama_full", extensive=True),
        ],
        tolerance=Tolerance.exact(),
    )

    input_buffer_size: int = param(repr=False)
    output_buffer_size: int | None = param(default=None, repr=False)
    src: Walk | None = param(default=None)  # None: the whole input
    dst: Walk | None = param(default=None)  # None: the whole output
    transfer_size: int = auto()  # None: the per-channel share of the walk
    num_aie_channels: int = auto(1)
    dtype: Any = field(default=bfloat16, repr=False)

    x = In(
        input_buffer_size,
        dtype=dtype,
        tile=(transfer_size,),
        per=(num_aie_channels,),
        depth=1,
    )
    y = Out(
        output_buffer_size,
        dtype=dtype,
        tile=(transfer_size,),
        per=(num_aie_channels,),
        depth=1,
    )
    # Per-call addends on the two base addresses, in elements.
    in_offset = Scratchpad(np.int32)
    out_offset = Scratchpad(np.int32)

    def validate(self) -> None:
        if self.src is None:
            self.src = Walk.of((self.input_buffer_size,))
        if self.output_buffer_size is None:
            self.output_buffer_size = self.src.elements
        if self.dst is None:
            self.dst = Walk.of((self.output_buffer_size,))
        if self.src.elements != self.dst.elements:
            raise ValueError(
                f"a copy moves the same element count both ways: src {self.src} "
                f"has {self.src.elements} elements, dst {self.dst} has "
                f"{self.dst.elements}"
            )

    def resolve(self, dev):
        """The transfer size is the per-channel share of the copy unless given."""
        assert self.src is not None  # validate() filled it
        transfer_size = self.transfer_size or self.src.elements // self.num_aie_channels
        return dataclasses.replace(self, transfer_size=transfer_size)

    def uses_value(self, name: str) -> bool:
        # An offset is patched only when a graph binds a handle to it.
        return name in self.used_values

    def array(self, target) -> list:
        from aie.iron import ObjectFifo

        for c in range(self.num_aie_channels):
            fifo_in = ObjectFifo(self.x.tile, name=f"fifo_in_{c}", depth=1)
            fifo_out = fifo_in.cons().forward(name=f"fifo_out_{c}", depth=1)
            self.x.lane(c).bind(fifo_in.prod())
            self.y.lane(c).bind(fifo_out.cons())
        return []

    @property
    def walks(self) -> tuple[Walk, Walk]:
        """The two walks, as :meth:`validate` filled them."""
        assert self.src is not None and self.dst is not None
        return self.src, self.dst

    def compatible(self) -> None:
        channels = self.num_aie_channels
        src, dst = self.walks
        for label, walk in (("src", src), ("dst", dst)):
            sizes, _ = _pad4(walk.sizes, walk.strides)
            highest = max(i for i, sz in enumerate(sizes) if sz >= 1)
            if sizes[highest] % channels:
                raise Incompatible(
                    f"the highest axis of {label} {walk} must be divisible by "
                    f"num_aie_channels ({channels})"
                )
        per_channel = src.elements // channels
        if per_channel % self.transfer_size:
            raise Incompatible(
                f"transfer_size {self.transfer_size} must divide the per-channel "
                f"transfer {per_channel} (= {src.elements} / {channels} channels)"
            )

    def _taps(self, buffer, walk: Walk, offset: int = 0):
        """Per channel, the descriptors of its share of the walk.

        The highest non-unit axis is split across the channels; each share
        is then legalized for the shim (an axis past its slot's wrap is
        factored or unrolled, order preserved), so a reorder as wide as a
        sequence lowers rather than failing three tools down.
        """
        sizes, strides = _pad4(walk.sizes, walk.strides)
        highest = max(i for i, sz in enumerate(sizes) if sz >= 1)
        channels = self.num_aie_channels
        share = sizes[highest] // channels
        split = sizes[:highest] + [share] + sizes[highest + 1 :]
        return [
            legalize(
                buffer.elements,
                walk.offset + offset + c * share * strides[highest],
                split,
                strides,
                buffer.dtype,
            )
            for c in range(channels)
        ]

    def reference(self, x, y=None, *, in_offset=0, out_offset=0):
        """CPU reference: gather by ``src``, scatter by ``dst``.

        ``x`` is the whole input buffer and ``y`` the whole output buffer,
        written in place when given (a cache the graph passes as an output
        keeps everything the copy does not touch); otherwise a zeroed buffer
        of ``output_buffer_size``. The offsets are the per-call values, in
        elements.
        """
        src, dst = self.walks
        out = reference(
            x.reshape(-1),
            src,
            self.output_buffer_size,
            dst,
            self.num_aie_channels,
            input_offset_addend=int(in_offset),
            output_offset_addend=int(out_offset),
            into=None if y is None else y.reshape(-1),
        )
        return out if y is None else y

    def sequence(self, rt):
        src, dst = self.walks
        ins = self._taps(self.x, src)
        outs = self._taps(self.y, dst)
        in_off = self.in_offset if self.uses_value("in_offset") else None
        out_off = self.out_offset if self.uses_value("out_offset") else None
        with rt.group() as tg:
            for c in range(self.num_aie_channels):
                for acc in ins[c]:
                    rt.fill(self.x.lane(c), acc, group=tg, offset_by=in_off)
                for acc in outs[c]:
                    rt.drain(
                        self.y.lane(c),
                        acc,
                        group=tg,
                        wait=acc is outs[c][-1],
                        offset_by=out_off,
                    )


# --------------------------------------------------------------------------
# The CPU reference this operator is checked against.
# --------------------------------------------------------------------------


def _walk_offsets(sizes, strides, offset):
    """Flat element offsets a walk visits, in issue order."""
    grids = np.meshgrid(*[np.arange(s) for s in sizes], indexing="ij")
    flat = np.full(grids[0].shape, offset, dtype=np.int64)
    for grid, stride in zip(grids, strides):
        flat = flat + grid * stride
    return flat.reshape(-1)


def _channel_offsets(walk: Walk, addend: int, num_aie_channels: int):
    """Per channel, the flat offsets of its share: design's split, exactly."""
    sizes, strides = _pad4(walk.sizes, walk.strides)
    highest = max(idx for idx, sz in enumerate(sizes) if sz >= 1)
    per_channel = sizes[highest] // num_aie_channels
    split = sizes[:highest] + [per_channel] + sizes[highest + 1 :]
    return [
        _walk_offsets(
            split, strides, walk.offset + addend + c * per_channel * strides[highest]
        )
        for c in range(num_aie_channels)
    ]


def reference(
    input_flat,
    src: Walk,
    output_buffer_size,
    dst: Walk,
    num_aie_channels=1,
    input_offset_addend=0,
    output_offset_addend=0,
    into=None,
):
    """Gather by ``src``, scatter by ``dst``, one channel at a time.

    The addends are the per-call offsets. They are element counts, not byte
    offsets: the firmware multiplies the scratchpad word by the element size
    before adding it into the BD address register. ``into`` is an existing
    flat output buffer to scatter into in place; without it the output
    starts zeroed.
    """
    gather = _channel_offsets(src, input_offset_addend, num_aie_channels)
    scatter = _channel_offsets(dst, output_offset_addend, num_aie_channels)
    out = (
        np.zeros(int(output_buffer_size), dtype=input_flat.dtype)
        if into is None
        else into
    )
    for src_c, dst_c in zip(gather, scatter):
        if len(src_c) != len(dst_c):
            raise ValueError(
                f"walk element counts differ ({len(src_c)} vs {len(dst_c)}); "
                "src and dst must move the same number of elements"
            )
        out[dst_c] = input_flat[src_c]
    return out
