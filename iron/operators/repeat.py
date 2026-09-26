# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import field

import numpy as np
from ml_dtypes import bfloat16

from aie.utils.verify import Tolerance

from iron.common.declare import In, Operator, Out, param, auto
from iron.common.tiling import Access, granule_elements
from iron.common.testing import Case, Testing
from iron.common.tiling import DMA_BD_MAX_WRAP


class Repeat(Operator):
    """AIE-accelerated repeat-interleave operator: a memtile pass-through of
    ``transfer_size`` elements, no cores.

    The repeat is entirely in the runtime sequence's descriptors: the input
    is re-read ``repeat`` times and the output interleaved.
    """

    # rows, cols, repeat, transfer_size. The sequence splits cols into chunks
    # <= 1023 by the smallest divisor that gets under the hardware limit, so
    # cols on either side of 1023 take different paths and both need
    # covering. The llama arm is the shape the only caller dispatches:
    # n_kv_groups=8 groups expanded to n_heads=32 over a max_seq_len=2048
    # context of head_dim=64, i.e. repeat=4 with cols=2048*64.
    #
    # Repeat moves data and computes nothing, so the gate is exact equality.
    # A tolerance would accept a permutation that reads the wrong group,
    # which is the failure mode here: a misrouted KV group is numerically
    # plausible.
    test = Testing(
        [
            Case(dict(rows=8, cols=64, repeat=4, transfer_size=None)),
            Case(dict(rows=8, cols=512, repeat=4, transfer_size=64)),
            Case(dict(rows=4, cols=1024, repeat=2, transfer_size=None)),
            Case(dict(rows=4, cols=2048, repeat=2, transfer_size=None), extensive=True),
            Case(
                dict(rows=8, cols=2048 * 64, repeat=4, transfer_size=64),
                extensive=True,
            ),
        ],
        tolerance=Tolerance.exact(),
    )

    rows: int = param()
    cols: int = param()
    repeat: int = param()
    # rows * repeat; derived unless given, since a shape may not be an expression.
    out_rows: int | None = param(default=None, repr=False)
    transfer_size: int = auto(repr=False)  # None: cols
    dtype: object = field(default=bfloat16, repr=False)

    x = In(rows, cols, dtype=dtype, tile=(transfer_size,))
    y = Out(out_rows, cols, dtype=dtype, tile=(transfer_size,))

    def validate(self) -> None:
        expected = self.rows * self.repeat
        if self.out_rows is None:
            self.out_rows = expected
        elif self.out_rows != expected:
            raise ValueError(
                f"out_rows={self.out_rows} is not rows * repeat ({expected})"
            )
        self._cols_split()  # reject an unsplittable cols at construction

    def resolve(self, dev):
        return dataclasses.replace(self, transfer_size=self.transfer_size or self.cols)

    def _cols_split(self) -> int:
        """Split cols into cols_split chunks of cols // cols_split.

        The chunk length is the innermost descriptor dimension, at most 1023
        and a whole number of 32-bit words; the chunk count is the next
        dimension out, at most 1023. An odd cols has only odd divisors, so
        no split of it is ever word-aligned at bf16; that is reported here
        rather than left to the BD verifier.
        """
        cols = self.cols
        granule = granule_elements(self.dtype)
        for divisor in range(1, cols + 1):
            if cols % divisor:
                continue
            chunk = cols // divisor
            if (
                chunk <= DMA_BD_MAX_WRAP
                and divisor <= DMA_BD_MAX_WRAP
                and chunk % granule == 0
            ):
                return divisor
        elem_bytes = np.dtype(self.dtype).itemsize
        raise ValueError(
            f"Cannot split cols={cols} at {elem_bytes} bytes/element: need a divisor d "
            f"with cols//d <= 1023, d <= 1023, and cols//d a multiple of {granule} "
            f"({granule} elements = one 32-bit word). No divisor of {cols} satisfies all three."
        )

    def array(self, target) -> list:
        from aie.iron import ObjectFifo

        fifo_in = ObjectFifo(self.x.tile, name="fifo_in", depth=2)
        fifo_out = fifo_in.cons().forward(name="fifo_out", depth=2)
        self.x.bind(fifo_in.prod())
        self.y.bind(fifo_out.cons())
        return []

    def sequence(self, rt):
        rows, cols, repeat = self.rows, self.cols, self.repeat
        cols_split = self._cols_split()
        chunk = cols // cols_split
        # The chunk length is innermost so the contiguous run is the innermost
        # dimension; the chunk count sits outside it. The input's outermost
        # (iteration) dimension re-reads the whole matrix ``repeat`` times with
        # a zero stride; the output's interleaves.
        input_tap = Access(
            self.x.elements, 0, (repeat, rows, cols_split, chunk), (0, cols, chunk, 1)
        )
        output_tap = Access(
            self.y.elements,
            0,
            (repeat, rows, cols_split, chunk),
            (cols, cols * repeat, chunk, 1),
        )
        with rt.group() as tg:
            rt.fill(self.x, input_tap, group=tg)
            rt.drain(self.y, output_tap, group=tg, wait=True)

    def reference(self, x):
        """CPU reference: repeat-interleave along the leading dimension."""
        return reference(x, self.repeat)


# --------------------------------------------------------------------------
# The CPU reference this operator is checked against.
# --------------------------------------------------------------------------


def reference(x, repeat):
    """CPU reference: repeat-interleave along the leading dimension (ground truth)."""
    return np.repeat(x, repeat, axis=0)
