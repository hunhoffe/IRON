# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The descriptors StridedCopy issues for the copies llama makes, pinned.

llama's graphs spell every copy as sizes, strides and offsets; a copy that
takes a slice of the destination instead must issue these same descriptors,
so they are recorded here as (offset, sizes, strides) per channel, exactly.
"""

import aie.utils as aie_utils
import pytest

from iron.operators.strided_copy import StridedCopy, _kv_slot

G, D, L, E, N = 8, 64, 128, 2048, 16  # kv groups, head dim, context, embed, rows

# One token's (G, D) keys into row `cache_offset` of the (G, L, D) cache.
ROW_INTO_CACHE = dict(
    input_sizes=(G, D),
    input_strides=(D, 1),
    input_offset=0,
    input_buffer_size=G * D,
    output_sizes=(1, G, D),
    output_strides=(0, L * D, 1),
    output_offset=0,
    output_buffer_size=G * L * D,
    num_aie_channels=1,
)
# N tokens' (N, G, D) keys, heads interleaved per token, into the first N rows.
ROWS_INTO_CACHE = dict(
    input_sizes=(G, N, D),
    input_strides=(D, G * D, 1),
    input_offset=0,
    input_buffer_size=N * G * D,
    output_sizes=(G, N, D),
    output_strides=(L * D, D, 1),
    output_offset=0,
    output_buffer_size=G * L * D,
    transfer_size=1024,
    num_aie_channels=1,
)
# The last prompt row of (4, E), selected by the per-call offset `last`.
LAST_ROW = dict(
    input_sizes=(1, E),
    input_strides=(E, 1),
    input_offset=0,
    input_buffer_size=4 * E,
    output_sizes=(1, E),
    output_strides=(E, 1),
    output_offset=0,
    output_buffer_size=E,
    num_aie_channels=1,
)

PINNED = {
    "row_into_cache": (
        ROW_INTO_CACHE,
        [[(0, (1, 1, 1, 512), (0, 0, 0, 1))]],
        [[(0, (1, 1, 8, 64), (0, 0, 8192, 1))]],
    ),
    "rows_into_cache": (
        ROWS_INTO_CACHE,
        [[(0, (1, 8, 16, 64), (0, 64, 512, 1))]],
        [[(0, (1, 8, 16, 64), (0, 8192, 64, 1))]],
    ),
    "last_row": (
        LAST_ROW,
        [[(0, (1, 1, 1, 2048), (0, 0, 0, 1))]],
        [[(0, (1, 1, 1, 2048), (0, 0, 0, 1))]],
    ),
    "kv_slot5": (
        _kv_slot(128, 5),
        [[(0, (1, 1, 1, 512), (0, 0, 0, 1))]],
        [[(320, (1, 1, 8, 64), (0, 0, 8192, 1))]],
    ),
    "kv_slot5_two_channels": (
        _kv_slot(128, 5, num_aie_channels=2),
        [[(0, (1, 1, 8, 32), (0, 0, 64, 1))], [(32, (1, 1, 8, 32), (0, 0, 64, 1))]],
        [
            [(320, (1, 1, 8, 32), (0, 0, 8192, 1))],
            [(352, (1, 1, 8, 32), (0, 0, 8192, 1))],
        ],
    ),
}


def _taps(op, buffer, sizes, strides, offset):
    return [
        [(a.offset, a.sizes, a.strides) for a in channel]
        for channel in op._taps(buffer, sizes, strides, offset)
    ]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_strided_copy_issues_these_descriptors(name):
    kwargs, ins, outs = PINNED[name]
    op = StridedCopy(**kwargs).resolved(aie_utils.get_current_device())
    assert _taps(op, op.x, op.input_sizes, op.input_strides, op.input_offset) == ins
    assert _taps(op, op.y, op.output_sizes, op.output_strides, op.output_offset) == outs
