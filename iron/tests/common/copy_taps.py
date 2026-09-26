# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The descriptors Copy issues for the copies llama makes, pinned.

The descriptors are recorded as (offset, sizes, strides) per channel,
exactly, and the copies spelled as views must issue them.
"""

from typing import Any

import aie.utils as aie_utils
import pytest

from iron.common.tiling import Walk
from iron.operators.copy import Copy, _kv_slot

pytestmark = pytest.mark.usefixtures("npu2")

G, D, L, E, N = 8, 64, 128, 2048, 16  # kv groups, head dim, context, embed, rows

# One token's (G, D) keys into row `pos` of the (G, L, D) cache:
# Copy(k, keys[:, pos]), the row a per-call value.
ROW_INTO_CACHE = dict(
    src=Walk.of((G, D)),
    dst=Walk.slice((G, L, D), (slice(None), 0)),
    input_buffer_size=G * D,
    output_buffer_size=G * L * D,
    num_channels=1,
)
# N tokens' (N, G, D) keys, heads interleaved per token, into the first N
# rows: Copy(k.reshape(N, G, D).transpose(1, 0, 2), keys[:, :N]).
ROWS_INTO_CACHE = dict(
    src=Walk.permuted((N, G, D), (1, 0, 2)),
    dst=Walk.slice((G, L, D), (slice(None), slice(0, N))),
    input_buffer_size=N * G * D,
    output_buffer_size=G * L * D,
    tile_size=1024,
    num_channels=1,
)
# The last prompt row of (4, E), selected by the per-call index `last`:
# Copy(x[last]).
LAST_ROW = dict(
    src=Walk.slice((4, E), (0,)),
    input_buffer_size=4 * E,
    output_buffer_size=E,
    num_channels=1,
)

PINNED: dict[str, tuple[dict[str, Any], list, list]] = {
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
        _kv_slot(128, 5, num_channels=2),
        [[(0, (1, 1, 8, 32), (0, 0, 64, 1))], [(32, (1, 1, 8, 32), (0, 0, 64, 1))]],
        [
            [(320, (1, 1, 8, 32), (0, 0, 8192, 1))],
            [(352, (1, 1, 8, 32), (0, 0, 8192, 1))],
        ],
    ),
}


def _taps(op, buffer, walk):
    return [
        [(a.offset, a.sizes, a.strides) for a in channel]
        for channel in op._taps(buffer, walk)
    ]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_copy_issues_these_descriptors(name):
    kwargs, ins, outs = PINNED[name]
    op = Copy(**kwargs).resolved(aie_utils.get_current_device())
    assert _taps(op, op.x, op.src) == ins
    assert _taps(op, op.y, op.dst) == outs
