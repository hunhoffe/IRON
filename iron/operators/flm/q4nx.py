# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The q4nx weight format that FastFlowLM quantizes Gemma 4 into.

A block holds M_TILE out-features by K_TILE in-features: bf16 scales, then
bf16 minima, one each per GROUP in-features of an out-feature, then 4-bit
codes. A weight is ``min + scale * code``.

A whole weight is held as ``flm.DequantBFP`` reads it (its README's layers
6 to 9): ``(out, in * 5 // 8)`` bytes, each 64 out-features one run of rows.
"""

import numpy as np
from ml_dtypes import bfloat16

M_TILE, K_TILE, GROUP = 32, 256, 32

BITS_PER_WEIGHT = 4 + 2 * 16 // GROUP
BLOCK_BYTES = M_TILE * K_TILE * BITS_PER_WEIGHT // 8


def packed_bytes(n_weights: int) -> int:
    """Bytes holding n_weights in q4nx packing."""
    return n_weights * BITS_PER_WEIGHT // 8


def unpack(grid):
    """A weight's codes ``(rows, cols)`` uint8 and its scales and minima
    ``(rows, cols // GROUP)`` float32, from its blocks laid out as a
    ``(rows // M_TILE, cols // K_TILE)`` grid.

    A block stores the scales, then the minima, at ``[k // GROUP, m]`` and
    the codes at ``[m // 16, k, m % 16]``, the lower index in a byte's low
    nibble.
    """
    grid = np.asarray(grid, np.uint8)
    nb_r, nb_c = grid.shape[:2]
    b = grid.reshape(-1, BLOCK_BYTES)
    groups = K_TILE // GROUP
    params = b[:, : 4 * groups * M_TILE].copy().view(bfloat16).astype(np.float32)
    scales, mins = params.reshape(-1, 2, groups, M_TILE).transpose(1, 0, 3, 2)
    packed = b[:, 4 * groups * M_TILE :]
    codes = np.empty((len(b), M_TILE * K_TILE), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    codes = codes.reshape(-1, M_TILE // 16, K_TILE, 16).transpose(0, 1, 3, 2)
    return tuple(
        part.reshape(nb_r, nb_c, M_TILE, -1)
        .transpose(0, 2, 1, 3)
        .reshape(nb_r * M_TILE, -1)
        for part in (codes, scales, mins)
    )


def dequantize(grid):
    """The ``(rows, cols)`` float64 weights of a grid of blocks, as ``unpack``'s."""
    codes, scales, mins = unpack(grid)
    group = np.arange(codes.shape[1]) // GROUP
    w = scales.astype(np.float64)[:, group]
    w *= codes
    w += mins[:, group]
    return w


def quantize(w):
    """``w`` ``(out, in)`` as q4nx bytes ``(out, in * 5 // 8)``: per GROUP
    in-features of an out-feature, the bf16 minimum and a bf16 scale
    spanning the group in 15 steps, and each weight's nearest code.

    Raises:
        ValueError: ``out`` is not a multiple of 64 or ``in`` of 512, the
            blocks of the layout.
    """
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if N % (2 * M_TILE) or K % (2 * K_TILE):
        raise ValueError(
            f"q4nx: a weight ({N}, {K}) must tile to ({2 * M_TILE}, {2 * K_TILE})"
        )
    g = w.reshape(N, K // GROUP, GROUP)
    mins = g.min(axis=2).astype(bfloat16)
    low = mins.astype(np.float32)[..., None]
    scales = ((g.max(axis=2) - low[..., 0]) / 15).astype(bfloat16)
    step = scales.astype(np.float32)[..., None]
    codes = np.divide(g - low, step, out=np.zeros_like(g), where=step > 0)
    codes = np.clip(np.rint(codes), 0, 15).astype(np.uint8)
    # Per block: scales and minima at [k // GROUP, m], codes at
    # [m // 16, k, m % 16], two to a byte, the lower index low.
    nb_r, nb_c, groups = N // M_TILE, K // K_TILE, K_TILE // GROUP
    params = [
        np.ascontiguousarray(
            p.reshape(nb_r, M_TILE, nb_c, groups).transpose(0, 2, 3, 1)
        )
        .reshape(nb_r, nb_c, -1)
        .view(np.uint8)
        for p in (scales, mins)
    ]
    codes = codes.reshape(nb_r, M_TILE // 16, 16, nb_c, K_TILE).transpose(0, 3, 1, 4, 2)
    codes = codes.reshape(nb_r, nb_c, -1)
    grid = np.concatenate([*params, codes[..., 0::2] | codes[..., 1::2] << 4], axis=2)
    # By column block of 64 out-features: k-tile pairs, k-tiles, then the
    # two 32-out-feature halves.
    grid = grid.reshape(N // 64, 2, K // (2 * K_TILE), 2, BLOCK_BYTES)
    return grid.transpose(0, 2, 3, 1, 4).reshape(N, -1)


def grid(weight):
    """The blocks of q4nx bytes ``(out, in * 5 // 8)`` as ``unpack``'s grid."""
    weight = np.asarray(weight, np.uint8)
    N, K = weight.shape[0], weight.shape[1] * 8 // BITS_PER_WEIGHT
    blocks = weight.reshape(N // 64, K // (2 * K_TILE), 2, 2, BLOCK_BYTES)
    return blocks.transpose(0, 3, 1, 2, 4).reshape(N // M_TILE, K // K_TILE, -1)
