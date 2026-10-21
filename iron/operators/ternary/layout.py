# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ternary weights as mlir-aie's ternary kernels read them.

Each weight is -1, 0 or +1 times a bf16 scale shared by GROUP in-features of
an out-feature, 2.125 bits a weight. A block holds M_TILE out-features by
K_TILE in-features (``aie.iron.kernels.quant.ternary_dequant_ref`` documents
its bytes). A whole weight is ``(out, in * 17 // 64)`` bytes: each M_TILE
rows hold one run of out-features' blocks, k-tile by k-tile.
"""

import numpy as np
from aie.iron.kernels.quant import TERNARY_GROUP as GROUP
from aie.iron.kernels.quant import ternary_dequant_ref
from ml_dtypes import bfloat16

M_TILE, K_TILE = 32, 256
BLOCK_BYTES = M_TILE * K_TILE // 4 + 2 * M_TILE * K_TILE // GROUP


def quantize(w):
    """``w`` ``(out, in)`` as ternary bytes ``(out, in * 17 // 64)``.

    Per GROUP in-features of an out-feature, each code is the weight over
    the group's mean magnitude, rounded and clipped to -1..1 (BitNet
    b1.58's rounding), and the scale is the mean magnitude of the weights
    whose code is not 0, the least-squares fit. A weight already ternary
    is held exactly, up to its scale's bf16 rounding.

    Raises:
        ValueError: ``out`` is not a multiple of M_TILE or ``in`` of
            K_TILE, the block.
    """
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if N % M_TILE or K % K_TILE:
        raise ValueError(
            f"ternary: a weight ({N}, {K}) must tile to ({M_TILE}, {K_TILE})"
        )
    g = w.reshape(N, K // GROUP, GROUP)
    size = np.abs(g)
    mean = size.mean(axis=2, keepdims=True)
    codes = np.divide(g, mean, out=np.zeros_like(g), where=mean > 0)
    codes = np.clip(np.rint(codes), -1, 1)
    count = np.count_nonzero(codes, axis=2)
    total = (size * (codes != 0)).sum(axis=2)
    scales = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    # 2-bit codes, 1 for +1 and 2 for -1; byte i of a group holds its
    # columns i, i + 32, i + 64 and i + 96 at bits 0, 2, 4 and 6.
    codes = np.where(codes < 0, 2, codes).astype(np.uint8).reshape(N, -1, 4, 32)
    packed = np.bitwise_or.reduce(
        codes << np.arange(0, 8, 2, dtype=np.uint8)[:, None], 2
    )
    nb_r, nb_c = N // M_TILE, K // K_TILE
    parts = [
        np.ascontiguousarray(p.reshape(nb_r, M_TILE, nb_c, -1).transpose(0, 2, 1, 3))
        .view(np.uint8)
        .reshape(nb_r, nb_c, -1)
        for p in (packed, scales.astype(bfloat16)[..., None])
    ]
    return np.concatenate(parts, axis=2).reshape(N, -1)


def dequantize(weight):
    """The ``(out, in)`` float32 weights of ternary bytes ``(out, in * 17 // 64)``."""
    weight = np.asarray(weight, np.uint8)
    N = weight.shape[0]
    blocks = weight.reshape(N // M_TILE, -1, BLOCK_BYTES)
    w = ternary_dequant_ref(blocks, m_tile=M_TILE, k_tile=K_TILE)
    return w.transpose(0, 2, 1, 3).reshape(N, -1)
