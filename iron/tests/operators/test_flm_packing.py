# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from iron.operators.flm.dequant.op import CT_K, K_TILE_B, N_TILE, S, T
from iron.operators.flm.packing import pack_b, unpack_b


def test_unpacking_a_packed_b_gives_back_its_values():
    # Integers, each bfp16 block (8 k of one n) holding 100: exact in bfp16.
    B = np.random.default_rng(0).integers(-127, 128, (1024, 128)).astype(np.float32)
    B[::8] = 100
    tiling = dict(k_tile=K_TILE_B, n_tile=N_TILE, s=S, t=T, ct_k=CT_K)
    packed = pack_b(B, **tiling, bfp16=True)
    np.testing.assert_array_equal(unpack_b(packed, 1024, 128, **tiling), B)
