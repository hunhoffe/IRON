# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.operators.flm import ProjectionGEMM


def _exact_b(K, N):
    # Integers, each bfp16 block (8 k of one n) holding 100: exact in bf16
    # and in bfp16.
    B = np.random.default_rng(0).integers(-127, 128, (K, N)).astype(np.float32)
    B[::8] = 100
    return B


@pytest.mark.parametrize("name", ["npu1", "npu2"])
def test_a_packed_b_gives_the_reference_its_values(name, request):
    dev = request.getfixturevalue(name)
    op = ProjectionGEMM(M=256, K=1024, N=128).resolved(dev)
    A = np.random.default_rng(1).standard_normal((256, 1024)).astype(bfloat16)
    B = _exact_b(1024, 128)
    packed = op.pack_B(B)
    assert packed.ndim == 1
    np.testing.assert_array_equal(op.reference(A, packed), op.reference(A, B))
    bound = op.tolerance().bound
    np.testing.assert_array_equal(bound(A, packed), bound(A, B))
