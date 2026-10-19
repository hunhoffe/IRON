# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MHA's ``window`` counts the positions a query attends to before its own:
the key ``window`` back is attended and the one past it is not. Every
logit is 0, so each attended key weighs ``1 / (window + 1)``, and a
one-hot value per key shows which were.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.operators.mha import MHA

pytestmark = pytest.mark.usefixtures("npu2")


@pytest.mark.parametrize("window", [7, 8, 63])
def test_a_window_attends_the_key_window_back_and_not_the_one_before(window):
    n, d = 64, 64
    q = np.zeros((1, n, d), bfloat16)
    k = np.zeros((1, n, d), bfloat16)
    v = np.zeros((1, n, d), bfloat16)
    last = n - 1
    v[0, last - window, 0] = 1
    if window < last:
        v[0, last - window - 1, 1] = 1
    mha = MHA(num_heads=1, num_KV_heads=1, seq_len=n, d=d, window=window)
    out = np.asarray(mha.reference(q, k, v), np.float32)[0, last]
    assert out[0] == pytest.approx(1 / (window + 1), rel=1e-2)
    assert out[1] == 0
