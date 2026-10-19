# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GELU's two forms differ below -3, where the tanh form's bf16 output is
zero and ``x * Phi(x)`` is not; elsewhere they agree to within a few ulps.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.operators.gelu import GELU


def gelu(approximate: str) -> GELU:
    return GELU(size=1024, num_aie_columns=1, tile_size=1024, approximate=approximate)


def test_the_exact_form_is_not_the_tanh_form_below_minus_three():
    x = np.resize(np.array([-5.0, -4.0, -3.5], bfloat16), 1024)
    exact = np.asarray(gelu("none").reference(x), np.float32)[:3]
    tanh = np.asarray(gelu("tanh").reference(x), np.float32)[:3]
    assert np.all(exact < 0) and np.all(tanh == 0), (exact, tanh)


def test_an_unknown_approximation_is_refused():
    with pytest.raises(ValueError, match="approximate"):
        gelu("sigmoid")
