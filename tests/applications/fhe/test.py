# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encrypted dot products with the NTTs' int8 products on the NPU: exactly
numpy's.
"""

import numpy as np
import pytest

from iron.applications import fhe


@pytest.mark.supported_devices("npu1", "npu2")
def test_encrypted_dot_products_on_the_npu_are_numpys():
    rng = np.random.default_rng(0)
    docs, x = rng.integers(-128, 128, (1024, 128)), rng.integers(-128, 128, 128)
    key = fhe.BFV().keygen()
    assert (key.decrypt(docs @ key.encrypt(x)) == docs @ x).all()
