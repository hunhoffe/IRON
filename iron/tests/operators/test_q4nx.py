# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from iron.operators.flm import q4nx


def test_a_quantized_weight_is_within_half_a_step_of_its_group():
    w = np.random.default_rng(0).standard_normal((128, 1024)).astype(np.float32)
    w[:64, :32] = 0.5  # a constant group: a scale of 0
    got = q4nx.dequantize(q4nx.grid(q4nx.quantize(w)))
    g = w.reshape(128, -1, q4nx.GROUP)
    step = (g.max(axis=2) - g.min(axis=2)) / 15
    # Half a step, and the bf16 roundings of the minimum and the scale.
    bound = step / 2 + 2**-8 * (np.abs(g).max(axis=2) + 15 * step)
    assert np.all(np.abs(got - w).reshape(g.shape) <= bound[..., None])


def test_a_run_of_64_out_features_is_its_own_rows():
    w = np.random.default_rng(1).standard_normal((192, 512)).astype(np.float32)
    whole = q4nx.quantize(w)
    assert whole.shape == (192, 512 * 5 // 8)
    np.testing.assert_array_equal(whole[64:128], q4nx.quantize(w[64:128]))


def test_a_weight_off_the_block_grid_is_refused():
    with pytest.raises(ValueError, match=r"\(96, 512\) must tile to \(64, 512\)"):
        q4nx.quantize(np.zeros((96, 512)))
