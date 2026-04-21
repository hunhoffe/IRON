#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.operators.repeat.op import Repeat
from iron.operators.repeat.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    configs = [
        # (rows, cols, repeat)
        (4, 64, 2),
        (4, 64, 4),
        (8, 32, 2),
        (8, 32, 4),
        (2, 128, 3),
    ]

    return [
        pytest.param(rows, cols, repeat, id=f"r{rows}_c{cols}_x{repeat}")
        for rows, cols, repeat in configs
    ]


@pytest.mark.parametrize("rows,cols,repeat", get_params())
def test_repeat(rows, cols, repeat, aie_context):
    golden_ref = generate_golden_reference(rows=rows, cols=cols, repeat=repeat)

    operator = Repeat(
        rows=rows,
        cols=cols,
        repeat=repeat,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.01, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
