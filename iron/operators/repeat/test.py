#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.operators.repeat.op import Repeat
from iron.operators.repeat.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    # cols must be > 1023 so that design.py computes cols_split >= 2,
    # ensuring the innermost BD dimension is >= 2 elements (4 bytes for bf16).
    configs = [
        # (rows, cols, repeat, transfer_size)
        (4, 1024, 2, 64),
        (4, 1024, 4, 64),
        (8, 1024, 2, 64),
        (8, 1024, 4, 64),
        (2, 1024, 3, 64),
    ]

    return [
        pytest.param(
            rows, cols, repeat, transfer_size,
            id=f"r{rows}_c{cols}_x{repeat}_ts{transfer_size}",
        )
        for rows, cols, repeat, transfer_size in configs
    ]


@pytest.mark.parametrize("rows,cols,repeat,transfer_size", get_params())
def test_repeat(rows, cols, repeat, transfer_size, aie_context):
    golden_ref = generate_golden_reference(rows=rows, cols=cols, repeat=repeat)

    operator = Repeat(
        rows=rows,
        cols=cols,
        repeat=repeat,
        transfer_size=transfer_size,
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
