#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.operators.strided_copy.op import StridedCopy
from iron.operators.strided_copy.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    params = [
        # (input_sizes, input_strides, input_offset,
        #  output_sizes, output_strides, output_offset,
        #  input_buffer_size, output_buffer_size)

        # Contiguous 32-element copy
        ([32], [1], 0, [32], [1], 0, 32, 32),
        # 2D: 4 rows x 8 cols, contiguous read, contiguous write
        ([4, 8], [8, 1], 0, [4, 8], [8, 1], 0, 32, 32),
        # 2D: 4 rows x 8 cols, strided read (stride-16 rows), contiguous write
        ([4, 8], [16, 1], 0, [4, 8], [8, 1], 0, 64, 32),
        # 2D: contiguous read, strided write (stride-16 rows)
        ([4, 8], [8, 1], 0, [4, 8], [16, 1], 0, 32, 64),
        # With input offset
        ([2, 8], [8, 1], 16, [2, 8], [8, 1], 0, 32, 16),
    ]

    return [
        pytest.param(
            isz, ist, ioff, osz, ost, ooff, ibs, obs,
            id=f"isz{'x'.join(map(str, isz))}_ist{'x'.join(map(str, ist))}_ioff{ioff}"
               f"_osz{'x'.join(map(str, osz))}_ost{'x'.join(map(str, ost))}_ooff{ooff}",
        )
        for isz, ist, ioff, osz, ost, ooff, ibs, obs in params
    ]


@pytest.mark.parametrize(
    "input_sizes,input_strides,input_offset,"
    "output_sizes,output_strides,output_offset,"
    "input_buffer_size,output_buffer_size",
    get_params(),
)
def test_strided_copy(
    input_sizes,
    input_strides,
    input_offset,
    output_sizes,
    output_strides,
    output_offset,
    input_buffer_size,
    output_buffer_size,
    aie_context,
):
    golden_ref = generate_golden_reference(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=input_offset,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=output_offset,
        input_buffer_size=input_buffer_size,
        output_buffer_size=output_buffer_size,
    )

    operator = StridedCopy(
        input_sizes=input_sizes,
        input_strides=input_strides,
        input_offset=input_offset,
        output_sizes=output_sizes,
        output_strides=output_strides,
        output_offset=output_offset,
        input_buffer_size=input_buffer_size,
        output_buffer_size=output_buffer_size,
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
