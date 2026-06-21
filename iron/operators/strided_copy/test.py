#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from ml_dtypes import bfloat16

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.operators.strided_copy.op import StridedCopy
from iron.operators.strided_copy.reference import generate_golden_reference
from iron.common.test_utils import verify_buffer


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


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
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

    # StridedCopy.get_arg_spec() passes scalar ints to AIERuntimeArgSpec.shape,
    # which causes run_test to allocate a 1-element output XRTTensor instead of
    # the correct size. Manually create XRTTensors with proper tuple shapes.
    operator.compile()
    op_func = operator.get_callable()

    input_buf = XRTTensor.from_torch(golden_ref["input"])
    output_buf = XRTTensor((output_buffer_size,), dtype=bfloat16)

    # Warmup
    op_func(input_buf, output_buf)

    # Timed run
    start = time.perf_counter()
    op_func(input_buf, output_buf)
    elapsed_us = (time.perf_counter() - start) * 1e6

    total_bytes = input_buf.buffer_object().size() + output_buf.buffer_object().size()
    bandwidth_gbps = total_bytes / (elapsed_us * 1e-6) / 1e9

    print(f"\nLatency (us): {elapsed_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    output_torch = output_buf.to_torch()
    errors = verify_buffer(
        output_torch, "output", golden_ref["output"], rel_tol=0.01, abs_tol=1e-6
    )
    assert not errors, f"Test failed with errors: {errors}"
