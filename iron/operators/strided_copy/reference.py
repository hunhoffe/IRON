# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from ml_dtypes import bfloat16
from itertools import product


def generate_golden_reference(
    input_sizes,
    input_strides,
    input_offset,
    output_sizes,
    output_strides,
    output_offset,
    input_buffer_size,
    output_buffer_size,
    dtype=bfloat16,
    seed=42,
):
    torch.manual_seed(seed)

    val_range = 4
    input_flat = torch.rand(input_buffer_size, dtype=torch.bfloat16) * val_range
    output_flat = torch.zeros(output_buffer_size, dtype=torch.bfloat16)

    # Enumerate all multi-dimensional indices from the sizes
    ranges = [range(s) for s in input_sizes]
    for idx in product(*ranges):
        src = input_offset + sum(i * s for i, s in zip(idx, input_strides))
        dst = output_offset + sum(i * s for i, s in zip(idx, output_strides))
        output_flat[dst] = input_flat[src]

    return {
        "input": input_flat,
        "output": output_flat,
    }
