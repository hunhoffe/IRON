# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from ml_dtypes import bfloat16


def generate_golden_reference(rows, cols, repeat, transfer_size=None, dtype=bfloat16, seed=42):
    torch.manual_seed(seed)

    val_range = 4
    input_data = torch.rand(rows, cols, dtype=torch.bfloat16) * val_range
    output_data = torch.repeat_interleave(input_data, repeats=repeat, dim=0)

    return {
        "input": input_data.flatten(),
        "output": output_data.flatten(),
    }
