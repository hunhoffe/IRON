# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aie.iron.kernels import eltwise

from iron.common import BinaryElementwise
from iron.common.testing import Testing, binary_elementwise_cases


class ElementwiseAdd(BinaryElementwise):
    """AIE-accelerated element-wise addition"""

    test = Testing(binary_elementwise_cases([1024, 2048, 4096, 8192]))

    def kernel(self, target):
        return eltwise.add_sized(self.line_size)

    def reference(self, a, b):
        return a + b
