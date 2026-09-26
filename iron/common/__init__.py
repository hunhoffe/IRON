# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities and base classes for IRON operators."""

# First: image.artifacts before design, or design's import of image completes
# a cycle back into design before DesignGenerator exists.
from .image.artifacts import Artifacts, Design, Step  # isort: skip

from .declare import (
    DeclarationError,
    DispatchTime,
    In,
    Incompatible,
    InOut,
    Operator,
    Out,
    Overlay,
    Resident,
    Scratchpad,
    Shim,
    StreamIn,
    StreamOut,
    Untunable,
    Xclbin,
    dim,
    from_spec,
    optional,
    select,
    tunable,
)
from .design import DesignGenerator
from .elementwise import (
    BinaryElementwiseOperator,
    BinaryElementwiseOverlay,
    ChanneledUnaryOperator,
    ChanneledUnaryOverlay,
    ElementwiseOperator,
    ElementwiseOverlay,
)

__all__ = [
    "Artifacts",
    "BinaryElementwiseOperator",
    "BinaryElementwiseOverlay",
    "ChanneledUnaryOperator",
    "ChanneledUnaryOverlay",
    "DeclarationError",
    "Design",
    "DesignGenerator",
    "DispatchTime",
    "ElementwiseOperator",
    "ElementwiseOverlay",
    "In",
    "InOut",
    "Incompatible",
    "Operator",
    "Out",
    "Overlay",
    "Resident",
    "Scratchpad",
    "Shim",
    "Step",
    "StreamIn",
    "StreamOut",
    "Untunable",
    "Xclbin",
    "dim",
    "from_spec",
    "optional",
    "select",
    "tunable",
]
