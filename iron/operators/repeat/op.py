# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict
from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class Repeat(MLIROperator):
    """AIE-accelerated repeat-interleave operator"""

    rows: int
    cols: int
    repeat: int
    transfer_size: int | None = None
    dtype: object = field(default=bfloat16, repr=False)
    num_invocations: int = 1
    input_fusion_group: str | None = None
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "repeat": "by",
        "transfer_size": "ts",
        "input_fusion_group": "ifg",
    }

    def __post_init__(self):
        if self.num_invocations < 1:
            raise ValueError(
                f"num_invocations must be >= 1, got {self.num_invocations}"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        callback_kwargs = {}
        if self.input_fusion_group is not None:
            callback_kwargs["input_fusion_group"] = self.input_fusion_group
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "repeat",
                (
                    aie_utils.get_current_device(),
                    self.dtype,
                    self.rows,
                    self.cols,
                    self.repeat,
                    self.transfer_size,
                    self.num_invocations,
                ),
                callback_kwargs,
            ),
        )

    def get_kernel_artifacts(self):
        return []

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.rows, self.cols)),
            AIERuntimeArgSpec("out", (self.rows * self.repeat, self.cols)),
        ]
