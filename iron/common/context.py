# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
import os

from . import compilation as comp
import aie.utils.config


@dataclass
class AIEContext:
    """Context for managing AIE operator compilation state.

    Attributes:
        base_dir: Repository root directory (three levels above this file).
        build_dir: Directory where compiled artifacts are written.
        mlir_verbose: Enable verbose MLIR output during compilation.
    """

    # Repo root: iron/common/../../.. = three levels up from this file.
    base_dir: ClassVar[Path] = Path(__file__).parent.parent.parent

    build_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "build")
    mlir_verbose: bool = False
    use_conduit: bool = False
    conduit_fusion_passes: list[str] = field(default_factory=list)
    bank_aware_placement: bool = True
    mlir_aie_install_dir: Path | None = None

    def __post_init__(self) -> None:
        """Normalize build_dir to a Path object."""
        self.build_dir = Path(self.build_dir)

    @property
    def compilation_rules(self):
        """Return the ordered list of compilation rules for this context.

        Returns:
            List of ``CompilationRule`` instances configured for the current
            mlir-aie and peano installation paths.
        """
        mlir_aie_dir = Path(self.mlir_aie_install_dir) if self.mlir_aie_install_dir else Path(aie.utils.config.root_path())
        peano_dir = Path(aie.utils.config.peano_install_dir())
        return [
            comp.FusePythonGeneratedMLIRCompilationRule(),
            comp.GenerateMLIRFromPythonCompilationRule(),
            comp.PeanoCompilationRule(peano_dir, mlir_aie_dir),
            comp.ArchiveCompilationRule(peano_dir, mlir_aie_dir),
            comp.AieccXclbinInstsCompilationRule(
                self.build_dir, peano_dir, mlir_aie_dir,
                use_conduit=self.use_conduit,
                conduit_fusion_passes=self.conduit_fusion_passes,
                bank_aware_placement=self.bank_aware_placement,
            ),
            comp.AieccFullElfCompilationRule(
                self.build_dir, peano_dir, mlir_aie_dir,
                use_conduit=self.use_conduit,
                conduit_fusion_passes=self.conduit_fusion_passes,
                bank_aware_placement=self.bank_aware_placement,
            ),
        ]
