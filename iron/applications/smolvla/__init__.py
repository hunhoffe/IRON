# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA on SmolVLM's vision tower and text layers; see ``model.py``."""

from iron.lm.cli import exports

__all__ = ["SmolVLA"]
__getattr__ = exports(__name__, __all__)
