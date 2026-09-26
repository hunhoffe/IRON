# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every exported operator is a declared one, or a graph-function factory.

The regression net the arg-spec snapshot used to be: each module imports,
each class is an ``Operator`` against an ``Overlay``, and its arg spec
comes from declared buffers.
"""

import importlib

import pytest

import iron.operators as ops
from iron.common.declare import Operator, Overlay

FACTORIES = {"SwiGLUDecode", "SwiGLUPrefill"}


@pytest.mark.parametrize("name", sorted(ops._OPERATOR_MODULES))
def test_exported_operator_is_declared(name):
    cls = getattr(ops, name)
    if name in FACTORIES:
        assert callable(cls) and not isinstance(cls, type)
        return
    assert isinstance(cls, type) and issubclass(cls, Operator), name
    # One class, or an operator on an overlay of its own: either way an array.
    assert cls._overlay_class is None or issubclass(cls._overlay_class, Overlay), name
    assert [b.name for b in cls._members if hasattr(b, "direction")], name


def test_flm_declares_one_operator_and_its_shipped_form():
    module = importlib.import_module("iron.exports.flm")
    cls, shipped = module.GEMM, module.Shipped
    assert issubclass(cls, Operator) and cls._overlay_class is None
    assert issubclass(shipped, cls) and shipped._external is not None
    assert cls._external is None
