# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator model's declaration layer: operators and their members.

An operator is one class. Its fields sort by when a change rebuilds: the
array tier is what an operand's tile names, plus what says ``array=True``,
and one array serves every extent; the other fields size the host buffers
and reach only the runtime sequence. Values a graph binds per call are
:class:`Value`, :class:`Scratchpad` or :class:`DispatchTime` members.

Declarations are class-level. A dimension is a dataclass field declared with
:func:`param`, a knob the library resolves is one declared with :func:`auto`,
and a shape is written in the class body using the field's bare name; an
operand with ``tile=`` is its own stream into the array::

    class GEMV(Operator):
        M: int = param()
        K: int = param()
        num_batches: int = param(default=1)
        num_aie_columns: int = auto()
        tile_size_output: int = auto(64)

        A = In(optional(num_batches), M, K, tile=(tile_size_output, K), per=(num_aie_columns,))
        B = In(optional(num_batches), K, tile=(K,), per=(num_aie_columns,))
        C = Out(optional(num_batches), M, tile=(tile_size_output,), per=(num_aie_columns,))
        tiles = Value(np.int32, derive=lambda op: op.M // (op.num_aie_columns * op.tile_size_output))

The shape rule: a host buffer's dimension is a ``param()`` field or an
integer literal, nothing else. Not a knob, not a per-call value, not an
expression. That is what makes inference a lookup (:mod:`.infer`) and what
lets the checks in :mod:`.creation` run once, as a class body finishes. A
tile's dimension may also be a knob: choosing the tile is what resolution
is for, and inference never reads a tile.

Generating MLIR is :mod:`iron.common.design`'s job, not this package's; it
reads the declarations made here. What little mlir-aie reaches this far --
the device a name is keyed on, the shim's DMA budget (:mod:`.shim`) -- is a
question about the target, not a design being built.

The package reads bottom-up: :mod:`.field` is what a class body writes,
:mod:`.member` what it declares alongside its fields, :mod:`.bound` what an
instance's attribute gives back, :mod:`.infer` how operand shapes reach a
declaration's dimension fields, :mod:`.operator` the class itself, and
:mod:`.creation` the checks it goes through as its body finishes.
:mod:`.naming` is how an instance spells its own label.
"""

from .bound import BoundBuffer, BoundStream, BoundValue, BufferView
from .field import (
    DeclarationError,
    DimRef,
    Incompatible,
    Unresolvable,
    auto,
    optional,
    param,
    select,
)
from .infer import infer, infer_kwargs
from .member import DispatchTime, In, Out, Scratchpad, Shim, Value, ValueSpec, Xclbin
from .operator import Operator
from .shim import get_shim_dma_limit
from .spec import from_spec

__all__ = [
    "BoundBuffer",
    "BoundStream",
    "BoundValue",
    "BufferView",
    "DeclarationError",
    "DimRef",
    "DispatchTime",
    "In",
    "Incompatible",
    "Operator",
    "Out",
    "Scratchpad",
    "Value",
    "Shim",
    "Unresolvable",
    "ValueSpec",
    "Xclbin",
    "param",
    "from_spec",
    "get_shim_dma_limit",
    "infer",
    "infer_kwargs",
    "optional",
    "select",
    "auto",
]
