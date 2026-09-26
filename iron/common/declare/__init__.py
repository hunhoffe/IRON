# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator model's declaration layer: overlays, operators, and their members.

An operator's fields sort by what a change rebuilds. Fields that configure the
array (tile shapes, columns, dtypes, kernel flags) live on an :class:`Overlay`;
fields that size the host buffers (extents, batch counts) live on an
:class:`Operator` declared against that overlay; values that change per call
are :class:`Scratchpad` or :class:`DispatchTime` members. Each layer has an
ABI: the overlay's is its **streams** (in tile units), the operator's is its
**buffers** (in extents), and a buffer names the stream it feeds or drains, so
direction, dtype, tile shape and shim binding agree by construction.

Declarations are class-level. A dimension is a dataclass field declared with
:func:`param`, a knob the library resolves is one declared with :func:`auto`, and a shape is
written in the class body using the field's bare name::

    class GEMVOverlay(Overlay):
        K: int = param()
        num_aie_columns: int = auto(8)
        tile_size_output: int = auto(64)

        a = StreamIn(tile_size_output, K, per=num_aie_columns)
        b = StreamIn(K, broadcast=True)
        c = StreamOut(tile_size_output, per=num_aie_columns)

    class GEMV(Operator[GEMVOverlay]):
        M: int = param()
        num_batches: int = param(default=1)

        A = In(optional(num_batches), M, GEMVOverlay.K, to=GEMVOverlay.a)
        B = In(optional(num_batches), GEMVOverlay.K, to=GEMVOverlay.b)
        C = Out(optional(num_batches), M, from_=GEMVOverlay.c)

The shape rule: a host buffer's dimension is a ``param()`` field or an integer
literal, nothing else. Not a tunable, not a per-call value, not an
expression. That is what makes inference a lookup (:mod:`.infer`)
and what lets the checks in :mod:`.creation` run once, as a class body finishes.
A stream's tile dimension may also be a tunable: choosing the tile is what
tuning is for, and inference never reads a stream.

Generating MLIR is :mod:`iron.common.design`'s job, not this package's; it
reads the declarations made here. What little mlir-aie reaches this far --
the device a name is keyed on, the shim's DMA budget -- is a question about
the target, not a design being built.

The package reads bottom-up: :mod:`.field` is what a class body writes,
:mod:`.member` what it declares alongside its fields, :mod:`.bound` what an
instance's attribute gives back, :mod:`.infer` how operand shapes reach a
declaration's dimension fields, :mod:`.overlay` and :mod:`.operator` the two
layers themselves, and :mod:`.creation` the checks both go through as their
bodies finish. :mod:`.naming` is how either one spells its own label.
"""

from .bound import (
    BoundBuffer,
    BoundResident,
    BoundStream,
    BoundValue,
    BufferView,
)
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
from .member import (
    DispatchTime,
    In,
    InOut,
    Out,
    Resident,
    Scratchpad,
    Shim,
    StreamIn,
    StreamOut,
    Value,
    ValueSpec,
    Xclbin,
)
from .operator import OV, Operator
from .overlay import Overlay, get_shim_dma_limit
from .spec import from_spec

__all__ = [
    "BoundBuffer",
    "BoundResident",
    "BoundStream",
    "BoundValue",
    "BufferView",
    "DeclarationError",
    "DimRef",
    "DispatchTime",
    "In",
    "InOut",
    "Incompatible",
    "OV",
    "Operator",
    "Out",
    "Overlay",
    "Resident",
    "Scratchpad",
    "Value",
    "Shim",
    "StreamIn",
    "StreamOut",
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
