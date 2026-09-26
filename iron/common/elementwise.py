# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared elementwise design: N flat buffers in, one of the same size out.

One array serves every elementwise kernel IRON ships. It places one core per
(column, channel), each streaming fixed-size lines in and out; an operator
declares a flat buffer per stream, the line as its tile, and its runtime
sequence is derived -- the buffer is split evenly across the cores' fifos
and drained back the same way.

:class:`UnaryElementwise` and :class:`BinaryElementwise` are the two operand
shapes, and nothing more: the array reads the operands it was declared
with, so an operator with a third input needs no new code here.

The core's trip count is a :class:`~iron.common.declare.Value` the sequence
writes before the first transfer, so the array does not depend on the
extent and one array serves every size (OPERATOR_MODEL_PLAN.md §3). This is
where the template parts company with upstream's
:func:`aie.iron.algorithms.transform_parallel`, which is otherwise the same
design: that one takes the tensor at build time and folds the trip count
into the core program, and owns the runtime sequence so it can issue the
taps. An array here returns workers and leaves the sequence to the
library, which is what lets several operators fuse into one image.

A concrete operator is one small subclass, naming the kernel each core
calls::

    class ReLU(UnaryElementwise):
        def kernel(self, target):
            return eltwise.relu_sized(self.tile_size)

        def reference(self, x):
            return np.maximum(x, 0)

:mod:`aie.iron.kernels` is where a kernel comes from: its factories return
the ``ExternalFunction`` for a symbol, its source and its argument types,
and handle aie2's LUT tables. An operator whose kernel takes more than the
line length (leaky_relu's alpha) or takes its arguments in another order
(axpy's scalar) overrides :meth:`Elementwise.kernel_call`; what it reads
there is declared ``param(..., array=True)``, since the array bakes it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from aie.iron import ObjectFifo, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.verify import Tolerance

from .declare import In, Incompatible, Operator, Out, Unresolvable, Value, auto, param
from .tiling import fifo_depth

if TYPE_CHECKING:
    from .design.target import Target

# The line an elementwise core streams when nothing else is asked for: small
# enough to divide any extent a model has, at some cost in DMA efficiency.
# Call sites that know their extent pass tile_size for performance.
DEFAULT_TILE = 256

_I32 = np.ndarray[(1,), np.dtype[np.int32]]  # type: ignore[misc]


class Elementwise(Operator):
    """The array for an elementwise kernel over lines of ``tile_size`` elements.

    Subclasses declare the operands with the line as their tile, one lane
    per (column, channel) (see the two below), and implement :meth:`kernel`.
    ``tile_cap`` is the largest line the kernel holds, so a larger tile is
    refused rather than split; a line spanning more than one local-memory
    bank drops the fifo depth to one.
    """

    # None: the most columns the device's shim budget allows that leave
    # every core whole lines, one channel each, default_tile lines.
    num_aie_columns: int = auto()
    num_channels: int = auto(1)
    tile_size: int = auto()

    # The lines each core processes: written once per build, before the
    # first transfer, so the array does not depend on the extent.
    count = Value(np.int32, derive=lambda op: op.lines // op.cores)

    default_tile: ClassVar[int] = DEFAULT_TILE
    tile_cap: ClassVar[int] = 4096

    def resolve(self, dev) -> Self:
        tile_size = self.default_tile if self.tile_size is None else self.tile_size
        if tile_size > self.tile_cap:
            raise Unresolvable(
                f"tile_size={tile_size} exceeds the {self.tile_cap}-element line "
                f"one core holds ({type(self).__name__}.tile_cap)"
            )
        (out,) = self.outputs
        cols = self.resolve_columns(
            dev,
            self.num_aie_columns,
            self.num_channels,
            fits=lambda c: out.elements % (c * self.num_channels * tile_size) == 0,
        )
        return dataclasses.replace(self, num_aie_columns=cols, tile_size=tile_size)

    def compatible(self) -> None:
        (out,) = self.outputs
        share = self.cores * self.tile_size
        if out.elements % share:
            raise Incompatible(
                f"{type(self).__name__}: {out.elements} elements do not divide "
                f"into whole {self.tile_size}-element lines over "
                f"{self.num_aie_columns} columns x {self.num_channels} channels "
                f"({share} per pass); give a tile_size= or num_aie_columns= "
                f"that divides it"
            )

    @property
    def cores(self) -> int:
        return self.num_aie_columns * self.num_channels

    @property
    def lines(self) -> int:
        """How many lines the operands hold; each core streams an equal share."""
        (out,) = self.outputs
        return out.elements // self.tile_size

    # -- the kernel --------------------------------------------------------

    def kernel(self, target: Target) -> ExternalFunction:
        """The ``ExternalFunction`` each core calls, over one line.

        Usually a factory from :mod:`aie.iron.kernels` at ``self.tile_size``;
        ``target.kernel(...)`` declares one upstream does not offer.
        """
        raise NotImplementedError(f"{type(self).__name__} declares no kernel()")

    def tolerance(self, target: Target) -> Tolerance | None:
        """The contract of the one kernel every core runs; ``None`` for a
        kernel declared without one.
        """
        contract = self.kernel(target).contract
        return None if contract is None else contract.tolerance

    def kernel_call(self, kernel, *elements) -> None:
        """Call the kernel on this core's acquired elements: inputs, then the
        output, then the line length.
        """
        kernel(*elements, self.tile_size)

    # -- the array ----------------------------------------------------------

    def array(self, target) -> list:
        streams = list(self.streams.values())
        ins = [s for s in streams if s.direction == "in"]
        outs = [s for s in streams if s.direction == "out"]
        n_in = len(ins)
        cores = self.cores
        kernel = self.kernel(target)

        def slot(k: int) -> str:
            col, chan = divmod(k, self.num_channels)
            return f"{col}" if self.num_channels == 1 else f"{col}_{chan}"

        def fifos(stream, name):
            # A line spanning more than one bank cannot be double-buffered in
            # what is left of local memory.
            depth = fifo_depth(stream.elements, stream.dtype)
            return [
                ObjectFifo(stream.tile, name=f"{name}_{slot(k)}", depth=depth)
                for k in range(cores)
            ]

        of_ins = [fifos(s, f"in{i}") for i, s in enumerate(ins)]
        of_outs = [
            fifos(s, f"out{i}" if len(outs) > 1 else "out") for i, s in enumerate(outs)
        ]
        counts = [target.rtp(_I32, name=f"count_{slot(k)}") for k in range(cores)]
        barriers = [target.barrier() for _ in range(cores)]

        def core_fn(*args):
            fifos_in = args[:n_in]
            fifos_out = args[n_in : n_in + len(outs)]
            kernel_fn, count, barrier = args[-3:]
            barrier.wait_for_value(1)
            for _ in range_(count[0]):
                elements = [f.acquire(1) for f in fifos_in + fifos_out]
                self.kernel_call(kernel_fn, *elements)
                for f in fifos_in + fifos_out:
                    f.release(1)

        workers = [
            Worker(
                core_fn,
                [of[k].cons() for of in of_ins]
                + [of[k].prod() for of in of_outs]
                + [kernel, counts[k], barriers[k]],
            )
            for k in range(cores)
        ]
        for k in range(cores):
            for stream, of in zip(ins, of_ins):
                stream[k].bind(of[k].prod())
            for stream, of in zip(outs, of_outs):
                stream[k].bind(of[k].cons())
        self.count.bind(counts)
        return workers


# --------------------------------------------------------------------------
# The two operand shapes
# --------------------------------------------------------------------------


class UnaryElementwise(Elementwise):
    """A flat buffer in, a flat buffer of the same size out."""

    size: int = param()

    x = In(
        size,
        tile=(Elementwise.tile_size,),
        per=(Elementwise.num_aie_columns, Elementwise.num_channels),
    )
    y = Out(
        size,
        tile=(Elementwise.tile_size,),
        per=(Elementwise.num_aie_columns, Elementwise.num_channels),
    )


class BinaryElementwise(Elementwise):
    """Two flat buffers in, one of the same size out. Each core's two input
    channels halve the columns the shim budget allows, so ``num_channels``
    stays at one.
    """

    size: int = param()

    a = In(
        size,
        tile=(Elementwise.tile_size,),
        per=(Elementwise.num_aie_columns, Elementwise.num_channels),
    )
    b = In(
        size,
        tile=(Elementwise.tile_size,),
        per=(Elementwise.num_aie_columns, Elementwise.num_channels),
    )
    y = Out(
        size,
        tile=(Elementwise.tile_size,),
        per=(Elementwise.num_aie_columns, Elementwise.num_channels),
    )
