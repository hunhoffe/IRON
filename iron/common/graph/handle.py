# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a traced graph passes around in place of a buffer."""

from __future__ import annotations

from math import prod

import numpy as np
from aie.utils import bfp
from ml_dtypes import bfloat16

from ..declare import Operator, Overlay
from ..tiling import Walk


class Handle:
    """A traced tensor: a buffer of the graph, with a shape and a dtype.

    Carries no data. ``h[key]`` takes numpy's basic indexing (integers,
    unit-step slices, an ellipsis) and, on one axis, a per-call
    :class:`Value`: ``keys[:, pos]``. A contiguous static region is a slice:
    part of the parent's buffer, which any operator takes. Any other view
    (a strided region, a transpose, a per-call index) is a walk over the
    parent's buffer, which only a copy takes, since a DMA walks it.
    """

    __slots__ = (
        "shape",
        "dtype",
        "name",
        "role",
        "parent",
        "start",
        "walk",
        "index_by",
    )

    def __init__(
        self, shape, dtype, name, role, parent=None, start=0, walk=None, index_by=None
    ):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.name = name
        # input | output | weight | state | intermediate | slice | view
        self.role = role
        self.parent = parent
        self.start = start  # element offset into the parent, for a slice
        self.walk: Walk | None = walk  # over the parent's buffer, for a view
        self.index_by: tuple[Value, int] | None = index_by  # (value, axis stride)

    @property
    def elements(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.elements * bfp.itemsize(self.dtype)

    @property
    def buffer_name(self) -> str:
        """The name the runlist uses: a slice is ``parent[start:stop]`` in bytes;
        a view is walked over its parent's buffer, so it is the parent's.
        """
        if self.parent is None:
            return self.name
        if self.walk is not None:
            return self.parent.buffer_name
        item = bfp.itemsize(self.dtype)
        return f"{self.parent.buffer_name}[{self.start * item}:{(self.start + self.elements) * item}]"

    def reshape(self, *shape) -> "Handle":
        """The same buffer seen with another shape (no data moves)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if prod(shape) != self.elements:
            raise ValueError(f"cannot reshape {self!r} to {list(shape)}")
        if self.walk is not None:
            raise ValueError(f"cannot reshape a view {self!r}; reshape what it views")
        return Handle(shape, self.dtype, self.name, self.role, self.parent, self.start)

    def transpose(self, *axes) -> "Handle":
        """The same buffer walked with its axes permuted (no data moves)."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if self.walk is not None:
            raise ValueError(
                f"cannot transpose a view {self!r}; transpose what it views"
            )
        walk = Walk.permuted(self.shape, axes)
        shape = tuple(self.shape[a] for a in axes)
        return Handle(shape, self.dtype, self.name, "view", self, 0, walk)

    def __getitem__(self, key) -> "Handle":
        if self.walk is not None or self.parent is not None:
            raise TypeError("slicing a slice is not supported; slice the parent")
        entries = list(key) if isinstance(key, tuple) else [key]
        rank = len(self.shape)
        if entries.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if Ellipsis in entries:
            at = entries.index(Ellipsis)
            fill = [slice(None)] * (rank - (len(entries) - 1))
            entries = entries[:at] + fill + entries[at + 1 :]
        if len(entries) > rank:
            raise IndexError(f"too many indices for shape {self.shape}")
        entries += [slice(None)] * (rank - len(entries))
        index_by = None
        static, shape = [], []
        for axis, (entry, n) in enumerate(zip(entries, self.shape)):
            if isinstance(entry, Value):
                if index_by is not None:
                    raise ValueError("one axis at most is indexed by a per-call value")
                if entry.kind != "scratchpad":
                    raise ValueError(
                        f"{entry.name} is {entry.kind}; only a Scratchpad value can "
                        f"index a view, since it moves a transfer's base address"
                    )
                index_by = (entry, prod(self.shape[axis + 1 :]))
                static.append(0)
            elif isinstance(entry, slice):
                if entry.step not in (None, 1):
                    raise ValueError("only unit steps are supported")
                static.append(entry)
                shape.append(len(range(*entry.indices(n))))
            else:
                static.append(int(entry))
        walk = Walk.slice(self.shape, tuple(static))  # checks ranges, empties
        if index_by is None and walk.contiguous:
            return Handle(shape, self.dtype, self.name, "slice", self, walk.offset)
        return Handle(shape, self.dtype, self.name, "view", self, 0, walk, index_by)

    def __repr__(self) -> str:
        return f"Handle({self.buffer_name!r}, {list(self.shape)}, {bfp.dtype_name(self.dtype)})"


class State:
    """A tensor that persists on the device across calls (a KV cache).

    Created outside the graph function with :func:`state` and closed over.
    Zero when the graph is first uploaded; read and written through
    :meth:`CompiledGraph.buffer`.
    """

    __slots__ = ("shape", "dtype", "name", "host")

    def __init__(self, shape, dtype=bfloat16, name=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.name = name
        # The reference path's copy, made on first use.
        self.host: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"State({self.name or ''}{list(self.shape)})"

    # Inside a graph function a state is viewed like a handle: the tracer
    # says what stands for it (a handle when tracing, its host tensor when
    # the reference runs).
    def _as_operand(self):
        from .trace import current  # imports this module

        tracer = current()
        if tracer is None:
            raise TypeError(f"{self!r} is viewed inside a graph function")
        return tracer.state_as(self)

    def __getitem__(self, key):
        return self._as_operand()[key]

    def reshape(self, *shape):
        return self._as_operand().reshape(*shape)

    def transpose(self, *axes):
        return self._as_operand().transpose(*axes)


def state(shape, dtype=bfloat16, name=None) -> State:
    """Declare device-resident state a graph function closes over."""
    return State(shape, dtype, name)


class Value:
    """A per-call scalar parameter of a graph function."""

    __slots__ = ("name", "kind", "dtype")

    def __init__(self, name, kind, dtype):
        self.name, self.kind, self.dtype = name, kind, dtype

    def __repr__(self) -> str:
        return f"Value({self.name!r}, {self.kind}[{np.dtype(self.dtype).name}])"


def is_operand(x) -> bool:
    """A graph handle, a state (or a view of one), or a host tensor (a weight)."""
    if isinstance(x, (Handle, State, _HostView)):
        return True
    if isinstance(x, (Overlay, Operator, type)):
        return False
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _tensor_dtype(t):
    dt = getattr(t, "dtype", None)
    name = str(dt)
    return {
        "bfloat16": bfloat16,
        "float32": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
    }.get(name, dt)


class _HostViews:
    """A state as the reference views it: ``[key]`` keeps the key for the
    operator; a reshape or transpose is numpy's own view of the host tensor.
    """

    def __init__(self, state: State) -> None:
        self.state = state

    def __getitem__(self, key):
        return _HostView(self.state, key)

    def reshape(self, *shape):
        assert self.state.host is not None
        return self.state.host.reshape(*shape)

    def transpose(self, *axes):
        assert self.state.host is not None
        return self.state.host.transpose(*axes)


class _HostView:
    def __init__(self, state: State, key) -> None:
        self.state, self.key = state, key
