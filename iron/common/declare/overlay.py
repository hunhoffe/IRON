# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The array-configuring half of a declaration.

An overlay fixes everything a change to which rebuilds the design: tile
shapes, column counts, dtypes, kernel flags. Its tunables start out ``None``
and :meth:`Overlay.tuning` fills them for a device, raising
:class:`~iron.common.declare.field.Unresolvable` when the device admits no legal
choice. :meth:`Overlay.array` writes the dataflow; an external overlay
declares :class:`~iron.common.declare.member.Xclbin` instead and supplies a
binary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, dataclass_transform

from aie.dialects.aie import (
    WireBundle,
    get_target_model,  # pyright: ignore[reportAttributeAccessIssue]  # not in _aie.pyi
)
from aie.utils.verify import Tolerance

from .bound import BoundResident, BoundStream, BoundValue
from .creation import declare
from .field import DeclarationError, Unresolvable, param
from .member import (
    DispatchTime,
    Resident,
    Value,
    Xclbin,
    _Buffer,
    _Member,
    _Stream,
    _Value,
)
from .naming import label_parts

if TYPE_CHECKING:
    from ..design.target import Target
    from .operator import Operator


def shim_columns(cls, dev, num_channels: int = 1) -> int:
    """How many of ``dev``'s columns a class's streams leave within the shim budget.

    One core per (column, channel) fills one fifo per input stream from the
    shim and drains one per output, so a column costs
    ``max(inputs, outputs) * num_channels`` channels in the busier
    direction. A ``replicate`` stream is shared by every column of a
    channel, so it is paid once per channel rather than per column.
    """
    streams = [m for m in cls._members if isinstance(m, _Stream)]
    shared = [m for m in streams if m.replicate]
    per_core = [m for m in streams if not m.replicate]
    directions = [m.direction for m in per_core]
    cost = max(directions.count("in"), directions.count("out")) * num_channels
    fixed = len(shared) * num_channels
    limit = get_shim_dma_limit(dev)
    return max(1, min(dev.cols, (limit - fixed) // cost))


def check_shim_columns(obj, dev, cols: int, num_channels: int = 1) -> None:
    """Raise :class:`Unresolvable` if ``cols`` exceeds ``obj``'s shim budget."""
    allowed = shim_columns(type(obj), dev, num_channels)
    if cols > allowed:
        raise Unresolvable(
            f"{type(obj).__name__} with {cols} columns x {num_channels} "
            f"channels exceeds this device's shim DMA budget; "
            f"{allowed} columns fit"
        )


def get_shim_dma_limit(dev) -> int:
    """Return the total number of ShimDMA output channels available on the device.

    Each shim tile exposes a fixed number of DMA source connections; summing
    across all shim tiles gives the device-wide ShimDMA budget.
    """
    tm = get_target_model(dev.resolve())
    return sum(
        tm.get_num_source_shim_mux_connections(col, row, WireBundle.DMA)
        for col in range(tm.columns())
        for row in range(tm.rows())
        if tm.is_shim_noc_or_pl_tile(col, row)
    )


# ``param`` is a field specifier, so a checker sees a ``param()`` without a
# ``default=`` as a required constructor argument. ``auto`` is not listed: it
# always has a default, but pyright reads a specifier's default only from a
# ``default=`` keyword, and ``auto(2)`` gives it positionally; unlisted, an
# ``auto()`` field is one with a default of type Any, which is what it is.
@dataclass_transform(field_specifiers=(param,))
@dataclasses.dataclass(eq=False)
class Overlay:
    """What configures the array. Subclass it.

    Declare ``param()`` and ``auto()`` fields, streams, and residents in the
    class body; implement :meth:`tuning` to fill the knobs from the device and
    :meth:`design` to build the array and bind each stream to a fifo's shim
    end. See the module docstring for the shape. Every subclass is a
    dataclass and is checked as its body finishes (:mod:`.creation`).
    """

    _members: ClassVar[tuple[_Member, ...]] = ()
    _param_fields: ClassVar[tuple[str, ...]] = ()
    _auto_fields: ClassVar[tuple[str, ...]] = ()
    _external: ClassVar[Xclbin | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        declare(cls, repr=True)
        images = [v for v in vars(cls).values() if isinstance(v, Xclbin)]
        if len(images) > 1:
            raise DeclarationError(f"{cls.__name__} declares more than one Xclbin")
        if images:
            cls._external = images[0]
        for m in cls._members:
            if isinstance(m, (_Buffer, DispatchTime)):
                raise DeclarationError(
                    f"{cls.__name__}.{m.name}: an Overlay declares streams, residents "
                    f"and core-read Scratchpad values; buffers and DispatchTime values "
                    f"belong on the Operator"
                )
            if images and isinstance(m, _Stream) and m.via is None:
                raise DeclarationError(
                    f"{cls.__name__}.{m.name}: a stream of an external overlay must be "
                    f"pinned with via=; nothing else says which shim it uses"
                )
            if images and isinstance(m, Resident) and m.address is None:
                raise DeclarationError(
                    f"{cls.__name__}.{m.name}: a resident of an external overlay needs "
                    f"an address; the sequence writes it there"
                )
        if not images:
            return
        for hook in ("prebuilt", "build"):
            if getattr(cls, hook) is getattr(Overlay, hook):
                raise DeclarationError(
                    f"{cls.__name__} declares an Xclbin, so nothing builds its array: "
                    f"it must supply {hook}() (iron.common.external.External "
                    f"does, for a downloaded image)"
                )

    @property
    def external(self) -> Xclbin | None:
        """The downloaded image this overlay is, if IRON did not build it."""
        return type(self)._external

    # -- placement ---------------------------------------------------------

    @classmethod
    def shim_columns(cls, dev, num_channels: int = 1) -> int:
        """How many of ``dev``'s columns this overlay's shim budget allows."""
        return shim_columns(cls, dev, num_channels)

    def check_shim_columns(self, dev, cols: int, num_channels: int = 1) -> None:
        """Raise :class:`Unresolvable` if ``cols`` exceeds the shim budget."""
        check_shim_columns(self, dev, cols, num_channels)

    # -- an overlay IRON does not design() ---------------------------------

    def prebuilt(self) -> Path:
        """The file the declared :class:`Xclbin` names, fetched if it is not
        already in the cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares an Xclbin but no prebuilt()"
        )

    def build(self, dev, op: "Operator"):
        """The MLIR module for ``op`` on this overlay, when ``design()`` does
        not build the array: a runtime sequence against the prebuilt image.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares an Xclbin but no build()"
        )

    # -- the sequence, when the overlay owns it -----------------------------

    def sequence(self, op: "Operator", rt) -> None:
        """The runtime sequence for ``op`` on this overlay, when the overlay
        rather than the operator knows it: a external image consumes its
        transfers in the order it was built for, whatever operator drives it.
        Takes precedence over the operator's ``sequence(rt)``.
        """
        raise NotImplementedError

    @classmethod
    def has_sequence(cls) -> bool:
        return cls.sequence is not Overlay.sequence

    def resident_values(self, op: "Operator") -> dict[str, Any]:
        """The words for this overlay's residents, from ``op``. By default the
        operator's own ``resident_values()`` and what a ``Value(derive=)``
        derives; an external overlay lays the operator's values out into
        the block its image reads.
        """
        derived = {
            m.name: m.derive(op)
            for m in self._members
            if isinstance(m, Value) and m.derive is not None
        }
        return {**derived, **op.resident_values()}

    def __post_init__(self) -> None:
        self._resolved = False
        self.validate()
        self._bind()

    # -- declared surface --------------------------------------------------

    def validate(self) -> None:
        """Check the compile-time fields. Runs at construction and after resolution."""

    def resolve(self, dev) -> Self:
        """Return a copy with every ``auto()`` filled for ``dev``; raise :class:`Unresolvable`.

        Sees the device and this overlay's own fields, so a resolved overlay
        serves every extent; an operator whose extents decide a knob fills it
        in its own :meth:`Operator.resolve` before this runs. The default
        fills nothing.
        """
        return self

    def device(self, target):
        """The device the Program is built for; the current device by default.

        An overlay that builds for a column subset (gemm's NPU1Col1/NPU1Col2)
        returns that variant.
        """
        return target.dev

    def array(self, target) -> list:
        """Build the array for ``target`` and return its workers.

        ``target`` (:class:`iron.common.design.Target`) carries the device,
        the kernel tree, and ``kernel()``/``barrier()`` helpers that apply
        the fusion prefix so the overlay never sees it. Must call
        ``.bind(handle)`` on every declared stream (or on every slot of a
        ``per=`` stream) with the shim end of the fifo that carries it, and
        ``.bind(buffers)`` on every declared resident.
        """
        raise NotImplementedError(f"{type(self).__name__}.array() is not implemented")

    def tolerance(self, target: Target) -> Tolerance | None:
        """How close this array's output comes to the operator's reference:
        the contract of the kernel it runs.

        ``None`` here: an overlay that builds several kernels in
        :meth:`design` has no one contract that speaks for its output, so its
        operator states a tolerance itself. An overlay running one kernel
        overrides this with that kernel's contract.
        """
        return None

    # -- library surface ---------------------------------------------------

    def resolved(self, dev) -> Self:
        """This overlay resolved for ``dev``: itself if it already is, else a
        resolved copy, every knob checked filled. The one place :meth:`resolve`
        is called.
        """
        if self._resolved:
            return self
        new = self.resolve(dev)
        if not isinstance(new, type(self)):
            raise TypeError(
                f"{type(self).__name__}.resolve() must return a {type(self).__name__}, "
                f"got {type(new).__name__}"
            )
        missing = [n for n in self._auto_fields if getattr(new, n) is None]
        if missing:
            raise Unresolvable(
                f"{type(self).__name__}.resolve() left {missing} unset for {dev}"
            )
        new.validate()
        new._resolved = True
        new._bind()
        return new

    def value_symbol(self, value: "BoundValue") -> str | None:
        """An explicit device symbol for a core-read per-call value, or ``None``."""
        return None

    def design_key(self) -> tuple:
        """Identity for sharing: the class and every compared field value."""
        return (type(self).__qualname__,) + tuple(
            (f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.compare
        )

    def copy(self) -> Self:
        """A fresh instance with the same fields and resolution state.

        A build works on a copy, so anything ``compatible()`` records on the
        overlay for one operator never reaches another that shares it.
        """
        new = dataclasses.replace(self)
        new._resolved = self._resolved
        new._bind()
        return new

    def __eq__(self, other) -> bool:
        if not isinstance(other, Overlay):
            return NotImplemented
        return self.design_key() == other.design_key()

    def __hash__(self) -> int:
        return hash(self.design_key())

    @property
    def streams(self) -> dict[str, BoundStream]:
        return {
            m.name: self._bound[m.name] for m in self._members if isinstance(m, _Stream)
        }

    @property
    def residents(self) -> dict[str, Any]:
        """What the preamble writes once per build: residents, and derived values."""
        return {
            m.name: self._bound[m.name]
            for m in self._members
            if isinstance(m, Resident)
            or (isinstance(m, Value) and m.derive is not None)
        }

    @property
    def values(self) -> list[BoundValue]:
        """Core-read per-call values this overlay declares."""
        return [
            self._bound[m.name]
            for m in self._members
            if isinstance(m, _Value)
            and not (isinstance(m, Value) and m.derive is not None)
        ]

    def build_array(self, target) -> list:
        """Run :meth:`array` for the build."""
        return self.array(target)

    def _bind(self) -> None:
        bound: dict[str, Any] = {}
        for m in self._members:
            if isinstance(m, _Stream):
                bound[m.name] = BoundStream(m, self)
            elif isinstance(m, Resident):
                bound[m.name] = BoundResident(m, self)
            elif isinstance(m, _Value):
                bound[m.name] = BoundValue(m, self)
        self._bound = bound

    def name_parts(self) -> list[str]:
        """This instance's fragments of an operator's name. Overridable: an
        external overlay names the binary it was built as, not its fields.
        """
        return label_parts(self)
