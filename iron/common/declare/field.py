# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field specifiers and the dimension references a class body writes.

A compile-time parameter is a dataclass field declared with :func:`param`, a
knob the library resolves one declared with :func:`auto`. Naming either in a
shape expression yields a :class:`DimRef`, which class creation resolves
against the class it lands on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING, Field
from typing import Any, Callable


class Unresolvable(ValueError):
    """No legal resolution exists for this operator on this device.

    An expected outcome, not a bug: raised by :meth:`Operator.resolve` so the
    caller learns at resolution rather than from a design that compiles and
    then hangs.
    """


class Incompatible(ValueError):
    """An operator's extents do not fit its resolved knobs."""


class DeclarationError(TypeError):
    """A class body violates the declaration rules; raised at class creation."""


_TIER = "iron.tier"  # dataclass Field.metadata key: "param" | "auto"
_CHOICES = "iron.choices"
_LEGAL = "iron.legal"
_ARRAY = "iron.array"  # the field is array-tier though no tile names it


def param(
    *, default: Any = MISSING, array: bool = False, repr: bool = True, init: bool = True
) -> Any:
    """Declare a compile-time parameter: given by the caller or inferred from
    the operands, and fixed from then on.

    A ``param()`` may appear in a shape. Which tier it is follows from use: a
    field named in an operand's ``tile=``/``per=``/``depth=`` configures the
    array (changing it rebuilds the array); any other rebuilds the
    instruction stream only, unless it says ``array=True``, which is how a
    field the array reads but no tile names (a kernel's epilogue) declares
    itself. ``default`` is keyword-only so a checker reads it: a ``param()``
    without one is a required constructor argument.
    """
    return _specifier("param", default, repr, init, array=array)


def auto(
    default: Any = None,
    /,
    *,
    choices: tuple | None = None,
    legal: Callable[..., bool] | None = None,
    array: bool = False,
    repr: bool = True,
    init: bool = True,
) -> Any:
    """Declare a knob the library resolves for the device when the caller
    does not: a compile-time value that starts at ``default`` (``None``:
    :meth:`~iron.common.declare.Operator.resolve` must fill it) and that
    ``resolve`` may replace. Annotate it with the resolved type: the field
    is ``None`` only until resolution, and every hook after it sees the
    value.

    An ``auto()`` never appears in a host shape (inference would cycle
    through resolution); a stream tile may name one. ``choices`` and ``legal``
    describe the knob for a tuner and are recorded, not yet read.
    ``init=False`` fixes a subclass's value of an inherited knob (a kernel
    that only works with one channel per column).
    """
    return _specifier(
        "auto", default, repr, init, choices=choices, legal=legal, array=array
    )


def _specifier(
    tier: str, default: Any, repr_: bool, init: bool = True, **extra: Any
) -> Field:
    metadata: dict[str, Any] = {_TIER: tier}
    if extra.get("choices") is not None:
        metadata[_CHOICES] = tuple(extra["choices"])
    if extra.get("legal") is not None:
        metadata[_LEGAL] = extra["legal"]
    if extra.get("array"):
        metadata[_ARRAY] = True
    kwargs: dict[str, Any] = {"metadata": metadata, "repr": repr_, "init": init}
    if default is not MISSING:
        kwargs["default"] = default
    else:
        # Keyword-only, so a field with no default may follow one with a
        # default -- which is what a subclass does when it pins an inherited
        # knob to a shape-bearing parameter of its own. Every declared field
        # is passed by keyword anyway; only ``ov`` is positional.
        kwargs["kw_only"] = True
    return dataclasses.field(**kwargs)


def _tier_of(f: Field) -> str | None:
    return f.metadata.get(_TIER) if f.metadata else None


def _declares_array(f: Field) -> bool:
    return bool(f.metadata and f.metadata.get(_ARRAY))


# --------------------------------------------------------------------------
# Dimension references
# --------------------------------------------------------------------------


class DimRef:
    """A reference to a ``param()`` or ``auto()`` field of a declared class.

    As a class is created, each field is re-attached to the
    class as a ``DimRef``, so ``GEMV.K`` names the dimension from
    outside the class body while ``ov.K`` on an instance is the integer. A
    non-data descriptor: instance attributes take precedence.
    """

    __slots__ = ("owner", "name", "tier", "default")

    def __init__(
        self, owner: type, name: str, tier: str | None, default=MISSING
    ) -> None:
        self.owner = owner
        self.name = name
        self.tier = tier
        self.default = default

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        # An init=False field is read from the class attribute, which is now
        # this object: serve its default. Anything else has no value yet.
        if self.default is not MISSING:
            return self.default
        raise AttributeError(self.name)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DimRef)
            and other.owner is self.owner
            and other.name == self.name
        )

    def __hash__(self) -> int:
        return hash((id(self.owner), self.name))

    def __repr__(self) -> str:
        return f"{self.owner.__qualname__}.{self.name}"


class _Optional:
    """A leading dimension that is present only when greater than one.

    ``In(optional(num_batches), M, K)`` declares ``(M, K)`` for a single batch
    and ``(num_batches, M, K)`` otherwise, which is how batched operators
    already spell their host shapes. Inference reads the rank to tell the two
    apart.
    """

    __slots__ = ("ref",)

    def __init__(self, ref) -> None:
        self.ref = ref

    def __repr__(self) -> str:
        return f"optional({self.ref!r})"


def optional(ref) -> _Optional:
    """Mark a leading dimension as omitted when it equals one. See :class:`_Optional`."""
    return _Optional(ref)


class _Select:
    """A shape chosen by a flag: ``select(b_col_maj, (N, K), (K, N))``.

    The flag is a field with a default or one the caller passes explicitly;
    it is never inferred. The only conditional shapes in the tree are GEMM's
    layout flags, which transpose a declared shape rather than resize it.
    """

    __slots__ = ("flag", "when_true", "when_false")

    def __init__(self, flag, when_true, when_false) -> None:
        self.flag = flag
        self.when_true = tuple(when_true)
        self.when_false = tuple(when_false)

    def __repr__(self) -> str:
        return f"select({self.flag!r}, {self.when_true!r}, {self.when_false!r})"


def select(flag, when_true, when_false) -> _Select:
    """A conditional shape. See :class:`_Select`."""
    return _Select(flag, when_true, when_false)


_DimSpec = Any  # Field (own class, pre-processing) | DimRef | int | _Optional


def _describe(spec) -> str:
    if isinstance(spec, Field):
        return spec.name if spec.name else "<field>"
    return repr(spec)
