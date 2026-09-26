# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator: one class declaring the array, the buffers and the sequence.

Its fields sort by when a change rebuilds: the array tier is what a tile
names (plus what says ``array=True``), and one array serves every extent;
the rest sizes the host buffers and reaches the runtime sequence alone.
Calling the class binds it: :func:`~iron.common.declare.infer` turns operand
shapes into the extents, and the instance's buffer attributes answer in
elements.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Self,
    TypeVar,
    dataclass_transform,
    overload,
)

import aie.utils as aie_utils
from aie.utils.npukernel import NPUKernel
from aie.utils.verify import Tolerance

from ..device import device_name
from ..kernels import kernels_dir
from ..testing import Testing
from .bound import BoundBuffer, BoundStream, BoundValue
from .creation import declare
from .field import DeclarationError, Unresolvable, param
from .infer import infer, infer_kwargs
from .member import Value, _Buffer, _Member, _Stream, _Value
from .naming import label_parts
from .profile import current as current_profile
from .shim import check_shim_columns, shim_columns

if TYPE_CHECKING:
    from ..graph.handle import Handle
    from ..image.artifacts import Artifacts

_T = TypeVar("_T")


class _OperatorMeta(type):
    """``GEMV(w, h)`` inside a graph function records a step; anything else constructs.

    The class tells the two apart by whether it received graph handles (or
    host tensors, which a graph closes over as weights); see
    :mod:`iron.common.graph`. Outside a graph the call constructs as usual.
    To a checker the two are the two overloads below: a call with operands
    is a graph step and yields a handle, a call by keyword constructs.
    """

    if TYPE_CHECKING:

        @overload
        def __call__(
            cls: type[_T], operand: Any, /, *operands: Any, **kwargs: Any
        ) -> Handle: ...
        @overload
        def __call__(cls: type[_T], **kwargs: Any) -> _T: ...

    def __call__(cls, *args, **kwargs):
        from .. import graph as _graph  # imports this package: a cycle at module scope

        tracer = _graph.current()
        if tracer is not None and args and all(_graph.is_operand(a) for a in args):
            return tracer.call(cls, args, kwargs)
        if args:
            raise TypeError(
                f"{cls.__name__} is constructed by keyword ({cls.__name__}(M=..., "
                f"K=...)); operands are given inside an @iron.graph function"
            )
        profile = current_profile()
        if profile is not None:
            # The knobs this call leaves open, where the profile names them.
            kwargs = {**profile.knobs_for(cls, kwargs), **kwargs}
        return super().__call__(*args, **kwargs)


def _check_shipped(cls: type) -> None:
    """What a class declared with ``image=`` must say, since nothing builds it."""
    if "array" in vars(cls):
        raise DeclarationError(
            f"{cls.__name__} runs a shipped image, so nothing builds its array(); "
            f"drop the override"
        )
    for m in cls._members:
        if isinstance(m, _Stream) and m.via is None:
            raise DeclarationError(
                f"{cls.__name__}.{m.name}: a stream into a shipped image must be "
                f"pinned with via=; nothing else says which shim it uses"
            )
        if isinstance(m, Value) and m.derive is not None and m.address is None:
            raise DeclarationError(
                f"{cls.__name__}.{m.name}: a value written into a shipped image "
                f"needs an address; the sequence writes it there"
            )


class _ArrayView:
    """What :meth:`Operator.array` sees of its operator: the array tier.

    Reading a field no tile names and that does not declare ``array=True``
    raises, so an array cannot come to depend on an extent by accident
    (one array serves every extent). The operator's methods and properties
    run on the view too, so a ``kernel()`` reading an extent is caught as
    well.
    """

    __slots__ = ("_op",)

    def __init__(self, op: "Operator") -> None:
        object.__setattr__(self, "_op", op)

    @property
    def __class__(self):  # type: ignore[override]
        # super() and isinstance() inside a hook see the operator's class.
        return type(object.__getattribute__(self, "_op"))

    def __getattr__(self, name: str):
        op = object.__getattribute__(self, "_op")
        fields = {f.name for f in dataclasses.fields(op)}
        if name in fields and name not in op._array_fields:
            raise TypeError(
                f"{type(op).__name__}.array() reads {name}, which no tile names: "
                f"an array serves every extent. Declare it param(..., array=True) "
                f"if the array does read it, or move the dependence into the "
                f"sequence or a Value"
            )
        attr = inspect.getattr_static(type(op), name, None)
        if isinstance(attr, FunctionType):
            return attr.__get__(self, type(op))
        if isinstance(attr, property) and attr.fget is not None:
            return attr.fget(self)
        return getattr(op, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(object.__getattribute__(self, "_op"), name, value)


# Keyword-only to a checker, as every declared field is passed by keyword
# (a required field after one with a default is keyword-only at runtime too,
# see field._specifier). ``param`` is a field specifier, so a ``param()``
# without a ``default=`` is a required constructor argument; ``auto`` is not
# listed, since pyright reads a specifier's default only from a ``default=``
# keyword and ``auto(2)`` gives it positionally: unlisted, an ``auto()`` field
# is one with a default of type Any, which is what it is.
@dataclass_transform(kw_only_default=True, field_specifiers=(param,))
@dataclasses.dataclass(eq=False, repr=True)
class Operator(metaclass=_OperatorMeta):
    """An operator. Subclass it.

    One class declares the whole thing: ``param()``/``auto()`` fields,
    ``In``/``Out`` operands (with ``tile=`` an operand is its own stream),
    ``Value`` members, :meth:`array` for the dataflow, :meth:`sequence` when
    the derived one is not wanted, :meth:`resolve`/:meth:`compatible` and
    :meth:`reference`. A class declared with ``image=`` runs a shipped
    binary instead of building an array. Every subclass is a dataclass and
    is checked as its body finishes (:mod:`.creation`).
    """

    _members: ClassVar[tuple[_Member, ...]] = ()
    _param_fields: ClassVar[tuple[str, ...]] = ()
    _derived_params: ClassVar[dict[str, Callable[[Any], Any]]] = {}
    _auto_fields: ClassVar[tuple[str, ...]] = ()
    _array_fields: ClassVar[tuple[str, ...]] = ()
    _external: ClassVar[Any] = None
    # The cases iron/operators/test.py runs this operator at; None for an
    # operator tested by its own test.py, or not on its own.
    test: ClassVar[Testing | None] = None

    def __init_subclass__(cls, image=None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if image is not None:
            # A shipped image: ``class Shipped(GEMM, image=Xclbin(...))``.
            # Nothing builds its array, so it declares where every stream
            # enters and every value lives, and may not define array().
            cls._external = image
        declare(cls)
        if image is not None:
            _check_shipped(cls)

    def __post_init__(self) -> None:
        self._resolved = False
        self._derive_params()
        self.validate()
        self._bind()
        # Every rule is answerable once every knob is known, so the extents
        # are checked where the operator is written rather than at resolution.
        if not any(getattr(self, n) is None for n in self._auto_fields):
            self.compatible()

    def _derive_params(self) -> None:
        """Compute each ``param(default=<callable>)`` a shape or the caller
        left open, in declaration order.
        """
        for name, derive in self._derived_params.items():
            if getattr(self, name) is None:
                value = derive(self)
                if value is None:
                    raise ValueError(
                        f"{type(self).__name__}.{name}: not given, and nothing "
                        f"to compute it from"
                    )
                setattr(self, name, value)

    def check_derived(self, *names: str) -> None:
        """Raise ``ValueError`` if a computed-default parameter was given a
        value its rule disagrees with: for one a shape may bind that the
        other fields nonetheless determine (``out_rows = rows * repeat``).
        """
        for name in names:
            given, expected = getattr(self, name), self._derived_params[name](self)
            if given != expected:
                raise ValueError(
                    f"{type(self).__name__}.{name}={given!r} is not what its "
                    f"other fields make it ({expected!r})"
                )

    # -- declared surface --------------------------------------------------

    def validate(self) -> None:
        """Check the sequence-tier fields on their own. Runs at construction."""

    def compatible(self) -> None:
        """Check the extents against the resolved knobs; raise :class:`Incompatible`."""

    def resolve(self, dev) -> Self:
        """Return a copy resolved for ``dev``: every ``auto()`` filled, from
        the device and from this operator's extents; raise :class:`Unresolvable`.

        The one hook that sees both. The default fills nothing. A knob left
        ``None`` is an error once this returns::

            def resolve(self, dev):
                cols = self.columns or self.shim_columns(dev)
                return dataclasses.replace(self, columns=cols)

        Identity for sharing a build is taken after this runs, so two ways
        of spelling one array resolve to one design.
        """
        return dataclasses.replace(self)

    def array(self, target) -> list:
        """Build the array for ``target`` and return its workers.

        ``target`` (:class:`iron.common.design.Target`) carries the device,
        the kernel tree, and ``kernel()``/``rtp()``/``barrier()``. Bind the
        shim end of a fifo to every operand's lane (``self.A.lane(i).bind(
        fifo.prod())``) and every ``Value`` to the buffer a core reads it
        from. Sees the array tier alone: a field no tile names raises.
        """
        raise NotImplementedError(f"{type(self).__name__}.array() is not implemented")

    def build_array(self, target) -> list:
        """Run :meth:`array` for the build, through the array-tier view."""
        return type(self).array(_ArrayView(self), target)  # type: ignore[arg-type]

    def tolerance(self, target) -> Tolerance | None:
        """The contract of the kernel this array runs; ``None`` when the
        operator states its own (see :attr:`test`).
        """
        return None

    def device(self, target):
        """The device the Program is built for; the current device by default."""
        return target.dev

    @property
    def external(self):
        """The downloaded image this operator runs on, if IRON did not build it."""
        return type(self)._external

    def prebuilt(self):
        """The file the declared image names, fetched if it is not in the cache."""
        from ..external import fetch  # imports this package: a cycle at module scope

        return fetch(self.external)

    @classmethod
    def shim_columns(cls, dev, num_channels: int = 1) -> int:
        """How many of ``dev``'s columns this operator's shim budget allows."""
        return shim_columns(cls, dev, num_channels)

    def check_shim_columns(self, dev, cols: int, num_channels: int = 1) -> None:
        check_shim_columns(self, dev, cols, num_channels)

    def resolve_columns(
        self,
        dev,
        given: int | None,
        num_channels: int = 1,
        *,
        fits: Callable[[int], bool] | None = None,
    ) -> int:
        """The column count to resolve to: ``given``, checked against the
        shim budget, or the most the budget allows that ``fits``.

        ``fits(c)`` is the operator's own rule for ``c`` columns leaving
        whole tiles; when no count within the budget does, one column is
        returned and :meth:`compatible` names the rule. With no device
        bound and no count given, :class:`Unresolvable`.
        """
        if given is not None:
            if dev is not None:
                self.check_shim_columns(dev, given, num_channels)
            return given
        if dev is None:
            raise Unresolvable(
                f"{type(self).__name__}: the column count defaults from the "
                f"device; none is bound and none was given"
            )
        budget = self.shim_columns(dev, num_channels)
        return next((c for c in range(budget, 0, -1) if fits is None or fits(c)), 1)

    def reference(self, *inputs):
        raise NotImplementedError(
            f"{type(self).__name__}.reference() is not implemented"
        )

    def sequence(self, rt) -> None:
        """Override to write the runtime sequence by hand; otherwise it is derived.

        ``rt`` is an :class:`iron.common.design.Sequence`: ``rt.fill(stream,
        view)``, ``rt.drain(stream, view)``, ``rt.group()``. The preamble
        (residents, barriers, parameter sync) has already run.
        """
        raise NotImplementedError

    def resident_values(self) -> dict[str, Any]:
        """What the preamble writes once per build: every ``Value(derive=)``
        no graph bound per call, derived from this instance.
        """
        return {
            m.name: m.derive(self)
            for m in self._members
            if isinstance(m, Value)
            and m.derive is not None
            and not self.uses_value(m.name)  # bound per call: not a resident
        }

    @classmethod
    def has_sequence_override(cls) -> bool:
        return cls.sequence is not Operator.sequence

    # -- library surface ---------------------------------------------------

    def value_symbol(self, value: "BoundValue") -> str | None:
        """An explicit device symbol for a per-call value, or ``None`` for the default."""
        return None

    def design_key(self):
        """Identity for sharing a build: the class, the array's key, every compared field.

        Two operators with equal keys generate byte-identical MLIR, so a
        sequence builds, prefixes and configures the design once.
        """
        own = tuple(
            (f.name, getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.compare
        )
        # A per-call value a graph bound is built in (a device parameter, a
        # patched descriptor, a core that reads it), so it tells designs apart.
        if self.used_values:
            own += (("values", tuple(sorted(self.bound_values.items()))),)
        return (type(self).__qualname__, own)

    def array_key(self):
        """Identity for sharing an array: the class and its array-tier fields."""
        return (type(self).__qualname__,) + tuple(
            (name, getattr(self, name)) for name in self._array_fields
        )

    def resolved(self, dev) -> Self:
        """This operator resolved for ``dev``: itself if it already is, else
        :meth:`resolve`'s copy, every knob filled and :meth:`validate` and
        :meth:`compatible` checked. The one place :meth:`resolve` is called.
        """
        if self._resolved:
            return self
        new = self.resolve(dev)
        if new is self:
            raise TypeError(
                f"{type(self).__name__}.resolve() must return a copy, "
                f"dataclasses.replace(self, ...), not self"
            )
        if not isinstance(new, type(self)):
            raise TypeError(
                f"{type(self).__name__}.resolve() must return a {type(self).__name__}"
            )
        missing = [n for n in self._auto_fields if getattr(new, n) is None]
        if missing:
            raise Unresolvable(
                f"{type(self).__name__}.resolve() left {missing} unset for {dev}"
            )
        new.validate()
        # What a graph bound on this instance is part of it, not of a field:
        # the build works on the copy, and a copy that forgot would silently
        # drop the per-call value from the sequence.
        if self.used_values:
            vars(new)["_used_values"] = dict(self.bound_values)
        new.compatible()
        new._resolved = True
        return new

    def copy(self) -> Self:
        """A fresh instance for one build, so the streams a build binds are
        this build's alone; resolution state and the per-call values a graph
        bound are kept.
        """
        new = dataclasses.replace(self)
        if self.used_values:
            vars(new)["_used_values"] = dict(self.bound_values)
        new._resolved = self._resolved
        if self._resolved:
            # What compatible() records is part of a resolved instance; a
            # replace() resets an init=False field to its default, so the
            # copy records it again.
            new.compatible()
        return new

    @property
    def buffers(self) -> list[BoundBuffer]:
        return [self._bound[m.name] for m in self._members if isinstance(m, _Buffer)]

    @property
    def inputs(self) -> list[BoundBuffer]:
        return [b for b in self.buffers if b.direction in ("in", "inout")]

    @property
    def outputs(self) -> list[BoundBuffer]:
        return [b for b in self.buffers if b.direction in ("out", "inout")]

    @property
    def values(self) -> list[BoundValue]:
        """The per-call values this instance uses (see :meth:`uses_value`)."""
        return [
            self._bound[m.name]
            for m in self._members
            if isinstance(m, _Value) and self.uses_value(m.name)
        ]

    @property
    def streams(self) -> dict[str, BoundStream]:
        """Every operand's own stream, by the operand's name."""
        return {
            m.name: self._bound[m.name].lanes
            for m in self._members
            if isinstance(m, _Buffer) and m.stream is not None
        }

    @property
    def residents(self) -> dict[str, BoundValue]:
        """What the preamble writes once per build: every ``Value(derive=)``
        no graph bound per call.
        """
        return {
            m.name: self._bound[m.name]
            for m in self._members
            if isinstance(m, Value)
            and m.derive is not None
            and not self.uses_value(m.name)
        }

    def uses_value(self, name: str) -> bool:
        """Whether this instance drives the declared per-call value ``name``.

        A value an instance does not use gets no device parameter and no
        sync. The default is every declared value, except a ``Value`` with a
        derivation, which is per-call only when a graph binds it
        (:meth:`use_value`); an operator whose values are optional (a copy
        with or without a patched offset) overrides this.
        """
        member = next((m for m in self._members if m.name == name), None)
        if isinstance(member, Value) and member.derive is not None:
            return name in self.used_values
        return True

    def use_value(self, name: str, bound_to: str | None = None) -> None:
        """Record that a graph binds the per-call value ``name`` on this
        instance, to its own value ``bound_to``.

        The graph value is part of what is built: two instances alike in
        every field that read different graph values are two designs with
        two device symbols, not one.
        """
        if not any(isinstance(m, _Value) and m.name == name for m in self._members):
            raise TypeError(
                f"{type(self).__name__} declares no per-call value {name!r}"
            )
        vars(self).setdefault("_used_values", {})[name] = bound_to

    @property
    def used_values(self) -> frozenset:
        """The names of the per-call values a graph binds on this instance."""
        return frozenset(self.bound_values)

    @property
    def bound_values(self) -> dict[str, str | None]:
        """Per-call value name -> the graph value it is bound to."""
        return dict(self.__dict__.get("_used_values", {}))

    # -- graph functions ---------------------------------------------------

    @classmethod
    def resolve_class(cls, n_operands: int, kwargs: dict) -> type:
        """The class a graph call with ``n_operands`` operands constructs.

        The default is the class itself; a family that picks a subclass from
        its arguments (RMSNorm with a weight) overrides.
        """
        return cls

    def __call__(self, *args, **kwargs):
        """An explicit instance applied to graph handles records a step."""
        from .. import graph as _graph  # as above

        tracer = _graph.current()
        if tracer is None:
            raise TypeError(
                f"{type(self).__name__} instances are called on graph handles inside "
                f"an @iron.graph function; outside one, compile() and get_callable()"
            )
        return tracer.call(self, args, kwargs)

    def _bind(self) -> None:
        bound: dict[str, Any] = {}
        for m in self._members:
            if isinstance(m, _Buffer):
                bound[m.name] = BoundBuffer(m, self)  # its own stream with it
            elif isinstance(m, _Value):
                bound[m.name] = BoundValue(m, self)
        self._bound = bound

    # -- construction from operand shapes ----------------------------------

    @classmethod
    def from_operands(cls, *operand_shapes, **overrides) -> Self:
        """Construct an operator from operand shapes."""
        values = infer(cls, *operand_shapes, **infer_kwargs(cls, overrides))
        return cls(**{**overrides, **values})

    def reference_tolerance(self) -> Tolerance | None:
        """How close the NPU output must come to :meth:`reference`: the
        resolved operator's :meth:`tolerance` for this device, ``None`` when
        it states none.
        """
        from ..design.target import (
            Target,
        )  # imports this package: a cycle at module scope

        return self.resolved(self.dev).tolerance(Target(self.dev, kernels_dir()))

    # -- the image of one operator on its own -------------------------------

    @property
    def dev(self):
        """The device a design is generated for, bound as the current one
        (:func:`~iron.common.device.bound_device`); ``None`` on a host
        without one, where an operator can still be checked and lowered.
        """
        return aie_utils.ensure_current_device()

    # Bytes of trace buffer to emit; 0 disables tracing. A plain attribute
    # rather than a property: OperatorSequence and LayerNorm assign it.
    trace_size = 0

    @property
    def name(self) -> str:
        """This instance's label: the class, every shown field as resolved
        for the device, the device. It names the per-call value
        symbols a host writes through and the kernel instances a chained
        image carries; nothing on disk, which the compile cache keys by
        content. A name describes what is built, so it is the resolved
        operator's.
        """
        dev = aie_utils.get_current_device()
        if dev is None:
            raise RuntimeError(f"{type(self).__name__}.name needs a bound device")
        own = label_parts(self.resolved(dev))
        base = type(self).__name__ + "_" + "_".join(own)
        # Upstream annotates Device.resolve() -> None; it returns the AIEDevice.
        return f"{base}_{device_name(dev)}"

    def generator(self, image: str = "elf"):
        """The design generator :class:`CompilableDesign` runs for this operator.

        An override point, not a forwarder: an operator whose design is
        exported text rather than derived from the declaration replaces this
        (see :func:`from_spec`, and swiglu_prefill_stream, which loads its
        group from the exported module). Everything else takes the default,
        which is ``build_design`` over the declaration.
        """
        from ..design import (
            generator_for,
        )  # reads this package: a cycle at module scope

        return generator_for(self, image=image)

    def compile(self, record: str = "memory") -> "Operator":
        """Build this operator's own image, once; sets :attr:`artifacts`.

        ``record="disk"`` also writes the :class:`~iron.common.image.artifacts.Artifacts`
        record beside the image; by default it is only kept in memory.
        """
        if getattr(self, "_artifacts", None) is None:
            self._artifacts = self._build()
            if record == "disk":
                self._artifacts.dump()
        return self

    @property
    def artifacts(self) -> "Artifacts":
        """The record of what :meth:`compile` produced."""
        artifacts = getattr(self, "_artifacts", None)
        if artifacts is None:
            raise RuntimeError(
                f"{type(self).__name__} is not compiled; compile() first"
            )
        return artifacts

    def _members_io(self):
        """The declared buffers, without resolving a shape: their names alone."""
        return [m for m in self._members if isinstance(m, _Buffer)]

    def buffer_map(self) -> dict[str, tuple[str, int, int]]:
        """Each buffer as ``(arena, position, nbytes)``, for an image's record.

        From the tuned operator: a shape may follow a tunable the device
        fills (flm/gemm's B layout), and the built image's buffers are the
        tuned ones. A standalone operator has no arena plan -- its buffers
        are the kernel's positional arguments.
        """
        resolved = self.resolved(self.dev)
        return {b.name: ("arg", i, b.nbytes) for i, b in enumerate(resolved.buffers)}

    def _build(self):
        """Compile to an xclbin and an instruction stream, or, on a shipped
        image, to the stream alone against the download.
        """
        # image/ reads this package, so naming it at module scope would make
        # the two import each other.
        from ..image.artifacts import Artifacts, Design, Step
        from ..image.jit_compile import cache_entry, insts_design, xclbin_design

        image = self.external
        if image is None:
            picture = None
            design = xclbin_design(self.generator("xclbin"), kernel_name="MLIR_AIE")
        else:
            picture = self.prebuilt()
            design = insts_design(self.generator())
        entry = cache_entry(design)
        insts = entry.insts
        assert insts is not None, "no instruction stream"
        if picture is None:
            picture = entry.xclbin
            assert picture is not None, "no xclbin"
        self._design = design
        return Artifacts(
            kind="xclbin",
            image=picture,
            insts=insts,
            entry=entry,
            designs=(
                Design(
                    name=self.name,
                    operators=(self.name,),
                    entry=entry,
                    image=picture,
                    insts=insts,
                ),
            ),
            steps=(Step(0, self.name, self.name, tuple(b.name for b in self.buffers)),),
            buffers={b.name: ("arg", i, b.nbytes) for i, b in enumerate(self.buffers)},
        )

    def get_callable(self):
        """The loaded image, ready to call on device tensors."""
        artifacts = self.compile()._artifacts
        image = self.external
        npu_kernel = NPUKernel(
            xclbin_path=str(artifacts.image),
            kernel_name="MLIR_AIE" if image is None else image.kernel_name,
            insts_path=str(artifacts.insts),
        )
        handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

        def call(*args):
            return aie_utils.DefaultNPURuntime.run(handle, list(args))

        return call

    def __repr__(self) -> str:
        own = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(self)
            if f.repr
        )
        return f"{type(self).__name__}({own})"

    def explain(self) -> str:
        """What a build of this operator compiles in and what it takes per call.

        One line per tier: the array's fields (every core is built from them;
        one array serves every operator with the same), the sequence's (the
        host's alone), then each value: written once per build, with its
        number once resolved; per call, as a scratchpad word or a regenerated
        instruction stream; or unused by this instance.
        """
        fields = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.compare
        }

        def spell(names):
            return ", ".join(f"{n}={fields[n]!r}" for n in names) or "nothing"

        array = [n for n in fields if n in self._array_fields]
        sequence = [n for n in fields if n not in self._array_fields]
        lines = [
            repr(self) + (" (resolved)" if self._resolved else " (unresolved)"),
            f"  array, compiled into every core: {spell(array)}",
            f"  sequence, the host's alone: {spell(sequence)}",
        ]
        for m in self._members:
            if not isinstance(m, _Value):
                continue
            if (
                isinstance(m, Value)
                and m.derive is not None
                and not self.uses_value(m.name)
            ):
                given = f", {m.derive(self)!r} here" if self._resolved else ""
                how = "written once per build" + given
            elif self.uses_value(m.name):
                how = "per call, " + (
                    "the instruction stream regenerated around it"
                    if m.kind == "dispatch"
                    else "a scratchpad word patched or read"
                )
            else:
                how = "unused here"
            lines.append(f"  {m.name}: {how}")
        return "\n".join(lines)
