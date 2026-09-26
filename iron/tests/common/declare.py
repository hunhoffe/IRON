# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The declaration layer, device-free.

Everything here runs without a device and without generating MLIR: what
class creation records and rejects, how bound members resolve on instances,
how inference binds fields from operand shapes, and how resolution behaves.
The design-generating half is ``iron/common/design/`` and needs the
toolchain.
"""

import dataclasses

import numpy as np
import pytest
from ml_dtypes import bfloat16

import iron
from iron.common import (
    DeclarationError,
    DispatchTime,
    In,
    Incompatible,
    Operator,
    Out,
    Scratchpad,
    Shim,
    Unresolvable,
    Value,
    auto,
    from_spec,
    optional,
    param,
)
from iron.common.declare.field import DimRef
from iron.common.declare.infer import infer


class FakeDev:
    def __init__(self, cols=8):
        self.cols = cols

    def columns(self):
        return self.cols


# --------------------------------------------------------------------------
# A worked operator, close to GEMV
# --------------------------------------------------------------------------


class MV(Operator):
    M: int = param()
    K: int = param()
    num_batches: int = param(default=1)
    columns: int | None = auto()
    tile_out: int = auto(64)
    vec: int | None = auto(repr=False)
    epilogue: str = param(default="none", array=True)

    A = In(optional(num_batches), M, K, tile=(tile_out, K), per=columns)
    B = In(optional(num_batches), K, tile=(K,), broadcast=True)
    C = Out(optional(num_batches), M, tile=(tile_out,), per=columns)
    count = Value(np.int32, derive=lambda op: op.M // (op.columns * op.tile_out))
    start = Value(np.int32)  # per-call when a graph binds it, else unused

    def resolve(self, dev):
        cols = self.columns or dev.columns()
        vec = self.vec or next((w for w in (64, 32, 16) if self.K % w == 0), None)
        if vec is None:
            raise Unresolvable(f"K={self.K}: no vector width divides it")
        return dataclasses.replace(self, columns=cols, vec=vec)

    def compatible(self):
        assert self.columns is not None
        unit = self.columns * self.tile_out
        if self.M % unit:
            raise Incompatible(f"M={self.M} is not a multiple of {unit}")

    def uses_value(self, name):
        return name in self.used_values if name == "start" else super().uses_value(name)

    def array(self, target):
        return [self.K, self.columns, self.tile_out, self.epilogue]

    def reference(self, A, B):
        return A @ B


# --------------------------------------------------------------------------
# Class creation: names, order, re-attached fields
# --------------------------------------------------------------------------


def test_fields_are_reattached_as_dim_refs():
    assert isinstance(MV.K, DimRef) and MV.K.owner is MV
    assert MV.K.name == "K" and MV.K.tier == "param"
    assert isinstance(MV.tile_out, DimRef) and MV.tile_out.tier == "auto"


def test_members_keep_declaration_order_and_names():
    # An operand's own stream follows it, under its name.
    assert [m.name for m in MV._members] == [
        "A",
        "A",
        "B",
        "B",
        "C",
        "C",
        "count",
        "start",
    ]
    assert MV.A.direction == "in" and MV.C.direction == "out"
    assert MV.A.stream is not None and MV.A.stream.direction == "in"


def test_shapes_captured_bare_names_resolve_to_refs():
    # ``M`` and ``num_batches`` were Field objects in the class body; class
    # creation rewrote them to DimRefs on the class.
    dims = MV.A.dims
    assert dims[0].ref.name == "num_batches"
    assert dims[1] == MV.M and dims[2] is MV.K
    assert MV.A.stream is not None
    assert MV.A.stream.dims == (MV.tile_out, MV.K) and MV.A.stream.per == (MV.columns,)


def test_dataclass_constructor_is_typed_by_real_fields():
    assert [f.name for f in dataclasses.fields(MV)] == [
        "M",
        "K",
        "num_batches",
        "columns",
        "tile_out",
        "vec",
        "epilogue",
    ]


# --------------------------------------------------------------------------
# Rules rejected at class creation
# --------------------------------------------------------------------------


def test_a_knob_in_a_buffer_shape_is_rejected():
    with pytest.raises(DeclarationError, match="host shape may not depend on tuning"):

        class Bad(Operator):
            M: int = param()
            t: int = auto(64)
            A = In(M, t)


def test_a_knob_in_a_tile_is_allowed():
    assert MV.A.stream is not None and MV.A.stream.dims[0] is MV.tile_out


def test_plain_defaulted_field_in_a_shape_is_its_literal():
    # A plain field with a default is bound to that default in the class
    # body, so a shape written against it captures the literal, not the
    # field. This is why anything a shape names must be declared with param().
    class Plain(Operator):
        n: int = 4
        x = In(n, tile=(n,))

    assert Plain.x.dims == (4,)
    assert Plain(n=8).x.shape == (4,)


def test_plain_field_reference_from_outside_is_rejected():
    class Plain(Operator):
        n: int = 4
        x = In(4)

    with pytest.raises(DeclarationError, match="not declared with param"):

        class Bad(Plain):
            M: int = param()
            A = In(M, Plain.n)


def test_expression_in_a_shape_is_rejected():
    with pytest.raises(DeclarationError, match="Expressions are not allowed"):

        class Bad(Operator):
            n: int = param()
            x = In("n // 2")


def test_annotated_member_is_rejected():
    with pytest.raises(DeclarationError, match="without an annotation"):

        class Bad(Operator):
            M: int = param()
            A: In = In(M)


def test_float_scratchpad_is_rejected():
    with pytest.raises(DeclarationError, match="floating point"):
        Scratchpad(np.float32)


def test_per_and_broadcast_are_exclusive():
    with pytest.raises(DeclarationError, match="either per"):

        class Bad(Operator):
            n: int = param()
            c: int = auto(2)
            x = In(n, tile=(4,), per=(c,), broadcast=True)


# --------------------------------------------------------------------------
# Bound members on instances
# --------------------------------------------------------------------------


def test_buffers_resolve_shape_dtype_and_direction():
    specs = MV(M=64, K=256, num_batches=2).buffers
    assert [(s.direction, s.shape) for s in specs] == [
        ("in", (2, 64, 256)),
        ("in", (2, 256)),
        ("out", (2, 64)),
    ]
    assert specs[0].dtype is bfloat16
    op = MV(M=1024, K=256)
    assert [b.name for b in op.inputs] == ["A", "B"]
    assert [b.name for b in op.outputs] == ["C"]


def test_optional_leading_dim_is_omitted_when_one():
    assert MV(M=64, K=256).A.shape == (64, 256)
    assert MV(M=64, K=256, num_batches=3).A.shape == (3, 64, 256)
    assert MV(M=64, K=256, num_batches=3).C.shape == (3, 64)


def test_buffers_carry_the_declared_dtype_and_size():
    """The sizing contract: the sequence layout and the test harness allocate
    from ``b.dtype`` and ``b.nbytes`` of a declared buffer.
    """
    from iron.operators.repeat import Repeat

    x, y = Repeat(rows=8, cols=64, repeat=4, dtype=np.int32).buffers
    assert x.dtype == np.int32 and y.dtype == np.int32
    assert (x.direction, y.direction) == ("in", "out")
    assert y.nbytes == 8 * 64 * 4 * 4


def test_instance_values_shadow_dim_refs():
    op = MV(M=64, K=256, columns=2)
    assert op.K == 256 and op.columns == 2
    assert isinstance(MV.K, DimRef) and MV.K.name == "K"


def test_a_stream_is_bound_lane_by_lane():
    op = MV(M=1024, K=256, columns=2)
    op.A.lane(0).bind("h0")
    op.A.lane(1).bind("h1")
    op.B.bind("hb")
    assert op.A.handles == ["h0", "h1"]
    assert op.B.handle == "hb"
    with pytest.raises(ValueError, match="already bound"):
        op.A.lane(0).bind("again")
    with pytest.raises(ValueError, match="never bound"):
        op.C.handles
    with pytest.raises(ValueError, match="index it"):
        op.A.handle

    class NoTile(Operator):
        n: int = param()
        x = In(n)

    with pytest.raises(TypeError, match="without a tile"):
        NoTile(n=4).x.tile


def test_per_call_values_bind_on_the_operator():
    class Copy(Operator):
        n: int = param()
        src = In(n, tile=(n,))
        off = Scratchpad(np.int32)
        live = DispatchTime(np.int32)

    op = Copy(n=256)
    assert [v.name for v in op.values] == ["off", "live"]
    assert op.off.kind == "scratchpad" and op.live.kind == "dispatch"


def test_shim_pins_declare():
    class Pinned(Operator):
        n: int = param()
        c: int = auto(2)
        x = In(n, tile=(n,), via=Shim(col=1, channel=0))
        y = Out(n, tile=(n,), via=[Shim(col=c, channel=0) for c in range(2)], per=(c,))

    op = Pinned(n=2)
    pins = [op.x.lane().shim] + [op.y.lane(i).shim for i in range(2)]
    assert op.y.count == 2
    assert [p.col for p in pins if p is not None] == [1, 0, 1]


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def test_infer_binds_the_fields_from_operands():
    assert infer(MV, (1024, 256), (256,)) == {"M": 1024, "K": 256, "num_batches": 1}
    assert infer(MV, (3, 1024, 256), (3, 256)) == {
        "num_batches": 3,
        "M": 1024,
        "K": 256,
    }


def test_infer_reports_conflicts_naming_both_operands():
    with pytest.raises(
        ValueError, match=r"K is 128 from B.shape\[0\] but 256 from A.shape\[1\]"
    ):
        infer(MV, (1024, 256), (128,))
    with pytest.raises(ValueError, match="rank"):
        infer(MV, (1, 2, 3, 4), (256,))
    with pytest.raises(ValueError, match="K is 512 from A.shape"):
        infer(MV, (1024, 512), (512,), K=256)


def test_from_spec_builds_an_operator_from_literal_shapes():
    # swiglu_prefill_stream's escape: shapes from an exported graph, a
    # design that is not derived, an identity for sharing.
    Group = from_spec(
        "Group",
        inputs={"input": (64, 128), "w_gate": (128, 256)},
        outputs={"left": (64, 256)},
        key="abc123",
        params={"seq_len": 64, "k": 2},
        generator=lambda self, image="elf": "generator",
    )
    op = Group()
    assert [b.name for b in op.buffers] == ["input", "w_gate", "left"]
    assert [b.shape for b in op.buffers] == [(64, 128), (128, 256), (64, 256)]
    assert (op.seq_len, op.k) == (64, 2)
    assert op.design_key() == "abc123"
    assert op.generator() == "generator"
    # Literal shapes bind no field; inference only checks them.
    assert infer(Group, (64, 128), (128, 256)) == {}
    with pytest.raises(ValueError):
        infer(Group, (64, 128), (128, 512))


# --------------------------------------------------------------------------
# The array tier, resolution, identity
# --------------------------------------------------------------------------


def test_the_array_tier_is_what_the_tiles_name_and_what_says_so():
    assert MV._array_fields == ("K", "columns", "tile_out", "epilogue")
    assert MV._param_fields == ("M", "K", "num_batches", "epilogue")
    assert MV._auto_fields == ("columns", "tile_out", "vec")


def test_an_operand_with_a_tile_is_its_own_stream():
    op = MV(M=1024, K=128).resolved(FakeDev(cols=8))
    assert {k: (s.count, s.shape) for k, s in op.streams.items()} == {
        "A": (8, (64, 128)),
        "B": (1, (128,)),
        "C": (8, (64,)),
    }
    assert op.A.count == 8 and op.A.tile == np.ndarray[(64, 128), np.dtype[op.A.dtype]]
    assert op.A.shape == (1024, 128) and op.C.shape == (1024,)
    op.A.lane(3).bind("h3")
    assert op.A.lane(3).handle == "h3"
    op.B.bind("hb")
    assert op.B.handle == "hb"


def test_a_derived_value_is_written_once_per_build():
    op = MV(M=1024, K=128).resolved(FakeDev(cols=8))
    assert list(op.residents) == ["count"] and op.values == []
    assert op.resident_values() == {"count": 2}


def test_a_value_a_graph_binds_is_per_call():
    @iron.graph
    def g(a, b, *, pos: Scratchpad[np.int32]):
        # A per-call value at a call site is not a field a checker knows (yet).
        return MV(a, b, columns=8, start=pos)  # pyright: ignore[reportCallIssue]

    t = g.trace(a=(1024, 128), b=(128,))
    (op,) = t.operators
    assert op.uses_value("start") and [v.name for v in op.values] == ["start"]
    assert [b.member.name for b in t.bindings] == ["start"]


def test_a_derived_value_a_graph_binds_is_per_call_and_no_longer_a_resident():
    @iron.graph
    def g(a, b, *, n: Scratchpad[np.int32]):
        return MV(a, b, columns=8, count=n)  # pyright: ignore[reportCallIssue]

    (op,) = g.trace(a=(1024, 128), b=(128,)).operators
    assert [v.name for v in op.values] == ["count"] and op.residents == {}
    assert op.resident_values() == {}  # the preamble writes nothing for it
    # What an instance binds per call is part of its identity: an array
    # reading the value from the scratchpad is not the one reading a resident.
    assert op.design_key() != MV(M=1024, K=128, columns=8).design_key()


def test_identity_is_the_array_tier_for_sharing_and_every_field_for_a_build():
    a = MV(M=1024, K=128).resolved(FakeDev(cols=8))
    b = MV(M=2048, K=128).resolved(FakeDev(cols=8))
    assert a.array_key() == b.array_key()
    assert a.design_key() != b.design_key()
    assert a.array_key() == (
        "MV",
        ("K", 128),
        ("columns", 8),
        ("tile_out", 64),
        ("epilogue", "none"),
    )


def test_array_sees_the_array_tier_alone():
    class Leaky(MV):
        def array(self, target):
            return self.M

    op = MV(M=1024, K=128).resolved(FakeDev(cols=8))
    assert op.build_array(None) == [128, 8, 64, "none"]
    with pytest.raises(TypeError, match="reads M, which no tile names"):
        Leaky(M=1024, K=128).resolved(FakeDev(cols=8)).build_array(None)


def test_resolution_fills_every_knob_or_says_which_it_left():
    with pytest.raises(Unresolvable, match="no vector width"):
        MV(M=1024, K=24).resolved(FakeDev())
    with pytest.raises(Incompatible, match="not a multiple"):
        MV(M=1000, K=128).resolved(FakeDev(cols=8))
    ok = MV(M=1024, K=128, columns=2)
    assert not ok._resolved
    r = ok.resolved(FakeDev(cols=8))
    assert r._resolved and r.resolved(FakeDev()) is r and (r.columns, r.vec) == (2, 64)
    assert ok.columns == 2 and ok.vec is None  # the original is untouched


def test_inference_binds_the_fields_from_the_operands():
    op = MV.from_operands(
        (3, 1024, 128),
        (
            3,
            128,
        ),
    )
    assert (op.M, op.K, op.num_batches) == (1024, 128, 3)
    assert op.A.shape == (3, 1024, 128)


# --------------------------------------------------------------------------
# Two behaviours the old shape functions existed to express, now declared
# --------------------------------------------------------------------------


def test_gemm_layout_flags_transpose_rather_than_resize():
    from iron.operators.gemm.op import GEMM

    plain = GEMM(M=256, K=64, N=512).buffers
    b_major = GEMM(M=256, K=64, N=512, b_col_maj=True).buffers
    c_major = GEMM(M=256, K=64, N=512, c_col_maj=True).buffers
    assert plain[1].shape == (64, 512) and b_major[1].shape == (512, 64)
    assert plain[2].shape == (256, 512) and c_major[2].shape == (512, 256)
    # Transposing a layout must not change how many bytes move.
    assert plain[1].nbytes == b_major[1].nbytes
    assert plain[2].nbytes == c_major[2].nbytes


def test_mha_pads_the_sequence_and_groups_kv():
    from iron.operators.mha.op import MHA

    grouped = MHA(num_heads=8, seq_len=100, num_KV_heads=2).buffers
    plain = MHA(num_heads=8, seq_len=100).buffers
    # 100 rounds up to 128, so Q is 8 heads x 128 x 64.
    assert grouped[0].shape == (8, 128, 64)
    # Grouped K/V are narrower than Q; plain K/V are exactly as wide.
    assert grouped[1].shape == (2, 128, 64)
    assert plain[1].shape == plain[0].shape
    assert [spec.direction for spec in grouped] == ["in", "in", "in", "out"]
