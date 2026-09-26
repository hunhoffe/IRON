# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one-class operator: what it declares, and what the library reads of it.

A worked operator close to GEMV: its operands are their own streams
(``tile=``), a ``Value`` derives the core's trip count from the extents, one
field the array reads is declared so. No device, no toolchain.
"""

import dataclasses

import numpy as np
import pytest

import iron
from iron.common.declare import (
    In,
    Incompatible,
    Operator,
    Out,
    Scratchpad,
    Unresolvable,
    Value,
    auto,
    optional,
    param,
)
from iron.tests.common.declare import FakeDev


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


def test_the_array_tier_is_what_the_tiles_name_and_what_says_so():
    assert MV._array_fields == ("K", "columns", "tile_out", "epilogue")
    assert MV._param_fields == ("M", "K", "num_batches", "epilogue")
    assert MV._auto_fields == ("columns", "tile_out", "vec")


def test_an_operand_with_a_tile_is_its_own_stream():
    op = MV(M=1024, K=128).resolved(FakeDev(cols=8))
    assert op.ov is op and op.merged
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
