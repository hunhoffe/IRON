# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The elementwise template: what an operator that names one kernel gets."""

import pytest
from aie.iron.device import from_name

from iron.common import Incompatible, UnaryElementwise, Unresolvable
from iron.operators.elementwise_add import ElementwiseAdd
from iron.operators.relu import ReLU

pytestmark = pytest.mark.usefixtures("npu2")
NPU2 = from_name("npu2", n_cols=8)


def test_a_knob_free_operator_takes_the_widest_split_that_leaves_whole_lines():
    for size, cols in ((256, 1), (1024, 4), (2048, 8), (3072, 6), (8192, 8)):
        op = ReLU(size=size).resolved(NPU2)
        assert (op.num_aie_columns, op.num_channels, op.tile_size) == (cols, 1, 256)
        assert op.lines % op.cores == 0
    assert ElementwiseAdd(size=2048).resolved(NPU2).num_aie_columns == 8


def test_the_refusals_name_what_to_change():
    with pytest.raises(Unresolvable, match="tile_size=8192 exceeds the 4096"):
        ReLU(size=8192, tile_size=8192).resolved(NPU2)
    with pytest.raises(Incompatible, match="give a tile_size= or num_aie_columns="):
        ReLU(size=1000).resolved(NPU2)
    with pytest.raises(Unresolvable, match="none is bound and none was given"):
        ReLU(size=1024).resolved(None)


def test_the_trip_count_is_a_resident_the_build_writes():
    op = ReLU(size=2048, tile_size=512).resolved(NPU2)
    assert op.cores == 4 and op.lines == 4
    assert op.resident_values() == {"count": 1}
    lines = op.explain().splitlines()
    assert lines[0].endswith("(resolved)") and "tile_size=512" in lines[1]
    assert lines[-1] == "  count: written once per build, 1 here"


def test_an_operator_written_by_inheritance_inherits_the_sweep():
    class Neg(UnaryElementwise):
        def kernel(self, target):
            raise NotImplementedError

        def reference(self, x):
            return -x

    assert Neg.test is not None
    cases = Neg.test.resolve(Neg)
    assert all(c.kwargs["tile_size"] <= Neg.tile_cap for c in cases)
    assert {c.kwargs["num_aie_columns"] for c in cases} == {1, 2, 4, 8}  # divide 2^n
