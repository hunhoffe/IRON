# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A full ELF in a store of prebuilt images loads with no compile, and a
graph whose image no store holds compiles, toolchain-free: a build is
stood in by one that refuses.
"""

import pytest
from aie.utils.compile.jit.compilabledesign import CompilableDesign

from iron.common.image import prebuilt
from iron.tests.support.tools import DEVICES, swiglu


class Compiled(Exception):
    pass


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store, the only one, where every build refuses; and the
    keys looked up in it.
    """
    monkeypatch.setenv("IRON_PREBUILT", str(tmp_path))

    def compile(self, *args, **kwargs):
        raise Compiled

    monkeypatch.setattr(CompilableDesign, "compile", compile)
    keys = []
    find = prebuilt.find
    monkeypatch.setattr(prebuilt, "find", lambda key: keys.append(key) or find(key))
    return tmp_path, keys


def _compile():
    fn, E = swiglu()
    return fn.compile(dev=DEVICES["npu2"](), full_elf=True, x=(1, E))


def test_an_image_the_store_holds_is_loaded_and_one_it_lacks_is_built(store):
    directory, keys = store
    with pytest.raises(Compiled):
        _compile()
    [key] = keys
    (directory / key).mkdir()
    (directory / key / "design.elf").write_bytes(b"\x7fELF")
    (directory / key / "params.txt").write_text("1\nposition 0 i32 core 0 0\n")
    net = _compile()
    assert keys == [key, key], "the same graph composes to the same key"
    assert net.image == directory / key / "design.elf"
    assert [p.name for p in net.artifacts.parameter_table] == ["position"]
