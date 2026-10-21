# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full ELFs another host built, found by what aiecc would be handed, so a
host without the toolchain loads them with no compile.

mlir-aie's cache keys each entry on the tools' paths and times and on its
modules' files, so it misses on any other install. Here an image is keyed
on its composed module's text, its aiecc flags, the mlir-aie release that
compiles it (which pins Peano) and the kernel sources' content: what
decides the bytes aiecc writes. A store is a directory holding
``<key>/design.elf`` and ``<key>/params.txt``; ``IRON_PREBUILT`` names the
stores, separated as ``PATH`` is, and a build adds its image to the first.
"""

import functools
import hashlib
import importlib.metadata
import os
import shutil
from pathlib import Path

from aie.utils.benchmark import kernel_tree_digest
from aie.utils.compile.jit.compilabledesign import CacheEntry

#: What a full ELF's load reads of its build.
FILES = ("design.elf", "params.txt")


def stores() -> list[Path]:
    """The stores ``IRON_PREBUILT`` names, in its order."""
    return [Path(p) for p in os.environ.get("IRON_PREBUILT", "").split(os.pathsep) if p]


@functools.cache
def _toolchain() -> str:
    return f"{importlib.metadata.version('mlir_aie')}|{kernel_tree_digest()}"


def key(text: str, flags) -> str:
    """The key of the image aiecc builds from ``text`` with ``flags``."""
    h = hashlib.sha256(text.encode())
    h.update(f"|{sorted(flags)}|{_toolchain()}".encode())
    return h.hexdigest()[:24]


def find(key: str) -> CacheEntry | None:
    """The image under ``key`` in the first store holding it, as an entry."""
    for store in stores():
        directory = store / key
        if all((directory / f).is_file() for f in FILES):
            elf, params = (directory / f for f in FILES)
            return CacheEntry(
                directory=directory,
                xclbin=None,
                insts=None,
                elf=elf,
                full_elf_kernel=None,
                pdis=(),
                params=params,
                lowered_mlir=None,
                objects=(),
                manifest=None,
                dispatch_library=None,
            )
    return None


def add(key: str, entry: CacheEntry) -> None:
    """Copy ``entry``'s image into the first store, under ``key``."""
    if not stores():
        return
    directory = stores()[0] / key
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in zip(FILES, (entry.elf, entry.params)):
        assert path is not None, f"a full ELF's build has its {name}"
        shutil.copyfile(path, directory / name)
