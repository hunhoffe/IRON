# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An image someone else built, declared as an operator's ``image=``."""

from __future__ import annotations

import functools
from pathlib import Path

from aie.utils.compile import NPU_CACHE_HOME
from aie.utils.npukernel import NPUKernel

from iron.common.fetch import fetch


class Xclbin(NPUKernel):
    """A downloaded xclbin, pinned by its SHA-256 and fetched on first read
    of ``xclbin_path``.

    Declared as a class's image (``class Shipped(GEMM, image=Xclbin(...))``)
    it has no instructions; a build runs ``with_instructions`` of it, the
    same image with the stream IRON generates for one shape.

    Args:
        url: Where the image is downloaded from, over https.
        sha256: The digest the download must have.
        filename: Its name in the JIT cache's ``prebuilt/`` directory.
        **kernel: `NPUKernel`'s other arguments (``kernel_name``, ...).

    Attributes:
        settings: Those other arguments, which ``with_instructions`` keeps.
    """

    def __init__(self, url: str, sha256: str, filename: str, **kernel) -> None:
        super().__init__(**kernel)
        self.url, self.sha256, self.filename = url, sha256, filename
        self.settings = kernel

    @functools.cached_property
    def xclbin_path(self) -> Path:
        return self.fetch()

    def with_instructions(self, insts_path, **kernel) -> Xclbin:
        """This image running ``insts_path``, with ``kernel`` (as `NPUKernel`
        takes it) over the declared settings.
        """
        run = Xclbin(
            self.url,
            self.sha256,
            self.filename,
            **{**self.settings, **kernel, "insts_path": insts_path},
        )
        if "xclbin_path" in vars(self):
            vars(run)["xclbin_path"] = self.xclbin_path
        return run

    def __call__(self, *args, **kwargs):
        if self.insts_path is None:
            raise TypeError(f"{self!r} has no instructions; call with_instructions()")
        return super().__call__(*args, **kwargs)

    def fetch(self, directory=None) -> Path:
        """The downloaded file, by digest: fetched unless a file of the pinned
        content is already there.

        Into the JIT cache's own root by default (``NPU_CACHE_HOME``'s
        ``prebuilt/``), so an external image is found where every other built
        artifact is and no caller has to name a directory for it.

        Raises:
            ValueError: The URL is not https.
            RuntimeError: The download has another digest.
        """
        if directory is None:
            directory = Path(NPU_CACHE_HOME) / "prebuilt"
        return fetch(self.url, self.sha256, Path(directory) / self.filename)

    def __repr__(self) -> str:
        return f"Xclbin({self.filename})"
