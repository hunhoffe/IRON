# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A file downloaded once, pinned by its digest."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


def fetch(url: str, sha256: str, target: Path) -> Path:
    """``target``, by digest: downloaded from ``url`` unless a file of the
    pinned ``sha256`` content is already there.

    Raises:
        ValueError: ``url`` is not https.
        RuntimeError: What ``url`` serves has another digest.
    """
    if target.exists():
        with open(target, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == sha256:
                return target
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download over {url!r}")
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if (got := hashlib.sha256(content).hexdigest()) != sha256:
        raise RuntimeError(f"{url} has SHA-256 {got}, expected {sha256}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # A file of this fetch's own, renamed over the target whole: concurrent
    # fetches never share one, and none sees a partial target.
    partial = tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f"{target.name}.", suffix=".part", delete=False
    )
    try:
        with partial:
            partial.write(content)
            # A temporary file is created owner-only; the cache is read as any file is.
            os.fchmod(partial.fileno(), 0o644)
        os.replace(partial.name, target)
    except BaseException:
        os.unlink(partial.name)
        raise
    return target
