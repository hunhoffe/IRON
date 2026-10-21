# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m iron.applications.compressor FILE`` writes ``FILE.irnz``
and prints its bits per byte beside xz's, bzip2's and gzip's;
``-d FILE.irnz`` writes ``FILE`` back.
"""

import argparse
import bz2
import lzma
import time
import zlib
from pathlib import Path

from iron.applications.qwen3 import Qwen3
from iron.lm.cli import parse_args

from . import compress, decompress

SUFFIX = ".irnz"


def main(args=None):
    parser = argparse.ArgumentParser(description="Lossless text compression on the NPU")
    parser.add_argument("file", type=Path)
    parser.add_argument("-d", "--decompress", action="store_true")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="a Qwen3 checkpoint's directory or repository id (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="the context each block of tokens is coded in; decompressing "
        "needs the same (default: %(default)s)",
    )
    args = parse_args(parser, args)
    if args.decompress and args.file.suffix != SUFFIX:
        parser.error(f"{args.file} does not end in {SUFFIX}")
    model = Qwen3.load(args.model, max_seq_len=args.max_seq_len)
    data = args.file.read_bytes()
    began = time.perf_counter()
    if args.decompress:
        out, path = decompress(data, model), args.file.with_suffix("")
    else:
        out, path = compress(data, model), args.file.with_name(args.file.name + SUFFIX)
    seconds = time.perf_counter() - began
    path.write_bytes(out)
    text = out if args.decompress else data
    print(f"{path}: {len(text) / seconds / 1e3:.2f} KB/s of text")
    if data and not args.decompress:
        for name, size in [
            ("irnz", len(out)),
            ("xz -9e", len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))),
            ("bzip2 -9", len(bz2.compress(data, 9))),
            ("gzip -9", len(zlib.compress(data, 9))),
        ]:
            print(
                f"{name:>9}: {size:>9} bytes, {8 * size / len(data):.3f} bits per byte"
            )


if __name__ == "__main__":
    main()
