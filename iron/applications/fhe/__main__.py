# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private search: score drawn int8 documents against a query the scorer
only ever holds encrypted, and time each side.

    python -m iron.applications.fhe [--docs 4096] [--dim 128] [--cpu]
"""

import argparse
import time

import numpy as np

from iron.applications import fhe


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--cpu", action="store_true", help="no NPU: numpy's int8 products"
    )
    args = parser.parse_args()

    rng = np.random.default_rng()
    docs = rng.integers(-128, 128, (args.docs, args.dim))
    query = rng.integers(-128, 128, args.dim)
    key = fhe.BFV(matmul=fhe.cpu_matmul if args.cpu else fhe.npu_matmul).keygen()

    def timed(label, f, *a):
        start = time.perf_counter()
        out = f(*a)
        print(f"{label:>8}: {1e3 * (time.perf_counter() - start):8.1f} ms")
        return out

    encrypted = timed("encrypt", key.encrypt, query)
    scores = timed("score", lambda: docs @ encrypted)
    found = timed("decrypt", key.decrypt, scores)
    assert (found == docs @ query).all()
    print(f"best of {args.docs}: document {found.argmax()}, score {found.max()}")


if __name__ == "__main__":
    main()
