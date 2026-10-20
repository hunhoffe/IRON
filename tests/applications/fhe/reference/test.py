# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FHE showcase without a device: the NTT against schoolbook
negacyclic products, its int8 products through GEMM's own reference and
lowered to NPU instructions, and BFV's encrypted dot products against
numpy's.
"""

import numpy as np
import pytest

from iron.applications import fhe
from iron.applications.fhe import bfv
from iron.applications.fhe.ntt import NTT, cpu_matmul, gemm, primes
from iron.tests.support.lowering import lower
from iron.tests.support.tools import requires

RNG = np.random.default_rng(0)


def negacyclic(a, b, p):
    """``a * b`` in ``Z_p[x]/(x^n + 1)``, term by term."""
    n = a.shape[-1]
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), np.int64)
    for i in range(n):
        term = a[..., i : i + 1] * b % p
        out[..., i:] += term[..., : n - i]
        out[..., :i] -= term[..., n - i :]
        out %= p
    return out


@pytest.mark.parametrize("n", [16, 32, 4096])
def test_ntt_multiplies_negacyclically(n):
    p = primes(n, count=1)[0]
    ntt = NTT(n, p, cpu_matmul)
    a, b = RNG.integers(0, p, (2, 3, n))
    assert (ntt.inverse(ntt.forward(a)) == a).all()
    product = ntt.inverse(ntt.forward(a) * ntt.forward(b) % p)
    assert (product == negacyclic(a, b, p)).all()


def test_ntt_runs_on_gemms_reference(npu2):
    """The int8 products GEMM declares give the host's, so the NPU's
    (an integer GEMM is held to its reference exactly).
    """
    matmul = lambda a, b: gemm(*a.shape, b.shape[1]).resolved(npu2).reference(a, b)
    p = primes(bfv.N, count=1)[0]
    a = RNG.integers(0, p, (2, bfv.N))
    on_gemm, on_host = NTT(bfv.N, p, matmul), NTT(bfv.N, p, cpu_matmul)
    assert (on_gemm.forward(a) == on_host.forward(a)).all()


@requires("aiecc", "peano")
@pytest.mark.parametrize("polynomials", [1, 32])
def test_ntt_gemm_lowers_to_instructions(npu2, polynomials, tmp_path):
    ntt = NTT(bfv.N, primes(bfv.N, count=1)[0], cpu_matmul)
    M, K = ntt.step1.weight.shape
    lower(gemm(M, K, ntt.n2 * polynomials).resolved(npu2), tmp_path)


@pytest.fixture(scope="module")
def key():
    return fhe.BFV(matmul=cpu_matmul).keygen()


def test_decrypts_what_it_encrypts(key):
    x = RNG.integers(-128, 128, 100)
    assert (key.decrypt(key.encrypt(x)) == x).all()


@pytest.mark.parametrize("rows,d", [(1000, 128), (5, 4096), (33, 1)])
def test_encrypted_dot_products_are_numpys(key, rows, d):
    docs = RNG.integers(-128, 128, (rows, d))
    x = RNG.integers(-128, 128, d)
    assert (key.decrypt(docs @ key.encrypt(x)) == docs @ x).all()


@pytest.mark.parametrize("v", [-128, 127])
def test_the_largest_dot_products_are_exact(key, v):
    docs = np.array([[-128] * bfv.N, [127] * bfv.N])
    x = np.full(bfv.N, v)
    assert (key.decrypt(docs @ key.encrypt(x)) == docs @ x).all()


def product(key):
    return np.ones((1, 1), int) @ (np.ones((1, 2), int) @ key.encrypt([1, 2]))


@pytest.mark.parametrize(
    "call,match",
    [
        pytest.param(lambda key: key.encrypt([128]), "int8", id="wide"),
        pytest.param(lambda key: key.encrypt([0.5]), "int8", id="float"),
        pytest.param(lambda key: key.encrypt([0] * (bfv.N + 1)), "at most", id="long"),
        pytest.param(
            lambda key: np.zeros((2, 3), int) @ key.encrypt([1, 2]),
            "2 columns",
            id="columns",
        ),
        pytest.param(product, "freshly", id="product"),
    ],
)
def test_refuses(key, call, match):
    with pytest.raises(ValueError, match=match):
        call(key)
