# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The negacyclic number-theoretic transform as int8 matrix products.

Multiplying polynomials in ``Z_p[x]/(x^n + 1)`` is pointwise after a
negacyclic NTT, and an NTT of length ``n = n1 * n2`` is two dense
transforms, ``n2`` of length ``n1`` then ``n1`` of length ``n2``, with a
pointwise twist between them (the four-step NTT). Each dense transform is a
constant matrix times data, modulo ``p``.

A constant matrix ``W`` times residues ``x`` modulo ``p`` runs on an int8
matrix unit exactly: split ``x`` into four signed bytes, ``x = sum_k x_k
256^k``; pre-reduce ``W 256^k mod p`` and split it into bytes too. Then
``W x = sum_j 256^j sum_k W_kj x_k``: one int8 matrix product of a ``(4m,
4n)`` weight by the ``(4n, columns)`` stacked bytes, whose four row blocks
are recombined by shifts and one reduction. Every partial sum fits int32.
This is CROSS's basis-aligned transformation (Georgia Tech and Google,
arXiv 2501.07047, MIT) on IRON's int8 ``GEMM``.
"""

import functools

import numpy as np

from iron.operators.gemm import GEMM

#: Signed bytes a residue below 2^30 splits into.
DIGITS = 4


def digits(x: np.ndarray) -> np.ndarray:
    """``x`` (int64, ``|x| < 2^31``) as ``DIGITS`` signed bytes stacked on a
    new first axis, least significant first: ``x = sum_k out[k] 256^k``.
    """
    out = np.empty((DIGITS, *x.shape), np.int8)
    for k in range(DIGITS):
        d = ((x + 128) & 255) - 128
        out[k] = d
        x = (x - d) >> 8
    return out


def centred(x: np.ndarray, p: int) -> np.ndarray:
    """``x mod p`` in ``[-p/2, p/2)``."""
    return (x + p // 2) % p - p // 2


def cpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` of int8 operands in int32, on the host."""
    return a.astype(np.int32) @ b.astype(np.int32)


@functools.cache
def gemm(M: int, K: int, N: int) -> GEMM:
    """The int8 ``GEMM`` that computes an ``(M, K) @ (K, N)`` product."""
    return GEMM(M=M, K=K, N=N, input_dtype=np.int8, output_dtype=np.int32)


def npu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` of int8 operands in int32, on the NPU."""
    return gemm(*a.shape, b.shape[1])(a, b)


class ModMatmul:
    """``w @ x mod p`` for a constant ``w``, as one int8 matrix product."""

    def __init__(self, w: np.ndarray, p: int, matmul):
        self.p, self.matmul = p, matmul
        m, n = w.shape
        shifted = [centred(w * pow(256, k, p) % p, p) for k in range(DIGITS)]
        # Row block j, column block k: byte j of w 256^k mod p.
        blocks = digits(np.stack(shifted))  # (j, k, m, n)
        self.weight = blocks.transpose(0, 2, 1, 3).reshape(DIGITS * m, DIGITS * n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``w @ x mod p`` of residues ``x`` (int64, ``(n, columns)``)."""
        stacked = digits(centred(x, self.p)).reshape(-1, x.shape[1])
        z = self.matmul(self.weight, stacked).astype(np.int64)
        z = z.reshape(DIGITS, -1, x.shape[1])
        return sum(z[j] << (8 * j) for j in range(DIGITS)) % self.p


class NTT:
    """The negacyclic NTT over ``Z_p[x]/(x^n + 1)`` and its inverse, each
    two int8 matrix products by ``matmul`` and a pointwise twist.

    ``forward`` and ``inverse`` take residues (int64) of any leading shape
    and length ``n``; ``forward(a) * forward(b) % p`` is ``forward`` of the
    negacyclic product.
    """

    def __init__(self, n: int, p: int, matmul=npu_matmul):
        if (p - 1) % (2 * n):
            raise ValueError(f"NTT: p ({p}) is not 1 mod 2n ({2 * n})")
        self.n, self.p = n, p
        self.n1 = 1 << (n.bit_length() - 1) // 2
        self.n2 = n // self.n1
        psi = _root(n, p)
        powers = np.array([pow(psi, e, p) for e in range(2 * n)], np.int64)
        inverse = np.roll(powers[::-1], 1)  # psi^-e

        k1 = np.arange(self.n1)[:, None]
        i1, i2 = np.arange(self.n1), np.arange(self.n2)
        # psi^(n2 i1 (2 k1 + 1)), psi^(i2 (2 k1 + 1)), psi^(2 n1 i2 k2)
        e1 = self.n2 * i1 * (2 * k1 + 1) % (2 * n)
        twist = i2 * (2 * k1 + 1) % (2 * n)
        e2 = 2 * self.n1 * np.outer(i2, i2) % (2 * n)
        n_inv = pow(n, -1, p)
        self.twist, self.untwist = powers[twist], inverse[twist]
        self.step1 = ModMatmul(powers[e1], p, matmul)
        self.step2 = ModMatmul(powers[e2], p, matmul)
        self.unstep2 = ModMatmul(inverse[e2], p, matmul)
        self.unstep1 = ModMatmul(inverse[e1].T * n_inv % p, p, matmul)

    def forward(self, a: np.ndarray) -> np.ndarray:
        n1, n2, shape = self.n1, self.n2, a.shape
        # Coefficient n2 i1 + i2 at (i1, i2), each polynomial n2 columns.
        x = a.reshape(-1, n1, n2).transpose(1, 0, 2).reshape(n1, -1)
        y = self.step1(x).reshape(n1, -1, n2) * self.twist[:, None] % self.p
        z = self.step2(y.transpose(2, 1, 0).reshape(n2, -1))
        # Value k1 + n1 k2 at (k2, k1).
        return z.reshape(n2, -1, n1).transpose(1, 0, 2).reshape(shape)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        n1, n2, shape = self.n1, self.n2, a.shape
        z = a.reshape(-1, n2, n1).transpose(1, 0, 2).reshape(n2, -1)
        y = self.unstep2(z).reshape(n2, -1, n1) * self.untwist.T[:, None] % self.p
        x = self.unstep1(y.transpose(2, 1, 0).reshape(n1, -1))
        return x.reshape(n1, -1, n2).transpose(1, 0, 2).reshape(shape)


def primes(n: int, count: int, bits: int = 30) -> tuple[int, ...]:
    """The ``count`` largest primes below ``2^bits`` that are 1 mod ``2n``,
    so each has the ``2n``-th roots of unity a negacyclic NTT needs.
    """
    found, p = [], (1 << bits) - (1 << bits) % (2 * n) + 1
    while len(found) < count:
        p -= 2 * n
        if _prime(p):
            found.append(p)
    return tuple(found)


def _prime(p: int) -> bool:
    """Miller-Rabin with the bases that decide every ``p < 3.4e14``."""
    d, s = p - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % p == 0:
            continue
        x = pow(a, d, p)
        if x in (1, p - 1):
            continue
        if all((x := x * x % p) != p - 1 for _ in range(s - 1)):
            return False
    return True


def _root(n: int, p: int) -> int:
    """A primitive ``2n``-th root of unity modulo ``p``: one whose ``n``-th
    power is -1.
    """
    for g in range(2, p):
        psi = pow(g, (p - 1) // (2 * n), p)
        if pow(psi, n, p) == p - 1:
            return psi
    raise ValueError(f"no primitive {2 * n}-th root modulo {p}")
