# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encrypted dot products: a server multiplies a plaintext matrix by a
vector it only ever sees encrypted, and the key holder decrypts the exact
integer products.

The scheme is BFV (Brakerski 2012; Fan and Vercauteren 2012) with
secret-key encryption in the ring ``Z_q[x]/(x^n + 1)``, ``q`` the product
of two 30-bit primes. A vector is encrypted as the low coefficients of one
polynomial. ``matrix @ encrypted`` packs ``n // d`` rows of length ``d`` per
plaintext polynomial, each reversed at its own offset, so that one
ciphertext-times-plaintext product holds each row's dot product at one
coefficient: no rotations, no key switching, no bootstrapping. This is
Cheetah's coefficient encoding (Huang et al., USENIX Security 2022). The
products are NTTs (``ntt.NTT``), which run on the NPU as int8 matrix
products.

``n = 4096`` with ``log2 q`` about 60, a ternary secret and centred
binomial noise of standard deviation about 3.2 is within the Homomorphic
Encryption Standard's 128-bit bound (109 bits of ``q`` at that ``n``).
Randomness is ``os.urandom``. A showcase, not an audited library.
"""

import dataclasses
import os

import numpy as np

from .ntt import NTT, npu_matmul, primes

#: Ring degree, and the plaintext modulus: room for any int8 dot product
#: of length up to ``N``: ``|score| <= N 128^2 = 2^26 < T / 2``.
N = 4096
T = 1 << 28


class BFV:
    """The ring and moduli keys are made in; ``matmul`` runs the NTTs' int8
    matrix products (``ntt.cpu_matmul`` for the host).
    """

    def __init__(self, matmul=npu_matmul):
        self.primes = primes(N, count=2)
        self.q = int(np.prod(self.primes, dtype=object))
        self.ntts = [NTT(N, p, matmul) for p in self.primes]

    def keygen(self) -> "SecretKey":
        s = _uniform(N, 3) - 1
        return SecretKey(self, np.stack([ntt.forward(s % ntt.p) for ntt in self.ntts]))

    def multiply(self, c: np.ndarray, p_hat: np.ndarray) -> np.ndarray:
        """Each prime's residues ``c`` (``(L, ..., N)``) times polynomials
        already in the NTT domain, broadcast over leading axes.
        """
        return np.stack(
            [
                ntt.inverse(ntt.forward(ci) * pi % ntt.p)
                for ntt, ci, pi in zip(self.ntts, c, p_hat)
            ]
        )


@dataclasses.dataclass(frozen=True, eq=False)
class SecretKey:
    """A ternary secret ``s``, held as its NTT for each prime."""

    scheme: BFV
    s_hat: np.ndarray

    def encrypt(self, values) -> "Encrypted":
        """An int8 vector of length at most ``N``, encrypted."""
        m = _int8(values, "encrypt")
        if m.ndim != 1 or len(m) > N:
            raise ValueError(f"encrypt: a vector of at most {N} values, not {m.shape}")
        d, m = len(m), np.pad(m, (0, N - len(m)))
        scheme, e = self.scheme, _noise(N)
        delta = scheme.q // T
        c0, c1 = [], []
        for ntt, s_hat in zip(scheme.ntts, self.s_hat):
            p = ntt.p
            a = _uniform(N, p)
            a_s = ntt.inverse(ntt.forward(a) * s_hat % p)
            c0.append((e - a_s + delta % p * (m % p)) % p)
            c1.append(a)
        c = np.stack([c0, c1])[:, :, None]  # (2, L, 1, N)
        return Encrypted(scheme, c, np.arange(d))

    def decrypt(self, encrypted: "Encrypted") -> np.ndarray:
        """The integers ``encrypted`` holds."""
        scheme, (c0, c1) = encrypted.scheme, encrypted.c
        v = (c0 + scheme.multiply(c1, self.s_hat[:, None])) % np.array(
            scheme.primes
        ).reshape(-1, 1, 1)
        v = v.reshape(len(scheme.primes), -1)[:, encrypted.at].astype(object)
        q = scheme.q
        x = sum(vi * (q // p) * pow(q // p, -1, p) for vi, p in zip(v, scheme.primes))
        x = (x + q // 2) % q - q // 2
        m = (x * T + q // 2) // q
        return ((m + T // 2) % T - T // 2).astype(np.int64)


@dataclasses.dataclass(frozen=True, eq=False)
class Encrypted:
    """An encrypted integer vector: ciphertext residues ``c`` (``c0, c1`` by
    prime by polynomial, ``(2, L, K, N)``), the vector's elements at flat
    coefficient positions ``at``. ``matrix @ encrypted`` multiplies by an
    int8 matrix the server holds in the clear.
    """

    scheme: BFV
    c: np.ndarray
    at: np.ndarray

    # numpy defers ``ndarray @ Encrypted`` to ``__rmatmul__``.
    __array_ufunc__ = None

    def __len__(self) -> int:
        return len(self.at)

    def __rmatmul__(self, matrix) -> "Encrypted":
        rows = _int8(matrix, "@")
        d = len(self)
        if rows.ndim != 2 or rows.shape[1] != d:
            raise ValueError(f"@: a matrix of {d} columns, not {rows.shape}")
        if self.c.shape[2] != 1 or not (self.at == np.arange(d)).all():
            raise ValueError("@: only a freshly encrypted vector")
        # Row k * per + j, reversed, at coefficients j d .. j d + d - 1 of
        # plaintext k, so its dot product lands at coefficient j d + d - 1.
        per = N // d
        count = -(-len(rows) // per)
        packed = np.zeros((count * per, d), np.int64)
        packed[: len(rows)] = rows[:, ::-1]
        packed = np.pad(packed.reshape(count, per * d), ((0, 0), (0, N - per * d)))
        p_hat = [ntt.forward(packed % ntt.p) for ntt in self.scheme.ntts]
        c = np.stack([self.scheme.multiply(ci, p_hat) for ci in self.c])
        r = np.arange(len(rows))
        return Encrypted(self.scheme, c, r // per * N + r % per * d + d - 1)


def _int8(values, name: str) -> np.ndarray:
    x = np.asarray(values)
    if (
        not np.issubdtype(x.dtype, np.integer)
        or x.size
        and (x.min() < -128 or x.max() > 127)
    ):
        raise ValueError(f"{name}: int8 values, not {x.dtype} in that range")
    return x.astype(np.int64)


def _uniform(size: int, bound: int) -> np.ndarray:
    """``size`` integers uniform in ``[0, bound)``, within 2^-34 for any
    ``bound`` below 2^30.
    """
    words = np.frombuffer(os.urandom(8 * size), np.uint64)
    return (words % np.uint64(bound)).astype(np.int64)


def _noise(size: int) -> np.ndarray:
    """Centred binomial noise: 21 coin flips less 21 more, sigma about 3.24."""
    words = np.frombuffer(os.urandom(8 * size), np.uint64)
    half = np.uint64((1 << 21) - 1)
    ones = np.bitwise_count(words & half).astype(np.int64)
    return ones - np.bitwise_count((words >> np.uint64(21)) & half)
