# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulated annealing of a QUBO, many replicas at once, on the NPU.

``anneal(Q)`` returns the lowest-energy binary ``x`` it finds for
``E(x) = x @ Q @ x``. Each replica is a textbook heat-bath annealer (Kirkpatrick
et al., 1983) on a geometric temperature schedule. Its variables are split
into a few blocks, and a block takes its draws at once from the fields of all
variables, so a sweep is a matrix product per block over every replica.

The device runs ``Sweeps``: per block, the fields ``W @ x`` (one GEMM over the
replicas), minus the host's noise, clamped to ``[0, 1]``. The noise holds all
the randomness and the temperature: ``x_i`` becomes 1 with probability
``sigmoid(f_i / T)`` exactly when the field ``f_i`` exceeds a logistic draw of
scale ``T``. Couplings and noise are scaled by one power of two, so that clamp
is a step and every variable stays 0 or 1.
"""

import dataclasses

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron.common import Precision
from iron.operators.clamp import Clamp
from iron.operators.gemm import GEMM

#: Variables are padded to a whole number of blocks of this many.
ALIGN = 64
#: The largest coupling after scaling; any power of two well inside bf16 and
#: f32's sums would do.
SCALED = 2.0**64


class Sweeps(iron.Graph):
    """Annealing sweeps over ``replicas`` copies of a QUBO's variables,
    one sweep per leading row of ``noise``.

    Args:
        W: The fields' couplings, ``-2 * (Q + Q.T) / 2`` off the diagonal,
            already scaled.
        replicas: How many replicas each sweep updates.
        blocks: How many blocks a sweep updates in turn.
    """

    def __init__(self, W: np.ndarray, replicas: int, blocks: int):
        n = len(W) // blocks
        self.couplings = [
            iron.param(W[b * n : (b + 1) * n].astype(bfloat16)) for b in range(blocks)
        ]
        # Variables by row, replicas by column: a block is a run of rows.
        self.x = iron.state((len(W), replicas))

    def body(self, noise):
        n = self.couplings[0].shape[0]
        for z in noise:
            for b, w in enumerate(self.couplings):
                rows = slice(b * n, (b + 1) * n)
                field = GEMM.step(w, self.x, precision=Precision.HIGHEST)
                Clamp.step(field - z[rows], self.x[rows], low=0.0, high=1.0)


@dataclasses.dataclass(frozen=True)
class Result:
    """What ``anneal`` found.

    Attributes:
        x: The lowest-energy sample, ``(n,)`` of 0 and 1.
        energy: Its ``x @ Q @ x``.
        samples: Every replica's final sample, ``(replicas, n)``.
        energies: Each sample's energy.
    """

    x: np.ndarray
    energy: float
    samples: np.ndarray
    energies: np.ndarray


def energies(Q, samples) -> np.ndarray:
    """``x @ Q @ x`` for each row ``x`` of ``samples``, in float64."""
    s = np.asarray(samples, np.float64)
    return np.einsum("ri,ij,rj->r", s, np.asarray(Q, np.float64), s)


def temperatures(Q, sweeps: int) -> np.ndarray:
    """A geometric schedule from hot, where the largest change one flip
    can make is taken half the time, to cold, where the smallest is taken
    one time in a hundred (dwave-neal's default).
    """
    Q = (Q + Q.T) / 2
    off = np.abs(Q - np.diag(np.diag(Q)))
    largest = (np.abs(np.diag(Q)) + 2 * off.sum(1)).max()
    steps = np.abs(np.concatenate([np.diag(Q), 2 * off[off > 0]]))
    smallest = steps[steps > 0].min()
    return np.geomspace(largest / np.log(2), smallest / np.log(100), sweeps)


def anneal(
    Q,
    *,
    replicas: int = 64,
    sweeps: int = 1000,
    blocks: int = 4,
    per_call: int = 10,
    seed=None,
    device: bool = True,
) -> Result:
    """Anneal ``replicas`` copies of the QUBO ``Q`` and return the best.

    Args:
        Q: ``(n, n)``; only ``(Q + Q.T) / 2`` matters.
        replicas: Independent annealers run side by side.
        sweeps: Updates of every variable, on ``temperatures(Q, sweeps)``.
        blocks: Groups of variables a sweep updates in turn.
        per_call: Sweeps the device runs per call.
        seed: For ``numpy.random.default_rng``.
        device: On the NPU, else the graph's reference on the host.
    """
    Q = np.asarray(Q, np.float64)
    n = len(Q)
    sym = (Q + Q.T) / 2
    size = -(-n // (blocks * ALIGN)) * blocks * ALIGN
    W = np.zeros((size, size))
    W[:n, :n] = -2 * (sym - np.diag(np.diag(sym)))
    scale = 2.0 ** np.floor(np.log2(SCALED / max(np.abs(W).max(), 1e-30)))
    bias = np.zeros(size)
    bias[:n] = np.diag(sym)

    graph = Sweeps(W * scale, replicas, blocks)
    rng = np.random.default_rng(seed)
    schedule = temperatures(Q, sweeps)
    schedule = np.concatenate([schedule, np.full(-sweeps % per_call, schedule[-1])])
    x0 = rng.integers(0, 2, (size, replicas)).astype(bfloat16)
    if device:
        version = graph.compile(noise=((per_call, size, replicas), bfloat16))
        version.write(graph.x, x0)
    else:
        graph.x.host = x0
    for T in schedule.reshape(-1, per_call):
        u = rng.random((per_call, size, replicas))
        z = T[:, None, None] * np.log(u / (1 - u)) + bias[:, None]
        noise = (z * scale).astype(bfloat16)
        if device:
            version(noise)
        else:
            graph.reference(noise)
    x = version.read(graph.x) if device else graph.x.host
    samples = (np.asarray(x, np.float32)[:n].T > 0.5).astype(np.int8)
    e = energies(Q, samples)
    best = int(np.argmin(e))
    return Result(samples[best], float(e[best]), samples, e)
