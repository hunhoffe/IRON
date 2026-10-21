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

import argparse
import dataclasses
import time

import aie.utils as aie_utils
import numpy as np
from aie.iron.device import NPU2
from ml_dtypes import bfloat16

import iron
from iron.common import Precision
from iron.lm.cli import parse_args
from iron.operators.clamp import Clamp
from iron.operators.gemm import GEMM

#: Blocks of variables a sweep updates in turn: four match sequential
#: annealing's energies on dense and sparse instances, where one oscillates.
BLOCKS = 4
#: Variables are padded to a whole number of blocks of this many.
ALIGN = 64
#: Sweeps the device runs per call.
PER_CALL = 10
#: The hottest temperature after scaling: a power of two that keeps the
#: fields and draws well inside bf16's range.
SCALED = 2.0**64


class Sweeps(iron.Graph):
    """Annealing sweeps over ``replicas`` copies of a QUBO's variables,
    one sweep per leading row of ``noise``.

    Args:
        W: The fields' couplings, ``-(Q + Q.T)`` off the diagonal, scaled.
        replicas: How many replicas each sweep updates.
    """

    def __init__(self, W: np.ndarray, replicas: int):
        n = len(W) // BLOCKS
        self.couplings = [
            iron.param(W[b * n : (b + 1) * n].astype(bfloat16)) for b in range(BLOCKS)
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


def planted(n: int, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """A dense QUBO whose every coupling agrees with a drawn ``x``, so
    ``x`` is its ground state, and that ``x``: an Ising ferromagnet with
    Gaussian couplings, its spins flipped where ``x`` is 0.
    """
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], n)
    J = np.triu(np.abs(rng.standard_normal((n, n))), 1)
    J = (J + J.T) * np.outer(s, s)
    # -s @ J @ s / 2 with s = 2x - 1, up to a constant.
    return -2 * J + np.diag(2 * J.sum(1)), (s > 0).astype(np.int8)


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
    Q, *, replicas: int = 64, sweeps: int = 1000, seed=None, device: bool = True
) -> Result:
    """Anneal ``replicas`` copies of the QUBO ``Q`` and return the best.

    Every replica starts at zero, which the first, hottest sweeps forget.

    Args:
        Q: ``(n, n)``; only ``(Q + Q.T) / 2`` matters.
        replicas: Independent annealers run side by side.
        sweeps: Updates of every variable, on ``temperatures(Q, sweeps)``.
        seed: For ``numpy.random.default_rng``.
        device: On the NPU, else the same sweeps by the graph's reference
            on the host, which resolves its operators for the bound device.
    """
    Q = np.asarray(Q, np.float64)
    n = len(Q)
    sym = (Q + Q.T) / 2
    size = -(-n // (BLOCKS * ALIGN)) * BLOCKS * ALIGN
    schedule = temperatures(Q, sweeps)
    schedule = np.concatenate([schedule, np.full(-sweeps % PER_CALL, schedule[-1])])
    # The hottest temperature bounds every field.
    scale = 2.0 ** np.floor(np.log2(SCALED / schedule[0]))
    W = np.zeros((size, size))
    W[:n, :n] = -2 * scale * (sym - np.diag(np.diag(sym)))
    bias = np.zeros((size, 1), np.float32)
    bias[:n, 0] = scale * np.diag(sym)

    graph = Sweeps(W, replicas)
    shape = (PER_CALL, size, replicas)
    version = graph.compile(noise=(shape, bfloat16)) if device else None
    rng = np.random.default_rng(seed)
    for T in (scale * schedule).astype(np.float32).reshape(-1, PER_CALL):
        u = rng.random(shape, np.float32)
        with np.errstate(divide="ignore"):  # u = 0 draws -inf: a certain 1
            noise = (T[:, None, None] * np.log(u / (1 - u)) + bias).astype(bfloat16)
        if version is None:
            graph.reference(noise)
        else:
            version(noise)
    x = graph.x.host if version is None else version.read(graph.x)
    samples = (np.asarray(x, np.float32)[:n].T > 0.5).astype(np.int8)
    e = energies(Q, samples)
    best = int(np.argmin(e))
    return Result(samples[best], float(e[best]), samples, e)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Anneal a planted QUBO, whose ground state is known."
    )
    parser.add_argument("n", type=int, help="variables")
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--host", action="store_true", help="the graph's reference")
    args = parse_args(parser, argv)
    if args.host:  # the reference resolves its operators for any device
        aie_utils.set_current_device(NPU2())
    Q, x = planted(args.n, seed=0)
    start = time.perf_counter()
    r = anneal(Q, replicas=args.replicas, sweeps=args.sweeps, device=not args.host)
    seconds = time.perf_counter() - start
    print(f"energy {r.energy:.6g}, ground state {energies(Q, [x])[0]:.6g}")
    print(f"{args.sweeps * args.replicas / seconds:.4g} replica-sweeps/s")


if __name__ == "__main__":
    main()
