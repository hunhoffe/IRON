#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The annealer by its graph's reference, device-free: one sweep against
numpy, the heat-bath draw's odds, and its answers against brute force, a
planted ground state and sequential annealing.
"""

import itertools

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.applications.anneal.model import (
    BLOCKS,
    Sweeps,
    anneal,
    energies,
    planted,
    temperatures,
)

pytestmark = pytest.mark.usefixtures("npu2")


def sequential(Q, replicas: int, sweeps: int, seed: int) -> np.ndarray:
    """Textbook annealing, one variable at a time, on ``anneal``'s schedule:
    each replica's final energy.
    """
    rng = np.random.default_rng(seed)
    sym = (Q + Q.T) / 2
    x = np.zeros((replicas, len(Q)))
    for T in temperatures(Q, sweeps):
        for i in rng.permutation(len(Q)):
            delta = sym[i, i] + 2 * (x @ sym[i] - x[:, i] * sym[i, i])
            x[:, i] = rng.random(replicas) < np.exp(-np.logaddexp(0, delta / T))
    return energies(Q, x)


def test_a_sweep_takes_each_block_from_the_fields_of_the_last():
    size, replicas = BLOCKS * 64, 8
    rng = np.random.default_rng(0)
    W = rng.standard_normal((size, size)).astype(bfloat16)
    noise = rng.standard_normal((1, size, replicas)).astype(bfloat16)
    graph = Sweeps(W, replicas)
    x = rng.integers(0, 2, (size, replicas)).astype(np.float32)
    graph.x.host = x.astype(bfloat16)

    for rows in np.split(np.arange(size), BLOCKS):
        field = (W[rows].astype(np.float32) @ x).astype(bfloat16)
        d = field.astype(np.float32) - noise[0, rows].astype(np.float32)
        x[rows] = np.clip(d.astype(bfloat16), 0, 1)
    graph.reference(noise)
    np.testing.assert_array_equal(np.asarray(graph.x.host, np.float32), x)


def test_a_variable_is_one_at_the_heat_bath_odds():
    # One variable costing q when 1: at the last temperature, q / ln 100,
    # it is 1 one time in 101.
    replicas = 4096
    r = anneal([[1.0]], replicas=replicas, sweeps=10, seed=0, device=False)
    assert abs(r.samples.sum() - replicas / 101) < 4 * np.sqrt(replicas / 101)


def test_anneal_finds_a_small_qubos_ground_state():
    n = 12
    Q = np.random.default_rng(1).standard_normal((n, n))
    every = np.array(list(itertools.product([0, 1], repeat=n)))
    r = anneal(Q, replicas=16, sweeps=200, seed=0, device=False)
    assert np.isclose(r.energy, energies(Q, every).min())
    assert np.isclose(r.energy, energies(Q, [r.x])[0])


def test_anneal_finds_a_planted_ground_state():
    Q, x = planted(300, seed=2)
    r = anneal(Q, replicas=8, sweeps=100, seed=0, device=False)
    assert np.isclose(r.energy, energies(Q, [x])[0])


def test_blocked_sweeps_anneal_as_well_as_sequential_ones():
    n, replicas, sweeps = 128, 16, 200
    Q = np.random.default_rng(3).standard_normal((n, n))
    want = sequential(Q, replicas, sweeps, seed=0)
    got = anneal(Q, replicas=replicas, sweeps=sweeps, seed=0, device=False)
    # Lower is better: within half a percent of the best, one of the mean.
    assert got.energy <= want.min() * 0.995
    assert got.energies.mean() <= want.mean() * 0.99
