# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A circuit's state vector on the NPU: fused gates as float32 GEMMs, and
the qubits each needs brought next to each other by copies.

The state is complex64 as numpy holds it, each amplitude a ``(re, im)``
pair of float32, viewed as rows of ``WIDTH`` qubits: a row is the ``2**WIDTH``
amplitudes the lowest ``WIDTH`` qubit positions index. Gates are cut into
runs that touch at most ``WIDTH`` qubits (``fuse``); a run is one GEMM of
the rows by its unitary as a real ``(2 * 2**WIDTH)``-square matrix, exact
in float32 through ``Limbs``. Before a run, copies move the qubits it
touches into the row; each copy swaps two adjacent blocks of qubit
positions, which is one access pattern of at most four dimensions. The
GEMM writes its row's qubits in the order the next run wants them, so the
qubits leaving the row are always its highest.
"""

import numpy as np
from aie.helpers.taplib import TensorAccessPattern

import iron
from iron.common import Precision
from iron.common.graph import Handle
from iron.operators.copy import Copy
from iron.operators.gemm import GEMM
from iron.operators.limbs import Limbs, stacked

from .circuit import Circuit, Gate, apply, fuse

# The qubits a row holds: 32 amplitudes, a 64-column GEMM tile.
WIDTH = 5


class Simulator(iron.Graph):
    """``circuit`` applied to a state: its ``2**num_qubits`` complex64
    amplitudes viewed as float32 (``psi.view(np.float32)``), at least
    ``WIDTH + 5`` qubits so the rows fill GEMM's 32-row tiles::

        sim = Simulator(circuit)
        out = sim(psi.view(np.float32)).numpy().view(np.complex64)

    ``steps`` is what the body records: an access pattern is a copy, an
    array a GEMM by it.
    """

    def __init__(self, circuit: Circuit):
        self.num_qubits = circuit.num_qubits
        self.steps = [
            step if isinstance(step, TensorAccessPattern) else iron.param(step)
            for step in plan(circuit)
        ]

    def body(self, state: Handle) -> Handle:
        exact = dict(tile_m=32, output_dtype=np.float32, precision=Precision.HIGHEST)
        size, rows = 2 << self.num_qubits, 1 << self.num_qubits - WIDTH
        for step in self.steps:
            if isinstance(step, TensorAccessPattern):
                state = Copy.step(state, src=step, dtype=np.float32)
            else:
                a = Limbs.step(state.reshape(rows, 2 << WIDTH))
                state = GEMM.step(a, step, **exact).reshape(size)
        return state


def plan(circuit: Circuit) -> list:
    """The copies (``TensorAccessPattern``) and GEMM matrices (each
    ``stacked``) that apply ``circuit`` to its state."""
    n = circuit.num_qubits
    runs = fuse(circuit.gates, WIDTH)
    # The qubit at each position, highest first; the row is the last WIDTH.
    order = list(range(n - 1, -1, -1))
    edge = n - WIDTH
    steps: list = []

    def swap(i: int, j: int, e: int):
        nonlocal order
        new = order[:i] + order[j:e] + order[i:j] + order[e:]
        steps.append(permutation(order, new))
        order = new

    for i, run in enumerate(runs):
        touched = {q for gate in run for q in gate.qubits}
        # Gather the qubits the run needs into one block just above the row,
        # then swap that block with the row's highest positions.
        blocks = _blocks(order[:edge], touched)
        while blocks and (len(blocks) > 1 or blocks[0][1] != edge):
            start, end = blocks[-1] if blocks[-1][1] != edge else blocks[-2]
            swap(start, end, blocks[-1][0] if blocks[-1][1] == edge else edge)
            blocks = _blocks(order[:edge], touched)
        if blocks:
            start, _ = blocks[0]
            # Only before the first run can the row's highest be needed:
            # every GEMM after writes them last.
            while touched & set(order[edge : 2 * edge - start]):
                inner = _blocks(order[edge:], touched) + [(WIDTH, WIDTH)]
                (top, mid), (end, _) = next(
                    (a, b) for a, b in zip(inner, inner[1:]) if a[1] < b[0]
                )
                swap(edge + top, edge + mid, edge + end)
            swap(start, edge, 2 * edge - start)
        # The next run's qubits lowest, so those leaving are highest; after
        # the last run, the row's own qubits in order.
        later = (
            {q for gate in runs[i + 1] for q in gate.qubits}
            if i + 1 < len(runs)
            else set(range(WIDTH))
        )
        row = order[edge:]
        out = [q for q in row if q not in later] + sorted(
            (q for q in row if q in later), reverse=True
        )
        steps.append(stacked(real(fused(run, row, out))))
        order = order[:edge] + out
    # Back to the circuit's order, a block at a time.
    for i, q in enumerate(range(n - 1, -1, -1)):
        j = order.index(q)
        if j != i:
            e = j
            while e < n and order[e] == q - (e - j):
                e += 1
            swap(i, j, e)
    return steps


def _blocks(qubits: list[int], members: set[int]) -> list[tuple[int, int]]:
    """The ``[start, end)`` runs of ``qubits`` that are all ``members``."""
    blocks: list[tuple[int, int]] = []
    for i, q in enumerate(qubits):
        if q in members:
            if blocks and blocks[-1][1] == i:
                blocks[-1] = (blocks[-1][0], i + 1)
            else:
                blocks.append((i, i + 1))
    return blocks


def fused(run: list[Gate], row: list[int], out: list[int]) -> np.ndarray:
    """``run``'s unitary on the row, from its qubits in ``row`` order to
    ``out`` order, each highest position first."""
    k = len(row)
    position = {q: k - 1 - i for i, q in enumerate(row)}
    u = np.eye(2**k, dtype=np.complex128)
    for gate in run:
        u = apply(Gate(gate.matrix, tuple(position[q] for q in gate.qubits)), u, k)
    axes = [row.index(q) for q in out]
    return u.reshape((2,) * k + (2**k,)).transpose(axes + [k]).reshape(2**k, 2**k)


def real(u: np.ndarray) -> np.ndarray:
    """``u`` as the real matrix that multiplies a row of ``(re, im)`` pairs
    from the right: ``row @ real(u)`` is ``u @ amplitudes``."""
    size = len(u)
    w = np.empty((size, 2, size, 2))
    w[:, 0, :, 0] = w[:, 1, :, 1] = u.real.T
    w[:, 0, :, 1] = u.imag.T
    w[:, 1, :, 0] = -u.imag.T
    return w.reshape(2 * size, 2 * size)


def permutation(order: list[int], new: list[int]) -> TensorAccessPattern:
    """The walk of a state held in ``order`` that reads it in ``new``, both
    qubit positions highest first, the ``(re, im)`` pair innermost, with
    the dimensions a contiguous run of ``order`` keeps merged."""
    n = len(order)
    dims: list[list[int]] = []
    for size, stride in [(2, 2 << (n - 1 - order.index(q))) for q in new] + [(2, 1)]:
        if dims and dims[-1][1] == size * stride:
            dims[-1] = [dims[-1][0] * size, stride]
        else:
            dims.append([size, stride])
    sizes, strides = zip(*dims)
    return TensorAccessPattern((2 << n,), 0, sizes, strides)
