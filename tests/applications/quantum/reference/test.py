#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quantum simulator device-free: ``Circuit.apply`` against closed forms
and dense matrices, and the ``Simulator`` graph's reference against
``Circuit.apply``.
"""

import functools

import numpy as np
import pytest
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.device import from_name

from iron.applications.quantum import Circuit, Simulator, ghz, qft
from iron.applications.quantum.circuit import fuse
from iron.applications.quantum.simulator import descriptors, plan

#: The simulator's error against complex128, both in units of the state's
#: norm: measured under 1e-7 and 3e-7 on the circuits here, float32 limbs
#: through about 30 GEMMs.
MAX_ERROR = 1e-6
INFIDELITY = 1e-6


def random_state(n, rng):
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return psi / np.linalg.norm(psi)


def random_circuit(n, gates, rng):
    circuit = Circuit(n)
    for _ in range(gates):
        u, _ = np.linalg.qr(
            rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        )
        circuit.unitary(u, rng.choice(n, 2, replace=False))
    return circuit


def dense(circuit):
    """The circuit's unitary, each gate a Kronecker product padded with the
    identity and its qubits moved by swaps of index bits."""
    n = circuit.num_qubits
    total = np.eye(2**n)
    for gate in circuit.gates:
        m = len(gate.qubits)
        full = np.kron(np.eye(2 ** (n - m)), gate.matrix)
        # Bit i of the row is gate.qubits[i]; the rest follow in order.
        rest = [q for q in range(n) if q not in gate.qubits]
        bits = list(gate.qubits) + rest
        index = np.arange(2**n)
        moved = sum(((index >> i) & 1) << q for i, q in enumerate(bits))
        p = np.zeros((2**n, 2**n))
        p[moved, index] = 1
        total = p @ full @ p.T @ total
    return total


def test_apply_matches_the_dense_unitary():
    rng = np.random.default_rng(0)
    circuit = random_circuit(4, 12, rng).h(3).cp(0.3, 2, 0).swap(1, 3)
    psi = random_state(4, rng)
    np.testing.assert_allclose(circuit.apply(psi), dense(circuit) @ psi, atol=1e-12)


def test_apply_puts_qubit_0_in_bit_0():
    assert np.argmax(np.abs(Circuit(3).x(0).apply())) == 1


def test_ghz():
    want = np.zeros(2**6)
    want[[0, -1]] = 2**-0.5
    np.testing.assert_allclose(ghz(6).apply(), want, atol=1e-12)


def test_qft_is_the_discrete_fourier_transform():
    n = 5
    psi = random_state(n, np.random.default_rng(1))
    want = np.fft.ifft(psi, norm="ortho")
    np.testing.assert_allclose(qft(n).apply(psi), want, atol=1e-12)


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        (np.eye(2), (0, 0)),
        (np.eye(2), (3,)),
        (np.eye(4), (0,)),
        (np.ones((2, 2)), (0,)),
    ],
    ids=["repeated", "outside", "wrong_size", "not_unitary"],
)
def test_unitary_refuses(matrix, qubits):
    with pytest.raises(ValueError):
        Circuit(3).unitary(matrix, qubits)


def test_fuse_cuts_greedily_in_order():
    gates = random_circuit(8, 30, np.random.default_rng(2)).gates
    runs = fuse(gates, 3)
    assert [g for run in runs for g in run] == gates
    for run, after in zip(runs, runs[1:]):
        touched = {q for g in run for q in g.qubits}
        assert len(touched) <= 3
        assert len(touched | set(after[0].qubits)) > 3


def test_fuse_refuses_a_wider_gate():
    with pytest.raises(ValueError):
        fuse(Circuit(4).unitary(np.eye(8), (0, 1, 2)).gates, 2)


CIRCUITS = {
    "ghz": lambda: ghz(12),
    "qft": lambda: qft(12),
    "random": lambda: random_circuit(12, 60, np.random.default_rng(3)),
}


@pytest.mark.parametrize("name", CIRCUITS)
def test_every_copy_is_one_four_dimensional_pattern(name):
    copies = [s for s in plan(CIRCUITS[name]()) if isinstance(s, TensorAccessPattern)]
    assert copies and all(len(c.sizes) <= 4 for c in copies)


@pytest.mark.usefixtures("npu2")
@pytest.mark.parametrize("name", CIRCUITS)
def test_simulator_reference_matches_complex128(name):
    circuit = CIRCUITS[name]()
    psi = random_state(circuit.num_qubits, np.random.default_rng(4))
    got = Simulator(circuit).reference(psi.astype(np.complex64).view(np.float32))
    got = np.asarray(got).view(np.complex64)
    want = circuit.apply(psi)
    assert np.abs(got - want).max() < MAX_ERROR
    assert 1 - abs(np.vdot(want, got)) ** 2 < INFIDELITY


def indices(tap):
    """The elements ``tap`` walks, in order."""
    index = np.zeros(tap.sizes, np.int64) + tap.offset
    for axis, stride in enumerate(tap.strides):
        shape = [1] * len(tap.sizes)
        shape[axis] = tap.sizes[axis]
        index += (np.arange(tap.sizes[axis]) * stride).reshape(shape)
    return index.reshape(-1)


def test_descriptors_move_what_the_copy_moves():
    """Past 20 qubits a copy is cut into pairs; together they send each
    element where the whole walk does, each within four dimensions."""
    n = 22
    dev = from_name("npu2", n_cols=8)
    copies = [s for s in plan(qft(n)) if isinstance(s, TensorAccessPattern)]
    cut = 0
    for src in copies[:12]:
        strides = np.cumprod((1,) + src.sizes[:0:-1])[::-1].tolist()
        dst = TensorAccessPattern((2 << n,), 0, src.sizes, strides)
        pairs = descriptors(src, dst, dev)
        cut += len(pairs) > 1
        assert all(len(t.sizes) <= 4 for pair in pairs for t in pair)
        moved = np.empty(2 << n, np.int64)
        moved[np.concatenate([indices(d) for _, d in pairs])] = np.concatenate(
            [indices(s) for s, _ in pairs]
        )
        assert np.array_equal(moved[indices(dst)], indices(src))
    assert cut
