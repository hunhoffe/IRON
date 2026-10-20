# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantum circuits, and the state vector they produce on the host.

Qubit ``q`` is bit ``q`` of an amplitude's index, as Qiskit numbers them, so
``Circuit(2).x(0)`` takes ``|00>`` to amplitude 1. A gate's matrix is
indexed the same way over its own qubits: bit ``i`` of a row is
``qubits[i]``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]])
# Controlled-X with qubits (control, target): bit 0 is the control.
CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@dataclasses.dataclass(frozen=True)
class Gate:
    """A unitary ``matrix``, ``(2**m, 2**m)``, on ``m`` distinct ``qubits``."""

    matrix: np.ndarray
    qubits: tuple[int, ...]


class Circuit:
    """A list of gates on ``num_qubits`` qubits, built by calls that return
    the circuit, so they chain::

        ghz = Circuit(3).h(0).cx(0, 1).cx(1, 2)

    ``apply(state)`` is its effect on a state, gate by gate in complex128:
    the golden a simulation is judged by.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: list[Gate] = []

    def __len__(self) -> int:
        return len(self.gates)

    def unitary(self, matrix, qubits) -> Circuit:
        """Append ``matrix`` on ``qubits``.

        Raises:
            ValueError: ``qubits`` repeat or fall outside the circuit, or
                ``matrix`` is not unitary on as many qubits.
        """
        qubits = tuple(int(q) for q in qubits)
        matrix = np.asarray(matrix, np.complex128)
        size = 2 ** len(qubits)
        if len(set(qubits)) != len(qubits) or not all(
            0 <= q < self.num_qubits for q in qubits
        ):
            raise ValueError(f"Circuit: qubits {qubits} on {self.num_qubits} qubits")
        if matrix.shape != (size, size) or not np.allclose(
            matrix @ matrix.conj().T, np.eye(size), atol=1e-9
        ):
            raise ValueError(f"Circuit: no {size}x{size} unitary on qubits {qubits}")
        self.gates.append(Gate(matrix, qubits))
        return self

    def h(self, q: int) -> Circuit:
        return self.unitary(H, (q,))

    def x(self, q: int) -> Circuit:
        return self.unitary(X, (q,))

    def cx(self, control: int, target: int) -> Circuit:
        return self.unitary(CX, (control, target))

    def cp(self, theta: float, control: int, target: int) -> Circuit:
        """The phase ``exp(i theta)`` on ``|11>``."""
        return self.unitary(np.diag([1, 1, 1, np.exp(1j * theta)]), (control, target))

    def swap(self, a: int, b: int) -> Circuit:
        return self.unitary(SWAP, (a, b))

    def apply(self, state=None) -> np.ndarray:
        """This circuit's effect on ``state`` (``|0...0>`` if None), a
        complex128 vector of ``2**num_qubits`` amplitudes. Trailing axes of
        ``state`` past the first are a batch, each column its own state.
        """
        if state is None:
            state = np.zeros(2**self.num_qubits, np.complex128)
            state[0] = 1
        state = np.asarray(state, np.complex128)
        for gate in self.gates:
            state = apply(gate, state, self.num_qubits)
        return state


def apply(gate: Gate, state: np.ndarray, n: int) -> np.ndarray:
    """``gate`` on ``state``, ``(2**n, ...)``."""
    m = len(gate.qubits)
    psi = state.reshape((2,) * n + state.shape[1:])
    # The gate's input axes, highest bit first, against the state's, whose
    # axis n - 1 - q is qubit q.
    axes = [n - 1 - q for q in reversed(gate.qubits)]
    g = gate.matrix.reshape((2,) * (2 * m))
    psi = np.tensordot(g, psi, axes=(list(range(m, 2 * m)), axes))
    return np.moveaxis(psi, list(range(m)), axes).reshape(state.shape)


def fuse(gates: list[Gate], width: int) -> list[list[Gate]]:
    """``gates`` in order, cut greedily into runs that touch at most
    ``width`` qubits: each gate joins the run before it unless that would
    take the run past ``width``.

    Raises:
        ValueError: A gate on more than ``width`` qubits.
    """
    runs: list[list[Gate]] = []
    touched: set[int] = set()
    for gate in gates:
        if len(gate.qubits) > width:
            raise ValueError(f"fuse: a gate on {gate.qubits} is wider than {width}")
        if not runs or len(touched | set(gate.qubits)) > width:
            runs.append([])
            touched = set()
        runs[-1].append(gate)
        touched |= set(gate.qubits)
    return runs


def ghz(num_qubits: int) -> Circuit:
    """``(|0...0> + |1...1>) / sqrt(2)`` from ``|0...0>``."""
    circuit = Circuit(num_qubits).h(0)
    for q in range(num_qubits - 1):
        circuit.cx(q, q + 1)
    return circuit


def qft(num_qubits: int) -> Circuit:
    """The quantum Fourier transform: ``|x>`` to the sum over ``y`` of
    ``exp(2 pi i x y / 2**num_qubits) |y>``, normalized."""
    circuit = Circuit(num_qubits)
    for j in reversed(range(num_qubits)):
        circuit.h(j)
        for m in reversed(range(j)):
            circuit.cp(math.pi / 2 ** (j - m), m, j)
    for q in range(num_qubits // 2):
        circuit.swap(q, num_qubits - 1 - q)
    return circuit
