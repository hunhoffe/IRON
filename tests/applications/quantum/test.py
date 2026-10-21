#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The quantum simulator on a device: a 20-qubit QFT of a drawn state
against ``Circuit.apply`` in complex128 and against the graph's own
reference.
"""

import time

import numpy as np
import pytest

from iron.applications.quantum import Simulator, qft

pytestmark = pytest.mark.supported_devices("npu2")

QUBITS = 20
#: Against complex128, in units of the state's norm: the device-free
#: reference measures under 1e-7.
MAX_ERROR = 1e-6
#: Against the graph's reference, which rounds as the kernels do.
REFERENCE = 1e-7


def test_qft(record_property):
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(2**QUBITS) + 1j * rng.standard_normal(2**QUBITS)
    psi /= np.linalg.norm(psi)
    x = psi.astype(np.complex64).view(np.float32)
    circuit = qft(QUBITS)
    sim = Simulator(circuit)
    sim.compile(state=(x.shape, np.float32))
    got = np.asarray(sim(x).numpy()).view(np.complex64)
    reference = np.asarray(sim.reference(x)).view(np.complex64)
    assert np.abs(got - reference).max() < REFERENCE
    assert np.abs(got - circuit.apply(psi)).max() < MAX_ERROR
    times = []
    for _ in range(5):
        start = time.perf_counter()
        sim(x)
        times.append(time.perf_counter() - start)
    record_property("Latency", min(times) * 1e6)
