<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A quantum circuit simulator on the NPU

This package simulates a quantum circuit on the NPU. It holds the full
state vector, one complex64 amplitude per basis state, and applies gates to
it with float32 GEMMs.

```python
import numpy as np
from iron.applications.quantum import Circuit, Simulator

circuit = Circuit(20).h(0)
for q in range(19):
    circuit.cx(q, q + 1)

psi = np.zeros(2**20, np.complex64)
psi[0] = 1
sim = Simulator(circuit)
sim.compile(state=((2 << 20,), np.float32))
out = sim(psi.view(np.float32)).numpy().view(np.complex64)
```

Qubit `q` is bit `q` of an amplitude's index, as in Qiskit.
`circuit.apply(psi)` runs the same circuit on the host in complex128. It is
the golden the simulator is tested against. `ghz(n)` and `qft(n)` build the
two demonstration circuits.

## How it runs

- **Fusion.** `fuse` cuts the gates, in order, into runs that touch at most
  five qubits. It is greedy: a gate joins the current run unless that would
  take the run past five qubits. Each run becomes one 32x32 complex
  unitary.
- **One GEMM per run.** The state is viewed as rows of the 32 amplitudes
  the lowest five qubit positions index, each `(re, im)` pair kept as numpy
  holds it. A run's unitary becomes a real 64x64 matrix, and the run is the
  rows times that matrix. `Limbs` splits the rows into bf16 limbs, so the
  GEMM's products are float32, as `LogMel` computes its rfft.
- **Copies move qubits into the row.** Before a run, copies bring the
  qubits it touches into the lowest five positions. Each copy swaps two
  adjacent blocks of qubit positions, which is one access pattern of at most
  four dimensions. The GEMM writes its row in the order the next run wants,
  so the qubits that leave the row are always its highest. Copies at the
  end put the state back in the circuit's qubit order.

`Simulator.steps` lists the copies and GEMMs in order. Its docstring and
`plan`'s say the rest.

## Accuracy

Against `Circuit.apply` in complex128, the graph's reference keeps every
amplitude within 1e-7 of a unit-norm state and the infidelity under 3e-7, on
12-qubit GHZ, QFT and random two-qubit circuits. Gate-by-gate complex64
numpy is about ten times closer, because each run's unitary is rounded to
float32 once. So the simulator is float32-class, not bit-exact float32.

## Limits

- The state needs at least 10 qubits, so its rows fill GEMM's 32-row
  tiles.
- A gate on more than five qubits is refused.
- The fuser must stay greedy and windowed. Do not replace it with a
  cost-based search over fusion choices, fuse measurements into gates, or
  cache fusion plans keyed by circuit structure. Each of these reads on an
  IBM simulator patent (US11586792, US11157667 and US12675718).
- Measurement, noise and batches of circuits are not built.

The device-free tests are in `tests/applications/quantum/reference/`. The
device test, `tests/applications/quantum/test.py`, runs a 20-qubit QFT.
