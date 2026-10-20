<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Annealing a QUBO on the NPU

`anneal(Q)` searches for the binary `x` that minimises `x @ Q @ x`, the
form max-cut, scheduling and many other combinatorial problems take. It
runs many textbook simulated annealers side by side, one per replica, and
returns the best sample with every replica's:

```python
from iron.applications.anneal import anneal

r = anneal(Q, replicas=64, sweeps=1000)
r.x, r.energy          # the best sample and its energy
r.samples, r.energies  # every replica's
```

`model.py`'s docstring says how a sweep maps onto the NPU: per block of
variables, one GEMM of the couplings by every replica's variables, then
a subtraction and a clamp. The host draws the noise, which carries the
temperature, so the device never branches or draws.

## Running

```
python -m iron.applications.anneal.model 4096          # on the NPU
python -m iron.applications.anneal.model 4096 --host   # the same sweeps on the host
```

Each anneals a dense QUBO with a planted ground state (`planted`) and
prints the energy found beside the ground state's, and the replica-sweeps
per second.

## Tests

`tests/applications/anneal/reference/` holds the graph's reference to
numpy's blocked sweep, the heat-bath draw to its odds, and the answers to
brute force, a planted ground state and one-variable-at-a-time annealing.
`tests/applications/anneal/test.py` finds a planted ground state of 1024
variables on the NPU.

## Limits

- Couplings are bf16 on the device; `energies` scores the samples in
  float64 against the `Q` given. A dense instance with a wide range of
  weights can have its ground state moved by the rounding.
- The host draws every replica's noise for every sweep, a float32 logistic
  per variable; at large sizes that, not the NPU, may bound a sweep.
