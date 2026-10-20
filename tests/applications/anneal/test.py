#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The annealer on the NPU: a planted ground state of 1024 variables found,
and its throughput recorded.
"""

import time

import numpy as np
import pytest

from iron.applications.anneal import anneal, energies, planted


@pytest.mark.supported_devices("npu1", "npu2")
def test_anneal_finds_a_planted_ground_state_on_the_npu(record_property):
    Q, x = planted(1024, seed=0)
    replicas, sweeps = 64, 200
    start = time.perf_counter()
    r = anneal(Q, replicas=replicas, sweeps=sweeps, seed=0)
    record_property(
        "replica_sweeps_per_s", replicas * sweeps / (time.perf_counter() - start)
    )
    assert np.isclose(r.energy, energies(Q, [x])[0])
