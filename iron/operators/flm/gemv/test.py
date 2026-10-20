#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q4nxGEMV's sequence, replayed on the host: each core's blocks gathered
through the fill patterns it is given, multiplied out, and scattered through
its drain patterns must give the reference. A wrong offset, stride or lane
moves real weights to the wrong out-features, which no lowering notices.
"""

import types
from collections import defaultdict

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.common.declare.operand import Operand
from iron.common.design import OperatorDesign
from iron.common.image.coresidence import fits
from iron.operators import quantized
from iron.operators.flm import q4nx
from iron.operators.flm.gemv import op as gemv
from iron.operators.flm.gemv.op import Q4nxGEMV
from iron.operators.flm.q4nx import BLOCK_BYTES, K_TILE, M_TILE


class _Fifo:
    def __init__(self):
        self.acquired = 0

    def acquire(self, n):
        self.acquired += n

    def release(self, n):
        pass


def _rounds(op, monkeypatch):
    """Each core's (w blocks, y tiles) taken, its body run on the host."""
    monkeypatch.setattr(quantized, "range_", range)
    taken = []
    for worker in op.build().objects:
        x, w, y = _Fifo(), _Fifo(), _Fifo()
        rest = [lambda *_: None] * (len(worker.fn_args) - 3)
        worker.core_fn(x, w, y, *rest)
        taken.append((w.acquired, y.acquired))
    return taken


def _replay(op, monkeypatch):
    """Each lane's w fills and y drains, in the order the sequence issues them."""
    moves = defaultdict(list)

    def record(kind):
        def transfer(self, tap=None, *, lane=None, **_):
            moves[self.name, kind, lane].append(tap)
            return types.SimpleNamespace(free=lambda: None)

        return transfer

    monkeypatch.setattr(Operand, "fill", record("fill"))
    monkeypatch.setattr(Operand, "drain", record("drain"))
    monkeypatch.setattr(
        quantized, "TaskGroup", lambda: types.SimpleNamespace(finish=lambda: None)
    )
    op.sequence(None)
    return moves


@pytest.mark.parametrize(
    "K, N, cols",
    [(512, 256, None), (1024, 320, None), (1024, 320, 3), (2048, 128, 1)],
)
def test_each_core_reads_its_tiles_blocks_and_writes_its_rows(
    K, N, cols, npu2, monkeypatch
):
    op = Q4nxGEMV(K=K, N=N, cols=cols).resolved(npu2)
    rng = np.random.default_rng(0)
    w = q4nx.quantize(rng.standard_normal((N, K))).reshape(-1)
    x = rng.standard_normal(K).astype(bfloat16)
    moves = _replay(op, monkeypatch)
    taken = _rounds(op, monkeypatch)
    y = np.full(N, np.nan)
    for lane in range(op.cols):
        stream = np.concatenate([t.gather(w) for t in moves["w", "fill", lane]])
        rounds = len(op.lane_tiles(lane))
        assert stream.size == rounds * K // K_TILE * BLOCK_BYTES
        drains = moves["y", "drain", lane]
        assert len(drains) == rounds
        assert taken[lane] == (rounds * K // K_TILE, rounds)
        for blocks, tap in zip(stream.reshape(rounds, 1, -1, BLOCK_BYTES), drains):
            tile = q4nx.dequantize(blocks) @ x.astype(np.float64)
            assert tile.shape == (M_TILE,)
            tap.scatter(tile, out=y)
    ref = op.reference(w, x).astype(np.float64)
    np.testing.assert_allclose(ref, y, rtol=2**-8, atol=0)


def test_a_llama_8b_down_projection_places(npu2):
    op = Q4nxGEMV(K=14336, N=4096).resolved(npu2)
    assert fits({"gemv": OperatorDesign(op)}) is None
