#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ternary layout, and its operators' sequences replayed on the host:
each core's blocks gathered through the fill patterns it is given, worked
out, and scattered through its drain patterns must give the reference. A
wrong offset, stride or lane moves real weights to the wrong out-features,
which no lowering notices.
"""

import types
from collections import defaultdict

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.common.declare.operand import Operand
from iron.operators import quantized
from iron.operators.ternary import TernaryDequant, TernaryGEMV
from iron.operators.ternary import layout
from iron.operators.ternary import op as ternary
from iron.operators.ternary.layout import BLOCK_BYTES, K_TILE, M_TILE


def test_a_ternary_weight_is_held_exactly():
    rng = np.random.default_rng(0)
    trits = rng.integers(-1, 2, (64, 512))
    trits[:32, :128] = 0  # a group of zeros: a scale of 0
    scales = rng.uniform(0.01, 1, (64, 4)).astype(bfloat16).astype(np.float32)
    w = trits * np.repeat(scales, 128, axis=1)
    q = layout.quantize(w)
    assert q.shape == (64, 512 * 17 // 64)
    np.testing.assert_array_equal(layout.dequantize(q), w)


def test_a_run_of_32_out_features_is_its_own_rows():
    w = np.random.default_rng(1).standard_normal((96, 512))
    np.testing.assert_array_equal(layout.quantize(w)[32:64], layout.quantize(w[32:64]))


def test_a_weight_off_the_block_grid_is_refused():
    with pytest.raises(ValueError, match=r"\(48, 256\) must tile to \(32, 256\)"):
        layout.quantize(np.zeros((48, 256)))


class _Fifo:
    def __init__(self):
        self.acquired = 0

    def acquire(self, n):
        self.acquired += n

    def release(self, n):
        pass


def _taken(op, monkeypatch):
    """Each core's fifo acquisitions, its body run on the host."""
    monkeypatch.setattr(quantized, "range_", range)
    monkeypatch.setattr(ternary, "range_", range)
    taken = []
    for worker in op.build().objects:
        fifos = [_Fifo() for _ in range(len(op.declaration.operands))]
        rest = [lambda *_: None] * (len(worker.fn_args) - len(fifos))
        worker.core_fn(*fifos, *rest)
        taken.append([f.acquired for f in fifos])
    return taken


def _replay(op, monkeypatch):
    """Each lane's fills and drains, in the order the sequence issues them."""
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


SHAPES = [(512, 256, None), (1024, 288, None), (1024, 288, 3), (512, 64, 1)]


@pytest.mark.parametrize("K, N, cols", SHAPES)
def test_each_gemv_core_reads_its_tiles_blocks_and_writes_its_rows(
    K, N, cols, npu2, monkeypatch
):
    op = TernaryGEMV(K=K, N=N, cols=cols).resolved(npu2)
    rng = np.random.default_rng(0)
    w = layout.quantize(rng.standard_normal((N, K))).reshape(-1)
    x = rng.standard_normal(K).astype(bfloat16)
    moves = _replay(op, monkeypatch)
    taken = _taken(op, monkeypatch)
    y = np.full(N, np.nan)
    for lane in range(op.cols):
        rounds = len(op.lane_tiles(lane))
        stream = np.concatenate([t.gather(w) for t in moves["w", "fill", lane]])
        assert stream.size == rounds * K // K_TILE * BLOCK_BYTES
        drains = moves["y", "drain", lane]
        assert len(drains) == rounds
        assert taken[lane] == [1, rounds * K // K_TILE, rounds]
        for blocks, tap in zip(stream.reshape(rounds, -1, BLOCK_BYTES), drains):
            tile = layout.dequantize(blocks.reshape(M_TILE, -1)) @ x.astype(np.float64)
            tap.scatter(tile, out=y)
    np.testing.assert_allclose(op.reference(w, x).astype(np.float64), y, rtol=2**-8)


@pytest.mark.parametrize("K, N, cols", SHAPES)
def test_each_dequant_core_reads_its_tiles_blocks_and_writes_them_in_place(
    K, N, cols, npu2, monkeypatch
):
    op = TernaryDequant(K=K, N=N, cols=cols).resolved(npu2)
    w = layout.quantize(np.random.default_rng(0).standard_normal((N, K))).reshape(-1)
    moves = _replay(op, monkeypatch)
    taken = _taken(op, monkeypatch)
    out = np.full(N * K, np.nan)
    for lane in range(op.cols):
        blocks = len(op.lane_tiles(lane)) * K // K_TILE
        stream = np.concatenate([t.gather(w) for t in moves["w", "fill", lane]])
        drains = moves["out", "drain", lane]
        assert taken[lane] == [blocks, blocks]
        tiles = [
            layout.dequantize(b.reshape(M_TILE, -1)).ravel()
            for b in stream.reshape(blocks, BLOCK_BYTES)
        ]
        k_tiles = K // K_TILE
        for r, tap in enumerate(drains):
            tap.scatter(np.concatenate(tiles[r * k_tiles : (r + 1) * k_tiles]), out=out)
    np.testing.assert_array_equal(op.reference(w).astype(np.float64), out)
