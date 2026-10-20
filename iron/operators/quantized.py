# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dataflow the quantized-weight operators share: each column's core
takes ROWS out-features a round, streaming their blocks of ``w``, and a
round's transfers share one TaskGroup.

An operator here declares ``K``, ``N``, ``cols`` and ``w`` and gives
``w_tap(t)``, the walk over ``w`` for out-feature tile ``t``'s blocks; a
GEMV also declares ``x`` and ``y``.
"""

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ObjectFifo, TaskGroup, Worker
from aie.iron.controlflow import range_

from iron.common import Built, Unresolvable

# Out-features a core takes a round.
ROWS = 32

# The sequence finishes each round's TaskGroup DEPTH rounds later, freeing
# its BDs; a shim tile holds 16.
DEPTH = 2


def cols(op, dev) -> int:
    """``op.cols``, else one column per tile up to the device's width."""
    if dev is None:
        raise Unresolvable(
            f"{type(op).__name__}: the grid defaults from the device; none given"
        )
    return min(dev.cols, op.N // ROWS) if op.cols is None else op.cols


def lane_tiles(op, lane: int) -> range:
    """The ROWS-out-feature tiles lane ``lane``'s core takes, in order."""
    return range(lane, op.N // ROWS, op.cols)


def rounds(op):
    """Each round's (lane, tile) pairs and its TaskGroup, finished DEPTH
    rounds on.
    """
    window = []
    tiles = [lane_tiles(op, c) for c in range(op.cols)]
    for r in range(len(tiles[0])):
        tg = TaskGroup()
        yield [(c, t[r]) for c, t in enumerate(tiles) if r < len(t)], tg
        window.append(tg)
        if len(window) > DEPTH:
            window.pop(0).finish()
    for tg in window:
        tg.finish()


def gemv_build(op, k_tile, zero, block, store, sums=None) -> Built:
    """A GEMV's cores: per round, ``zero`` the float32 accumulator, add
    each k-tile's block (``block(w, x, acc, *sums, k)``), ``store`` it.
    ``sums``, a kernel and its buffer's type, first reads ``x`` into its
    buffer, once a call.
    """
    w_ty, x_ty, acc_ty = block.arg_types()[:3]
    y_ty = store.arg_types()[0]

    def core(n):
        def core_fn(x_in, w_in, y_out, k_zero, k_block, k_store, acc, *extra):
            x = x_in.acquire(1)
            if extra:
                k_sums, buf = extra
                k_sums(x, buf)
            for _ in range_(n):
                k_zero(acc)
                for k in range_(op.K // k_tile):
                    w = w_in.acquire(1)
                    k_block(w, x, acc, *extra[1:], k)
                    w_in.release(1)
                y = y_out.acquire(1)
                k_store(y, acc)
                y_out.release(1)
            x_in.release(1)

        return core_fn

    # Acquired once a call, so one buffer: a second would cap K.
    of_x = ObjectFifo(x_ty, depth=1)
    op.x.bind(of_x.prod())
    workers, ws, ys = [], [], []
    for c in range(op.cols):
        of_w, of_y = ObjectFifo(w_ty), ObjectFifo(y_ty)
        ws.append(of_w.prod())
        ys.append(of_y.cons())
        extra = (
            [] if sums is None else [sums[0], Buffer(type=sums[1], name=f"sums_{c}")]
        )
        workers.append(
            Worker(
                core(len(lane_tiles(op, c))),
                [
                    of_x.cons(),
                    of_w.cons(),
                    of_y.prod(),
                    zero,
                    block,
                    store,
                    Buffer(type=acc_ty, name=f"acc_{c}"),
                    *extra,
                ],
            )
        )
    op.w.bind(ws)
    op.y.bind(ys)
    return Built(workers)


def gemv_sequence(op, rt) -> None:
    """A GEMV's sequence: ``x`` once, then each round's rows out and blocks in."""
    x = op.x.fill(managed=False)
    for lanes, tg in rounds(op):
        for col, t in lanes:
            op.y.drain(
                TensorAccessPattern((op.N,), t * ROWS, [ROWS], [1]),
                lane=col,
                wait=True,
                group=tg,
            )
            op.w.fill(op.w_tap(t), lane=col, group=tg)
    x.free()
