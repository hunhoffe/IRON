# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from aie.dialects._aie_enum_gen import AIEArch
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, In, ObjectFifo, Out, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.utils import config
from aie.utils.compile.jit.params import param, tunable, widths
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from iron.common import Built, Operator, Unresolvable
from iron.common.testing import Case, Testing
from iron.operators.flm import q4nx
from iron.operators.flm.q4nx import BLOCK_BYTES, GROUP, K_TILE, M_TILE, packed_bytes

# The error bound, a share of the largest output: the cores narrow each
# 32-column dot product and each 32-column sum of x to bf16, as LMHead's.
PROJECTION_ERROR = 0.025

# The sequence finishes each round's TaskGroup DEPTH rounds later, freeing
# its BDs; a shim tile holds 16.
DEPTH = 2

# mlir-aie's q4nx block product (flm_gemma4/q4nx.h), the LM head's, without
# its RMS norm and softcap.
SOURCE = """
#include "common/zero.h"
#include "flm_gemma4/q4nx.h"

#include <aie_api/aie.hpp>

using block_t = q4nx_block<32, 256>;

extern "C" {
void q4nx_gemv_sums(const bf16 *x, bf16 *sums, int k_tiles) {
  for (int k = 0; k < k_tiles; k++)
    group_sums_32<256>(sums + k * 8, x + k * 256);
}

void q4nx_gemv_zero(float *y_acc) { zero_vectorized<float, 32, 1, false>(y_acc); }

void q4nx_gemv_block(const block_t *w, const bf16 *x, float *y_acc,
                     const bf16 *sums, int k) {
  q4nx_accumulate<32, 256>(w, x + k * 256, y_acc, sums + k * 8);
}

void q4nx_gemv_store(bf16 *y, const float *y_acc) { narrow_to_bf16<32>(y, y_acc); }
}
"""


def _draw(op):
    """Weights quantized from normal draws, and a normal x."""
    rng = np.random.default_rng(0)
    w = q4nx.quantize(rng.standard_normal((op.N, op.K)) / np.sqrt(op.K))
    x = rng.standard_normal(op.K).astype(bfloat16)
    return dict(given={op.w: w.reshape(-1), op.x: x})


class Q4nxGEMV(Operator):
    """``y = w @ x`` for one row ``x`` of ``K`` values and ``N`` out-features
    of q4nx weights ``w``, laid out as ``q4nx.quantize`` lays them out.

    Each column's core takes 32 out-features a round, streaming their
    blocks past ``x``. AIE2P only, since it runs mlir-aie's q4nx block
    product. See README.md.
    """

    test = Testing(
        [
            Case(dict(K=512, N=256)),
            # More out-feature tiles than a round takes, unevenly.
            Case(dict(K=1024, N=320), bench=True),
            # Llama 3.2 1B's MLP down projection.
            Case(dict(K=8192, N=2048), extensive=True),
        ],
        draw=_draw,
    )

    K: int = param(array=True)
    N: int = param(array=True)
    w_bytes: int = param(derive=lambda op: packed_bytes(op.N * op.K), repr=False)
    cols: int = tunable(repr=False, domain=widths)

    # The runtime sequence's argument order: one block a fill object, 32
    # out-features a drain object, x broadcast to every core.
    y = Out(N, dtype=bfloat16, obj_shape=(M_TILE,), per=(cols,))
    w = In(w_bytes, dtype=np.uint8, obj_shape=(BLOCK_BYTES,), per=(cols,))
    x = In(K, dtype=bfloat16, obj_shape=(K,))

    def validate(self) -> None:
        if self.K % (2 * K_TILE) or self.N % (2 * M_TILE):
            raise ValueError(
                f"Q4nxGEMV: K ({self.K}) and N ({self.N}) must be multiples of "
                f"{2 * K_TILE} and {2 * M_TILE}, q4nx's layout"
            )

    def resolve(self, dev):
        if dev is None:
            raise Unresolvable(
                "Q4nxGEMV: the grid defaults from the device; none given"
            )
        if dev.arch != AIEArch.AIE2p:
            raise Unresolvable("Q4nxGEMV: the q4nx block product is AIE2P only")
        cols = min(dev.cols, self.N // M_TILE)
        return self.replace(cols=cols if self.cols is None else self.cols)

    def lane_tiles(self, lane: int) -> range:
        """The 32-out-feature tiles lane ``lane``'s core takes, in order."""
        return range(lane, self.N // M_TILE, self.cols)

    def ops(self) -> int:
        return 2 * self.N * self.K

    def build(self) -> Built:
        K = self.K
        x_ty = self.x.obj_type
        y_ty = self.y.obj_type
        w_ty = self.w.obj_type
        acc_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
        sums_ty = np.ndarray[(K // GROUP,), np.dtype[bfloat16]]
        block = ExternalFunction(
            "q4nx_gemv_block",
            source_string=SOURCE,
            arg_types=[w_ty, x_ty, acc_ty, sums_ty, np.int32],
            include_dirs=[config.aie_kernels_dir()],
            # As the LM head's: the bf16 products otherwise run about 8x slower.
            compile_flags=["-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"],
        )
        bind = block.object_file.bind
        kernels = [
            bind("q4nx_gemv_sums", [x_ty, sums_ty, np.int32]),
            bind("q4nx_gemv_zero", [acc_ty]),
            block,
            bind("q4nx_gemv_store", [y_ty, acc_ty]),
        ]

        def core(rounds):
            def core_fn(x_in, w_in, y_out, k_sums, k_zero, k_block, k_store, acc, sums):
                x = x_in.acquire(1)
                k_sums(x, sums, K // K_TILE)
                for _ in range_(rounds):
                    k_zero(acc)
                    for k in range_(K // K_TILE):
                        w = w_in.acquire(1)
                        k_block(w, x, acc, sums, k)
                        w_in.release(1)
                    y = y_out.acquire(1)
                    k_store(y, acc)
                    y_out.release(1)
                x_in.release(1)

            return core_fn

        # Acquired once a call, so one buffer: a second would cap K.
        of_x = ObjectFifo(x_ty, depth=1)
        self.x.bind(of_x.prod())
        workers, ws, ys = [], [], []
        for c in range(self.cols):
            of_w, of_y = ObjectFifo(w_ty), ObjectFifo(y_ty)
            ws.append(of_w.prod())
            ys.append(of_y.cons())
            workers.append(
                Worker(
                    core(len(self.lane_tiles(c))),
                    [
                        of_x.cons(),
                        of_w.cons(),
                        of_y.prod(),
                        *kernels,
                        Buffer(type=acc_ty, name=f"acc_{c}"),
                        Buffer(type=sums_ty, name=f"sums_{c}"),
                    ],
                )
            )
        self.w.bind(ws)
        self.y.bind(ys)
        return Built(workers)

    def sequence(self, rt):
        x = self.x.fill(managed=False)
        k_tiles = self.K // K_TILE
        window = []
        for r in range(len(self.lane_tiles(0))):
            tg = TaskGroup()
            for col in range(self.cols):
                if r >= len(self.lane_tiles(col)):
                    break
                t = self.lane_tiles(col)[r]
                self.y.drain(
                    TensorAccessPattern((self.N,), t * M_TILE, [M_TILE], [1]),
                    lane=col,
                    wait=True,
                    group=tg,
                )
                # Tile t's blocks, k-tile by k-tile, in q4nx.quantize's order:
                # its column block, its half of it, then every other block.
                # A block is split 10 x 512 to keep the innermost size inside
                # the BD's field.
                offset = (t // 2 * 2 * k_tiles + t % 2) * BLOCK_BYTES
                self.w.fill(
                    TensorAccessPattern(
                        (self.w_bytes,),
                        offset,
                        [k_tiles, BLOCK_BYTES // 512, 512],
                        [2 * BLOCK_BYTES, 512, 1],
                    ),
                    lane=col,
                    group=tg,
                )
            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()
        for tg in window:
            tg.finish()
        x.free()

    def tolerance(self) -> Tolerance:
        return Tolerance.bounded(
            lambda w, x: PROJECTION_ERROR * float(np.abs(self.reference(w, x)).max()),
            note="the cores narrow each 32-column dot product and sum of x to bf16",
        )

    def reference(self, w, x):
        """``w @ x`` in float64, rounded to bf16."""
        weights = q4nx.dequantize(q4nx.grid(np.reshape(w, (self.N, -1))))
        return (weights @ np.asarray(x, np.float64)).astype(bfloat16)
