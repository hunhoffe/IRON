# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from aie.dialects._aie_enum_gen import AIEArch
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, Out
from aie.iron.kernels import flm_gemma4
from aie.utils.compile.jit.params import param, tunable, widths
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from iron.common import Built, Operator, Unresolvable
from iron.common.testing import Case, Testing
from iron.operators import quantized
from iron.operators.flm import q4nx
from iron.operators.flm.q4nx import BLOCK_BYTES, GROUP, K_TILE, M_TILE, packed_bytes

# The error bound, a share of the largest output: the cores narrow each
# 32-column dot product and each 32-column sum of x to bf16, as LMHead's.
PROJECTION_ERROR = 0.025


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
        if dev is not None and dev.arch != AIEArch.AIE2p:
            raise Unresolvable("Q4nxGEMV: the q4nx block product is AIE2P only")
        return self.replace(cols=quantized.cols(self, dev))

    lane_tiles = quantized.lane_tiles

    def ops(self) -> int:
        return 2 * self.N * self.K

    def build(self) -> Built:
        kernel = flm_gemma4.flm_gemma4_q4nx_gemv(
            dim=self.K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP
        )
        sums_ty = kernel.arg_types()[3]
        return quantized.gemv_build(
            self,
            K_TILE,
            kernel.q4nx_lm_head_zero,
            kernel,
            kernel.q4nx_gemv_store,
            sums=(kernel.q4nx_gemv_sums, sums_ty),
        )

    def w_tap(self, t: int) -> TensorAccessPattern:
        """Tile t's blocks, k-tile by k-tile, in q4nx.quantize's order: its
        column block, its half of it, then every other block. A block is
        split 10 x 512 to keep the innermost size inside the BD's field.
        """
        k_tiles = self.K // K_TILE
        return TensorAccessPattern(
            (self.w_bytes,),
            (t // 2 * 2 * k_tiles + t % 2) * BLOCK_BYTES,
            [k_tiles, BLOCK_BYTES // 512, 512],
            [2 * BLOCK_BYTES, 512, 1],
        )

    sequence = quantized.gemv_sequence

    def tolerance(self) -> Tolerance:
        return Tolerance.bounded(
            lambda w, x: PROJECTION_ERROR * float(np.abs(self.reference(w, x)).max()),
            note="the cores narrow each 32-column dot product and sum of x to bf16",
        )

    def reference(self, w, x):
        """``w @ x`` in float64, rounded to bf16."""
        weights = q4nx.dequantize(q4nx.grid(np.reshape(w, (self.N, -1))))
        return (weights @ np.asarray(x, np.float64).reshape(-1)).astype(bfloat16)
