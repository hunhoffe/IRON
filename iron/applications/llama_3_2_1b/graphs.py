# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama as one graph function over one set of weights and caches.

:class:`LlamaGraph` holds ``forward(x, angles, *, cache_offset, vector_size,
last)``: ``x`` is the embedded tokens, ``(rows, emb_dim)``, and the function
branches on its static shape. One row is a decode step: GEMV projections,
attention against the KV caches, the row written into them at
``cache_offset``, the softmax masked to ``vector_size`` keys. Many rows are
a prompt: GEMM projections, causal MHA over the rows, the caches written in
full from row zero, and the final norm and output head for row ``last``
alone. Both end in the same norm and head.

Each shape compiles its own version of the one function, and every version
runs in the function's one scratch arena (:mod:`iron.common.graph.compiled`):
the weights and the caches -- the caches are :func:`iron.state`, the weights
closed over from ``config.weights`` -- sit at one offset in every image and
are uploaded once, so the caches a prompt writes are the ones the next
decode step reads, and there is nothing to hand over.

The knobs the operators run with are a :class:`Profile`
(:func:`profile`), the graph function's own, applied whenever its body runs: the tile choices decode
and prefill were tuned with, keyed by operator shape, and the GEMMs' width
and row tile and MHA's pipeline count, which follow the model's shape and
``max_seq_len``. A call site spells a knob only where a shape does not
determine it.

``config`` is the model's shape (``n_heads``, ``n_kv_groups``, ``head_dim``,
``emb_dim``, ``hidden_dim``) with the parameters as ``config.weights``
(:class:`.weights.LlamaWeights`); the depth is the number of layers it
holds. Each array is closed over as-is, so the tracer names and pins it by
identity. Traced here on handles; compiled by ``npu.py`` against a device,
or by a test against nothing.
"""

import math

import aie.utils as aie_utils
import numpy as np
from ml_dtypes import bfloat16

import iron
from iron.common import Profile, Scratchpad
from iron.operators.copy import Copy
from iron.operators.elementwise_add import ElementwiseAdd
from iron.operators.elementwise_mul import ElementwiseMul
from iron.operators.gemm.op import GEMM
from iron.operators.gemv.op import GEMV
from iron.operators.mha.op import MHA
from iron.operators.repeat import Repeat
from iron.operators.rms_norm import RMSNorm
from iron.operators.rope.op import RoPE
from iron.operators.silu import SiLU
from iron.operators.softmax import Softmax
from iron.operators.transpose import Transpose


def _device_columns() -> int:
    """The bound device's width: eight on NPU2, four on NPU1; eight unbound."""
    dev = aie_utils.get_current_device()
    return dev.cols if dev is not None else 8


def profile(config, max_seq_len) -> Profile:
    """The knobs the graph's operators run with, keyed by their shapes.

    Decode's tiles are the ones it was tuned with: the per-column share of a
    row on the bound device's width, four input rows per GEMV tile (one for
    the down projection), thirty-two rows of logits. A prompt spans the
    device's columns for its norms and elementwise ops, one row per tile,
    and its GEMMs take the widest column count their narrowest projection
    fills at 64-wide tiles; MHA's pipelines and the GEMMs' row tile fit
    ``max_seq_len``.

    Every value is one the graph ran with before it was a profile; none has
    been re-measured. A tuner writing this profile replaces these lines.
    """
    H, G, D = config.n_heads, config.n_kv_groups, config.head_dim
    E, F, V = config.emb_dim, config.hidden_dim, config.vocab_size
    L, cols = max_seq_len, _device_columns()
    p = Profile()
    # -- decode: one row --------------------------------------------------
    p.add(GEMV, tile_size_input=4)
    p.add(GEMV, M=E, K=H * D, tile_size_output=E // cols)  # o
    p.add(GEMV, M=F, K=E, tile_size_output=F // cols)  # gate, up
    p.add(GEMV, M=E, K=F, tile_size_input=1, tile_size_output=E // cols)  # down
    p.add(GEMV, M=L, K=D, num_batches=H, tile_size_output=L // cols)  # scores
    p.add(GEMV, M=V, K=E, tile_size_output=32)  # the head
    p.add(Transpose, M=L, N=D, num_batches=H, num_aie_columns=2, m=256, n=32)
    p.add(ElementwiseAdd, size=E, tile_size=E // cols)
    p.add(ElementwiseMul, size=F, tile_size=F // cols)
    p.add(SiLU, size=F, tile_size=F // cols)
    # -- a prompt: many rows ----------------------------------------------
    p.add(RMSNorm, tile_size=E, num_aie_columns=cols)
    p.add(RMSNorm, rows=1, tile_size=E, num_aie_columns=1)  # decode: one core
    p.add(ElementwiseAdd, tile_size=E)
    p.add(ElementwiseMul, tile_size=min(F, ElementwiseMul.tile_cap))  # the FFN row
    p.add(SiLU, tile_size=min(F, SiLU.tile_cap))  # exceeds one core's line
    narrowest = min(H * D, G * D, E, F)
    p.add(
        GEMM,
        num_aie_columns=max(c for c in range(1, cols + 1) if narrowest % (64 * c) == 0),
        tile_m=min(64, L // 4),
    )
    p.add(MHA, num_of_pipelines=min(8, L // 64))
    return p


class LlamaGraph:
    """The graph function and the state it closes over.

    ``keys[i]`` and ``values[i]`` are the layer caches, each ``(n_kv_groups,
    max_seq_len, head_dim)``: the per-group layout both phases write and
    decode's repeat reads. ``scale`` is the attention scale as a tensor,
    since the elementwise multiply takes one.

    A prompt of ``rows`` rows needs ``rows`` a multiple of 512 (MHA's eight
    pipelines of 64 rows, four GEMM row tiles of 64) and at most
    ``max_seq_len``; a ``max_seq_len`` under 512 lowers both to fit it.
    """

    def __init__(self, config, max_seq_len):
        W = config.weights
        H, G, D = config.n_heads, config.n_kv_groups, config.head_dim
        E = config.emb_dim
        L = max_seq_len
        self.max_seq_len = L
        self.profile = profile(config, max_seq_len)
        cols = _device_columns()
        self.keys = [
            iron.state((G, L, D), name=f"keys_cache_{i}") for i in range(len(W.layers))
        ]
        self.values = [
            iron.state((G, L, D), name=f"values_cache_{i}")
            for i in range(len(W.layers))
        ]
        # 1/sqrt(head_dim) over every score, as the elementwise multiply wants it.
        self.scale = np.full((H, L), 1.0 / math.sqrt(D), dtype=bfloat16)
        keys, values, scale = self.keys, self.values, self.scale

        # -- one row: a decode step ------------------------------------------

        def decode_block(i, lw, x, angles, cache_offset, vector_size):
            h = RMSNorm(x, lw.norm1)
            # <grouped query attention>
            # Matrices are read as the checkpoint ships them, (out, in):
            # GEMV's (M, K). The projections into heads write half a head
            # per tile; q's shape is o's when H * D == E, so it is said here.
            q, k, v = (GEMV(w, h, tile_size_output=D // 2) for w in (lw.q, lw.k, lw.v))
            q = RoPE(q.reshape(H, D), angles)
            k = RoPE(k.reshape(G, D), angles)
            Copy(k, keys[i][:, cache_offset])
            Copy(v.reshape(G, D), values[i][:, cache_offset])
            # Every head sees its group's keys and values.
            k_all = Repeat(keys[i], repeat=H // G)
            v_all = Repeat(values[i], repeat=H // G)
            scores = GEMV(k_all, q)
            # One row of scores per column; its size is the FFN's when
            # H * L == F, so the tile is said here.
            scores = ElementwiseMul(scores, scale, tile_size=L // cols)
            # The valid row length is the context length: the kernel masks
            # every column from there on, so the cache's unwritten tail
            # contributes nothing.
            weights = Softmax(scores, vector_size=vector_size)
            v_t = Transpose(v_all)
            ctx = GEMV(v_t, weights)
            o = GEMV(lw.o, ctx.reshape(H * D))
            # </grouped query attention>
            x = ElementwiseAdd(x, o)
            h = RMSNorm(x, lw.norm2)
            gate = GEMV(lw.gate, h)
            up = GEMV(lw.up, h)
            act = ElementwiseMul(SiLU(gate), up)
            down = GEMV(lw.down, act)
            return ElementwiseAdd(x, down)

        # -- many rows: a prompt ---------------------------------------------

        def gemm(x, weight):
            # Every projection is read as the checkpoint ships it, (out, in):
            # GEMM's column-major B, the layout the GEMVs read too.
            return GEMM(x, weight, b_col_maj=True)

        def prefill_block(i, lw, x, angles):
            n = x.shape[0]
            h = RMSNorm(x, lw.norm1)
            # <grouped query attention>
            q = gemm(h, lw.q)  # (n, H*D)
            k = gemm(h, lw.k)  # (n, G*D)
            v = gemm(h, lw.v)
            # One angle row per position, applied to that position's heads.
            q = RoPE(q.reshape(n * H, D), angles)
            k = RoPE(k.reshape(n * G, D), angles)
            # (n, G, D), the heads interleaved per token as the projection
            # wrote them, into the first n rows of the cache's (G, L, D).
            Copy(
                k.reshape(n, G, D).transpose(1, 0, 2),
                keys[i][:, :n],
                transfer_size=1024,
            )
            Copy(
                v.reshape(n, G, D).transpose(1, 0, 2),
                values[i][:, :n],
                transfer_size=1024,
            )
            o = MHA(
                q.reshape(n, H, D),
                k.reshape(n, G, D),
                v.reshape(n, G, D),
                heads_interleaved=True,
            )
            o = gemm(o.reshape(n, H * D), lw.o)
            # </grouped query attention>
            x = ElementwiseAdd(x, o)
            h = RMSNorm(x, lw.norm2)
            gate = gemm(h, lw.gate)
            up = gemm(h, lw.up)
            act = ElementwiseMul(SiLU(gate), up)
            down = gemm(act, lw.down)
            return ElementwiseAdd(x, down)

        @iron.graph(names_from=W, profile=self.profile)
        def forward(
            x,
            angles,
            *,
            cache_offset: Scratchpad[np.int32],
            vector_size: Scratchpad[np.int32],
            last: Scratchpad[np.int32],
        ):
            prompt = x.shape[0] > 1
            for i, lw in enumerate(W.layers):
                if prompt:
                    x = prefill_block(i, lw, x, angles)
                else:
                    x = decode_block(i, lw, x, angles, cache_offset, vector_size)
            if prompt:
                # The last prompt row alone: its logits are all the host reads.
                x = Copy(x[last]).reshape(1, E)
            x = RMSNorm(x, W.norm)
            return GEMV(W.out_head, x)

        self.graph = forward

    def shapes(self, config, rows):
        """The input shapes of the version that runs ``rows`` tokens."""
        return dict(x=(rows, config.emb_dim), angles=(rows, config.head_dim))

    def trace(self, config, rows):
        return self.graph.trace(**self.shapes(config, rows))

    def compile(self, config, rows, **kwargs):
        return self.graph.compile(**self.shapes(config, rows), **kwargs)
