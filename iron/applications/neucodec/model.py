#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeuCodec's decoder, which turns NeuTTS's speech codes into 24 kHz audio:
one graph from a clip's frames to their spectra (``NeuCodecDecoder``), the
codes before it and the inverse STFT after it on the host.

A code is eight digits of four levels, each a value of -1, -0.5, 0 or 0.5;
two linear maps take them to the hidden width (``inputs``). The graph runs
a convolution of seven frames, two residual blocks, the transformer layers,
two more residual blocks, a LayerNorm and a linear head, whose rows are
each frame's log-magnitudes and phases; ``audio`` overlaps their inverse
FFTs. A residual block is GroupNorm, swish and a convolution of three
frames, twice, added to its input.

The layers attend both ways. NeuCodec rotates q and k by RoPE with the
head as the position, the same angle at every frame of a head, which
leaves each head's scores as they are: no RoPE here.
"""

import dataclasses

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron.applications.phi3.model import Phi4Attention
from iron.lm import MLP, Config, Conv1d, DecoderLayer, LayerNorm, Linear, NoCache
from iron.lm.causal import Call
from iron.lm.checkpoint import renamed
from iron.operators.copy import Copy

#: GroupNorm's groups, every residual block's.
GROUPS = 32

#: A transformer layer's paths under the names ``iron.lm``'s modules read.
RENAMES = {
    "att_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "att.c_attn": "self_attn.qkv_proj",
    "att.c_proj": "self_attn.o_proj",
}


@dataclasses.dataclass(frozen=True)
class NeuCodecConfig(Config):
    """The decoder's shape, NeuCodec's by default; its checkpoint has no
    ``config.json``.
    """

    hidden_act = "silu"

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    hop_length: int = 480

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        return 4 * self.hidden_size

    @property
    def n_fft(self) -> int:
        """Each frame's FFT, four hops."""
        return 4 * self.hop_length


class NeuCodecLayer(DecoderLayer):
    """``DecoderLayer`` over NeuCodec's fused qkv, a plain ``MLP`` and no
    cache.
    """

    def __init__(self, c: NeuCodecConfig, i: int):
        super().__init__(c, i, Phi4Attention(c, i), MLP(c), NoCache())


class GroupNorm(iron.Graph):
    """``torch.nn.GroupNorm`` over ``(frames, channels)``: each group's
    channels over every frame gathered into one row, LayerNorm'd (at its
    kernel's eps), scattered back, then each channel's weight and bias.
    """

    def __init__(self, norm):
        self.weight, self.bias = norm.weight, norm.bias

    def body(self, x):
        T, C = x.shape
        G, W = GROUPS, C // GROUPS
        grouped = TensorAccessPattern((T, C), 0, [G, T, W], [W, C, 1])
        rows = Copy.step(x, src=grouped).reshape(G, T * W)
        h = F.layer_norm(rows, (T * W,))
        back = TensorAccessPattern((G, T * W), 0, [T, G, W], [W, T * W, 1])
        return Copy.step(h, src=back).reshape(T, C) * self.weight + self.bias


class ResnetBlock(iron.Graph):
    """GroupNorm, swish and a convolution of three frames, twice, added to
    the input; each convolution reads a zero frame either side.
    """

    def __init__(self, block, frames: int):
        C = block.conv1.bias.shape[0]
        self.norm1, self.norm2 = GroupNorm(block.norm1), GroupNorm(block.norm2)
        self.conv1 = Conv1d(block.conv1.weight, block.conv1.bias, C)
        self.conv2 = Conv1d(block.conv2.weight, block.conv2.bias, C)
        self.padded = [iron.state((frames + 2, C)) for _ in range(2)]

    def body(self, x):
        T = x.shape[0]
        h = x
        for norm, conv, padded in zip(
            (self.norm1, self.norm2), (self.conv1, self.conv2), self.padded
        ):
            padded[1 : T + 1] = F.silu(norm(h))
            h = conv(padded, rows=T)
        return x + h


class NeuCodecDecoder(iron.Graph):
    """NeuCodec's decoder over a clip of ``frames`` frames, one compile per
    length: ``inputs`` gives its buffer, the graph its spectra, ``audio``
    the waveform.

    Args:
        config: The decoder's shape.
        weights: NeuCodec's tree; ``fc_post_a`` and ``generator`` are read.
        frames: The clip's frames, 50 a second.
    """

    def __init__(self, config: NeuCodecConfig, weights, frames: int):
        c, g = config, weights.generator
        self.config, self.frames = c, frames
        b = g.backbone
        # The two linear maps from a code to the hidden width, as one, held on
        # the host (``_``: no weight of the graph).
        q, a = g.quantizer.project_out, weights.fc_post_a
        w, wa = np.asarray(q.weight, np.float32), np.asarray(a.weight, np.float32)
        self._codebook = (wa @ w).T
        self._offset = wa @ np.asarray(q.bias, np.float32) + a.bias
        self.embed = Conv1d(b.embed.weight, b.embed.bias, c.hidden_size)
        self.prior_net = [ResnetBlock(r, frames) for r in b.prior_net]
        self.layers = [NeuCodecLayer(c, i) for i in range(c.num_hidden_layers)]
        tree = renamed(b, RENAMES)
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(tree, path=f"transformers.{i}.")
        self.post_net = [ResnetBlock(r, frames) for r in b.post_net]
        self.final_layer_norm = LayerNorm(c.hidden_size)
        self.final_layer_norm.load_state_dict(b, path="final_layer_norm.")
        # The head's rows padded to a whole tile with zero rows.
        n, E = c.n_fft + 2, c.hidden_size
        self.out = Linear(E, -(-n // 64) * 64)
        weight = np.zeros((self.out.out_features, E), bfloat16)
        bias = np.zeros(self.out.out_features, bfloat16)
        weight[:n], bias[:n] = g.head.out.weight, g.head.out.bias
        self.out.weight, self.out.bias = iron.param(weight), iron.param(bias)

    def body(self, x):
        T = self.frames
        h = self.embed(x, rows=T)
        for block in self.prior_net:
            h = block(h)
        call = Call(prompt=True, start=0, rows=T, position=None)
        for layer in self.layers:
            h = layer(h, call)
        for block in self.post_net:
            h = block(h)
        return self.out(self.final_layer_norm(h))

    def inputs(self, codes) -> np.ndarray:
        """The graph's buffer for ``codes``, ``frames`` FSQ indices: their
        rows at the hidden width, three zero frames either side.

        Raises:
            ValueError: ``codes`` are not ``frames`` indices.
        """
        codes = np.asarray(codes)
        if codes.shape != (self.frames,):
            raise ValueError(
                f"NeuCodecDecoder: codes {codes.shape} are not ({self.frames},)"
            )
        digits = codes[:, None] // 4 ** np.arange(8) % 4
        buffer = np.zeros((self.frames + 6, self.config.hidden_size), bfloat16)
        buffer[3:-3] = (digits - 2) / 2 @ self._codebook + self._offset
        return buffer

    def audio(self, spectra) -> np.ndarray:
        """The waveform of the graph's ``spectra``: each frame's
        magnitudes (capped at 100) and phases inverse-FFT'd, windowed by
        Hann's, overlapped at the hop and divided by the windows' sum,
        trimmed by 1.5 hops either side; ``(frames * hop_length,)`` float32.
        """
        c = self.config
        N, H = c.n_fft, c.hop_length
        s = np.asarray(spectra, np.float64)[:, : N + 2]
        mag, phase = np.minimum(np.exp(s[:, : N // 2 + 1]), 1e2), s[:, N // 2 + 1 :]
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
        frames = np.fft.irfft(mag * np.exp(1j * phase), N, axis=1) * window
        T = len(frames)
        y, envelope = np.zeros((T + 3, H)), np.zeros((T + 3, H))
        for j in range(4):
            y[j : j + T] += frames[:, j * H : (j + 1) * H]
            envelope[j : j + T] += window[j * H : (j + 1) * H] ** 2
        trim = np.s_[(N - H) // 2 : -(N - H) // 2]
        return (y.reshape(-1)[trim] / envelope.reshape(-1)[trim]).astype(np.float32)
