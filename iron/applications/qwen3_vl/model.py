#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL over ``iron.lm``: its text model (``Qwen3VLText``, Qwen3's
layers under ``model.language_model``) and its vision tower
(``vision.py``) as one ``iron.lm.Multimodal`` (``Qwen3VL``), its shape
read from the checkpoint's nested ``config.json``.

A token's RoPE position has three parts, time, row and column, each
turning its own share of a head's frequencies, interleaved (MRoPE). A text
token's three are equal and count up; an image's soft token at row ``i``
and column ``j`` of its grid takes the image's start plus ``(0, i, j)``,
and the text after it continues from the start plus the grid's side. So
the RoPE row of a position follows from the tokens before it: the host
writes every position's row into the text model's table, a state, before
the calls that read it (``Qwen3VLText.place``).

The vision tower's DeepStack rows are added to the image's rows after the
first decoder layers, one set a layer, gathered by placeholder as the soft
tokens are; a text row's gather reads zeros.
"""

import dataclasses
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron.applications.qwen3.model import Qwen3, Qwen3Layer
from iron.applications.qwen3_vl.vision import VisionConfig, VisionTower
from iron.lm import Call, CausalLM, DecoderConfig, Multimodal, not_json
from iron.common.graph.handle import State
from iron.lm.blocks import Rope
from iron.lm.checkpoint import subtree
from iron.operators.copy import Copy
from iron.operators.merge import Merge


@dataclasses.dataclass(frozen=True)
class Qwen3VLConfig(DecoderConfig):
    """Qwen3-VL's text model's shape, from the checkpoint's nested
    ``text_config``, and the vision tower's, ``vision``.
    """

    json_prefix = "text_config."

    # Each part's frequencies, time's first: row's and column's every third
    # from the second and third, time's the rest.
    mrope_section: tuple[int, int, int] = dataclasses.field(
        default=(24, 20, 20),
        kw_only=True,
        metadata={"json": "rope_scaling.mrope_section", "convert": tuple},
    )
    image_token_id: int = dataclasses.field(
        default=151655, kw_only=True, metadata={"json": "/image_token_id"}
    )
    vision: VisionConfig = not_json(VisionConfig(), kw_only=True)

    @classmethod
    def from_checkpoint(cls, path) -> "Qwen3VLConfig":
        text = super().from_checkpoint(path)
        return dataclasses.replace(text, vision=VisionConfig.from_checkpoint(path))


class PlacedRope(Rope):
    """``Rope`` on a table the host writes, a row a position."""

    table: State  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, c: DecoderConfig):
        self.table = iron.state((c.max_seq_len, c.rope_width))
        self.q_table = None
        self.chunk, self.width = c.prefill_chunk, c.rope_width


@dataclasses.dataclass
class StackedCall(Call):
    """A placed chunk's ``Call``: the rows added after each first layer."""

    deepstack: tuple = ()


class Qwen3VLText(CausalLM):
    """Qwen3-VL's text model on ``config``'s shape and a checkpoint's
    ``weights``: the decoder under ``model.language_model``, its RoPE
    table written by the host (``place``).
    """

    config = Qwen3VLConfig
    layer_class = Qwen3Layer
    bos = Qwen3.bos

    def __init__(self, config: Qwen3VLConfig, weights):
        lm_head = {"lm_head": weights.lm_head} if hasattr(weights, "lm_head") else {}
        super().__init__(
            config, subtree(weights, model=weights.model.language_model, **lm_head)
        )
        self._placed = None

    def rotary(self, config) -> Rope:
        return PlacedRope(config)

    def rope_positions(self, tokens) -> np.ndarray:
        """Each position's three RoPE positions, ``(max_seq_len, 3)``, after
        ``tokens``: an image's run of placeholders on its grid, every other
        position text, past ``tokens`` too.
        """
        c = self.config
        side = c.vision.side // c.vision.spatial_merge_size
        grid = np.indices((side, side)).reshape(2, -1).T
        tokens = np.asarray(tokens)
        out = np.empty((c.max_seq_len, 3), np.int64)
        p = i = 0
        while i < tokens.size:
            if tokens[i] == c.image_token_id:
                out[i : i + side**2] = p
                out[i : i + side**2, 1:] += grid
                p, i = p + side, i + side**2
            else:
                out[i], p, i = p, p + 1, i + 1
        out[i:] = p + np.arange(c.max_seq_len - i)[:, None]
        return out

    def place(self, tokens) -> None:
        """Write the RoPE row of every position after ``tokens`` into the
        table, unless it holds them.
        """
        positions = self.rope_positions(tokens)
        if self._placed is not None and np.array_equal(positions, self._placed):
            return
        c, D = self.config, self.config.head_dim
        # rope_angles' rounding: float32 frequencies, each sine in float64.
        exponents = np.arange(0, D, 2, dtype=np.float32) / np.float32(D)
        inv = (1.0 / np.power(c.rope_theta, exponents.astype(np.float64))).astype(
            np.float32
        )
        part = np.zeros(D // 2, np.int64)
        for k in (1, 2):
            part[k : 3 * c.mrope_section[k] : 3] = k
        freqs = positions[:, part].astype(np.float32) * inv
        rows = np.empty((c.max_seq_len, D), np.float32)
        rows[:, ::2] = np.cos(freqs.astype(np.float64))
        rows[:, 1::2] = np.sin(freqs.astype(np.float64))
        rope = self.rotary_emb
        assert isinstance(rope, PlacedRope)
        table = rope.table
        if self.decode_step is None:
            table.host = rows.astype(bfloat16)
        else:
            self.decode_step.write(table, rows.astype(bfloat16))
        self._placed = positions

    def logits(self, tokens) -> np.ndarray:
        self.place(tokens)
        return super().logits(tokens)

    def _draw(self, tokens, max_tokens, sampler, each=True):
        self.place(tokens)
        return super()._draw(tokens, max_tokens, sampler, each)

    def from_rows(self, x, *, deepstack=(), token, position, start, rows, rope_start):
        """``CausalLM.from_rows``, ``deepstack[i]`` added after layer ``i``."""
        call = StackedCall(
            prompt=True,
            start=start,
            rows=rows,
            position=position,
            rope_start=rope_start,
            deepstack=deepstack,
        )
        return self.forward(x, call)

    def layer(self, call: Call, i: int, w, x):
        x = w(x, call)
        if isinstance(call, StackedCall) and i < len(call.deepstack):
            x = x + call.deepstack[i]
        return x


class Qwen3VL(Multimodal):
    """Qwen3-VL, its text model and vision tower as one
    ``iron.lm.Multimodal``, from its checkpoint and ``tokenizer.json``.
    """

    config = Qwen3VLConfig
    bos = Qwen3.bos

    def __init__(self, config: Qwen3VLConfig, weights):
        self.config, self.weights = config, weights
        self.tower = VisionTower(config.vision, weights, config.prefill_chunk)
        super().__init__(
            Qwen3VLText(config, weights), {config.image_token_id: self.tower}
        )

    def from_placed(self, rows, placed, counted: dict, text: dict):
        """The text model on ``rows``, with each DeepStack layer's rows
        gathered by placeholder from past the tower's zero rows.
        """
        c, T = self.config, self.config.vision.tokens
        deepstack = tuple(
            Copy.step(
                self.tower.stacked[
                    Merge.step(
                        placed,
                        placeholders=((c.image_token_id, c.prefill_chunk + k * T),),
                        **counted,
                    )
                ]
            )[: text["rows"]]
            for k in range(len(c.vision.deepstack_visual_indexes))
        )
        return self.text.inline(self.text.from_rows, rows, deepstack=deepstack, **text)
