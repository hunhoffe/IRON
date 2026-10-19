#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA over ``iron.lm``: a chunk of robot actions from one camera
image, a task's token ids and the robot's state, judged by lerobot's own
``VLAFlowMatching``.

Two versions share the caches. ``observe`` runs SmolVLM's vision tower
(``idefics3``'s) and the first layers of its text model over the
observation: the image's soft tokens, the task's tokens and one row
projected from the state, each layer keeping its keys and values. The
image's and the task's rows attend both ways among themselves and the
state's row over all of them (``ObservationCache``). ``denoise`` then
runs the action expert, a narrower stack of Llama layers, over the noisy
action chunk and the timestep: its layers alternate between attending
over the observation's keys and values with its own after them, causally
(``self_attn_every_n_layers``), and over the observation's keys and values
projected by its own ``k_proj`` and ``v_proj`` (``CrossCache``). The host
integrates the expert's velocity from noise to actions in ``num_steps``
Euler steps (``sample``).
"""

import dataclasses
import math

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Scratchpad
from iron.applications.idefics3.model import SmolVLMTextConfig
from iron.applications.idefics3.vision.model import VisionConfig, VisionTower
from iron.lm import (
    Call,
    Config,
    CrossAttention,
    CrossCache,
    DecoderLayer,
    Embedding,
    KVCache,
    Linear,
    Model,
    RMSNorm,
    Rope,
    not_json,
    subtree,
)
from iron.operators.copy import Copy
from iron.operators.mha import MHA


def padded(n: int) -> int:
    """``n`` rounded up to a whole 64: MHA's unit of queries, and GEMM's
    of the columns it reduces over.
    """
    return -(-n // 64) * 64


@dataclasses.dataclass(frozen=True)
class SmolVLAConfig(Config):
    """SmolVLA's shape, as lerobot's ``config.json`` names it, and the
    shapes of its VLM's text model (``text``, its first
    ``num_hidden_layers`` layers) and vision tower (``vision``).
    """

    text: SmolVLMTextConfig = not_json(None)
    vision: VisionConfig = not_json(None)
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenizer_max_length: int = 48
    expert_width_multiplier: float = 0.75
    self_attn_every_n_layers: int = 2
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    @property
    def prefix(self) -> int:
        """The rows an observation's call holds: the soft tokens, the most
        task tokens and the state's, padded (``padded``).
        """
        return padded(self.vision.tokens + self.tokenizer_max_length + 1)

    @property
    def chunk(self) -> int:
        """The rows a call of the expert holds: ``chunk_size``, padded."""
        return padded(self.chunk_size)

    @property
    def actions(self) -> int:
        """The columns an action row holds: ``max_action_dim``, padded,
        zero past it.
        """
        return padded(self.max_action_dim)

    def vlm(self) -> SmolVLMTextConfig:
        """The VLM's text model's shape over the observation, then the
        action chunk, RoPE at lerobot's base of 10000.
        """
        return dataclasses.replace(
            self.text,
            rope_theta=10000.0,
            max_seq_len=self.prefix + self.chunk,
            prefill_chunk=self.prefix,
        )

    def expert(self) -> SmolVLMTextConfig:
        """The action expert's shape: the VLM's heads, narrower, a chunk a
        call; its MLP width rounded as lerobot's ``get_intermediate_size``.
        """
        width = int(self.text.hidden_size * self.expert_width_multiplier)
        hidden = 4 * int(2 * width / 3)
        return dataclasses.replace(
            self.vlm(),
            hidden_size=width,
            intermediate_size=-(-hidden // 256) * 256,
            prefill_chunk=self.chunk,
        )


class ObservationCache(KVCache):
    """A VLM layer's keys and values of an observation, ``rows`` from the
    first: every row but the last, the state's, attends both ways over
    those rows, and the state's over all of them.
    """

    def attend(self, call, q, k, v, scale: float | None = None):
        n, H, D = q.shape
        for x, cache in ((k, self.keys), (v, self.values)):
            cache[: self.chunk][: call.rows] = x
        mha = dict(heads_interleaved=True, kv_interleaved=True, causal=False)
        o = MHA.step(q, self.keys[: call.position], self.values[: call.position], **mha)
        state = Copy.step(q[call.position]).reshape(1, H, D)
        span = np.s_[: call.position + 1]
        last = MHA.step(state, self.keys[span], self.values[span], **mha)
        o[call.position] = last.reshape(H, D)
        return o.reshape(n, H * D)


class Language(iron.Graph):
    """The VLM's text model's parts SmolVLA runs: ``embed_tokens`` and
    ``layers``, each keeping an observation in an ``ObservationCache``.
    """

    def __init__(self, c: SmolVLMTextConfig):
        self.embed_tokens = Embedding(c.vocab_size, c.hidden_size)
        self.layers = [
            DecoderLayer(c, i, cache=ObservationCache(c))
            for i in range(c.num_hidden_layers)
        ]


class Expert(iron.Graph):
    """The action expert: ``layers``, every ``self_attn_every_n_layers``-th
    attending over its VLM layer's cache, its own rows after the
    observation's, the rest over a ``CrossCache`` of the observation; and
    the final ``norm``.
    """

    def __init__(self, c: SmolVLAConfig, vlm: Language):
        e, kv = c.expert(), c.text.num_key_value_heads * c.text.head_dim
        self.layers = []
        for i, held in enumerate(vlm.layers):
            if i % c.self_attn_every_n_layers == 0:
                layer = DecoderLayer(e, i)
                layer.cache.keys, layer.cache.values = (
                    held.cache.keys,
                    held.cache.values,
                )
            else:
                attention = CrossAttention(e, i)
                attention.k_proj = Linear(kv, kv, e.attention_bias, e.precision)
                attention.v_proj = Linear(kv, kv, e.attention_bias, e.precision)
                layer = DecoderLayer(
                    e, i, self_attn=attention, cache=CrossCache(e, c.prefix)
                )
            self.layers.append(layer)
        self.norm = RMSNorm(e.hidden_size, e.eps)


class ActionTime(iron.Graph):
    """The noisy actions and the timestep's embedding into the expert's
    space: ``action_time_mlp_out(silu(action_time_mlp_in([action_in_proj(x),
    time])))``, the timestep's half of ``action_time_mlp_in`` one row added
    to every action's.
    """

    def __init__(self, c: SmolVLAConfig):
        E, p = c.expert().hidden_size, c.text.precision
        self.action_in_proj = Linear(c.max_action_dim, E, precision=p)
        self.action_time_mlp_in = Linear(2 * E, E, precision=p)
        self.action_time_mlp_out = Linear(E, E, precision=p)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        weight = np.asarray(self.action_time_mlp_in.weight)
        E = weight.shape[0]
        self.action, self.time = iron.param(weight[:, :E]), iron.param(weight[:, E:])
        weight = np.asarray(self.action_in_proj.weight)
        columns = padded(weight.shape[1]) - weight.shape[1]
        self.action_in_proj.weight = iron.param(np.pad(weight, ((0, 0), (0, columns))))

    def body(self, x, time):
        p = self.action_time_mlp_in.precision
        h = F.linear(self.action_in_proj(x), self.action, precision=p)
        bias = F.linear(time, self.time, self.action_time_mlp_in.bias, precision=p)
        h = h + bias.reshape(bias.shape[-1])
        return self.action_time_mlp_out(F.silu(h))


class SmolVLA(Model):
    """SmolVLA on ``config``'s shape and a checkpoint's ``weights``, by
    lerobot's names under ``model.``: an observation in, an action chunk out
    (``sample``).
    """

    config = SmolVLAConfig

    def __init__(self, config: SmolVLAConfig, weights):
        c, vlm = config, weights.model.vlm_with_expert.vlm.model
        self.config, self.weights = c, weights
        self.tower = VisionTower(c.vision, subtree(weights, model=vlm))
        text = vlm.text_model
        self.language = Language(c.vlm())
        self.language.load_state_dict(
            subtree(text, embed_tokens=text.embed_tokens, layers=text.layers)
        )
        self.expert = Expert(c, self.language)
        self.expert.load_state_dict(weights.model.vlm_with_expert.lm_expert)
        p = c.text.precision
        self.state_proj = Linear(c.max_state_dim, c.text.hidden_size, precision=p)
        self.state_proj.load_state_dict(weights.model.state_proj)
        self.action_time = ActionTime(c)
        m = weights.model
        self.action_time.load_state_dict(
            subtree(
                m,
                action_in_proj=m.action_in_proj,
                action_time_mlp_in=m.action_time_mlp_in,
                action_time_mlp_out=m.action_time_mlp_out,
            )
        )
        E = c.expert().hidden_size
        self.action_out_proj = Linear(E, c.max_action_dim, precision=p)
        self.action_out_proj.load_state_dict(weights.model.action_out_proj)
        out = self.action_out_proj
        rows = ((0, c.actions - c.max_action_dim),)
        out.weight = iron.param(np.pad(np.asarray(out.weight), rows + ((0, 0),)))
        out.bias = iron.param(np.pad(np.asarray(out.bias), rows))
        self.vlm_rope, self.expert_rope = Rope(c.vlm()), Rope(c.expert())
        for layer in self.language.layers:
            layer.rope = self.vlm_rope
        for i, layer in enumerate(self.expert.layers):
            self_attending = i % c.self_attn_every_n_layers == 0
            layer.rope = self.expert_rope if self_attending else "chunk"
        self.rows = iron.state((c.prefix, c.text.hidden_size))

    def body(
        self,
        rgb=None,
        ids=None,
        state=None,
        actions=None,
        time=None,
        *,
        rows: Scratchpad[np.int32],
    ):
        """``observe`` on ``rgb``, ``ids`` (``rows`` of them the task's) and
        ``state``, else ``denoise`` on ``actions`` and ``time``.
        """
        if rgb is not None:
            return self.observe(rgb, ids, state, rows)
        return self.denoise(actions, time, rows)

    def observe(self, rgb, ids, state, rows):
        """The VLM's layers over the observation, each keeping its keys and
        values, and each cross layer's of the expert projected from them.
        """
        c, T = self.config, self.config.vision.tokens
        E, L = c.text.hidden_size, c.tokenizer_max_length
        scale = float(bfloat16(math.sqrt(E)))
        self.rows[:T] = self.tower(rgb) * scale
        self.rows[T : T + L][:rows] = self.language.embed_tokens(ids[:rows]) * scale
        self.rows[T + rows] = self.state_proj(
            state.reshape(1, c.max_state_dim)
        ).reshape(E)
        call = Call(prompt=True, start=0, rows=T + rows + 1, position=T + rows)
        call.ropes[self.vlm_rope] = self.vlm_rope.rows(call)
        x = self.rows[: T + rows + 1]
        for layer in self.language.layers:
            x = layer(x, call)
        G, D, n = c.text.num_key_value_heads, c.text.head_dim, c.prefix
        for held, layer in zip(self.language.layers, self.expert.layers):
            if isinstance(layer.cache, CrossCache):
                projected = [
                    p(cache[:n][: T + rows + 1].reshape(n, G * D)).reshape(n, G, D)
                    for p, cache in (
                        (layer.self_attn.k_proj, held.cache.keys),
                        (layer.self_attn.v_proj, held.cache.values),
                    )
                ]
                layer.cache.write(*projected)

    def denoise(self, actions, time, rows):
        """The expert's velocity at ``actions``, its first ``chunk_size``
        rows, the timestep's embedding ``time``.
        """
        c = self.config
        P, C = c.vision.tokens + rows + 1, c.chunk_size
        # C as a per-call count: a run starting at the per-call P needs a
        # per-call bound, and C alone is a constant.
        whole = rows * 0 + C
        call = Call(prompt=True, start=P, rows=whole, position=P + C - 1, source=P)
        call.ropes[self.expert_rope] = self.expert_rope.rows(call)
        first = dataclasses.replace(call, start=0)
        call.ropes["chunk"] = self.expert_rope.rows(first)
        x = self.action_time(actions[:whole], time)
        for layer in self.expert.layers:
            x = layer(x, call)
        return self.action_out_proj(self.expert.norm(x))

    def shapes(self) -> list[dict]:
        """``observe``'s inputs, then ``denoise``'s."""
        c = self.config
        side = c.vision.image_size
        return [
            dict(
                rgb=((side, side * 3), bfloat16),
                ids=((c.tokenizer_max_length,), np.int32),
                state=((c.max_state_dim,), bfloat16),
            ),
            dict(
                actions=((c.chunk, c.actions), bfloat16),
                time=((1, c.expert().hidden_size), bfloat16),
            ),
        ]

    def time(self, t: float) -> np.ndarray:
        """The timestep ``t``'s sine-cosine embedding, its periods from
        ``min_period`` to ``max_period``, ``(1, width)`` bfloat16.
        """
        c, width = self.config, self.config.expert().hidden_size
        fraction = np.linspace(0.0, 1.0, width // 2)
        period = c.min_period * (c.max_period / c.min_period) ** fraction
        angle = 2 * np.pi / period * t
        return np.concatenate([np.sin(angle), np.cos(angle)])[None].astype(bfloat16)

    def sample(self, image, ids, state, noise) -> np.ndarray:
        """The action chunk, ``(chunk_size, max_action_dim)`` float32, for
        ``image`` (``VisionTower.inputs``), the task's token ``ids`` and the
        padded ``state``, from ``noise`` at ``t = 1`` to ``t = 0`` in
        ``num_steps`` Euler steps.
        """
        c = self.config
        rgb, _ = self.tower.inputs(image)
        tokens = np.zeros(c.tokenizer_max_length, np.int32)
        tokens[: len(ids)] = ids
        state = np.asarray(state).astype(bfloat16)
        self(rgb, tokens, state, rows=len(ids))
        x, dt = np.asarray(noise, np.float32), -1.0 / c.num_steps
        actions = np.zeros((c.chunk, c.actions), bfloat16)
        for step in range(c.num_steps):
            actions[: c.chunk_size, : c.max_action_dim] = x
            time = self.time(1.0 + step * dt)
            v = self(None, None, None, actions, time, rows=len(ids)).numpy()
            v = np.asarray(v, np.float32).reshape(-1, c.actions)
            x = x + dt * v[: c.chunk_size, : c.max_action_dim]
        return x
