#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA against lerobot's own ``VLAFlowMatching`` in float32 over the
same random weights, its action chunks recorded in ``goldens/``,
device-free, at a toy shape: an 8x8 patch grid shuffled 2x2, four VLM
layers, an expert three quarters as wide. Recording needs lerobot (its
SmolVLA modules, without its package's other policies), torch and
transformers.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.applications.idefics3.model import SmolVLMTextConfig
from iron.applications.idefics3.vision.model import VisionConfig
from iron.applications.smolvla.model import SmolVLA, SmolVLAConfig
from iron.lm.golden import flat
from iron.lm.testing import LOGITS
from iron.tests.support.llama_model import checkpoint, shapes

pytestmark = pytest.mark.usefixtures("npu2")

VISION = VisionConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    patch_size=16,
    image_size=128,
    scale_factor=2,
    text_dim=256,
    image_token=511,
)
TEXT = SmolVLMTextConfig(
    vocab_size=512,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=512,
    rope_theta=1e5,
    max_seq_len=1024,
)
TOY = SmolVLAConfig(text=TEXT, vision=VISION, chunk_size=16, tokenizer_max_length=16)


def names(c: SmolVLAConfig) -> dict[str, tuple[int, ...]]:
    """Each tensor of a SmolVLA checkpoint of ``c``'s shape, by lerobot's
    names.
    """
    v, E = c.vision, c.vision.hidden_size
    vlm, expert = "model.vlm_with_expert.vlm.", "model.vlm_with_expert.lm_expert."
    vision = f"{vlm}model.vision_model."
    out = {
        f"{vision}embeddings.patch_embedding.weight": (
            E,
            3,
            v.patch_size,
            v.patch_size,
        ),
        f"{vision}embeddings.patch_embedding.bias": (E,),
        f"{vision}embeddings.position_embedding.weight": (v.side**2, E),
        f"{vision}post_layernorm.weight": (E,),
        f"{vision}post_layernorm.bias": (E,),
        f"{vlm}model.connector.modality_projection.proj.weight": (
            v.text_dim,
            E * v.scale_factor**2,
        ),
        f"{vlm}lm_head.weight": (c.text.vocab_size, c.text.hidden_size),
    }
    for i in range(v.num_hidden_layers):
        p = f"{vision}encoder.layers.{i}."
        for norm in ("layer_norm1", "layer_norm2"):
            out |= {f"{p}{norm}.weight": (E,), f"{p}{norm}.bias": (E,)}
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out |= {
                f"{p}self_attn.{proj}.weight": (E, E),
                f"{p}self_attn.{proj}.bias": (E,),
            }
        out |= {
            f"{p}mlp.fc1.weight": (v.intermediate_size, E),
            f"{p}mlp.fc1.bias": (v.intermediate_size,),
            f"{p}mlp.fc2.weight": (E, v.intermediate_size),
            f"{p}mlp.fc2.bias": (E,),
        }
    for name, shape in shapes(c.text).items():
        out[name.replace("model.", f"{vlm}model.text_model.", 1)] = shape
    KV = c.text.num_key_value_heads * c.text.head_dim
    for name, shape in shapes(c.expert()).items():
        if "embed_tokens" in name:
            continue
        cross = (
            int(name.split(".")[2]) % c.self_attn_every_n_layers
            if "layers" in name
            else 0
        )
        if cross and name.endswith(("k_proj.weight", "v_proj.weight")):
            shape = (KV, KV)
        out[name.replace("model.", expert, 1)] = shape
    W, S, A = c.expert().hidden_size, c.max_state_dim, c.max_action_dim
    for name, shape in (
        ("state_proj", (c.text.hidden_size, S)),
        ("action_in_proj", (W, A)),
        ("action_out_proj", (A, W)),
        ("action_time_mlp_in", (W, 2 * W)),
        ("action_time_mlp_out", (W, W)),
    ):
        out |= {f"model.{name}.weight": shape, f"model.{name}.bias": shape[:1]}
    return out


class LeRobot:
    """lerobot's own ``VLAFlowMatching`` at ``c``'s shape over ``weights``,
    its VLM's config given rather than fetched; ``logits`` is the action
    chunk it samples.
    """

    def __init__(self, c: SmolVLAConfig, weights):
        import lerobot
        import torch
        import transformers

        # lerobot's policies package imports every policy; SmolVLA's
        # modules need none of them.
        policies = types.ModuleType("lerobot.policies")
        policies.__path__ = [str(Path(lerobot.__file__).parent / "policies")]
        sys.modules.setdefault("lerobot.policies", policies)
        from lerobot.policies.smolvla import configuration_smolvla, modeling_smolvla
        from lerobot.policies.smolvla import smolvlm_with_expert as expert

        json = c.vision.to_json()
        json["text_config"] = c.text.to_json()["text_config"]
        vlm = transformers.SmolVLMConfig(**json)
        expert.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *_: vlm)
        tokenizer = types.SimpleNamespace(
            fake_image_token_id=0, global_image_token_id=0
        )
        expert.AutoProcessor = types.SimpleNamespace(
            from_pretrained=lambda *_: types.SimpleNamespace(tokenizer=tokenizer)
        )

        class Cache(expert.DynamicCache):
            """transformers' cache, cropped to its first ``n`` positions as
            lerobot expects; transformers' own refuses a positive ``n``.
            """

            def crop(self, n):
                for layer in self.layers:
                    layer.keys, layer.values = (
                        layer.keys[..., :n, :],
                        layer.values[..., :n, :],
                    )

        expert.DynamicCache = Cache
        config = configuration_smolvla.SmolVLAConfig(
            num_vlm_layers=c.text.num_hidden_layers,
            chunk_size=c.chunk_size,
            n_action_steps=c.chunk_size,
            load_vlm_weights=False,
        )
        self.model = modeling_smolvla.VLAFlowMatching(config).eval()
        state = {
            k.removeprefix("model."): torch.from_numpy(t) for k, t in flat(weights)
        }
        self.model.load_state_dict(state, strict=True)

    def observation(self, tokens, image, state) -> tuple:
        """``sample_actions``' first arguments: the images and their masks,
        the task's ids and its mask, and the state.
        """
        import torch

        def tensor(x):
            return torch.from_numpy(np.asarray(x, np.float32))[None]

        ids = torch.from_numpy(np.asarray(tokens, np.int64))[None]
        return (
            [tensor(image).permute(0, 3, 1, 2)],
            [torch.ones(1, dtype=torch.bool)],
            ids,
            torch.ones_like(ids, dtype=torch.bool),
            tensor(state),
        )

    def logits(self, tokens, image, state, noise) -> np.ndarray:
        import torch

        noise = torch.from_numpy(noise)[None]
        with torch.no_grad():
            actions = self.model.sample_actions(
                *self.observation(tokens, image, state), noise=noise
            )
        return actions[0].numpy()


class Observed(LeRobot):
    """lerobot's pass over an observation: each VLM layer's keys, then its
    values, as ``logits``.
    """

    def logits(self, tokens, image, state) -> np.ndarray:
        import torch
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        model = self.model
        images, masks, ids, words, state = self.observation(tokens, image, state)
        with torch.no_grad():
            rows, pad, causal = model.embed_prefix(images, masks, ids, words, state)
            _, cache = model.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, causal),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                inputs_embeds=[rows, None],
                use_cache=True,
            )
        held = [x for layer in cache.layers for x in (layer.keys, layer.values)]
        return torch.stack(held)[:, 0].transpose(1, 2).numpy()


class Host(SmolVLA):
    """SmolVLA, each call the graph's reference."""

    def __call__(self, *tensors, **values):
        out = np.asarray(self.reference(*tensors, **values))
        return types.SimpleNamespace(numpy=lambda: out)


def observation(words: int) -> tuple:
    """An image, a task of ``words`` tokens and a state, the image and the
    state as the device holds them, and the noise an action chunk starts at.
    """
    rng = np.random.default_rng(words)
    image = rng.uniform(-1, 1, (128, 128, 3)).astype(bfloat16).astype(np.float32)
    tokens = rng.integers(0, TEXT.vocab_size, words)
    state = rng.standard_normal(TOY.max_state_dim).astype(bfloat16).astype(np.float32)
    noise = rng.standard_normal((TOY.chunk_size, TOY.max_action_dim)).astype(np.float32)
    return image, tokens, state, noise


def test_an_observation_s_keys_and_values_are_lerobot_s(golden):
    """Each VLM layer's keys and values of an observation, the image's and
    the task's rows attending both ways but not to the state's, within 5%
    of lerobot's largest, as a decoder's logits are held.
    """
    weights = checkpoint(TEXT, names(TOY)).tree("random")
    model = Host(TOY, weights)
    reference = golden("observed", lambda: Observed(TOY, weights), TOY, weights)
    image, tokens, state, _ = observation(5)
    rgb, _ = model.tower.inputs(image)
    ids = np.zeros(TOY.tokenizer_max_length, np.int32)
    ids[: tokens.size] = tokens
    model(rgb, ids, state.astype(bfloat16), rows=tokens.size)
    P = VISION.tokens + tokens.size + 1
    got = np.stack(
        [
            np.asarray(x.host[:P], np.float32)
            for layer in model.language.layers
            for x in (layer.cache.keys, layer.cache.values)
        ]
    )
    want = reference.logits(tokens, image, state)
    assert got.shape == want.shape
    err, scale = np.abs(got - want).max(), np.abs(want).max()
    assert err <= LOGITS * scale, f"|diff| {err:.4f} against |rows| {scale:.3f}"


@pytest.mark.parametrize("words", [5, 16])
def test_an_action_chunk_is_lerobot_s(golden, words):
    """An observation, a task of ``words`` tokens up to the most, then ten
    Euler steps of the expert from the same noise, within 5% of lerobot's
    largest action, as a decoder's logits are held.
    """
    weights = checkpoint(TEXT, names(TOY)).tree("random")
    model = Host(TOY, weights)
    reference = golden("toy", lambda: LeRobot(TOY, weights), TOY, weights)
    image, tokens, state, noise = observation(words)
    got = model.sample(image, tokens, state, noise)
    want = reference.logits(tokens, image, state, noise)
    err, scale = np.abs(got - want).max(), np.abs(want).max()
    assert err <= LOGITS * scale, f"|diff| {err:.4f} against |actions| {scale:.3f}"
