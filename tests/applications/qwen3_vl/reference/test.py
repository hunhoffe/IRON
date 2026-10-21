#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL against transformers' own in float32 over the same random
weights, on an image transformers' own image processor cuts into patches,
recorded in ``goldens/``, device-free, at a toy shape: the vision tower,
an 8x8 patch grid, its position table resampled from 12x12; and the tower
and the text model as one ``Qwen3VL``, its soft tokens placed across a
chunk boundary.
"""

import numpy as np
import pytest

from iron.applications.qwen3_vl.model import Qwen3VL, Qwen3VLConfig
from iron.applications.qwen3_vl.vision import VisionConfig, VisionTower
from iron.lm.golden import HuggingFace, flat
from iron.lm.testing import LOGITS, assert_close
from iron.tests.support.llama_model import checkpoint, shapes
from iron.tests.support.llama_reference import prompt

pytestmark = pytest.mark.usefixtures("npu2")

VISION = VisionConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    num_position_embeddings=144,
    deepstack_visual_indexes=(0,),
    text_dim=256,
    image_size=128,
)
#: Its text model: Qwen3's layer, head_dim 64's 32 frequencies in three parts.
TEXT = Qwen3VLConfig(
    vocab_size=512,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=512,
    rope_theta=1e5,
    max_seq_len=1024,
    prefill_chunk=256,
    mrope_section=(12, 10, 10),
    image_token_id=511,
    vision=VISION,
)
IMAGE = TEXT.image_token_id


def vision_names(c: VisionConfig) -> dict[str, tuple[int, ...]]:
    """The vision tower's tensors of a checkpoint of ``c``'s shape, by name."""
    E, M, P = c.hidden_size, c.intermediate_size, c.patch_size
    W = E * c.spatial_merge_size**2
    v = "model.visual"
    out = {
        f"{v}.patch_embed.proj.weight": (E, 3, c.temporal_patch_size, P, P),
        f"{v}.patch_embed.proj.bias": (E,),
        f"{v}.pos_embed.weight": (c.num_position_embeddings, E),
    }
    for i in range(c.num_hidden_layers):
        p = f"{v}.blocks.{i}"
        for name, shape in (
            ("norm1", (E,)),
            ("norm2", (E,)),
            ("attn.qkv", (3 * E, E)),
            ("attn.proj", (E, E)),
            ("mlp.linear_fc1", (M, E)),
            ("mlp.linear_fc2", (E, M)),
        ):
            out |= {f"{p}.{name}.weight": shape, f"{p}.{name}.bias": shape[:1]}
    mergers = ["merger"] + [
        f"deepstack_merger_list.{j}" for j in range(len(c.deepstack_visual_indexes))
    ]
    for j, merger in enumerate(mergers):
        p = f"{v}.{merger}"
        norm = W if j else E
        out |= {
            f"{p}.norm.weight": (norm,),
            f"{p}.norm.bias": (norm,),
            f"{p}.linear_fc1.weight": (W, W),
            f"{p}.linear_fc1.bias": (W,),
            f"{p}.linear_fc2.weight": (c.text_dim, W),
            f"{p}.linear_fc2.bias": (c.text_dim,),
        }
    return out


def text_names(c: Qwen3VLConfig) -> dict[str, tuple[int, ...]]:
    """The text model's tensors of a checkpoint of ``c``'s shape, by name:
    Qwen3's under ``model.language_model``, the head tied.
    """
    out = {}
    for name, shape in shapes(c).items():
        out[name.replace("model.", "model.language_model.", 1)] = shape
    for i in range(c.num_hidden_layers):
        p = f"model.language_model.layers.{i}.self_attn"
        out |= {
            f"{p}.q_norm.weight": (c.head_dim,),
            f"{p}.k_norm.weight": (c.head_dim,),
        }
    return out


def patches(c: VisionConfig, image):
    """transformers' image processor's patches of ``image``, as
    ``VisionTower.inputs`` takes it, unresized.
    """
    import transformers

    processor = transformers.Qwen2VLImageProcessorPil(
        do_resize=False,
        do_rescale=False,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        patch_size=c.patch_size,
        temporal_patch_size=c.temporal_patch_size,
        merge_size=c.spatial_merge_size,
    )
    return processor(
        images=[(image + 1) / 2], return_tensors="pt", input_data_format="channels_last"
    )


class Visual:
    """transformers' vision tower on its own image processor's patches:
    its merger's soft tokens, then each DeepStack merger's, as ``logits``.
    """

    def __init__(self, c: VisionConfig, weights):
        import torch
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

        config = qwen3_vl.Qwen3VLVisionConfig(**c.to_json()["vision_config"])
        self.model = qwen3_vl.Qwen3VLVisionModel(config).eval()
        prefix = "model.visual."
        state = {k.removeprefix(prefix): torch.from_numpy(v) for k, v in flat(weights)}
        self.model.load_state_dict(state, strict=True)

    def logits(self, tokens, image):
        import torch

        out = patches(VISION, image)
        with torch.no_grad():
            out = self.model(out["pixel_values"], grid_thw=out["image_grid_thw"])
        return torch.cat([out.pooler_output, *out.deepstack_features]).numpy()


def test_the_vision_tower_s_soft_tokens_match_transformers(golden):
    """The tower's reference against transformers' soft tokens and
    DeepStack rows: bf16 against float32, LayerNorm at the kernel's 1e-5
    against 1e-6, so within 5% of the largest row's magnitude, as a
    decoder's logits are held.
    """
    names = vision_names(VISION)
    weights = checkpoint(VISION, names).tree("random")
    tower = VisionTower(VISION, weights, 0)
    image = np.random.default_rng(1).uniform(-1, 1, (128, 128, 3)).astype(np.float32)
    buffer, values = tower.inputs(image)
    soft = tower.reference(buffer, **values)
    got = np.concatenate([soft, tower.stacked.host]).astype(np.float32)
    reference = golden("soft_tokens", lambda: Visual(VISION, weights), VISION, weights)
    want = reference.logits([], image)
    assert got.shape == want.shape == (2 * VISION.tokens, VISION.text_dim)
    err, scale = np.abs(got - want).max(), np.abs(want).max()
    assert err <= LOGITS * scale, f"|diff| {err:.4f} against |rows| {scale:.3f}"


class HuggingFaceVL(HuggingFace):
    """transformers' own Qwen3-VL, its image processor's patches and the
    image's token types given, as its processor gives them.
    """

    model_class = "AutoModelForImageTextToText"

    def logits(self, tokens, image) -> np.ndarray:
        import torch

        pixels = patches(VISION, image)
        ids = torch.from_numpy(np.asarray(tokens, np.int64)[None])
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                pixel_values=pixels["pixel_values"],
                image_grid_thw=pixels["image_grid_thw"],
                mm_token_type_ids=(ids == IMAGE).int(),
                use_cache=False,
            )
        return out.logits[0, -1].numpy()


class RopeIndex(HuggingFaceVL):
    """transformers' three RoPE positions of each token, then the cosine and
    sine of each of its frequencies, as ``logits``.
    """

    def logits(self, tokens, image) -> np.ndarray:
        import torch

        ids = torch.from_numpy(np.asarray(tokens, np.int64)[None])
        images = int((ids == IMAGE).sum()) // VISION.tokens
        grid = patches(VISION, image)["image_grid_thw"].repeat(images, 1)
        positions, _ = self.model.model.get_rope_index(
            ids, (ids == IMAGE).int(), image_grid_thw=grid
        )
        rotary = self.model.model.language_model.rotary_emb
        cos, sin = rotary(torch.zeros(1), positions)
        D = cos.shape[-1] // 2
        rows = [positions[:, 0].T.float(), cos[0, :, :D], sin[0, :, :D]]
        return torch.cat(rows, dim=1).numpy()


def test_each_token_s_rope_positions_are_transformers(golden):
    """Two images, text before, between and after, take transformers' own
    three positions, exactly, and the positions past them count on; the
    host writes each position's angles, to within bfloat16.
    """
    weights = checkpoint(TEXT, text_names(TEXT) | vision_names(VISION)).tree("random")
    model = Qwen3VL(TEXT, weights)
    reference = golden(
        "rope_index",
        lambda: RopeIndex.build(
            "qwen3_vl", VISION, weights, **TEXT.to_json(), tie_word_embeddings=True
        ),
        TEXT,
        VISION,
        weights,
    )
    T = VISION.tokens
    tokens = np.r_[np.arange(5), [IMAGE] * T, np.arange(3), [IMAGE] * T, np.arange(7)]
    image = np.zeros((128, 128, 3), np.float32)
    got = model.text.rope_positions(tokens)
    want = reference.logits(tokens, image)
    np.testing.assert_array_equal(got[: tokens.size], want[:, :3])
    after = want[-1, 0] + 1 + np.arange(3)
    np.testing.assert_array_equal(
        got[tokens.size : tokens.size + 3], np.stack([after] * 3, 1)
    )
    model.text.place(tokens)
    angles = np.asarray(model.text.rotary_emb.table.host[: tokens.size], np.float32)
    D = TEXT.head_dim // 2
    np.testing.assert_allclose(angles[:, ::2], want[:, 3 : 3 + D], atol=4e-3)
    np.testing.assert_allclose(angles[:, 1::2], want[:, 3 + D :], atol=4e-3)


def test_the_image_s_soft_tokens_take_its_placeholders_across_a_chunk(golden):
    """Each version's reference in turn, its states carried from one to the
    next: the tower, then each chunk, both holding some of the run of
    placeholders that crosses their boundary, then a decode step, against
    transformers' logits.
    """
    weights = checkpoint(TEXT, text_names(TEXT) | vision_names(VISION)).tree("random")
    model = Qwen3VL(TEXT, weights)
    reference = golden(
        "small",
        # transformers reads the tie from the top level.
        lambda: HuggingFaceVL.build(
            "qwen3_vl", VISION, weights, **TEXT.to_json(), tie_word_embeddings=True
        ),
        TEXT,
        VISION,
        weights,
    )
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, IMAGE, 300)
    tokens[250 : 250 + VISION.tokens] = IMAGE
    image = rng.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
    calls, towers = model.tower_calls(tokens, {IMAGE: (image,)})
    for inputs in calls:
        model.reference(
            *inputs, token=0, position=0, start=0, rows=1, rope_start=0, **towers
        )
    model.text.place(tokens)
    logits, carry = prompt(model, tokens, **towers)
    assert_close(logits, reference.logits(tokens, image))
    tokens = np.append(tokens, carry["token"])
    logits, _ = model.reference(start=0, rows=1, rope_start=0, **carry, **towers)
    assert_close(logits, reference.logits(tokens, image))
