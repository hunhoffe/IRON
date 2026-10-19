# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL's vision tower on the NPU: Hugging Face's ``Qwen3VLVisionModel``
over one square image: its patch merger's soft tokens, and each DeepStack
merger's rows in ``stacked``; its shape ``VisionConfig``.

The host resizes and normalizes the image and cuts it into patches in
merge-block order, each 2x2 block of patches a soft token's four rows in
turn, as the processor does. A still image repeats its one frame across
the patch embedding's two, so the two temporal halves of its kernel are
summed into one linear map. The patch grid is fixed, so the learned
position embedding's bilinear resample and the 2D RoPE angles, a patch's
row on half of each head's frequencies and its column on the other half,
are tables built once.
"""

import dataclasses

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron.lm import (
    Attention,
    Call,
    Config,
    EncoderLayer,
    LayerNorm,
    Linear,
    Tower,
    not_json,
)


@dataclasses.dataclass(frozen=True)
class VisionConfig(Config):
    """The vision tower's shape, from the checkpoint's ``vision_config``.
    Its norms take the kernel's LayerNorm eps, not the 1e-6 transformers'
    uses.
    """

    # Every projection biased (``Attention``, ``MLP``).
    attention_bias = mlp_bias = True
    json_prefix = "vision_config."

    hidden_size: int = 1024
    num_hidden_layers: int = dataclasses.field(default=24, metadata={"json": "depth"})
    num_attention_heads: int = dataclasses.field(
        default=16, metadata={"json": "num_heads"}
    )
    intermediate_size: int = 4096
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = dataclasses.field(
        default=(5, 11, 17), metadata={"convert": tuple}
    )
    text_dim: int = dataclasses.field(
        default=2048, metadata={"json": "out_hidden_size"}
    )
    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = not_json(10000.0)
    # The side every image is resized to, in pixels.
    image_size: int = not_json(448)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    @property
    def side(self) -> int:
        """The patches along each side of the image."""
        return self.image_size // self.patch_size

    @property
    def tokens(self) -> int:
        """The soft tokens an image gives."""
        return self.side**2 // self.spatial_merge_size**2


class VisionAttention(Attention):
    """``Attention`` with q, k and v the rows of one biased ``qkv``."""

    def projections(self, c: VisionConfig) -> None:
        self.qkv = Linear(c.hidden_size, 3 * c.hidden_size, precision=c.precision)

    def project(self, x, **gemv) -> list:
        E = self.heads * self.head_dim
        return self.qkv.split(x, E, E, E, **gemv)


class VisionLayer(EncoderLayer):
    """SigLIP's layer on a fused ``qkv``, q and k turned by the 2D RoPE."""

    def __init__(self, c: VisionConfig, i: int):
        super().__init__(c, i, VisionAttention(c, i, o_bias=True))
        self.rope = c.head_dim


class Merger(iron.Graph):
    """Each 2x2 block of patch rows as one row, normed before the block
    (or after, ``postshuffle``), through ``linear_fc1``, the exact GELU and
    ``linear_fc2`` into the text model's space.
    """

    def __init__(self, c: VisionConfig, postshuffle: bool):
        W = self.width = c.hidden_size * c.spatial_merge_size**2
        self.postshuffle = postshuffle
        self.norm = LayerNorm(W if postshuffle else c.hidden_size)
        self.linear_fc1 = Linear(W, W, precision=c.precision)
        self.linear_fc2 = Linear(W, c.text_dim, precision=c.precision)

    def body(self, x):
        rows = x.shape[0] * x.shape[1] // self.width
        if self.postshuffle:
            x = self.norm(x.reshape(rows, self.width))
        else:
            x = self.norm(x).reshape(rows, self.width)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionTower(Tower):
    """The vision tower over one ``image_size`` square image: its patches
    in, the patch merger's ``tokens`` soft tokens in the text model's space
    out. Each DeepStack merger, of a layer ``deepstack_visual_indexes``
    names, writes its ``tokens`` rows to ``stacked`` in turn after
    ``chunk`` rows no call writes, which a text row's gather reads as zero.

    Args:
        config: The tower's shape.
        weights: The checkpoint's tree; ``model.visual`` is read.
        chunk: The rows before the DeepStack rows: the text model's
            prompt chunk.
    """

    layer_class = VisionLayer
    renames = {
        "attn.proj": "self_attn.o_proj",
        "attn": "self_attn",
        "norm1": "input_layernorm",
        "norm2": "post_attention_layernorm",
        "linear_fc1": "fc1",
        "linear_fc2": "fc2",
    }

    def __init__(self, config: VisionConfig, weights, chunk: int):
        c = config
        N, E, D = c.side**2, c.hidden_size, c.head_dim
        stacks = len(c.deepstack_visual_indexes)
        super().__init__(c, N, c.tokens)
        visual = weights.model.visual
        self.load_layers(weights, "model.visual.blocks.")
        self.merger = Merger(c, postshuffle=False)
        self.merger.load_state_dict(weights, path="model.visual.merger.")
        self.deepstack = [Merger(c, postshuffle=True) for _ in range(stacks)]
        for j, merger in enumerate(self.deepstack):
            merger.load_state_dict(
                weights, path=f"model.visual.deepstack_merger_list.{j}."
            )
        proj = visual.patch_embed.proj
        patch = np.asarray(proj.weight, np.float32).sum(2).reshape(E, -1)
        self.patch = iron.weight(patch.astype(bfloat16))
        # Each patch's row and column, in merge-block order.
        m, blocks = c.spatial_merge_size, c.side // c.spatial_merge_size
        grid = np.indices((blocks, blocks, m, m))
        row, col = (grid[0] * m + grid[2]).ravel(), (grid[1] * m + grid[3]).ravel()
        # The learned table's bilinear resample to the grid, corners aligned,
        # plus the patch embedding's bias.
        table = np.asarray(visual.pos_embed.weight, np.float32)
        S = int(np.sqrt(c.num_position_embeddings))
        positions = np.asarray(proj.bias, np.float32) + sum(
            (wy * wx)[:, None] * table[y * S + x]
            for y, wy in self.taps(row, S)
            for x, wx in self.taps(col, S)
        )
        self.position_rows = iron.weight(positions.astype(bfloat16))
        inv = 1 / c.rope_theta ** (np.arange(0, D // 2, 2) / (D // 2))
        freqs = np.concatenate([np.outer(row, inv), np.outer(col, inv)], axis=1)
        angles = np.empty((N, D), np.float32)
        angles[:, ::2], angles[:, 1::2] = np.cos(freqs), np.sin(freqs)
        self.rope = iron.weight(angles.astype(bfloat16))
        self.chunk = chunk
        self.stacked = iron.state((chunk + stacks * c.tokens, c.text_dim))

    def taps(self, index, S: int):
        """Each of the two table rows a bilinear resample with corners
        aligned reads for ``index`` along a side of ``S``, with its weights.
        """
        src = index * (S - 1) / max(self.config.side - 1, 1)
        low = np.floor(src).astype(int)
        frac = src - low
        return (low, 1 - frac), (np.minimum(low + 1, S - 1), frac)

    def body(self, patches):
        """The soft-token rows of ``patches``, ``(side ** 2, 3 * patch_size
        ** 2)``.
        """
        c, T = self.config, self.config.tokens
        h = F.linear(patches, self.patch) + self.position_rows
        call = Call(
            prompt=True,
            start=0,
            rows=self.max_tokens,
            position=None,
            ropes={c.head_dim: (self.rope,) * 2},
            embeddings=h,
        )
        for i, layer in enumerate(self.layers):
            h = layer(h, call)
            if i in c.deepstack_visual_indexes:
                j = c.deepstack_visual_indexes.index(i)
                at = self.chunk + j * T
                self.stacked[at : at + T] = self.deepstack[j](h)
        return self.merger(h)

    def soft(self, values: dict) -> int:
        return self.config.tokens

    def shapes(self) -> list[dict]:
        c = self.config
        return [dict(patches=((c.side**2, 3 * c.patch_size**2), bfloat16))]

    def longest(self) -> dict:
        return {}

    def inputs(self, image) -> tuple[np.ndarray, dict]:
        """A call's patches of ``image``, ``(image_size, image_size, 3)``
        float32 in ``[-1, 1]`` (the processor's mean and std of 0.5 both
        ways): each patch's channels, rows and columns in merge-block order.
        """
        c = self.config
        P, m, blocks = (
            c.patch_size,
            c.spatial_merge_size,
            c.side // c.spatial_merge_size,
        )
        x = np.asarray(image).reshape(blocks, m, P, blocks, m, P, 3)
        x = x.transpose(0, 3, 1, 4, 6, 2, 5).reshape(c.side**2, 3 * P * P)
        return x.astype(bfloat16), {}
