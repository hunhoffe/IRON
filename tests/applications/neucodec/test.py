#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeuCodec's decoder at a toy shape, on drawn weights (no checkpoint),
against neucodec's own modules in float32 over the same weights, the audio
recorded in ``goldens/``: device-free, the graph's reference and the host's
inverse STFT; and the same on the NPU.
"""

import importlib.util
import sys
import types
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.applications.neucodec.model import NeuCodecConfig, NeuCodecDecoder
from iron.lm.checkpoint import leaves
from iron.tests.support.llama_model import checkpoint

TOY = NeuCodecConfig(
    hidden_size=256, num_hidden_layers=2, num_attention_heads=4, hop_length=64
)
FRAMES = 256


def names(c: NeuCodecConfig) -> dict[str, tuple[int, ...]]:
    """Each tensor NeuCodec's decoder reads at ``c``'s shape, by name."""
    E, V = c.hidden_size, 2048
    out = {
        "generator.quantizer.project_out.weight": (V, 8),
        "generator.quantizer.project_out.bias": (V,),
        "fc_post_a.weight": (E, V),
        "fc_post_a.bias": (E,),
        "generator.backbone.embed.weight": (E, E, 7),
        "generator.backbone.embed.bias": (E,),
        "generator.backbone.final_layer_norm.weight": (E,),
        "generator.backbone.final_layer_norm.bias": (E,),
        "generator.head.out.weight": (c.n_fft + 2, E),
        "generator.head.out.bias": (c.n_fft + 2,),
    }
    for net in ("prior_net", "post_net"):
        for i in range(2):
            p = f"generator.backbone.{net}.{i}"
            for n in (1, 2):
                out |= {
                    f"{p}.norm{n}.weight": (E,),
                    f"{p}.norm{n}.bias": (E,),
                    f"{p}.conv{n}.weight": (E, E, 3),
                    f"{p}.conv{n}.bias": (E,),
                }
    for i in range(c.num_hidden_layers):
        p = f"generator.backbone.transformers.{i}"
        out |= {
            f"{p}.att_norm.weight": (E,),
            f"{p}.ffn_norm.weight": (E,),
            f"{p}.att.c_attn.weight": (3 * E, E),
            f"{p}.att.c_proj.weight": (E, E),
            f"{p}.mlp.fc1.weight": (4 * E, E),
            f"{p}.mlp.fc2.weight": (E, 4 * E),
        }
    return out


class NeuCodec:
    """neucodec's decoder modules over the toy's weights, what records
    ``goldens/toy.npz``: the audio of a clip's codes.
    """

    def __init__(self, weights):
        import torch

        # Its decoder's modules alone: the package imports its encoder's
        # dependencies, and torchtune's its quantization's.
        root = Path(importlib.util.find_spec("neucodec").origin).parent
        sys.modules["neucodec"] = types.ModuleType("neucodec")
        sys.modules["neucodec"].__path__ = [str(root)]
        tune = Path(importlib.util.find_spec("torchtune").origin).parent
        rope = tune / "modules" / "position_embeddings.py"
        modules = types.ModuleType("torchtune.modules")
        exec(compile(rope.read_text(), str(rope), "exec"), modules.__dict__)
        sys.modules["torchtune.modules"] = modules
        from neucodec.codec_decoder_vocos import CodecDecoderVocos

        c = TOY
        self.generator = CodecDecoderVocos(
            hidden_dim=c.hidden_size,
            depth=c.num_hidden_layers,
            heads=c.num_attention_heads,
            hop_length=c.hop_length,
        ).eval()
        self.fc_post_a = torch.nn.Linear(2048, c.hidden_size)
        state = {
            n: torch.from_numpy(np.asarray(w, np.float32)) for n, w in leaves(weights)
        }
        own = {
            n[len("generator.") :]: w
            for n, w in state.items()
            if n.startswith("generator.")
        }
        missing = self.generator.load_state_dict(own, strict=False).missing_keys
        # Its encoder's projection, and the window it builds itself.
        assert len(missing) == 3 and "head.istft.window" in missing, missing
        self.fc_post_a.load_state_dict(
            {"weight": state["fc_post_a.weight"], "bias": state["fc_post_a.bias"]}
        )

    def logits(self, codes):
        import torch

        codes = torch.from_numpy(np.asarray(codes))[None, :, None]
        with torch.no_grad():
            x = self.generator.quantizer.get_output_from_indices(codes)
            x = self.fc_post_a(x)
            audio, _ = self.generator(x, vq=False)
        return audio[0, 0].numpy()


@pytest.fixture(scope="module")
def clip():
    """The toy's weights and a clip's codes."""
    weights = checkpoint(TOY, names(TOY)).tree("random")
    codes = np.random.default_rng(1).integers(0, 4**8, FRAMES)
    return weights, codes


@pytest.fixture(scope="module")
def reference(clip, golden):
    weights, _ = clip
    return golden("toy", lambda: NeuCodec(weights), TOY, weights)


def test_each_operator_resolves(clip, npu2):
    weights, _ = clip
    decoder = NeuCodecDecoder(TOY, weights, FRAMES)
    for op in decoder.record(x=((FRAMES + 6, TOY.hidden_size), bfloat16)).operators:
        op.resolved(npu2)


@pytest.mark.usefixtures("npu2")
def test_the_decoder_s_audio_is_neucodec_s(clip, reference):
    weights, codes = clip
    decoder = NeuCodecDecoder(TOY, weights, FRAMES)
    got = decoder.audio(decoder.reference(decoder.inputs(codes)))
    want = reference.logits(codes)
    assert got.shape == want.shape == (FRAMES * TOY.hop_length,)
    # bf16 against float32: measured 1.4% of the loudest sample.
    err, scale = np.abs(got - want).max(), np.abs(want).max()
    assert err <= 0.03 * scale, f"|diff| {err:.4f} against {scale:.3f}"


@pytest.mark.supported_devices("npu2")
def test_the_decoder_on_the_device(clip, reference):
    weights, codes = clip
    decoder = NeuCodecDecoder(TOY, weights, FRAMES)
    x = decoder.inputs(codes)
    decoder.compile(x=(x.shape, bfloat16))
    got = decoder.audio(decoder(aie_utils.DEFAULT_TENSOR_CLASS(x)).numpy())
    want = reference.logits(codes)
    err, scale = np.abs(got - want).max(), np.abs(want).max()
    assert err <= 0.03 * scale, f"|diff| {err:.4f} against {scale:.3f}"


def test_audio_caps_each_magnitude_at_100(clip):
    """NeuCodec's own clip: a log-magnitude past ln 100 sounds as 100."""
    decoder = NeuCodecDecoder(TOY, clip[0], FRAMES)
    half = TOY.n_fft // 2 + 1
    # Drawn phases: at phase 0 each frame is an impulse the window zeroes.
    phases = np.random.default_rng(2).uniform(-np.pi, np.pi, (FRAMES, half))
    spectra = np.concatenate([np.zeros_like(phases), phases], axis=1)
    capped = spectra.copy()
    spectra[:, :half], capped[:, :half] = 10, np.log(100)
    np.testing.assert_array_equal(decoder.audio(spectra), decoder.audio(capped))
