#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compressor device-free: the coder against its ideal length, the
integer distribution pinned, and round trips through a toy decoder whose
graph runs on the host, its tokens a text's bytes.
"""

import dataclasses
import hashlib
import math

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iron.applications.compressor import compress, decompress
from iron.applications.compressor.arithmetic import EXP2, Decoder, Encoder, cdf
from iron.applications.llama import Llama
from iron.lm import OnHost, Tokenizer
from iron.tests.support.llama_model import PROFILE, SMALL, checkpoint

pytestmark = pytest.mark.usefixtures("npu2")


class Bytes:
    """A text's UTF-8 bytes as its token ids."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")


class Toy(OnHost, Llama):
    profile = PROFILE
    bos = 256


def toy(seed: int, max_seq_len: int = 64) -> Toy:
    """Llama's small shape on the host, blocks of ``max_seq_len`` rows."""
    config = dataclasses.replace(SMALL, max_seq_len=max_seq_len, prefill_chunk=64)
    model = Toy(config, checkpoint(config).tree("random", seed))
    model.tokenizer = Tokenizer(Bytes(), bos=Toy.bos)
    return model


TEXT = "Lossless, on the NPU: naïve café ☕ — every byte comes back. " * 3


@pytest.fixture(scope="module")
def model():
    return toy(seed=0)


def test_a_text_of_several_blocks_comes_back(model):
    data = TEXT.encode()
    assert len(data) > 2 * model.config.max_seq_len
    blob = compress(data, model)
    assert decompress(blob, model) == data


def test_an_empty_text_comes_back(model):
    assert decompress(compress(b"", model), model) == b""


def test_bytes_that_are_not_utf8_are_refused(model):
    with pytest.raises(UnicodeDecodeError):
        compress(b"\xff\xfe", model)


def test_another_context_is_refused(model):
    blob = compress(b"hello", model)
    with pytest.raises(ValueError, match="max_seq_len=64"):
        decompress(blob, toy(seed=0, max_seq_len=128))


def test_another_model_fails_the_crc(model):
    blob = compress(TEXT[:40].encode(), model)
    with pytest.raises(ValueError, match="CRC"):
        decompress(blob, toy(seed=1))


def test_the_coder_spends_within_a_few_bytes_of_the_entropy():
    rng = np.random.default_rng(0)
    encoder, cdfs, symbols, ideal = Encoder(), [], [], 0.0
    for _ in range(2000):
        c = cdf(rng.normal(0, 3, 1000))
        p = np.diff(c) / c[-1]
        s = int(rng.choice(p.size, p=p))
        encoder.encode(s, c)
        cdfs.append(c)
        symbols.append(s)
        ideal -= math.log2(p[s])
    blob = encoder.finish()
    decoder = Decoder(blob)
    assert [decoder.decode(c) for c in cdfs] == symbols
    assert 8 * len(blob) - ideal < 16


def test_a_certain_token_costs_almost_nothing_and_the_rest_stay_codable():
    logits = np.full(151936, -100.0)
    logits[7] = 100.0
    c = cdf(logits)
    freq = np.diff(c)
    assert freq.min() == 1
    assert -math.log2(freq[7] / c[-1]) < 1e-4
    encoder = Encoder()
    for s in (7, 0, 151935):
        encoder.encode(s, c)
    decoder = Decoder(encoder.finish())
    assert [decoder.decode(c) for _ in range(3)] == [7, 0, 151935]


def test_the_distribution_is_pinned_and_follows_the_logits_values_not_their_dtype():
    # Changing either breaks every archive written before.
    assert hashlib.sha256(EXP2.tobytes()).hexdigest()[:16] == "e65fbff51100d27e"
    logits = (np.arange(-40, 40) * 0.37).astype(bfloat16)
    c = cdf(logits)
    assert int(c[-1]) == 13771476069
    assert np.array_equal(c, cdf(logits.astype(np.float32)))


def test_logits_that_are_not_finite_are_refused():
    with pytest.raises(ValueError, match="finite"):
        cdf(np.array([0.0, np.inf]))
