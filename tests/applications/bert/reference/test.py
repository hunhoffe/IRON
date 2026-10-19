#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A BERT cross-encoder against transformers' own
``BertForSequenceClassification`` in float32 over the same random weights,
its logits recorded in ``goldens/``, device-free, at a toy shape.
"""

import types
import zlib

import numpy as np
import pytest

from iron.applications.bert.model import SEP, BertConfig, BertCrossEncoder
from iron.lm.golden import HuggingFace
from iron.lm.testing import LOGITS
from iron.tests.support.llama_model import checkpoint

pytestmark = pytest.mark.usefixtures("npu2")

TOY = BertConfig(
    vocab_size=512,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=256,
    max_tokens=256,
)


def names(c: BertConfig) -> dict[str, tuple[int, ...]]:
    """Each tensor of a ``BertForSequenceClassification`` checkpoint of
    ``c``'s shape, one label, by name.
    """
    E, M = c.hidden_size, c.intermediate_size
    e = "bert.embeddings"
    out = {
        f"{e}.word_embeddings.weight": (c.vocab_size, E),
        f"{e}.position_embeddings.weight": (c.max_position_embeddings, E),
        f"{e}.token_type_embeddings.weight": (c.type_vocab_size, E),
        f"{e}.LayerNorm.weight": (E,),
        f"{e}.LayerNorm.bias": (E,),
        "bert.pooler.dense.weight": (E, E),
        "bert.pooler.dense.bias": (E,),
        "classifier.weight": (1, E),
        "classifier.bias": (1,),
    }
    for i in range(c.num_hidden_layers):
        p = f"bert.encoder.layer.{i}"
        for name, shape in (
            ("attention.self.query", (E, E)),
            ("attention.self.key", (E, E)),
            ("attention.self.value", (E, E)),
            ("attention.output.dense", (E, E)),
            ("attention.output.LayerNorm", (E,)),
            ("intermediate.dense", (M, E)),
            ("output.dense", (E, M)),
            ("output.LayerNorm", (E,)),
        ):
            out |= {f"{p}.{name}.weight": shape, f"{p}.{name}.bias": shape[:1]}
    return out


class Classifier(HuggingFace):
    """transformers' own sequence classifier; ``logits`` takes each token's
    segment.
    """

    model_class = "AutoModelForSequenceClassification"

    def logits(self, tokens, segments) -> np.ndarray:
        import torch

        ids = torch.from_numpy(np.asarray(tokens, np.int64)[None])
        kinds = torch.from_numpy(np.asarray(segments, np.int64)[None])
        with torch.no_grad():
            return self.model(input_ids=ids, token_type_ids=kinds).logits[0].numpy()


class Host(BertCrossEncoder):
    """The cross-encoder, each call the graph's reference."""

    def __call__(self, *tensors, **values):
        out = np.asarray(self.reference(*tensors, **values))
        return types.SimpleNamespace(numpy=lambda: out)


class Words:
    """A word a token, by its checksum, past BERT's special tokens."""

    def encode(self, text: str) -> list[int]:
        return [200 + zlib.crc32(w.encode()) % 300 for w in text.split()]


@pytest.mark.parametrize("words", [(3, 9), (40, 200)])
def test_a_pair_s_score_is_transformers(golden, words):
    """A query and a passage, the longer nearly filling ``max_tokens``,
    scored as transformers scores them: within 5% of the logit's magnitude,
    or of 1 where it is smaller, as a decoder's logits are held.
    """
    weights = checkpoint(TOY, names(TOY)).tree("random")
    model = Host(TOY, weights)
    reference = golden(
        "toy",
        lambda: Classifier.build("bert", TOY, weights, num_labels=1),
        TOY,
        weights,
    )
    model.tokenizer = types.SimpleNamespace(encoding=Words())
    rng = np.random.default_rng(words)
    query, passage = (" ".join(map(str, rng.integers(0, 10**6, n))) for n in words)
    got = model.score(query, passage)
    q, p = Words().encode(query), Words().encode(passage)
    tokens = [model.bos, *q, SEP, *p, SEP]
    segments = [0] * (len(q) + 2) + [1] * (len(p) + 1)
    [want] = reference.logits(tokens, segments)
    assert abs(got - want) <= LOGITS * max(abs(want), 1.0), (got, want)
