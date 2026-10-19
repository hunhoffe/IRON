#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A BERT cross-encoder over ``iron.lm``: a query and a passage in one
sequence, scored by ``BertForSequenceClassification``'s head, judged by
Hugging Face's own model in float32.

Each row is its token's embedding plus its position's and its segment's,
then a LayerNorm. Each layer normalizes after its residual add rather than
before (post-norm): the attention's sum through ``input_layernorm``, the
feed-forward's through ``post_attention_layernorm``. The head reads the
first row, ``[CLS]``: ``tanh`` of the pooler's projection, then the
classifier. Every LayerNorm runs at the kernel's epsilon, 1e-5
(``iron.functional.LAYER_NORM_EPS``), not BERT's 1e-12.
"""

import dataclasses

import numpy as np

import iron
from iron import Scratchpad
from iron.lm import Config, Encoder, EncoderLayer, LayerNorm, Linear, not_json
from iron.operators.copy import Copy
from iron.operators.tanh import Tanh

#: BERT's paths under the names ``iron.lm``'s modules read, the longer
#: first: ``renamed`` moves a path by the first that matches.
RENAMES = {
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.o_proj",
    "attention.output.LayerNorm": "input_layernorm",
    "intermediate.dense": "mlp.fc1",
    "output.dense": "mlp.fc2",
    "output.LayerNorm": "post_attention_layernorm",
}


@dataclasses.dataclass(frozen=True)
class BertConfig(Config):
    """BERT's shape, as its ``config.json`` names it; a call takes at most
    ``max_tokens`` rows, a whole number of 64.
    """

    # Every projection biased (``Attention``, ``MLP``).
    attention_bias = mlp_bias = True

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_act: str = "gelu"
    max_tokens: int = not_json(512)

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class BertLayer(EncoderLayer):
    """``EncoderLayer`` normalizing after each residual add."""

    def body(self, x, call):
        x = self.input_layernorm(x + self.mix(x, call))
        return self.post_attention_layernorm(x + self.mlp(x))


class BertCrossEncoder(Encoder):
    """BERT scoring a passage against a query (``score``), on ``config``'s
    shape and a ``BertForSequenceClassification`` checkpoint's ``weights``.

    A call takes ``ids``, each row's token, then its segment, ``(2,
    max_tokens)``, and ``rows``, the real rows. It returns the classifier's
    logits, zero past ``labels``.
    """

    layer_class = BertLayer
    renames = RENAMES

    def __init__(self, config: BertConfig, weights):
        c, e = config, weights.bert.embeddings
        super().__init__(c, c.max_tokens)
        self.config, self.weights = c, weights
        self.load_layers(weights, "bert.encoder.layer.")
        self.word_embeddings = iron.param(e.word_embeddings.weight)
        self.token_type_embeddings = iron.param(e.token_type_embeddings.weight)
        self.position_embeddings = iron.param(e.position_embeddings.weight)
        self.embeddings_norm = LayerNorm(c.hidden_size)
        self.embeddings_norm.load_state_dict(e.LayerNorm)
        E = c.hidden_size
        self.pooler = Linear(E, E, precision=c.precision)
        self.pooler.load_state_dict(weights.bert.pooler.dense)
        self.labels = weights.classifier.weight.shape[0]
        # Its rows padded to a whole 64, a GEMV's.
        rows = ((0, -self.labels % 64),)
        self.classifier = Linear(E, self.labels + rows[0][1], precision=c.precision)
        self.classifier.weight = iron.param(
            np.pad(np.asarray(weights.classifier.weight), rows + ((0, 0),))
        )
        self.classifier.bias = iron.param(
            np.pad(np.asarray(weights.classifier.bias), rows)
        )

    def body(self, ids, *, rows: Scratchpad[np.int32]):
        words = Copy.step(iron.weight(self.word_embeddings)[ids[0][:rows]])
        segments = Copy.step(iron.weight(self.token_type_embeddings)[ids[1][:rows]])
        x = words + segments + iron.weight(self.position_embeddings)[:rows]
        return self.forward(self.embeddings_norm(x), rows, None)

    def pool(self, x, rows):
        """``[CLS]``'s row."""
        return Copy.step(x[0]).reshape(1, self.config.hidden_size)

    def head(self, x):
        return self.classifier(Tanh.step(self.pooler(x)))

    def shapes(self) -> list[dict]:
        return [dict(ids=((2, self.max_tokens), np.int32))]

    def inputs(self, tokens, segments=()) -> tuple[np.ndarray, dict]:
        """A call's ``ids``: ``tokens`` and their ``segments`` (0 for
        each one left out), each padded to ``max_tokens``. Tokens past
        ``max_tokens`` are refused (``Encoder.inputs``), where transformers'
        tokenizer would truncate them.
        """
        ids, values = super().inputs(tokens)
        types = np.zeros_like(ids)
        types[: len(segments)] = segments
        return np.stack([ids, types]), values

    def logits(self, tokens, segments) -> np.ndarray:
        """The classifier's logits over ``tokens`` in ``segments``."""
        ids, values = self.inputs(tokens, segments)
        out = np.asarray(self(ids, **values).numpy(), np.float32)
        return out.reshape(-1)[: self.labels]

    def score(self, query: str, passage: str) -> float:
        """The relevance logit of ``passage`` to ``query``, as
        ``[CLS] query [SEP] passage [SEP]``, the passage's segment 1, each
        special token's id the tokenizer's.
        """
        encode = self.tokenizer.encoding.encode
        [cls], [sep] = encode("[CLS]"), encode("[SEP]")
        q, p = encode(query), encode(passage)
        tokens = [cls, *q, sep, *p, sep]
        segments = [0] * (len(q) + 2) + [1] * (len(p) + 1)
        return float(self.logits(tokens, segments)[0])
