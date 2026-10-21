<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A BERT cross-encoder on the NPU

A reranker: `BertCrossEncoder.score(query, passage)` reads the pair as one
sequence, `[CLS] query [SEP] passage [SEP]`, and returns the relevance
logit of a `BertForSequenceClassification` checkpoint with one label. Its
layers normalize after each residual add (post-norm), and it adds learned
position and segment embeddings. `model.py` says what is its own. The
shared library is [AGENTS.md](../../../AGENTS.md#language-models).

Its LayerNorms run at the LayerNorm kernel's epsilon, 1e-5
(`iron.functional.LAYER_NORM_EPS`), not BERT's 1e-12.

## Scope: toy shape

The port is checked at a toy shape over drawn weights; no released
checkpoint is loaded. MHA takes heads of a whole 64, so a checkpoint whose
`hidden_size / num_attention_heads` is 32 (MiniLM's) does not run. A pair
longer than `max_tokens` is refused, not truncated.

## Testing

```bash
pytest tests/applications/bert/
```

`reference/` checks, device-free, every row's last hidden state and the
score of a short pair and of one that nearly fills `max_tokens` against
transformers' own, at a toy shape over random weights.
