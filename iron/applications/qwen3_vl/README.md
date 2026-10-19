<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-VL on the NPU

A vision-language model: Qwen3's text layers and a vision tower as one
`iron.lm.Multimodal`, `Qwen3VL`, whose image placeholders take the tower's
soft tokens. Each position's three-part RoPE row (MRoPE) is written by the
host into a table the decoder reads (`Qwen3VLText.place`), and the tower's
DeepStack rows are added to the image's rows after the first decoder
layers. `model.py` and `vision.py` say what is their own. The shared
library is [AGENTS.md](../../../AGENTS.md#language-models).

## Scope: one square image, toy shape

The port takes one square image, resized and normalized on the host, and
is checked at a toy shape over drawn weights; the released checkpoint has
not run yet.

## Testing

```bash
pytest tests/applications/qwen3_vl/
```

`reference/` checks, device-free, the tower's soft tokens and DeepStack
rows, each token's RoPE positions, and the logits with an image's
placeholders across a chunk boundary, against transformers' own at a toy
shape over random weights.
