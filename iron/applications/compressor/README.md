<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lossless text compression on the NPU

A language model on the NPU predicts each next token of a text, and an
arithmetic coder on the host spends about `-log2 p` bits on the token that
came. Decompressing runs the same model over the tokens decoded so far.
The package docstring (`__init__.py`) says how blocks, the archive and the
model's logits fit together; `arithmetic.py` is the coder.

```bash
python -m iron.applications.compressor notes.txt      # writes notes.txt.irnz, prints bits per byte beside xz's
python -m iron.applications.compressor -d notes.txt.irnz
```

The model is Qwen3 0.6B (`--model` takes another Qwen3 checkpoint), at a
2048-token context (`--max-seq-len`). Any `iron.lm.CausalLM` with a
tokenizer works from Python: `compress(data, model)`, `decompress(blob, model)`.

## What decompressing needs

The same checkpoint at the same `max_seq_len`, and logits identical bit for
bit to the ones compressing saw. The host's part is integer arithmetic
after one IEEE multiply, so the same on every host; the NPU's logits are
the same run to run (`check_determinism` in every model's device test). An
archive written on one NPU generation, driver or mlir-aie build is not
known to decode on another. A wrong model or a logit that did not
reproduce fails the archive's CRC rather than returning other text.

The text must be UTF-8 that the tokenizer gives back from its tokens; text
naming a special token (`<|endoftext|>`) is refused.

## Sources

The coder follows Witten, Neal and Cleary, "Arithmetic coding for data
compression" (CACM, 1987); nothing is copied from another compressor. It
is lossless only, codes tokens with one pretrained model, and uses
arithmetic coding, not ANS. The approach is DeepMind's "Language Modeling
Is Compression" (ICLR 2024).

## Testing

`tests/applications/compressor/reference/` round-trips text through a toy
decoder on the host and pins the integer distribution;
`tests/applications/compressor/test.py` round-trips two blocks on the NPU
with Qwen3 0.6B's checkpoint and records bits per byte and speed.
