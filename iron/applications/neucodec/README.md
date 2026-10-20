<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeuCodec's decoder on the NPU

NeuCodec's decoder turns NeuTTS's speech codes, 50 a second, into 24 kHz
audio. `model.py`'s docstrings describe it (`NeuCodecDecoder`): the codes
and the inverse STFT run on the host, everything between them on the NPU.
The shared library is [AGENTS.md](../../../AGENTS.md#language-models).

## Status

The decoder runs at a toy shape on drawn weights
(`tests/applications/neucodec/test.py`): device-free, the graph's reference
against neucodec's own modules' audio recorded in
`tests/applications/neucodec/goldens/`, and the same check on the NPU.

Two gaps stand before the real checkpoint:

- A clip is 256 frames (5.12 s), one compile. GroupNorm normalizes a
  group's 32 channels over every frame as one LayerNorm row, at most 8192
  elements, and the GEMMs take frames in whole 256s. Other lengths need a
  GroupNorm whose statistics are bounded per call.
- The checkpoint is `pytorch_model.bin`, a torch pickle, which IRON does
  not read.
