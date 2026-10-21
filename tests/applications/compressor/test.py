#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compressor on the NPU over Qwen3 0.6B's checkpoint: a text of two
blocks comes back byte for byte, in fewer bits than xz spends, with its
bits per byte and speed recorded.
"""

import lzma
import time

import aie.utils as aie_utils
import pytest

from iron.applications.compressor import compress, decompress
from iron.applications.qwen3 import Qwen3
from iron.lm import prompt_text
from iron.lm.testing import require, weights_dir

WEIGHTS = weights_dir("qwen3-0.6b") / "model.safetensors"

pytestmark = pytest.mark.supported_devices("npu2")


@pytest.fixture(scope="module")
def model():
    require(WEIGHTS, WEIGHTS.with_name("tokenizer.json"), staged=False)
    yield Qwen3.load(WEIGHTS.parent, max_seq_len=2048)
    if aie_utils.DefaultNPURuntime is not None:
        aie_utils.DefaultNPURuntime.cleanup()


@pytest.mark.bench
def test_a_text_comes_back_in_fewer_bits_than_xz_spends(model, record_property):
    data = prompt_text(12000).encode()
    began = time.perf_counter()
    blob = compress(data, model)
    record_property("Compress KB/s", len(data) / (time.perf_counter() - began) / 1e3)
    began = time.perf_counter()
    assert decompress(blob, model) == data
    record_property("Decompress KB/s", len(data) / (time.perf_counter() - began) / 1e3)
    record_property("Bits per byte", 8 * len(blob) / len(data))
    assert len(blob) < len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))
