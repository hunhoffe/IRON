# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless text compression with a language model on the NPU: the model
predicts each next token, and an arithmetic coder spends about
``-log2 p`` bits on the token that came. Decompression runs the same
model over the tokens decoded so far, so it needs the same model, loaded
with the same ``max_seq_len``.

    >>> from iron.applications.compressor import compress, decompress
    >>> from iron.applications.qwen3 import Qwen3
    >>> model = Qwen3.load("Qwen/Qwen3-0.6B", max_seq_len=2048)
    >>> blob = compress(data, model)
    >>> decompress(blob, model) == data
    True

The text is tokenized once, then coded in blocks of ``max_seq_len`` rows,
the first holding the model's BOS, each starting from empty caches. Every
logit is a decode step's, compressing and decompressing alike, and the
coder reads them only through ``arithmetic.cdf``, which is integer
arithmetic after one IEEE multiply: the NPU's logits are the only thing
the two sides must reproduce.

An archive is ``MAGIC``, then ``max_seq_len``, the token count and the
CRC-32 of the text, each 4 bytes little-endian, then the coded bits.
"""

import struct
import zlib

from .arithmetic import Decoder, Encoder, cdf

MAGIC = b"IRNZ"
HEADER = struct.Struct("<4sIII")


def compress(data: bytes, model) -> bytes:
    """``data``, UTF-8 text, coded with ``model``'s predictions.

    Args:
        data: The text's bytes.
        model: A loaded ``iron.lm.CausalLM`` with a tokenizer.

    Raises:
        ValueError: ``data`` is not UTF-8, or the tokenizer does not give it
            back from its tokens (it names a special token).
    """
    text = data.decode("utf-8")
    tokens = model.tokenizer(text)[1:]
    if model.tokenizer.decode(tokens) != text:
        raise ValueError("the tokenizer does not round-trip this text")
    L = model.config.max_seq_len
    encoder = Encoder()
    for start in range(0, len(tokens), L - 1):
        history = _restart(model)
        for token in tokens[start : start + L - 1]:
            encoder.encode(token, cdf(model.logits(history)))
            history.append(token)
    header = HEADER.pack(MAGIC, L, len(tokens), zlib.crc32(data))
    return header + encoder.finish()


def decompress(blob: bytes, model) -> bytes:
    """The bytes ``compress`` coded into ``blob`` with ``model``.

    Raises:
        ValueError: ``blob`` is not an archive; ``model`` was loaded with
            another ``max_seq_len``; or the text decoded fails its CRC (a
            different model, or logits that did not reproduce).
    """
    magic, L, n, crc = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not a compressor archive")
    if L != model.config.max_seq_len:
        raise ValueError(
            f"compressed at max_seq_len {L}; load the model with "
            f"max_seq_len={L}, not {model.config.max_seq_len}"
        )
    decoder = Decoder(blob[HEADER.size :])
    tokens = []
    for start in range(0, n, L - 1):
        history = _restart(model)
        for _ in range(min(L - 1, n - start)):
            history.append(decoder.decode(cdf(model.logits(history))))
        tokens += history[1:]
    data = model.tokenizer.decode(tokens).encode("utf-8")
    if zlib.crc32(data) != crc:
        raise ValueError(
            "the text decoded fails its CRC: not the model it was coded with"
        )
    return data


def _restart(model) -> list[int]:
    """A block's history, the BOS alone, its first step a decode step from
    empty caches.
    """
    model.forget(0)
    return [model.bos]
