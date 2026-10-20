# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic coding over a model's logits: ``cdf`` turns logits into an
integer cumulative distribution, the same on every host, and ``Encoder``
and ``Decoder`` code symbols against it.

The coder is the textbook one of Witten, Neal and Cleary ("Arithmetic
coding for data compression", CACM 1987), at 64 bits of state on Python's
integers: each symbol narrows ``[low, high]`` to its share of the
distribution, and every bit both ends agree on is written out.
"""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

import numpy as np

#: A logit's resolution in ``cdf``: 1/STEPS of a bit.
STEPS = 256
#: log2(e), the float64 nearest it, so a logit times it is a log2.
LOG2E = 1.4426950408889634
#: 2^32 * 2^(-j/STEPS) for j in [0, STEPS), rounded: the fraction of each
#: power of two ``cdf`` reads. Computed in decimal, which is software, so
#: every host builds the same table.
with localcontext(prec=40):
    EXP2 = np.array(
        [round(Decimal(2) ** (32 - Decimal(j) / STEPS)) for j in range(STEPS)],
        dtype=np.int64,
    )


def cdf(logits) -> np.ndarray:
    """The cumulative frequencies of ``logits``' softmax in integers: entry
    ``s`` is where symbol ``s`` starts, the last entry the total.

    Each logit is floored to 1/STEPS of a bit by one float64 multiply, which
    IEEE rounds the same everywhere; everything after it is integer. A
    symbol's frequency is ``2^32 * 2^-d``, ``d`` its distance in bits below
    the largest logit, plus one, so every symbol stays codable.

    Raises:
        ValueError: A logit is not finite.
    """
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("logits must be finite")
    q = np.floor(x * (STEPS * LOG2E))
    d = (q.max() - q).astype(np.int64)
    freq = (EXP2[d % STEPS] >> np.minimum(d // STEPS, 63)) + 1
    return np.concatenate([[0], np.cumsum(freq)])


BITS = 64
HALF, QUARTER = 1 << (BITS - 1), 1 << (BITS - 2)
TOP = (1 << BITS) - 1


class Encoder:
    """Symbols in, bits out: ``encode`` each symbol with its ``cdf``, then
    ``finish`` for the bytes.
    """

    def __init__(self):
        self.low, self.high = 0, TOP
        self.pending = 0  # bits owed, the opposite of the next one written
        self.bits = bytearray()

    def encode(self, symbol: int, cdf: np.ndarray) -> None:
        span, total = self.high - self.low + 1, int(cdf[-1])
        self.high = self.low + span * int(cdf[symbol + 1]) // total - 1
        self.low += span * int(cdf[symbol]) // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self) -> bytes:
        """The bits so far and enough more to name the final interval."""
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return np.packbits(np.frombuffer(self.bits, dtype=np.uint8)).tobytes()

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0


class Decoder:
    """Bits in, symbols out: ``decode`` with the same ``cdf`` each symbol
    was encoded with returns it.
    """

    def __init__(self, data: bytes):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.low, self.high = 0, TOP
        self.value = 0
        self.read = 0
        for _ in range(BITS):
            self.value = 2 * self.value + self._next()

    def decode(self, cdf: np.ndarray) -> int:
        span, total = self.high - self.low + 1, int(cdf[-1])
        target = ((self.value - self.low + 1) * total - 1) // span
        symbol = int(np.searchsorted(cdf, target, side="right")) - 1
        self.high = self.low + span * int(cdf[symbol + 1]) // total - 1
        self.low += span * int(cdf[symbol]) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.value = 2 * self.value + self._next()
        return symbol

    def _next(self) -> int:
        """The next bit, zeros past the end."""
        self.read += 1
        return int(self.bits[self.read - 1]) if self.read <= self.bits.size else 0
