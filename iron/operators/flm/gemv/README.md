<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `iron.operators.flm.Q4nxGEMV`

One row times a q4nx weight: `y = w @ x` for `x` of `K` bf16 values and `N`
out-features, `w` as `q4nx.quantize` lays it out (the order
[`DequantBFP`](../dequant) reads). It reads 5 bits a weight where a bf16
GEMV reads 16.

```python
from iron.operators.flm import Q4nxGEMV, q4nx

Q4nxGEMV(K=2048, N=512)(y, q4nx.quantize(w).reshape(-1), x)
```

This is IRON's own operator, not a FastFlowLM port. Each column's core
takes 32 out-features a round: it sums each 32 values of `x` once, then
streams those out-features' blocks past `x` through mlir-aie's q4nx block
product, the one the [LM head](../lm_head) runs, without its RMS norm and
softcap.

## Numerics

The cores narrow each 32-column dot product and each 32-value sum of `x` to
bf16. The tolerance is 2.5% of the largest output, the LM head's bound.

## Constraints

`K % 512 == 0`, `N % 64 == 0`, AIE2P only.
