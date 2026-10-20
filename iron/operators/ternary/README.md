<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `iron.operators.ternary`

Weights of -1, 0 or +1, each times a bf16 scale shared by 128 in-features of
an out-feature: 2.125 bits a weight, where q4nx takes 5 and bf16 16.
`layout.quantize` holds a weight `(out, in)` as `(out, in * 17 // 64)`
bytes, the order mlir-aie's `quant/ternary.cc` reads, and `F.linear` runs a
weight of that width on these operators.

```python
from iron.operators.ternary import TernaryGEMV, layout

TernaryGEMV(K=2048, N=512)(y, layout.quantize(w).reshape(-1), x)
```

- `TernaryGEMV`: `y = w @ x` for one row `x`. Each column's core takes 32
  out-features a round and streams their blocks past `x`, adding and
  subtracting `x` and multiplying only by each group's scale. It sums in
  float32 and narrows each output to bf16 by floor.
- `TernaryDequant`: the weight as bf16 `(N, K)`, exactly, for a prompt's
  `flm.ProjectionGEMM`.

`layout.quantize` rounds each weight over its group's mean magnitude to
-1, 0 or +1 (BitNet b1.58's rounding) and fits the scale by least squares,
so a weight that is already ternary is held exactly, up to its scale's
bf16 rounding.

## Constraints

`K % 256 == 0`, `N % 32 == 0`. The kernels build for AIE2 and AIE2P; no
device run has checked them.
