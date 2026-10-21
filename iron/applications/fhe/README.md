<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Encrypted dot products on the NPU

A server scores a matrix it holds in the clear against a vector it only
ever sees encrypted; the key holder decrypts the exact integer scores.
That is private search: the server learns nothing about the query.

```python
import numpy as np
from iron.applications import fhe

key = fhe.BFV().keygen()            # the client's secret key
encrypted = key.encrypt(query)      # int8 vector, length <= 4096
scores = docs @ encrypted           # the server: int8 matrix @ ciphertext
assert (key.decrypt(scores) == docs @ query).all()
```

`python -m iron.applications.fhe` runs it on 4096 drawn documents and
times each side (`--cpu` without an NPU).

## How it maps onto the NPU

- **Scheme**, `bfv.py`: BFV with secret-key encryption at `n = 4096`,
  `q` two 30-bit primes, within the Homomorphic Encryption Standard's
  128-bit bound. `docs @ encrypted` packs `4096 // d` rows per plaintext,
  each reversed, so one ciphertext-times-plaintext product holds each row's
  dot product at one coefficient, as in Cheetah (Huang et al., USENIX
  Security 2022). No rotations, key switching or bootstrapping.
- **Arithmetic**, `ntt.py`: every product is a negacyclic NTT, done as two
  dense transforms of length 64 with a pointwise twist between them. Each
  dense transform modulo `p` is one int8 matrix product, a `(256, 256)`
  weight of pre-reduced twiddle bytes times the data's four signed bytes,
  accumulated in int32 by IRON's `GEMM` and recombined on the host. This
  follows CROSS's basis-aligned transformation and 3-step negacyclic NTT
  (arXiv 2501.07047). Written from the paper; no code is copied from its
  JAX implementation (MIT).

## Tests

`tests/applications/fhe/reference/` runs without a device: the NTT
against schoolbook negacyclic products, its int8 products through `GEMM`'s
own reference and lowered to NPU instructions, and the encrypted dot
products against numpy's, including the largest int8 scores.
`tests/applications/fhe/test.py` does the last on the NPU.

## Scope

BFV, not CKKS: CKKS's encoding and rescaling are covered by active
patents (CryptoLab US10778409, Duality US12362903), and the IBM patents
on BGV-era modulus and key switching have lapsed. No code from Zama
(TFHE-rs, Concrete) or HEaaN is used.

The host still splits, recombines and twists between the two `GEMM`
calls of each NTT, so for now it, not the NPU, bounds the time; one
mlir-aie kernel fusing those steps is the next step.
