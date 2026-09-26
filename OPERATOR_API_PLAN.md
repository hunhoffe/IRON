<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The operator API, redesigned

Branch `claude/iron-pr221-api-simplify`, off amd/IRON PR 221 (`26ce43f`).
Toolchain: mlir-aie branch `aie2p-kernel-peano-opt`, built from source (see
Step 0). This file is the plan and its progress log; it travels with the
branch and is updated as steps land.

## Why

PR 221 lands the declared operator model (`declare/`, `design/`, `graph/`,
`image/`). The model is sound; the surface is dense, and the density falls on
performance engineers and ML programmers. On `applications/llama_3_2_1b/
graphs.py` (276 lines): **46** tunable kwargs at call sites, **18**
hand-written DMA size/stride/offset fields, **13** rank-plumbing reshapes,
**32** names exported from `iron.common.declare`; and every operator is two
classes and two decorators.

Goals: fewer concepts, with everything inferable inferred (the
`aie.iron.algorithms` style); friendly authoring, where a newcomer customises
a hello-world by inheritance, in one file, with an inlined C++ kernel; a
design that leaves the door open to autotuning, profiles and per-kernel L1 /
tile tuning without implementing them; dynamic sequences and full-ELF assumed
to coexist one day; flm kept, as an export. Typed throughout: pyright and ruff
run in CI with mlir-aie's configuration.

## Decisions

- **One class per operator; one organizing axis: when a value resolves.**
  Compile-time = baked into the artifact (array *or* sequence; which one is
  derived from where the field is used); dispatch-time = never baked.
- **1A** The array's dependency set is *declared*: a field is array-tier iff
  it appears in an operand's `tile=`/`per=`/`depth=` or in `bakes = (...)`.
  `array(target)` receives a view exposing only those and raises on any other
  read. No learned read-sets.
- **2B** `In(*shape, dtype=, tile=, per=, depth=, via=)` absorbs `StreamIn`:
  one declaration per operand; `op.A[view]` a region, `op.A.lane(i)` a shim
  endpoint, `op.A.tile` the fifo type. Shape args stay extent-only; `tile=`/
  `per=` may be knobs. `Stream(...)` remains for the rare internal stream.
- **3B** `Value(dtype, derive=..., address=, lock=)` replaces `Resident` and
  `Scratchpad`; the call site decides the tier (an int bakes, a `DispatchTime`
  graph parameter never bakes); `.read()` in a core body picks the lowering
  per image. `design_key`/`explain()` print which values were baked.
- **4B** `Shipped(GEMM, image=Xclbin(url=, sha256=))`: the shipped binary is
  a subclass; `image=` is consumed by `__init_subclass__` (replaces
  `External`, the `Xclbin` member, `prebuilt()`/`build()`; every operand needs
  `via=`, `array()` forbidden). flm moves to `iron/exports/flm`.
- Factories may carry more kernel/profiling data (legality and L1 footprint
  move onto `KernelContract`); compilation is cheap, so `legal=` may be a
  compile probe and the C11 check runs per class in the device-free suite.
- Autotuning: a content-keyed **tuning digest** (below). Door open, not built.

## The API

Hello world (rung 1: the template owns array and sequence):

```python
from iron import BinaryElementwise, inline

class VectorAdd(BinaryElementwise):
    """y = a + b."""
    kernel = inline("vadd", """
        extern "C" void vadd(bfloat16 *a, bfloat16 *b, bfloat16 *y, int n) {
            for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
        }""")                                   # arg_types from the operands

    def reference(self, a, b):
        return a + b
# onto a shipped factory: kernel = eltwise.add_sized
```

Rung 2 overrides `array(target)`; rung 3 overrides `sequence(rt)`. Each rung
is additive.

GEMV (rung 3):

```python
class GEMV(Operator):
    M: int = param()                          # sequence-tier: only in a shape
    K: int = param()                          # array-tier: in tile=(tile_in, K)
    batches: int = param(default=1)
    columns: int = auto()                     # shim budget of the device
    tile_in: int = auto(2, choices=(1, 2, 4, 8))
    tile_out: int = auto()                    # largest divisor of M//columns that fits L1
    vector_width: int = auto()                # the kernel contract's vec_size
    epilogue: str = param(default="none")
    bakes = ("epilogue",)                     # read by array(), named by no tiling

    A = In(optional(batches), M, K, tile=(tile_in, K), per=columns, depth=2)
    B = In(optional(batches), K,    tile=(K,),         per=columns, depth=1)
    C = Out(optional(batches), M,   tile=(tile_out,),  per=columns, depth=2)
    rows = Value(np.int32, derive=lambda op: op.M // op.columns)

    def resolve(self, dev): ...   # device AND extents; proposals for auto fields
    def check(self): ...          # after resolution
    def array(self, target): ...  # sees only K, columns, tile_in, tile_out, vector_width, epilogue
    def sequence(self, rt):
        share = self.M // self.columns
        with rt.group():
            for i in range(self.columns):
                rt.fill(self.B.lane(i), self.B[...])
        with rt.group():
            for i in range(self.columns):
                rt.fill(self.A.lane(i), self.A[..., i*share:(i+1)*share, :])
            for i in range(self.columns):
                rt.drain(self.C.lane(i), self.C[..., i*share:(i+1)*share], wait=True)
    def reference(self, A, B): ...
```

A field is declared by a specifier, not an annotation alone: `M: int` binds
no name for a shape to use, and `In(M, K)` needs one. `param` is a field
specifier to a checker (`default=` is keyword-only, so a `param()` without one
is a required constructor argument, and `GEMV(K="4")` is an error); `auto` is
not listed, since it always has a default and gives it positionally, which
pyright does not read. Both are compile-time; a `Value` bound to a
`DispatchTime` graph parameter is the dispatch-time side.

llama decode block (46 knob kwargs → 4, 18 stride fields → 0, 13 reshapes →
3; `cols` and `get_current_device()` leave the graph):

```python
def decode_block(i, lw, x, angles, pos, n_keys):
    h = RMSNorm(x, lw.norm1)
    q, k, v = (GEMV(w, h, tile_in=4) for w in (lw.q, lw.k, lw.v))
    q = RoPE(q.reshape(H, D), angles); k = RoPE(k.reshape(G, D), angles)
    Copy(k, keys[i][:, pos]); Copy(v.reshape(G, D), values[i][:, pos])
    k_all, v_all = Repeat(keys[i], H // G), Repeat(values[i], H // G)
    w = Softmax(Mul(GEMV(k_all, q), scale), valid=n_keys)     # DispatchTime -> never baked
    o = GEMV(lw.o, GEMV(Transpose(v_all), w).reshape(H * D))
    x = Add(x, o); h = RMSNorm(x, lw.norm2)
    return Add(x, GEMV(lw.down, Mul(SiLU(GEMV(lw.gate, h)), GEMV(lw.up, h)), tile_in=1))
```

flm (`iron/exports/flm`): `class Shipped(GEMM, image=Xclbin(...))` pins every
knob, redeclares operands with `via=`, keeps `sequence()`.

## Resolution, profiles, digest

Precedence, once, before identity is taken: **explicit call-site value >
profile entry > `resolve(dev)` proposal > `auto(default)`**. Proposals apply
only to still-`auto` fields, so an explicit knob can never be overridden;
identity after resolution, so two spellings of one array build one image.
Profile entries name compile-time knobs only; dispatch fields never enter a
key; a miss never searches; search never runs inside `array()`/`sequence()`
or a build.

The library synthesises `GEMV.Tuning` from the `auto()` fields (`choices=`,
`legal=`), so profiles and a tuner have a typed object without an authored
nested class. A profile is a data artifact applied in a scope.

Per-kernel budget: the kernel declares `stack_bytes`/`static_bytes`/`lanes`
(optional; upstream `KernelContract.stack_bytes`); the template sums against
the *resolved* device's `core_memory_bytes`, chooses tile/depth, and checks
with a breakdown in the error. One legality predicate validates a pin, bounds
a default, prunes a search.

Tuning digest, two content-keyed tiers: kernel tier (slow, once per kernel
digest × knob point × device type × toolchain; built on upstream
`kernel_design` + `run_iters`; records metrics, L1 bytes, and the tolerance
it was judged at) and operator/graph tier (fast: `design_key` minus tuned
knobs; candidates from `auto()`, pruned by legality, ranked by the kernel
tier, top-k measured in situ). `iron.tune(graph, dev, level=, budget=,
digest=)` measures only what is missing; entries record their level so runs
extend, not restart; a kernel edit invalidates only that kernel. Per-model
profiles are exported slices. Nothing in an operator class knows it exists.

## Steps

Each separately reviewable; each ends at the full device-free run against the
baseline (80 failed / 1680 passed / 38 skipped on `26ce43f`: 40 need
`pyxrt`, 35 need a device, 5 re-bind a device after clearing it); each fix
minimal. `pyright` and `ruff check` clean at every commit.

0. **Toolchain from source** — done. mlir-aie `aie2p-kernel-peano-opt` (the
   only ref with `linalg.mv(output_rows=)`, which PR 221's GEMV calls) per
   `docs/Building.md`: cmake ≥ 3.30, lit, all submodules, `nanobind==2.12.0`,
   psutil; `aie.extras` vendored from `llvm/eudsl` `4853bb0`; the mlir wheel's
   `_mlir/*.pyi` stubs copied into `install/python/aie/_mlir_libs/_mlir/`
   (the build leaves them out; pyright needs them). Notes for the PR:
   `requirements.txt` pins `1d7b9ea`, which lacks `output_rows` and the
   `trace_to_json(colshift=, kernel=)` signature `tracing.py` calls.
1. **`@operator` → `__init_subclass__`** — done. Dead `pre_fields` gone; the
   nine jobs on the base hooks; `@dataclass_transform()` on both bases;
   `ABCMeta` dropped; pyright and ruff configured after mlir-aie's and run
   in the lint workflow.
2. **Field vocabulary**: `param()`/`auto()` (done: a rename on the two-class
   model, `auto(choices=, legal=)` recorded for a tuner); then `In(..., tile=,
   per=, depth=, via=)` (2B), `Value` (3B) and `bakes`/the restricted view
   (1A). Those three are one-class concepts (an operand's `tile=` is the
   stream the overlay owns today), so they land with the merge, operator by
   operator: a merged class answers as its own `ov`, so the graph, sequence
   and image layers keep working while both forms coexist.
3. **One resolution point** (`resolve(dev)` with extents, before
   `unique_designs()`); retire `overlay_defaults`, `tuning`, `for_extent`;
   `None`-guarded device defaults for gemv/softmax/rope/gemm. Gate: two
   spellings of one array → one design; llama image count before/after; the
   first profile registered is `llama_decode` with today's 46 values, and
   `applications/llama_3_2_1b/test.py`'s bit-identical-logits check runs
   against it before any inferred default is trusted.
4. **Views and `Copy`.** First a test pinning the exact `Access` lists
   `StridedCopy._taps` produces. Then `tiling.view` → `TensorAccessPattern.
   from_slice`; stop flattening in `infer.py`; `Handle.__getitem__` on tuple
   keys + per-call `Scratchpad` start; `Copy.sequence(rt)` passes views to
   `rt.fill`/`rt.drain`; `_taps`, `_pad4`, `_kv_slot`, `_flat` and the eight
   sizes/strides fields go. Name the channel-split axis explicitly.
5. **Collapse the four one-purpose templates**; land `inline()` and the
   hello-world operator (`source_text=` on `declare_kernel`, hashing the
   string; `arg_types` from operands).
6. **C11 device-free**: build each class's array at two extents, require
   byte-identical core ELFs. Gates the merge.
7. **Merge** each operator onto one class (elementwise; gemv/softmax/rope/
   transpose/repeat/mem_copy; gemm/mha; flm). Rename collisions first. flm →
   `iron/exports/flm`. Delete `_split_kwargs`, the `__init__` wrapper,
   `_overlay_class_of`, `Overlay.copy()`, the `residents()` bridge, both tier
   guards. An operator's constructor becomes visible to a checker here, so
   pyright's scope widens to `iron/operators`, `iron/applications` and the
   rest of `iron/tests`.
8. **Exports**: `declare` 32 → ~15; delete the dead re-exports in
   `iron/common/__init__.py`.

## Verification

Device-free net, in the order it catches things: `tests/common/graph.py`,
`tests/common/tiling.py`, `tests/common/declare.py`, `tests/operators/
rejected_shapes.py`, `tests/toolchain/lowering.py`, the compile-path tests
runnable with the built `aiecc`. Hardware only: `iron/operators/test.py` at
`Tolerance.exact()`, `strided_copy_offset.py`, `applications/llama_3_2_1b/
test.py`. Headline: 46/18/13 → ~4/0/3 on the llama graph; concepts 27 → ~15;
**no explicit knob resolves differently; no graph builds more images than
today**.

## Progress

- `d25f00a` Step 1: drop the pre-dataclass Field snapshot.
- `fd3ba37` Step 1: the bases process their subclasses; `@operator` is gone
  (38 files, +390/−509). Failure set identical to baseline.
- `610c926` Step 1: the bases are dataclasses to a checker; pyright in CI on
  the declare package. Members generic in their bound form; `Self` returns.
- (this commit) Step 3a: one resolution point. `Operator.resolve(dev)` is
  the hook that sees the device and the extents (`overlay_defaults` gone;
  StridedCopy's transfer size is its one override); `Overlay.resolve(dev)`
  (was `tuning`) sees the device and its own fields; `resolved(dev)` on
  both is idempotent and the only caller of `resolve`. The sequence
  resolves every operator in `prepare()`, before `unique_designs()` takes
  identity, so two spellings of one array are one design (test added; ffn
  6 designs / swiglu 4, as before). `for_extent`, `specialised` gone;
  `Untunable` → `Unresolvable`. Failure set identical to baseline.
- `da31c8b` Step 2a: `dim()`/`tunable()` → `param()`/`auto()`. `param`
  is a field specifier with keyword `default=`, so a missing required field
  is now a checker error too; `auto(choices=, legal=)` is accepted and
  recorded. Failure set identical to baseline.
- `e04956f` + `b9aaecb` ruff after mlir-aie's `ruff.toml` (D205/D401 off, for the codebase's
  sentence summaries); `pyrightconfig.json` after mlir-aie's; both scoped to
  all of `iron/common` and `iron/tests/common` (three tests wait on step 7),
  both clean with and without mlir-aie on the path. The ~95 findings fixed
  were narrowings of "None until prepared" state, `Transfers`' abstract
  surface, `ElementwiseOperator` typed over its own overlay, pyxrt behind
  `TYPE_CHECKING`, and one accessor (`jit_compile.cache_entry`) for "a
  compiled design has its entry". Upstream typing gaps worked around inline:
  `Device.resolve() -> None`, `region_op` decorators, `_aie.pyi` missing
  `get_target_model`, `_mlir/*.pyi` not installed by the build.
