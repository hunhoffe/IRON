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
  it appears in an operand's `tile=`/`per=`/`depth=` or says so itself, `param(..., array=True)`.
  `array(target)` receives a view exposing only those and raises on any other
  read. No learned read-sets.
- **2B** `In(*shape, dtype=, tile=, per=, depth=, via=)` absorbs `StreamIn`:
  one declaration per operand; `op.A[view]` a region, `op.A.lane(i)` a shim
  endpoint, `op.A.tile` the fifo type. Shape args stay extent-only; `tile=`/
  `per=` may be knobs. `Stream(...)` remains for the rare internal stream.
- **3B** `Value(dtype, derive=..., address=, lock=)` replaces `Resident` and
  `Scratchpad`; the call site decides the tier (an int is compiled in, a `DispatchTime`
  graph parameter never is); `.read()` in a core body picks the lowering
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
    epilogue: str = param(default="none", array=True)   # read by array(), named by no tile

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
baseline and the toolchain run (`iron/tests/toolchain`, its own conftest;
4 failures on `26ce43f`, all needing a device) against its (80 failed / 1680 passed / 38 skipped on `26ce43f`: 40 need
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
   per=, depth=, via=)` (2B), `Value` (3B) and `array=True`/the restricted view
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
4. **Views and `Copy`.** The pin test first (done: `tests/common/copy_taps.
   py`). Then:
   - A graph handle takes numpy basic indexing, `h[..., a:b]`, `h[i]`,
     `h.transpose(1, 0, 2)`, and a per-call `Scratchpad` value as the index
     on one axis, `keys[i][:, pos]`. A view is metadata: the base buffer and
     a `Walk` (offset, sizes, strides over it), plus the axis stride the
     per-call index scales. A contiguous static view is a sub-buffer, as
     today; any other view is an operand of `Copy` alone.
   - `Copy(src, dst=None)` replaces `StridedCopy`: two `Walk` params in
     place of eight sizes/strides/offset fields; its sequence legalizes each
     walk per channel (`_taps`' split of the highest non-unit axis, as
     pinned). `in_offset`/`out_offset` stay its per-call values; a per-call
     index binds to them with the axis stride as its scale, applied where
     the host writes the value (upstream's scratchpad offset is a raw
     element offset). The class declares which params take a view
     (`accept_views`), the one hook.
   - llama: caches become `(G, L, D)`; `Copy(k, keys[i][:, pos])`, `Copy(v.
     reshape(G, D), values[i][:, pos])`, prefill `Copy(k.reshape(n, G, D).
     transpose(1, 0, 2), keys[i][:, :n])`, `Copy(x[last]).reshape(1, E)`;
     the host passes row indices (`pos`, `n - 1`), not element offsets.
     Rank flattening in `infer.py` stays until the merge.
5. **Collapse the four one-purpose templates**; land `inline()` and the
   hello-world operator (`source_text=` on `declare_kernel`, hashing the
   string; `arg_types` from operands).
6. **C11 device-free**: build each class's array at two extents, require
   byte-identical core ELFs. Gates the merge.
7. **Merge** each operator onto one class. The mechanics, so both forms
   coexist and operators migrate one at a time with the suites green:
   - The merged base carries both surfaces. `Operator` gains what `Overlay`
     has (`resolve`/`resolved`, `array` (was `array(target)`), `streams`,
     `values`, `residents`, `design_key`, `tolerance`, `device`, `copy`,
     `name_parts`, `external`/`prebuilt`/`build`), and `op.ov` is `self`
     on a merged class (no `ov` field, no `_split_kwargs`, no `__init__`
     wrapper); a two-class operator keeps its separate `ov`. The graph,
     sequence and image layers keep reading `op.ov` until the last operator
     has moved, then `ov` goes.
   - `In(*shape, dtype=, tile=, per=, depth=, via=, replicate=)`: an operand
     with `tile=` is its own stream; `op.A` answers as a buffer (shape,
     elements, nbytes, views) and as a stream (`tile`, `lane(i)`/`[i]`,
     `bind`, `handles`, `count`). The array tier is declared: every field
     named in a `tile=`/`per=`/`depth=` or declared `param(..., array=True)`. `design_key`
     is the class and those fields, resolved; `array(target)` receives a
     view that raises on any other field, naming the rule.
   - `Value(dtype, derive=..., address=, lock=)`: a resident when it is
     derived or given a number (the preamble writes it), a per-call value
     when a graph binds it (`Scratchpad`'s lowering), `DispatchTime` when
     declared so. `residents()` goes: the derivation is on the member.
   - `array(self, target)`: the old overlay `design`; `sequence(self, rt)`:
     the old operator `sequence(rt)`; `reference` unchanged.
   - Rungs: the elementwise family (the four templates collapse to one
     `Elementwise` base whose operands say how many streams there are),
     then gemv/softmax/rope/transpose/repeat/mem_copy, then gemm/mha, then
     flm (→ `iron/exports/flm`; `Shipped(GEMM, image=Xclbin(...))`). Rename
     collisions first: flm `epilogue` resident → `mode`, dequant `qw`/`out`,
     `config_name`.
   - Open, noted: a per-call value given at a graph call site
     (`MV(a, b, start=pos)`) is not a constructor parameter a checker
     knows. Making a `Value` a field whose default is "unbound" would type
     it, and give the call site the tier choice (a number binds once per
     build, a handle per call); decide with the graph-layer rung.
   - Gates per rung: both suites identical to baseline; the C11 gate
     (GEMV's xfail turns strict-pass when `rows` becomes a `Value`); the
     design counts of step 3's test; llama's reference tests.
   - After the last rung: delete `Overlay`, `_overlay_class_of`, the tier
     guards, the `residents()` bridge; pyright's scope widens to
     `iron/operators`, `iron/applications` and the rest of `iron/tests`,
     since an operator's constructor is the dataclass one.
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

- (this commit) Step 7, rung 2: GEMV, Softmax, RoPE, Transpose, Repeat,
  MemCopy and Copy are one class each (7 files, −177 lines net). Their
  trip counts are derived `Value`s; GEMV's `tiles` replaces the rows per
  column it compiled into its core loop, so the C11 gate now passes for
  it too (the strict xfail is gone), and Softmax's `vector_size` is one
  `Value` that is the whole row unless a graph binds a per-call handle:
  `DynamicSoftmax` and its overlay are gone, `resolve_class` with them.
  What an array bakes beyond its tiles is declared (`kernel_vector_size`,
  `epilogue`, `method_type`, `s`, `tile_size`, `bypass`); a tile's dtype
  field is array-tier too. In a sequence an operand is its own stream:
  `rt.fill(self.A.lane(col), access)`, the buffer implied. The per-call
  values an instance binds are part of its `design_key`, and a derived
  value a graph binds is not written as a resident (the toolchain gate
  caught that on llama's decode graph). Both suites identical to
  baseline.
- `08e1f7a` Step 7, rung 1: the elementwise family is one class per
  operator. `iron/common/elementwise.py` is `Elementwise` (the array over
  lines, its `count` a derived `Value`) with `UnaryElementwise` and
  `BinaryElementwise` declaring the operand shapes as their own streams;
  the twelve elementwise operators (axpy, dequant, elementwise_add/mul,
  gelu, layer_norm, leaky_relu, relu, rms_norm and its weighted form,
  sigmoid, silu, tanh) and the inline-kernel hello world are each one
  subclass naming a kernel (12 files, −133 lines net). What a kernel call
  bakes (axpy's scalar, leaky_relu's alpha, rms_norm's epsilon, dequant's
  group size) is `param(..., array=True)`; the view `array()` runs on
  binds the operator's methods and properties too, so a `kernel()` that
  reads an extent is caught, and `super()` works inside it. `auto()`
  fields are annotated with their resolved type. Both suites identical to
  baseline; the C11 gate passes for the merged ReLU, ElementwiseAdd and
  RMSNorm.
- `404c939` Step 7, rung 0: the mechanics of the one-class operator,
  with every shipped operator still two-class and both suites identical to
  baseline. `Operator` carries the array surface (`resolve`, `array`,
  `build_array`, `streams`, `residents`, `tolerance`, `array_key`), and on
  a class declared without an overlay `op.ov` is the operator itself.
  `In(*shape, tile=, per=, depth=, via=, replicate=, broadcast=)`: an
  operand with a tile is its own stream (`op.A.tile`, `op.A.lane(i)`,
  `op.A.bind`, `op.A.count`). `Value(dtype, derive=, address=, lock=)`: a
  resident when derived, per-call when a graph binds it. The array tier
  is `_array_fields`: whatever a `tile=`/`per=` names plus
  `param(..., array=True)`; `array(target)` runs on a view that raises on
  any other field; `array_key()` is the class and those fields, so two
  extents of one array share a build. `tests/common/merged.py` exercises
  it on a GEMV-shaped operator: fields, lanes, a derived value, a per-call
  value in a graph, the two keys, the guard, resolution errors, inference.
- `d25f00a` Step 1: drop the pre-dataclass Field snapshot.
- `fd3ba37` Step 1: the bases process their subclasses; `@operator` is gone
  (38 files, +390/−509). Failure set identical to baseline.
- `610c926` Step 1: the bases are dataclasses to a checker; pyright in CI on
  the declare package. Members generic in their bound form; `Self` returns.
- `d782c44` Step 6: the C11 gate, `tests/toolchain/array_identity.py`:
  each operator built at two extents with the same knobs must leave
  byte-identical per-core ELFs (the real build; an insts-only lowering
  compiles no core). ReLU, ElementwiseAdd, Softmax, RoPE, RMSNorm and GEMM
  pass on both device widths; GEMV is a strict xfail, since it compiles the
  rows per column into its core loop (the `rows` `Value` of the merge
  fixes it); Repeat and Copy have no core.
- `a99ced5` Step 5a: `declare_kernel(source_text=)`: a kernel written
  in the operator's own file, its recipe digest over the text; the
  hello-world `VectorAdd` (`tests/toolchain/inline_kernel.py`) lowers
  through the toolchain with its `vadd` compiled from the text. The four
  elementwise templates collapse with the merge (step 7's first rung),
  where an operator's operands say how many streams there are.
- `6aa37f2` Step 4: views and `Copy`. A graph handle takes numpy's
  basic indexing plus a per-call `Scratchpad` index on one axis, and
  `transpose`; a contiguous static region is a sub-buffer as before, any
  other view a `Walk` over the parent that `Copy` alone takes (its
  `accept_views`). `Copy(src, dst=None)` replaces `StridedCopy`'s eight
  fields with two walks; the per-call index binds `in_offset`/`out_offset`
  with the axis stride as its scale, applied where the host writes the
  value. States are viewed inside a graph function (a handle when tracing,
  a remembered key over the host tensor in the reference). llama: caches
  `(G, L, D)`, `Copy(k, keys[i][:, cache_offset])`, prefill through a
  transpose view, `Copy(x[last]).reshape(1, E)`; the host passes rows.
  Descriptors identical to the pinned ones; llama's reference tests pass;
  both suites identical to baseline. Reshapes in the llama graph: 13 → 11
  (the two `Repeat` reshapes wait on rank-3 operands).
- `51e807f` `Operator.copy()` re-records what `compatible()` writes
  (GEMV's rows per column, an `init=False` field a `replace()` resets):
  the toolchain run caught the core loop running zero times, which the
  device-free run cannot see, so the toolchain run is now a gate of every
  step. Step 4's first item: `tests/common/copy_taps.py` pins the exact
  descriptors StridedCopy issues for llama's three copies and the KV slot.
- `5d63876` Step 3b: the column knob defaults from the device on GEMV,
  Softmax and RoPE (`auto()`; the operator's `resolve` picks the most
  columns the shim budget allows that divide its extents; RMSNorm keeps one
  core per row; GEMM keeps its given eight, since its host buffers are
  padded by it). An operator's `name` is the resolved operator's, so the
  per-call value symbols a graph names at trace time and the kernel
  instances a build names agree. Every device test gives the knob, so
  nothing measured changes. Failure set identical to baseline.
- `4bb8f82` Step 3a: one resolution point. `Operator.resolve(dev)` is
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
