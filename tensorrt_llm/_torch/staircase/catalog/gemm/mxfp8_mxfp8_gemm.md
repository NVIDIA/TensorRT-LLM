---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 191}
---

<!--
`tests:` is filled from the observed pytest count of the certifying run, never
from arithmetic.
-->


# mxfp8_mxfp8_gemm

**Wraps** `torch.ops.trtllm.mxfp8_mxfp8_gemm` (one call).

## Semantics

Dense matrix multiply in `nn.Linear` layout over two MXFP8 (OCP microscaling
FP8) operands, executed as a single CUTLASS `W8A8_MXFP8_MXFP8` GEMM:

```
out[m, n] = global_scale[0] * sum_k dequant(act)[m, k] * dequant(weight)[n, k]
```

where `dequant(x)[i, j] = x[i, j] * 2 ** (scale_byte[i, j // 32] - 127)`, i.e.
one UE8M0 power-of-two exponent per **32 contiguous elements along K**, applied
to e4m3 values. Accumulation is fp32 and the result is cast to `out_dtype`.

**The fusion boundary.** Only the `alpha` multiply is fused, in the epilogue.
There is no bias, no activation, no transpose and no quantization inside this
call. The caller owns:

* quantizing the activation and producing its swizzled scales — in this
  codebase that is the `quantization/mxfp8_quantize` entry, called with
  `swizzled_layout=True`;
* producing the weight's swizzled scales, which is a **load-time** relayout
  (`torch.ops.trtllm.block_scale_interleave`), not forward work;
* nothing about `weight`'s storage beyond ordinary contiguity — see below.

### Both scale buffers are 128x4-swizzled, and neither size is checked

Each scale buffer is a flat `uint8` tensor holding one byte per
`(row, 32-wide K block)`, written in the 128x4 swizzle. Its length is

```
pad_up(rows, 128) * pad_up(K // 32, 4)
```

Measured for both operands across the coverage table below — for example act
`M=1, K=5120` occupies 20,480 bytes, the same as `M=64` and `M=128`, because
rows pad to 128; weight `N=25600, K=6144` occupies 4,915,200 bytes exactly.

**The op performs no input-length validation of either scale buffer.** It reads
each byte by computed offset and uses whatever is at that address. Nothing
inspects the length, so an undersized buffer produces no diagnostic the caller
can rely on — see the next subsection for the three different things it has
actually been measured doing.

Measured on both buffers at `(5120,1280)`, `M=128`, with the bytes past the end
held **inside a mapped arena** and set to `0x00` (UE8M0 exponent `2**-127`).
That construction pins the read to memory the process owns, so the outcome is
reproducible; it is the shape the GPU test asserts on, and it is deliberately
the *mildest* of the possible outcomes:

| buffer length | what the op did, with the out-of-bounds read mapped |
|---|---|
| one byte short | returned, silently wrong: act `15.00` from correct, weight `11.41`, on outputs of scale ~320 |
| a tenth of the size | returned, grossly wrong: act `308.6`, weight `308.0` |
| oversized by 4096 | returned **bit-identical**; the tail is ignored |

So only the short side is dangerous and `>=` is the right guard. It is a
**wrapper guard rather than a documented precondition** precisely because the op
gives the caller nothing to catch: a precondition is only useful when the
violation is reported, and here it is not.

**Unmapped truncation is worse than that table, and it is not deterministic.** A
freshly allocated truncation leaves the bytes past the end to the allocator, and
whether that address is even mapped is out of the caller's control. All three
outcomes below were observed on this machine for the same code — the caller
cannot predict which one it gets, and only the third is reported at all:

* **arbitrary wrong numbers** — one byte short, freshly cloned: `1.50e+01` for
  act, and `5.37e+22` at `M=128` / `2.15e+23` at `M=64` for weight in the same
  run. The value is whatever that byte held, so the verdict reproduces and the
  number does not;
* **`inf` / `NaN`** — a tenth of the size, freshly cloned: weight returned `inf`
  with 4,096 NaN elements;
* **`AcceleratorError: CUDA error: an illegal memory access was encountered`** —
  the same tenth-size call in a different process, after which every later call
  in that process fails with `Failed to init`. Note what this is *not*: it is an
  asynchronous device fault surfaced at the next synchronization, not a
  validation error raised by the op, so it can be attributed to whatever call
  happens to synchronize next and it takes the CUDA context with it.

That is why the GPU test drives these cases inside an oversized arena (the same
out-of-bounds read, but mapped and deterministic) and the probe keeps the
freshly-allocated tenth-size variant in a **child process**: driven in-process
it took two whole probe sections down with it.

**One row count can hide the whole thing.** At `M=64` the act buffer's last
bytes belong to padding rows 64..127 — the swizzle packs rows in groups of 128
— so truncating it cannot change any real row's result, and it measures
**bit-identical**. A probe that asked this question only at `M=64` would report
that short act buffers are harmless. `M=128` is the first row count whose real
rows reach the buffer's last byte, and that is where the certified measurement
is taken. Both are in the test
(`act_short_at_M64_is_harmless`, `act_one_byte_short_M128`) so the exception
cannot be mistaken for the rule.

### `global_scale` is alpha, and only element 0 is read

Measured (`K=5120, N=1280, M=64`, bf16 out): `[2.0]` and `[0.5]` each reproduce
`alpha * (alpha=1 result)` **bit-exactly**, and a two-element `[1.0, 7.0]` is
accepted and gives a result bit-identical to `[1.0]` — element 0 is used and
the rest silently ignored.

### `weight` is an ordinary contiguous `[N, K]` tensor

The op calls `CHECK_INPUT(weight, ...)`, which is plain `.is_contiguous()`, and
a transposed view is rejected with `RuntimeError: weight must be contiguous`
(observed). So at the Python boundary this is exactly what `nn.Linear` already
holds: a row-major `[N, K]` tensor, no relayout, no `.t()`, no permute.

The CUTLASS-side description of the same bytes is what a source reading turns
into a false API requirement. A row-major `[N, K]` buffer *is* a column-major
`[K, N]` buffer — same memory, two names for it — and the kernel's B operand is
the `[K, N]` one, so the C++ comment calls it column-major. That is a statement
about how the kernel reads the buffer, not an extra obligation on the caller. A
caller who "helpfully" produces column-major storage by transposing a `[K, N]`
tensor gets `weight must be contiguous`.

## Signature

```python
mxfp8_mxfp8_gemm(act, act_scale, weight, weight_scale, global_scale,
                 out_dtype=None) -> Tensor
```

| name | shape | dtype | layout / device | notes |
|---|---|---|---|---|
| `act` | `[M, K]` | `float8_e4m3fn` | row-major contiguous, CUDA | `M >= 1`; `K % 32 == 0` |
| `act_scale` | `[pad_up(M,128) * pad_up(K//32,4)]` | `uint8` | contiguous, CUDA | UE8M0, 128x4 swizzle |
| `weight` | `[N, K]` | `float8_e4m3fn` | **contiguous** (ordinary row-major), CUDA | `N % 32 == 0`; same `K` as `act` |
| `weight_scale` | `[pad_up(N,128) * pad_up(K//32,4)]` | `uint8` | contiguous, CUDA | UE8M0, 128x4 swizzle |
| `global_scale` | `[1]` | `float32` | CUDA | epilogue alpha; element 0 only |
| `out_dtype` | — | `torch.dtype \| None` | — | bf16 / fp16 / fp32; `None` defaults to bf16 |
| **returns** | `[M, N]` | `out_dtype` | row-major contiguous, CUDA | |

## Metadata consumed

**The serving tactic cache.** This op calls the implementation with
`useTacticCache=true`, so it consults a process-global cache that
`torch.classes.trtllm.MXFP8GemmRunner.register_tactic(m, n, k, idx)` populates
— empty in a test process, warm in a served one whose autotuner has run. That
difference is the thing a contract for this op is most likely to get wrong, so
it is not merely observed: the GPU test **drives the runner class and asserts**
every statement below, at `K=5120, N=1280, M=128`, bf16 out.

* **10** compiled tactics; the cache-miss sentinel is `-2`.
* Calling `mxfp8_mxfp8_gemm` does **not** populate the cache — only
  `register_tactic` does, which is why a test process stays cold unless it asks
  not to. Asserted, because "the test path is the cold path" is an assumption
  everything else here rests on.
* All 10 tactics, **plus the `-1` generic fallback**, return results
  `torch.equal` to the cold-cache result — 11 assertions, not a printed
  `max_abs_diff`.
* Registering tactic 0 and tactic 9 and then calling this op again each leaves
  the result `torch.equal` to the cold-cache one, i.e. the served path and the
  test path cannot disagree here.

This is measured at that one shape and is stated as such. It is not a general
claim that tactic selection cannot matter — only that on the certified path,
cache state did not change the answer. The tactic count is itself asserted
(`test_compiled_tactic_count_and_miss_sentinel`), because the sweep is
parametrized over it and a build carrying more tactics would otherwise silently
narrow what this receipt covers.

Nothing else is consumed: no attention metadata, no workspace, no stream-order
state beyond the current stream.

## Certified coverage

What the receipt covers, exactly, and nothing wider. **Every row below is one or
more cases in
`tests/unittest/_torch/staircase/gemm/test_staircase_mxfp8_mxfp8_gemm.py`**, and
the receipt is that file passing on sm_103. Nothing in this table rests on the
domain probe: the probe measures, the test asserts, and only what the test
asserts is certified.

| axis | certified values | cases |
|---|---|---|
| **TARGET `(K, N)` x `M`** | **the full cross-product**: all 8 target surfaces x all 17 row buckets | 136 |
| target `(K, N)` surfaces | `(6144,25600)` engram.wkv, `(5120,2304)` shared w1/w3, `(2304,5120)` shared w2, **`(8192,5120)` wo_b**, **`(1280,32768)` wq_b**, `(5120,1280)` wq_a, `(5120,512)` wkv, **`(1280,4096)` indexer wq_b** | — |
| `M` row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| reference-only rank shards | `(2048,5120)`, `(1280,8192)`, `(1280,1024)` — the native implementation's shards of the three bold surfaces, `M` in {1, 129, 4096}. **Not target calls** | 9 |
| `out_dtype` | bf16 across the whole cross-product; fp16 and fp32 at `(5120,1280)`, `M=64` | 3 |
| `global_scale` | 0.5, 1.0, 2.0 at `(5120,1280)`, `M=64`; plus the two-element tensor | 3 + 1 |
| tactic cache | `(5120,1280)`, `M=128`, bf16 — count and miss sentinel, all 10 tactics plus the `-1` fallback, warm tactic 0 and 9 | 1 + 11 + 2 |
| scale-buffer domains | `(5120,1280)` — one byte short / a tenth / oversized on **both** `act_scale` (at `M=128`) and `weight_scale`, plus the `M=64` padding-row exception | 7 |
| loud rejections | `(5120,1280)`, `M=64` — all 13 in the Preconditions table | 13 |
| zero rows | `M=0` raises through the wrapper | 1 |
| tolerance | the scale-layout discrimination control, and the two floor cases that measure why the absolute floor moved | 1 + 2 |
| process-global state | `matmul.fp32_precision` is restored, so a receipt taken beside this file is still taken under torch defaults | 1 |
| total | | **191**, all passing; `tests: 191` in the receipt |

The cross-product is certified rather than sampled because that is what the
target calls: every dense projection runs at every token count the engine
serves, so certifying the surfaces at one `M` and the row buckets at one surface
would leave precisely the combinations in use uncovered.

### Three surfaces were the reference's rank shard, not the target's

The surfaces are derived from the **raw checkpoint's safetensors headers**
(`[out, in]`, so `(K, N) = (in, out)`), which is what the target's `weights.py`
reads. An earlier revision of this entry derived them from
`staircase-v41/ckpt-mp4/model0-mp4.safetensors` — the checkpoint already
**converted to the reference implementation's four tensor-parallel ranks** — and
so certified three widths the target never calls:

| what | raw header → target `(K, N)` | reference shard, and how | superseded value |
|---|---|---|---|
| `attn.wo_b` | `[5120, 8192]` → `(8192, 5120)` | `RowParallelLinear` splits the reduction dim | `(2048, 5120)` |
| `attn.wq_b` | `[32768, 1280]` → `(1280, 32768)` | `ColumnParallelLinear` splits the output dim | `(1280, 8192)` |
| `attn.indexer.wq_b` | `[4096, 1280]` → `(1280, 4096)` | `ColumnParallelLinear` splits the output dim | `(1280, 1024)` |

The staircase target shards none of them: `plan.md` line 30 fixes attention DP
at dep4 and replicates attention and every dense projection, lines 78–79 state
the target's own `5120 → 1280 → 32768` Q LoRA and `wo_b [5120, 8192]`, and line
82 requires all 32 index heads replicated with no TP score collective. The three
shard widths remain in the table as explicitly **reference-only** coverage,
because the module-parity leg runs the reference at them.

The other five surfaces are unaffected: `engram.wkv`, the three shared-expert
projections and `attn.wq_a` / `attn.wkv` are plain replicated `Linear`s in the
reference too, so shard and raw header agree.

The single-shape rows above are **not** claimed to generalize. Nothing here
establishes that alpha, the output dtype, the tactic cache or the scale-buffer
failure modes behave the same at another geometry; a target that needs one of
those at a different shape needs it certified there.

Unspecified and therefore uncertified: `M > 4096`, `K` or `N` outside the eleven
surfaces, non-bf16 output away from `(5120,1280)`, and any arch other than
sm_103.

### The tolerance the receipt is measured at

`rtol` is `torch.testing.assert_close`'s default for the output dtype,
unchanged. The **absolute floor** is not the default, and the substitution is
measured rather than argued: `atol=1e-5` is a floor for unit-scale results,
while these outputs measure **155 to 438**. It is replaced by two named terms,
each dominating exactly where it should:

```
atol = max(one output-dtype step, sqrt(K) * 2**-24) * |want|.max()
```

* the **output step** (`2**-8` bf16, `2**-11` fp16) dominates for the two
  narrow dtypes;
* **`sqrt(K) * 2**-24`** — fp32 accumulation over `K` terms summed in a
  different order from a single-pass reference — dominates for fp32 output,
  where no output cast rounds it away.

Two of the 191 cases hold that substitution to evidence, on the same realistic
operands every correctness case uses, at `(5120,1280)`, `M=64`:

| case | what it asserts | measured |
|---|---|---|
| `test_default_floor_is_too_tight_for_fp16` | the unmodified default floor rejects **correct** results, those results are near zero, and the scale-aware floor rejects none | 3 elements over the default floor, worst of them at `3.9e-05` of the tensor scale; 0 over the scale-aware floor |
| `test_fp32_output_needs_the_accumulation_term` | the default floor **and** an output-step-only floor both reject correct elements, and only the full `max(step, sqrt(K)*2**-24)` bound covers them | 856 over default, 236 over step-only, 0 over the full bound; `max_abs` 2.44e-04 against an accumulation bound of 1.36e-03 |

**The reference's TF32 setting is scoped, not global, and that matters to the
neighbours.** The expected value here is an fp32 matmul, so it must run with
`matmul.fp32_precision = ieee`; an earlier revision of the GPU test set that at
**module scope**, and pytest imports every selected module during collection, so
the assignment ran before any test did and left the whole process in `ieee` --
including the three other entry files in the combined receipt job, whose
contracts say their receipts are taken under torch defaults. The setting is now
a restoring context around the reference matmul only, and
`test_fp32_precision_is_restored` asserts three things: that the context really
forces `ieee` while open (so it is not a no-op), that it round-trips the prior
state, and that the state visible to the test is not `ieee` -- which is what a
surviving module-scope leak would look like.

**The tests assert the direction of those counts, never the number**, because
the count is a property of the *reference*, not of the op: how many near-zero
elements fall outside a floor depends on the accumulation order cuBLAS picks
for the fp32 reference matmul. Measured directly — with `allow_tf32` left at
the container default, the same seed and the same kernel output put 788 and 176
elements outside where pinning `allow_tf32 = False` puts 856 and 236. (The
operands are e4m3 values times powers of two, so TF32's wider significand still
represents every product exactly; only the summation order moves.) **0 elements
lie outside the final floor under either setting**, which is the claim the gate
rests on. The test and the domain probe now both pin `allow_tf32 = False`, so
the numbers in this contract and the numbers the test prints are the same
numbers.

The wider sweep behind the choice, 8 **target** surfaces x {64, 4096} x 3 output
dtypes = 48 combinations, lives in the domain probe's section 7 and is **not**
part of the receipt; it is how the bound was chosen, not what certifies it.
Across all 48 combinations `>final` is **0** — every surface, both row counts,
all three output dtypes. Re-measured after the surface list was corrected to the
target's replicated widths: the three per-dtype maxima above are unchanged
(they come from `(6144,25600)`, which never sharded), and the widest new
surface, `(8192,5120)` at `M=4096` fp32, is the one that moved — 933,348
elements over the default floor and 198,421 over step-only, still **0** over the
final bound.

**Discrimination.** Rolling the weight scales one 32-wide block along `K` —
same shape, same values, only the scale association changed — is asserted, with
its numbers printed on every run:
`correct max_abs=9.8987e-01 (0 elements over the gate) | wrong max_abs=1.1125e+02
(74,794 of 81,920 over) | wrong/correct=112.4x`. Across the 48-combination
sweep the same control puts `29,940` to `1.30e8` elements outside the same
bound at max-error ratios of `97.3x` to `1.43e6x`.

## Preconditions

### Rejected loudly by the op — every one driven, not read from source

All thirteen are exercised by `test_op_rejects_loudly` and each raises
`RuntimeError` with the text quoted here. The wrapper does not repeat any of
them: a guard on a domain the op already checks only changes which error the
caller sees.

| violation | observed message |
|---|---|
| `act` not `float8_e4m3fn` | `act dtype is BFloat16, while Float8_e4m3fn is expected` |
| `weight` not `float8_e4m3fn` | `weight dtype is BFloat16, while Float8_e4m3fn is expected` |
| either scale not `uint8` | `actScale dtype is Char, while Byte is expected` |
| `global_scale` not fp32 | `globalScale dtype is BFloat16, while Float is expected` |
| `act` not contiguous | `act must be contiguous` |
| `weight` not contiguous | `weight must be contiguous` |
| operand on CPU | `act must be a CUDA tensor` |
| `act` not rank 2 | `act must be a 2D tensor [M, K]` |
| `weight` not rank 2 | `weight must be a 2D tensor [N, K]` |
| `K` differs between operands | `act and weight K dims must match: act K=2048, weight K=5120` |
| `K % 32 != 0` | `K (33) must be divisible by MXFP8 block size 32` |
| `N % 32 != 0` | `N (33) must be divisible by 32` |
| unsupported `out_dtype` | `out_dtype must be one of fp16/bf16/fp32 (default bf16).` |

**`M >= 1` belongs to this list, not to the wrapper.** A zero-row call does not
return an empty result: it prints 20 TMA-descriptor failures to stderr and then
raises `RuntimeError: [TensorRT LLM Error][MXFP8xMXFP8 gemm Runner] Failed to
run cutlass MXFP8xMXFP8 gemm. Error: Error Internal` (observed at both
`K=6144,N=25600` and `K=5120,N=512`). It is loud, so it is a caller
precondition rather than a guard. **This matters for dep4:** a rank with zero
logical rows must branch around this call rather than pass an empty activation
to it.

### Not validated by the op — the two the wrapper guards

* **Both scale buffers at least their swizzled length.** See the tables above:
  the op does not check the length, so the outcome of violating this is not one
  behaviour but three — a silently wrong result, an `inf`/`NaN` result, or an
  asynchronous illegal-access fault that ends the CUDA context. None of them is
  a validation error the caller can catch, which is why the length is checked in
  the wrapper. Oversized is safe and must stay accepted. The guard is on the
  length alone, so it also rejects the `M=64` act truncation the op happens to
  survive — a caller cannot know which side of the padding boundary it is on.
* **`global_scale.numel() == 1`.** A two-element tensor is accepted and element
  0 used, so a caller who meant the second element gets a silently wrong
  answer.

A caller inside the coverage table above, violating none of the above, gets a
result within the bound stated under "The tolerance the receipt is measured at"
of an fp32 reference built from the dequantized operands. Outside that coverage
this entry makes no claim.

## Notes

* **The bound method names are snake_case**, while the C++ declares camelCase:
  `run_gemm`, `get_num_configs`, `register_tactic`, `get_cached_tactic`,
  `clear_tactic_cache`. Observed when the first tactic probe followed the
  source names: `AttributeError: __torch__.torch.classes.trtllm.MXFP8GemmRunner
  (of Python compilation unit at: 0) does not have a field with name
  'clearTacticCache'`.
* `torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned` is a *different* op — a Python
  `custom_op` that runs the `AutoTuner` and then calls `MXFP8GemmRunner` with a
  chosen tactic. It is not this entry; a caller wanting it needs its own.
* This checkpoint's dense FP8 weights are stored with a **32x32 tile** scale
  grid (one scale per 32 output channels x 32 input channels), while this op
  wants one scale per `(row, 32-wide K block)`. Expanding the tile grid to
  per-row is a post-load derivation, not forward work.
* **The accumulator is fp32, and with fp32 output that is visible.** Measured
  over all eight **target** surfaces at `M=64` and `M=4096`, fp32 output sits at
  most **1.5e-06 of the tensor's scale** from a single-pass fp32 reference —
  worst case `max_abs=7.63e-04` on a scale of 526 at `(8192,5120)`, `M=4096`,
  which is the widest-`K` surface and so the one that accumulates the most
  terms. That is inside `sqrt(K) * 2**-24` (5.39e-06 at `K=8192`) by 3.7x, and
  roughly 2,700x inside the `2**-8` a bf16 accumulator would give, which is what
  certifies the accumulator as fp32 rather than inheriting the claim from a
  comment. A caller comparing fp32 output against its own matmul must budget
  that accumulation term; with bf16 or fp16 output the coarser cast hides it
  entirely.
* **Probing a short scale buffer is itself hazardous, and the hazard is not
  stable.** A *slice* of the correct buffer shares storage and reads correct
  bytes past its end, which reports "harmless" for a reason that has nothing to
  do with the op. A fresh `.clone()` reads whatever the allocator left, so the
  same call has been measured returning `1.50e+01`, `5.37e+22`, `inf` with
  4,096 NaNs, and `CUDA error: an illegal memory access was encountered` — the
  verdict is stable, the number is not, and one of those outcomes ends the
  process. The test therefore uses an oversized arena with a `0x00` tail (the
  same out-of-bounds read, mapped and deterministic) and the probe runs the
  freshly-allocated tenth-size variant in a child process.
