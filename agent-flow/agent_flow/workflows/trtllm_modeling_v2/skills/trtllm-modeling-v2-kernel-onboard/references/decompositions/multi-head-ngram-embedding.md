# Decomposing `multi_head_ngram_embedding` (Qwen4-Exp PLE n-gram table read)

Why the requested catalog entry `embedding/multi_head_ngram_embedding` was not
written, what the step actually is, and which two calls serve it.

Derived on: tensorrt_llm 1.3.0rc26 | torch 2.12.0a0+5aff3928d8.nv26.05 |
NVIDIA GB300, `torch.cuda.get_device_capability()` `(10, 3)` → **sm_103** — the
stamp on the observations below, not a pin to assert. Receipts are keyed by GPU
architecture and nothing else, and each constituent carries its own in its own
contract; this document carries none.

Checkpoint: `Qwen3.8-Flash-Next-NVFP4`, `model_type` `qwen4_exp`.

## 1. Why decompose it

The request was one entry for the step the run's `vocabulary.yaml` states as
"PLE multi-head n-gram embedding lookup with per-head offsets and per-tensor
FP8 dequantization". Written out at this checkpoint's widths, the step is

```
rows    : [T, 16] int64 global row indices          (produced upstream)
table   : [320001536, 160] float8_e4m3fn            (the n-gram table)
scale   : scalar bfloat16, 0.00019931793212890625   (per-tensor dequant scale)

ple[t, 160*h : 160*h + 160] = table[rows[t, h]] * scale        for h in [0, 16)
ple                          : [T, 2560] bfloat16
```

That is **two kernel-launching calls**, not one: a row gather, then an
elementwise dequantize. There is no fused gather-and-dequantize anywhere in
this build — the registered op set was swept exhaustively (294 `trtllm::` /
`tensorrt_llm::` schemas after forcing the lazy registrations by importing
`tensorrt_llm._torch.custom_ops`) and the only ops whose names match
`dequant|embed|gather|lookup|e4m3` are listed in §3; none of them indexes.

Profiled directly, the two-call form launches **exactly two CUDA kernels**:

```
at::native::vectorized_gather_kernel<16, long>
tensorrt_llm::_v1::common::scaleMatrix<(QuantizeMode)1, false, __nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>
```

Both are legal single entries under the catalog's admission rule, and one of
them is already in the catalog. Writing a wrapper that chains them would be
Python-level composition of two available entries — the case
`references/entry-spec.md` names as forbidden and `SKILL.md`'s red-flag table
calls "the exact anti-pattern the catalog bans". Hence: **macro; onboard
nothing**.

Note the contrast with the sibling step, the n-gram *hash*, which this run did
land as a composed entry (`catalog/embedding/ngram_hash.py`). That one composes
roughly thirty launches out of integer ops for which the catalog has **no**
mirrors and the engine has **no** op, so composing was the only route to a
closed vocabulary and its contract says so. This step is the opposite case:
both constituents exist as atomic surfaces.

### 1.1 The named tier-2 candidate computes a different function

The dispatch sourced this step at tier 2 on
`tensorrt_llm._torch.modules.engram.engram.MultiHeadEmbedding`. That class
imports on this host and was exercised here; it does **not** serve, on four
independent axes, each measured rather than read:

1. **It cannot hold the checkpoint's table.** Constructing it with
   `dtype=torch.float8_e4m3fn` raises
   `NotImplementedError: "normal_kernel_cpu" not implemented for 'Float8_e4m3fn'`
   — it allocates an `nn.Embedding`, whose initializer is `normal_`. A bare
   `torch.nn.Embedding(dtype=torch.float8_e4m3fn)` raises the same thing. Its
   table is therefore a float parameter: at this checkpoint's row count a
   bfloat16 `nn.Embedding` over 320001446 rows × 160 is **95.37 GiB**, against
   **47.68 GiB** for the fp8 table the checkpoint ships.
2. **It re-applies the per-head offsets.** Its `forward` adds a registered
   `offsets` buffer to the indices it is given. This run's hash step already
   returns *global* rows (its contract: "The returned indices are global"), so
   feeding them in double-applies. Measured on a 4-head instance:
   `m(global_indices) == table[global_indices]` is **False**;
   `m(local_indices) == table[global_indices]` is **True**.
3. **It has no dequantization at all** — no scale appears in its `forward`.
4. **It silently clamps out-of-range indices** to the last row (its `forward`
   clamps to `num_embeddings - 1`), which turns a wrong index into a plausible
   embedding. The raw gather this step needs instead **raises** (§4.3), and
   losing that is a downgrade, not a convenience.

It is also an `nn.Module` holding its own parameter (`embedding.weight`) rather
than a stateless callable, so there is no callable to wrap that takes the
checkpoint's table as an argument.

### 1.2 Ruling R1 is not what excludes the tier-2 path here

`workspace/rulings.md` R1 embargoes the callables under
`tensorrt_llm/_torch/modules/qwen4_exp/`, which is where this checkpoint's own
PLE implementation lives, and it was honoured: nothing under that directory was
opened, and no constituent below routes through it. But the tier-2 candidate
named in the dispatch (`engram.py`) is **not** embargoed, and it was read and
run. It fails on its own merits, per §1.1. The two are separate findings.

## 2. Granularity and sourcing policy

See `../decomposition.md`. Both constituents below land at preference level 1
(a registered `torch.ops` op) or level 3 (a torch-native mirror already in
`catalog/torch/`); no policy exception is needed.

## 3. The resulting vocabulary

Entry paths are relative to
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/`.

| # | role | callable | state |
|---|---|---|---|
| 1 | gather 16 table rows per token | `torch.nn.functional.embedding` | **existing entry**: `torch/embedding.py` |
| 2 | per-tensor e4m3 → bfloat16 dequantize | `torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor` | **not onboarded**; proposed category `quantization/` |
| 3 | `[T, 16, 160] → [T, 2560]` (a view) | `torch.reshape` | **existing entry**: `torch/reshape.py` |

Symbol resolution, checked on this host before citing:

- `tensorrt_llm::dequantize_e4m3_per_tensor(Tensor input, Tensor scales) -> Tensor`
  is present in the registered-schema list, and
  `torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor` **resolves**.
- `torch.ops.trtllm.dequantize_e4m3_per_tensor` does **not**:
  `AttributeError: '_OpNamespace' 'trtllm' object has no attribute 'dequantize_e4m3_per_tensor'`.
  The namespace is `tensorrt_llm`, exactly as for the catalog entry
  `quantization/static_quantize_e4m3_per_tensor`, whose contract already names
  this op as "the inverse, multiplying".
- `catalog/torch/embedding.py`, `catalog/torch/reshape.py` and
  `catalog/embedding/ngram_hash.py` all import.
- `tensorrt_llm._torch.modules.engram.engram.MultiHeadEmbedding` imports (§1.1).

Ops in this build whose names match `dequant|embed|gather|lookup|e4m3`, for the
record, so the next reader does not re-sweep: `tensorrt_llm::` carries
`dequantize_e4m3_{per_tensor,weight,activation}`,
`quantize_e4m3_{per_tensor,weight,activation}`,
`static_quantize_e4m3_{per_tensor,weight,activation}` and the `mxe4m3` host
pair; `trtllm::` carries `embedding_skip_forward`, `mxfp4_dequantize_unswizzled`,
`indexer_k_cache_gather_op`, `nvfp4_mla_kv_cache_gather`,
`nvfp4_mla_context_kv_cache_gather` and the gather-fused grouped GEMM. None
combines an index lookup with a dequantize.

## 4. Assembly — the mainline path

Packed-token form, which is what a chunked-prefill engine produces:

```
rows      = ngram_hash(...)                       # [B, T, 16] int64, global rows
rows2     = reshape(rows, [-1, 16])               # [N, 16]        torch/reshape
gathered  = embedding(rows2, ngram_table)         # [N, 16, 160] fp8  torch/embedding
deq       = dequantize_e4m3_per_tensor(gathered,  # [N, 16, 160] bf16  NOT ONBOARDED
                                       scale_3d)
ple       = reshape(deq, [N, 2560])               # [N, 2560]      torch/reshape
```

`scale_3d` is the checkpoint's `weight_scale` broadcast to shape `[1, 1, 1]` —
see §4.2 for why its **rank** matters.

Run end to end on this host at the checkpoint's own constants (a `[2, 130]` id
history, 128 query tokens per row, `eos_token_id` 248044, the three
`layer_multipliers`, the sixteen `ngram_heads_vocab_sizes` and
`ngram_heads_offsets`) against a real-size `[320001536, 160]` fp8 table:

- row indices came out in `[95556, 319978804]`, all `< 320001446`;
- the result is `[256, 2560]` bfloat16 — `ple_embed_dim` exactly;
- it is **bitwise equal** to an independent per-row reference built from the
  payload rows directly;
- head `h` occupies columns `[160h, 160h + 160)`, checked for all 16 heads
  against an independent per-head gather.

At `T` in {1, 8, 4096} over a smaller table the pipeline is likewise **bitwise
equal** to `(table[rows].to(torch.float32) * scale).to(torch.bfloat16)`.

### 4.1 Where the table comes from (assembler-side, load time)

The checkpoint stores the table as **128 separate tensors**, keys
`...ple_embedding.ngram_embedding.shard_{0..127}.weight` in
`model-fp8-mtp-ple.safetensors`, each `[2500012, 160]` `F8_E4M3`, plus one
`...ngram_embedding.weight_scale`, `BF16`, shape `[1]`. `config.json`'s
`split_ngram_parts` is 128 and `128 * 2500012 == 320001536`.

The sharding is **contiguous chunks**, not interleaving — global row `r` lives
in shard `r // 2500012` at local row `r % 2500012`. Established from the
checkpoint alone, without reading any implementation: the sixteen
`ngram_heads_vocab_sizes` sum to 320001446, so rows 320001446…320001535 are
padding (`make_ngram_vocab_size_divisible_by` is 128). Scanning the last 200
rows of shards 0, 1, 37, 38, 39, 126 and 127, the first six are fully non-zero
to their last byte, while shard 127's rows go from "all 160 bytes non-zero" to
"about half the bytes zero" at local row 2499922 — **global row 320001446**,
the first padding row, to the row. Interleaving would have put the boundary in
90 different shards, including 38 and 39.

Consequences the assembler owns:

- A loader that allocates one contiguous `[128, 2500012, 160]` (or
  `[320001536, 160]`) fp8 buffer and fills it shard by shard produces a table
  that `torch.nn.functional.embedding` takes directly with **global** indices:
  `t3.reshape(-1, 160)` and the flat table were verified to be the same storage
  and byte-identical.
- The **padding rows are not zero and are not meaningful** — they hold what
  looks like uninitialized data. Nothing in this path range-checks against
  320001446; the hash step's own modulo is what keeps indices below it.
- `torch.nn.functional.embedding` requires a **2-D** weight
  (`RuntimeError: 'weight' must be 2-D`), so the `[128, …]` form must be
  reshaped — free, same storage.

### 4.2 The scale argument (post-onboarding detail, measured here)

`dequantize_e4m3_per_tensor` requires `scales.dim() == input.dim()` with every
`scales` dimension equal to 1; anything else raises
`RuntimeError: Expected scales.dim() == input.dim() to be true, but got false.`
or `Expected scales.size(i) == 1 to be true, but got false.`. So a rank-3
gathered tensor needs a `[1, 1, 1]` scale and a rank-4 one needs `[1, 1, 1, 1]`
— confirmed both ways.

The op **multiplies** (`out = input * scales[0]`, bitwise equal to
`input.to(torch.float32) * scale` with a float32 scale), and its **output dtype
follows the scales dtype**: float32 → float32, bfloat16 → bfloat16, float16 →
float16. Passing the checkpoint's bfloat16 `weight_scale` therefore yields
bfloat16 directly, with no cast anywhere in the path.

The multiply direction is independently corroborated by the checkpoint: `448`
(e4m3's max finite) times 0.00019931793212890625 is 0.08929443359375, and a
50 000-row sample of shard 0 dequantizes to mean −1.84781e-05, std 0.00747576,
range [−0.0478363, 0.0510254] — ordinary embedding weights. Dividing would give
a std of about 37.5.

### 4.3 Hard constraints a caller must respect (measured, this host)

On constituent 1, `torch.nn.functional.embedding` over an fp8 table:

- An out-of-range **or negative** row index is **not** silently clamped. It
  trips a device-side assert:
  `/opt/pytorch/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [0,0,0], thread: [0,0,0] Assertion 'ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"' failed.`
  surfacing as `AcceleratorError: CUDA error: device-side assert triggered`.
  This **poisons the CUDA context** for the rest of the process, so a test that
  provokes it must do so in its own process.
- Index dtype must be **Long or Int**; Short and Byte raise
  `Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int`.
  The checkpoint's largest legal row, 320001445, fits in int32.
- Index and table must share a device. A CUDA index against a pinned-CPU table
  raises `RuntimeError: Expected all tensors to be on the same device, but got
  index is on cuda:0, different from other tensors on cpu`. `pin_memory()` on an
  fp8 CPU tensor works and a **CPU** index against a CPU table gathers fine —
  so a host-offloaded table can be read on the host but not indexed from the
  device by this call. Anything host-offloaded needs a different path; that is
  outside both constituents.
- A `[320001536, 160]` fp8 table spans 51 200 245 760 bytes, past `2**31`;
  gathers at rows 0, 2500011, 2500012, 20000002, 160000374, 300001275,
  320001444 and 320001445 were all bitwise correct.

On constituent 2, `dequantize_e4m3_per_tensor` (these belong in its contract
once it is onboarded; recorded here so its smith does not re-derive them):

| condition | result |
|---|---|
| rank < 2 input | `RuntimeError: Invalid dim. The dim of input should be greater than or equal to 2` |
| non-contiguous input | `RuntimeError: input must be contiguous` |
| zero-row input | `RuntimeError: input should not be empty tensor` |
| non-fp8 input | `RuntimeError: Invalid datatype. input must be Int8 (Fp8)` |
| float64 scales | `RuntimeError: Invalid datatype. input must be BF16/FP16/FP32` (the message says "input"; the offending tensor is `scales`) |
| CPU scales | `RuntimeError: scales must be a CUDA tensor` |
| CPU input | `NotImplementedError: Could not run 'tensorrt_llm::dequantize_e4m3_per_tensor' with arguments from the 'CPU' backend` |
| `scales == 0.0` | **accepted**, output all zeros, no signal |
| `scales < 0` | **accepted**, every sign flips |

It is deterministic (two calls bitwise equal), leaves its input untouched, and
allocates its output. The gather output is contiguous and so is the dequantized
one, so the final reshape to `[N, 2560]` is a view — measured on the bfloat16
result, whose `flatten(-2)` has the same `data_ptr`.

### 4.4 CUDA graphs and cost

The two-call pipeline **captures and replays** under `torch.cuda.CUDAGraph`,
bitwise equal to eager against fresh indices copied into the static input.

Wall time per call, `[T, 16]` indices into a `2**23`-row fp8 table, mean of 50
iterations on sm_103:

| T | `embedding` + `dequantize_e4m3_per_tensor` | `embedding` + `.to(bfloat16)` + `* scale` |
|---|---|---|
| 8 | 0.0394 ms | 0.0721 ms |
| 512 | 0.0373 ms | 0.0716 ms |
| 4096 | 0.0469 ms | 0.1031 ms |

Over the real `[320001536, 160]` table at T=4096 the pipeline measured
0.0485 ms. The trtllm op is the faster of the two dequantize routes at every
size measured because it is one kernel where torch needs two; this is a
measurement, not a gate.

## 5. Certification status

**Nothing here is certified.** No entry was created, no test was written and no
receipt was recorded. Every number above comes from probe scripts, not from a
catalog test, and the probes are not part of any entry.

What the child smith for constituent 2 still owes: a contract, a GPU test at
`tests/unittest/_torch/modeling_v2/quantization/test_modeling_v2_dequantize_e4m3_per_tensor.py`,
and an `sm_103` receipt. Suggested coverage beyond §4.3: the whole 254-code
finite e4m3 domain against a native-torch reference (`x.to(torch.float32) *
scales[0]`), all three scales dtypes, ranks 2 through 4, and the boundary
`448 * scales[0]`.

Constituent 1 (`torch/embedding.py`) is a hand-added mirror and by catalog
policy carries no contract, no test and no receipt; its fp8 behaviour is
therefore **unconstrained by design** and the observations in §4.3 live here
rather than in that entry.

## 6. Out of scope

- **Host offload.** This checkpoint's table is 47.68 GiB and the device here
  has 275.9 GiB free, so it fits; whether the target keeps it resident is the
  target's decision, and §4.3 records only that a device-side gather against a
  CPU-resident table raises.
- **Tensor parallelism.** Single-rank only (`parallel: tp1`). Nothing above
  shards the table across ranks.
- **The head-to-column order.** That head `h` occupies columns
  `[160h, 160h + 160)` follows from `reshape` of `[T, 16, 160]` and from the
  head order the hash step fixes (`[2-gram block | 3-gram block]`, 8 heads
  each). It has **not** been checked against the trained `key_proj` /
  `value_proj` weights; that is an assembly-time comparison against the
  `modeling_v2=off` oracle, not something either constituent can settle.
- **A second PLE layer.** `ple_layer_ids` is `[2]`; one layer, one table, one
  scale.
