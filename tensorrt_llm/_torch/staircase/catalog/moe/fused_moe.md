---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 23}
---

# fused_moe

**Wraps** `torch.ops.trtllm.fused_moe` (one call).

## Semantics

One complete mixture-of-experts layer over already-routed tokens: expert
permutation, the grouped FC1 GEMM, the gated activation, the grouped FC2
GEMM, and the routing-weighted combine back to one row per token — all
inside a single call.

Let `T = input.shape[0]`, `H = hidden size`, `I = intermediate size per
expert`, `K = top-k`, `E = fc1_expert_weights.shape[0]` (the experts this
rank holds) and `first = ep_rank * E` (this rank's first *global* expert
id). For every token `t` and slot `j < K`:

```
g = token_selected_experts[t, j]           # a GLOBAL expert id
skip this slot unless first <= g < first + E
e = g - first                              # index into the weight tensors

up   = input[t] @ fc1_expert_weights[e, :I, :].T   (+ fc1_expert_biases[e, :I])
gate = input[t] @ fc1_expert_weights[e, I:, :].T   (+ fc1_expert_biases[e, I:])

# activation_type 5 (Swiglu) and 7 (SwigluBias) — identical on this path:
gate = min(gate, swiglu_limit[e])                  # only when swiglu_limit given
up   = clamp(up, -swiglu_limit[e], swiglu_limit[e])
act  = gate * sigmoid(swiglu_alpha[e] * gate) * (up + swiglu_beta[e])
       # swiglu_alpha defaults to 1.0, swiglu_beta to 0.0, no clamp by default,
       # so the default is plain SwiGLU: silu(gate) * up
# activation_type 6 (Geglu):
act  = gelu(gate) * up

act  = round_to(input.dtype)(act)                  # materialized between the GEMMs
y    = act @ fc2_expert_weights[e].T   (+ fc2_expert_biases[e])
out[t] += token_final_scales[t, j] * y             # fp32 accumulation
```

and finally `out` is cast to `output_dtype`. Both GEMMs accumulate in
fp32; only the FC1 activation result and the final store are in the
low-precision dtype. `token_final_scales=None` means every selected slot
combines with weight `1.0`.

**FC1 row order is `[up | gate]`.** The first `I` rows of
`fc1_expert_weights[e]` are the up projection (trtllm's `w3`, HF's
`up_proj`) and the last `I` rows are the gate projection (trtllm's `w1`,
HF's `gate_proj`) — the activation's sigmoid is applied to the *second*
half. `fc1_expert_biases[e]` is split the same way. Swapping the halves
produces a plausible-looking, entirely different result; nothing detects
it.

**Expert parallelism.** `ep_size`/`ep_rank` do not slice anything: they
only shift which global ids this rank answers for. This rank owns
`[ep_rank * E, (ep_rank + 1) * E)`; slots routed anywhere else contribute
nothing to its output, so summing the outputs of all `ep_size` ranks
reproduces the single-rank result. Certified at `ep_size = 2` with
`E = 8`, and at `ep_size = 4` with `E = 64` over a 256-expert global stack
(all four windows driven, each against its own reference, and their sum
against a 256-expert native-torch reference — see *Notes* for the
numbers). A rank whose window catches no slot at all returns exactly
zero. With `ep_size = 1` (the default) the ids are plain indices into the
weight tensors. `tp_size`/`tp_rank` are a separate axis and are **not**
certified here (held at `1`/`0`): under tensor parallelism a rank passes
its own `I/tp_size` slice of both weight tensors and the reduction across
ranks happens outside this call.

**No routing, no reduction.** The router GEMM and the top-k selection that
produce `token_selected_experts` / `token_final_scales` are the caller's
(a sibling op, `torch.ops.trtllm.renorm_moe_routing_op`, produces exactly
that pair), as are any shared/dense expert branch, the TP all-reduce or EP
gather of this call's output, and the residual add. Nothing is
renormalized inside: the combine weights are used exactly as given (they
need not sum to 1, and may be negative).

**Output.** The call returns a Python list. With `out_tensor=None` it is
`[out]`, one freshly allocated contiguous `[T, H]` tensor in
`output_dtype`. With `out_tensor` given the result is written into that
buffer (every element overwritten, nothing outside its rows touched) and
the returned list is **empty** — the caller uses its own buffer. Inputs
are never mutated.

## Signature

```python
def fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    swizzled_input_sf: bool = True,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1, tp_rank: int = 0,
    ep_size: int = 1, ep_rank: int = 0,
    cluster_size: int = 1, cluster_rank: int = 0,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_int8_woq_per_channel: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    tuner_num_tokens: Optional[int] = None,
    tuner_top_k: Optional[int] = None,
    activation_type: int = 5,
    unpadded_hidden_size: Optional[int] = None,
    out_tensor: Optional[torch.Tensor] = None,
    use_dynamic_fc2_scale: bool = False,
    use_mxfp8_weight_scaling: bool = False,
    # routed-expert LoRA, per-request and slot-indexed families
    fc1_lora_ranks=None, fc1_lora_weight_ptrs=None,
    fc2_lora_ranks=None, fc2_lora_weight_ptrs=None,
    gated_lora_ranks=None, gated_lora_weight_ptrs=None,
    host_request_types=None, host_context_lengths=None,
    lora_max_low_rank: int = 0,
    fc1_slot_lora_ranks=None, fc1_slot_lora_weight_ptrs=None,
    fc2_slot_lora_ranks=None, fc2_slot_lora_weight_ptrs=None,
    gated_slot_lora_ranks=None, gated_slot_lora_weight_ptrs=None,
    token_to_slot=None,
) -> List[torch.Tensor]
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[T, H]` | bf16 or fp16 | contiguous | CUDA |
| `token_selected_experts` | `[T, K]` | **int32** | contiguous | CUDA |
| `token_final_scales` | `[T, K]` or `None` | **fp32** | contiguous | CUDA |
| `fc1_expert_weights` | `[E, 2I, H]`, rows `[up; gate]` | = `input.dtype` | contiguous | CUDA |
| `fc1_expert_biases` | `[E, 2I]` or `None` | = `input.dtype` | contiguous | CUDA |
| `fc2_expert_weights` | `[E, H, I]` | = `input.dtype` | contiguous | CUDA |
| `fc2_expert_biases` | `[E, H]` or `None` | = `input.dtype` | contiguous | CUDA |
| `output_dtype` | scalar | must equal `input.dtype` here | — | — |
| `quant_scales` | `[]` (empty list) | — | — | — |
| `swiglu_alpha` / `swiglu_beta` / `swiglu_limit` | `[E]` or `None` | **fp32** | contiguous | CUDA |
| `ep_size` / `ep_rank` | scalars | Python int, `0 <= ep_rank < ep_size` | — | — |
| `use_fused_finalize` | scalar | bool — both values accepted, giving identical bits; not switchable per call (see *Metadata consumed*) | — | — |
| `activation_type` | scalar | int: 5 (Swiglu), 6 (Geglu), 7 (SwigluBias) | — | — |
| `out_tensor` | `[T, H]` or `None` | = `output_dtype` | contiguous | CUDA |
| returns `[0]` (absent when `out_tensor` given) | `[T, H]` | `output_dtype` | contiguous, newly allocated | same as `input` |

Biases are all-or-nothing: pass both or neither (see *Preconditions*).

### Arguments held inert (not certified)

`input_sf`, `swizzled_input_sf`, `use_deepseek_fp8_block_scale`,
`use_w4_group_scaling`, `use_int8_woq_per_channel`,
`use_mxfp8_act_scaling`, `use_mxfp8_weight_scaling`,
`use_dynamic_fc2_scale` — the quantized-activation/weight paths (fp8
per-tensor and block scale, NVFP4, MXFP4/MXFP8, INT8 weight-only), which
also populate `quant_scales`. `min_latency_mode` — NVFP4-only (it is
rejected here) and returns a different tensor list.
`enable_alltoall`, `tuner_num_tokens`,
`tuner_top_k`, `cluster_size`, `cluster_rank` — multi-rank
dispatch/smart-router paths. `tune_max_num_tokens` — an autotuner bucket
cap; left at its default. `unpadded_hidden_size` — for callers that padded
`H`; left at `None`. The whole LoRA family (`fc1_lora_*`, `fc2_lora_*`,
`gated_lora_*`, `host_request_types`, `host_context_lengths`,
`lora_max_low_rank`, `*_slot_lora_*`, `token_to_slot`). `tp_size` /
`tp_rank` are forwarded but were only exercised at `1` / `0`.

## Metadata consumed

None. The op reads no attention metadata, no KV cache, no registered
layer, and no module state — every tensor and every scalar it uses is an
argument.

Two process-global caches sit behind it. The runner cache is result-neutral;
the tactic cache is not:

- an **autotuner profiling cache** that picks the two GEMM tactics. Cold —
  the state most calls certified here run in — both tactics are the `-1`
  fallback, and nothing is printed (the cache-miss message is a
  `warning_once`, below trtllm's default `error` severity).
  **Tactic choice is not bitwise neutral.** After one `autotune()`
  pass the same inputs at `T=256, H=512, I=256, E=32, K=4` came back
  different in 83447 of 131072 elements (2.96 ulp of the token row's largest
  magnitude), and clearing the cache restored the cold bits exactly. It is a
  different accumulation order, not a different computation — cold and hot
  both pass this entry's gate against the torch reference (2.4 and 2.0 ulp).
  Almost all of the spread is the FC2 GEMM: over the tactic space at that
  shape, `gemm1`'s 209 tactics yield 2 distinct outputs, `gemm2`'s 309 yield
  101. **Serving runs on the hot side**, so a target exercises tactics a cold
  receipt never saw: `PyExecutor` warmup calls `_run_autotuner_warmup`
  (`_torch/pyexecutor/model_engine.py:1103,1278`), which wraps a real forward
  in `autotune()` (`:1293`) unless `enable_autotuner=False` (default `True`,
  `llmapi/llm_args.py:4669`).

  **The hot side is certified at `E=64, H=7168, I=2048, K=8`**, two ways:

  - The *whole tactic space*, driven directly. The tactic population is a
    property of the build, not of the geometry — the count comes from
    `get_tactic_num(gemm_idx)`, which takes no shape argument — and that was
    measured rather than assumed: the tactic space captured at
    `E=64, H=7168, I=2048` has exactly the same size as one captured at
    `E=32, H=512, I=256`, 209 x 309 = 64581 configurations. Every one of
    `gemm2`'s 309 tactics and `gemm1`'s 209 passes this entry's gate at
    `T=256`, and all 309 `gemm2` tactics pass at `T=8192`; none failed to
    run. The distinct-output counts match the small shape's exactly — 101
    from `gemm2`, 2 from `gemm1` — and the worst deviation over the whole
    space is **3.71 ulp** element-wise at `T=256` and **3.98 ulp** at
    `T=8192` (relative RMS 1.42 both), so the 8 / 4 ulp bounds below hold at
    this cell with 2.0x and 2.8x margin.
  - One real `autotune()` pass at that geometry: it fills exactly **28**
    cache entries (2 tunable GEMMs x 14 power-of-2 token buckets, 1 … 8192),
    every one carrying a profiled tactic rather than the `-1` fallback. The
    warm results move the bits (36172254 of 58720256 elements at `T=8192`)
    and still pass at 3.88 ulp; clearing the cache restores the cold bits
    exactly at every token count. The pass costs ~70 s at this geometry.
- a cache of C++ `FusedMoeRunner` objects keyed by
  `(activation dtype, weight dtype, output dtype, quant flags)`, each
  holding a GPU **workspace** buffer that is allocated on first use and
  grown as needed. It is never freed by the op;
  `tensorrt_llm._torch.custom_ops.torch_custom_ops.MoERunner.clear_all_workspaces()`
  releases the workspaces of every cached runner. **`use_fused_finalize` is
  baked into that object** and is not part of the key, so the first call in
  a process for a given dtype set fixes it for every later call — a caller
  cannot flip it per call, and the entry's own test therefore runs on a
  runner built with `True`. On this bf16 path the value makes no difference
  either way: two fresh processes driving `True` then `False`, and `False`
  then `True`, produced four bitwise identical outputs at
  `T=2048, H=512, I=256, E=32, K=8` (a probe, not part of the test), and
  `True` and `False` agree bitwise inside the test at
  `T=2048, E=64, H=7168, I=2048, K=8`.

Two identical calls return bitwise identical results **within one tuner
state** (verified at `T=256, H=512, I=256, E=32, K=4`, cold and hot
separately; and at `E=64, H=7168, I=2048, K=8` for `T` in {1, 256, 8192},
where a cold call repeated after an intervening tuning pass and a cache
clear returned the same bits); across a cold/hot change they differ by the
ulp above.

## Preconditions

Shapes and layout:

- `input` is a **2D contiguous CUDA** tensor with `T >= 1`. 3D input raises
  `input must be 2D`; a strided view raises `input must be contiguous`; a
  CPU tensor raises `input must be a CUDA tensor`; `T == 0` fails a C++
  assertion (`Assertion failed: input_activations`).
- `H % 8 == 0` and `I % 8 == 0`, else `hidden_size <n> must be divisible
  by 8 for weights` / `inter_size <n> must be divisible by 8 for weights`.
  `H = 8` and `I = 8` work.
- `fc1_expert_weights` is `[E, 2I, H]` and `fc2_expert_weights` is
  `[E, H, I]`, both 3D, contiguous, CUDA, same dtype as `input`, same `E`.
  Violations raise (`must be 3D`, `must be contiguous`,
  `must be a CUDA tensor`, `fc1_expert_weights and fc2_expert_weights must
  have the same number of experts.`). Certified: `E` in
  {1, 2, 7, 8, 16, 32, **64**, 128, 256} and `(H, I)` in {(8,8), (16,8),
  (128,64), (192,96), (256,128), (512,256), (1024,512), (2048,768),
  **(7168,2048)**}. The two enumerations are not a product: `(7168, 2048)`
  was driven at `E = 64` only, and `E = 64` only at `(7168, 2048)`. Nothing
  in the permutation or gather path keys on the expert count, or on which
  index a given expert sits at: permuting the 64-expert stack and
  relabelling the ids to match reproduces the output **bitwise**.
- `token_selected_experts` is `[T, K]` **int32** contiguous CUDA;
  `token_final_scales`, when given, is `[T, K]` **fp32** contiguous CUDA
  with the same `K`. Other dtypes raise (`token_selected_experts dtype is
  Long, while Int is expected`, `token_final_scales.value() dtype is
  BFloat16, while Float is expected`), as do mismatched `T` or `K`.
  Certified `K`: 1, 2, 3, 4, 8, 16. **A token's `K` ids must be distinct
  when `T > 256`** — a repeat there is accepted and then computed wrong, or
  faults; see *Notes*. At `T <= 256` a repeat is honoured once per slot, so
  that expert's output is added twice with its two combine weights.
- Biases are optional but **paired**: passing exactly one of
  `fc1_expert_biases` / `fc2_expert_biases` raises
  `RuntimeError: bad optional access`. Both must be `[E, 2I]` and `[E, H]`
  contiguous CUDA tensors in `input.dtype` (fp32/fp16 bias against a bf16
  input raises).
- `out_tensor`, when given, must be a contiguous CUDA tensor of shape
  exactly `[T, H]` in `output_dtype`; wrong dtype, wrong shape, or a
  strided view each raise. A leading row-slice of a larger contiguous
  buffer is a valid `out_tensor` — rows past `T` are left bitwise
  untouched.

Dtypes and modes:

- Activations and weights must be the **same** dtype: bf16 or fp16.
  fp32 activations, or weights whose dtype differs from `input`, raise
  `Could not construct fused moe op with the requested input combination
  ...`.
- On this unquantized path `output_dtype` **must equal** `input.dtype`.
  A mismatch does **not** raise — see *Notes*. The wrapper asserts it.
- `quant_scales` must carry the scale set of whatever quantization is
  requested; on the unquantized path pass `[]` (a non-empty list is
  accepted and ignored). fp8-e4m3 inputs with `[]` raise `Expecting 4
  quant scales for fp8 quantization`.
- `swiglu_alpha` / `swiglu_beta` / `swiglu_limit`, when given, are fp32
  CUDA tensors with exactly `E` elements (`swiglu_alpha must have
  num_experts_on_rank elements.`, `... must be a CUDA tensor`, `... dtype
  is BFloat16, while Float is expected`).
- `activation_type` must be a gated type (5 Swiglu, 6 Geglu, 7
  SwigluBias) for the `[E, 2I, H]` FC1 layout. The non-gated values of the
  same enum (1 Identity, 2 Gelu, 3 Relu, 4 Silu, 8 Relu2) — and any
  unrecognised value — expect `[E, I, H]` instead, so with a gated layout
  they raise `fc1_expert_weights inter size must be equal to
  fc2_expert_weights inter size.` (verified for 1, 2, 4, 8 and 99).
- `0 <= ep_rank < ep_size` (`Assertion failed: ep_rank < ep_size`).
- `min_latency_mode=True` is NVFP4-only and fails here with `Assertion
  failed: use_fp4 == true`. `cluster_size > 1` raises `smart_router is
  supported in min_latency mode`.
- `tuner_num_tokens` and `tuner_top_k` must both be `None` unless
  `enable_alltoall=True`, and both must be set when it is; the op's own
  Python `assert` fires otherwise (bare `AssertionError`).
- Any LoRA tensor implies `lora_max_low_rank > 0` (`MoE LoRA requires
  lora_max_low_rank > 0; got 0`), and the per-request and slot-indexed
  families are mutually exclusive.

A caller violating none of the above gets the result described under
*Semantics*, to within a few low-precision ulps (see *Notes*).

## Notes

- **`output_dtype != input.dtype` is a silent wrong answer.** On the
  unquantized path the output buffer is allocated with `output_dtype`
  while the epilogue stores elements in the activation dtype, so the
  caller reads reinterpreted bits: bf16 input with `output_dtype=fp32`
  returns finite, plausibly-scaled values that are pairs of bf16 results
  glued into fp32 words (element `j` carries roughly the true element
  `2j+1`). No error, no NaN. `use_fused_finalize=False` behaves the same.
  The wrapper carries the one guard this entry needs: when
  `input.dtype in (bf16, fp16)` and `fc1_expert_weights.dtype ==
  input.dtype`, `output_dtype` must equal `input.dtype`. The guard is
  scoped so it never fires on the quantized paths, where a differing
  `output_dtype` is the norm.
- **Expert ids outside this rank's range are silently dropped**, not
  clamped and not rejected: with `ep_size=1, E=8`, ids `8`, `100` and `-1`
  each contribute nothing, and the token's other slots still combine
  normally. A routing bug therefore shows up as a quietly weaker token,
  never as an error. Checking ids costs a device-to-host sync, so the
  wrapper does not.
- **Three quantization flags are accepted and ignored** on the bf16 path:
  `use_w4_group_scaling`, `use_mxfp8_act_scaling` and
  `use_dynamic_fc2_scale` return bitwise the unquantized result. The other
  three (`use_deepseek_fp8_block_scale`, `use_int8_woq_per_channel`,
  `use_mxfp8_weight_scaling`) raise.
- **Numerics.** Against a native-torch reference that consumes the same
  low-precision operands (fp32 GEMM accumulation, the FC1 activation
  rounded to the input dtype, fp32 combine), the kernel's worst
  element-wise deviation over the covered configurations was **2.8 ulp of
  the token row's largest magnitude** (bf16 ulp = 2^-8, fp16 ulp = 2^-11)
  and its relative RMS deviation **1.2 ulp**. Both were calibrated on the
  fallback tactic; walking the *whole* tactic space reaches **3.98 ulp**
  element-wise and **1.43 ulp** RMS (`gemm2`, `T = 8192`, at
  `E=64, H=7168, I=2048`) — still inside the test's 8 ulp / 4 ulp gates,
  with 2.0x and 2.8x margin. That is rounding noise, not
  a different computation: the same comparison against a gate/up-swapped
  reference lands at 176+ ulp element-wise and ~190 ulp RMS at the small
  shapes and 284 / 197 at `E=64, H=7168, I=2048`, and against a
  one-expert-dropped reference at 997 / 212 there (measured while
  calibrating the bound; the entry's test asserts only that such a
  reference is rejected). Default `torch.testing.assert_close` tolerances
  do not fit this op — their bf16 `atol` of 1e-5 is three orders of
  magnitude below one output ulp of a two-GEMM chain.
- **Scale is inert.** `(H, I) = (7168, 2048)` at `E = 64` is 3.5x the
  hidden size and 2.7x the intermediate size of anything else certified
  here, and behaves exactly like the small shapes: same tactic population,
  same distinct-output counts, same ulp band, no tactic failing to run.
  The deviation grows only with token count, and only slightly — 1.12 ulp
  at `T = 1` to 2.54 at `T = 8192`, the same trend the smaller shapes show.
- **Four-way expert parallelism**, measured at `E = 64` per rank over a
  256-expert global stack with global ids in `[0, 256)`, at
  `T` in {1, 1024, 8192}: each rank's output matches its own reference to
  at most 2.69 ulp, and the four outputs summed in fp32 match a 256-expert
  native-torch reference to 3.53 ulp element-wise / 1.34 ulp RMS. The sum
  sits slightly wider than any single rank because each rank rounds its own
  partial to bf16 before the add. A rank whose window catches no slot
  returns exactly zero, and the same weights driven under the wrong
  `ep_rank` land 90x outside the RMS gate — so the tiling check is a real
  check and not a formality.
- **Repeated expert ids in one token's row are a defect past `T = 256`,
  not a supported input.** At `T <= 256` a repeat is honoured once per
  slot, as *Preconditions* says. At `T >= 257` the op accepts the input and
  then either returns values ~200-460 ulp wrong on essentially every row,
  or dies with `CUDA error: an illegal memory access was encountered` from
  inside the call. Which of the two happens depends on whether the
  out-of-bounds address is mapped; both were seen at the same shape.
  `compute-sanitizer` names the fault: an invalid 4-byte global read in
  `tensorrt_llm::_v1::kernels::cutlass_kernels::finalizeMoeRoutingKernel`,
  launched from `finalizeMoeRoutingKernelLauncher` inside
  `CutlassMoeFCRunner<...>::gemm2`. The boundary is exactly 256 tokens
  (`T = 256` clean, `T = 257` faults) and matches the largest block size the
  single-CTA expert-map builder is instantiated for —
  `fusedBuildExpertMapsSortFirstTokenDispatch<256, …>` in
  `libth_common.so`, whose fallback is
  `threeStepBuildExpertMapsSortFirstToken`. `use_fused_finalize` does not
  change it. This is **not specific to the new geometry**: reproduced at
  `(E,H,I,K) = (8,256,128,4)`, `(32,512,256,8)` and `(128,2048,768,8)` —
  cells this entry already certified — as well as at `(64,7168,2048,8)`.
  Top-k selection returns distinct ids by construction, so a caller taking
  its ids from a selection op cannot reach this; a caller synthesizing
  `token_selected_experts` itself can.
- `activation_type=7` (SwigluBias) computed bitwise the same result as
  `5` (Swiglu) for every input tried, with and without
  `swiglu_alpha/beta/limit`.
- Geglu (6) matches a `gelu(gate) * up` reference; the tanh-approximate
  and exact gelu variants are indistinguishable at bf16 output resolution,
  so which one the kernel uses is **not** pinned by this entry.
- `T = 8192` (one token past the default `tune_max_num_tokens` bucket cap)
  is certified, at `(2048, 768, E=128)` and at `(7168, 2048, E=64)`;
  `T = 8193` was also observed correct during an earlier run's probing but
  is not in the entry's test.
- **What `(7168, 2048)` at `E = 64` does *not* cover.** bf16 only (no
  fp16), no biases, no `swiglu_alpha`/`beta`/`limit`, `activation_type = 5`
  only (no Geglu, no SwigluBias), `tp_size = 1`, and `K = 8` only. Those
  axes are certified at the smaller shapes and were not re-driven here.
- Sibling ops in this build address the same job differently:
  `torch.ops.trtllm.moe_custom_op` takes a registered layer by string id
  instead of explicit weights, and `torch.ops.trtllm.fp8_block_scale_moe_runner`
  / `torch.ops.trtllm.fp4_block_scale_moe_runner` are the trtllm-gen
  block-scaled MoE entry points. Catalog membership is `index.yaml`'s fact
  alone.
