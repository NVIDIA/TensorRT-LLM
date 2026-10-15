# Decomposing `mla_custom_op_inplace`

How TRT-LLM's MLA block macro was broken into Python-level atomic catalog
entries, how each batch mode reassembles from them, and what the current
coverage does not include.

This document has two roles. It is the reference record of the completed
`mla_custom_op_inplace` decomposition, and it is the **norm and exemplar
for handling any macro op**: when an onboarding request turns out to
target a macro, onboard nothing — produce a decomposition report shaped
like this document at `.claude/skills/staircase-kernel-onboard/references/decompositions/<op>.md`
instead. What such a report must contain is specified at the end of §2.

Derived on: tensorrt_llm 1.3.0rc21 | torch 2.11.0+cu130 | NVIDIA B200
(sm_100) — the stamp on the observations below, not a pin to assert.
Receipts themselves are keyed by GPU architecture and nothing else; each
entry cited below carries its own, in its own contract.

## 1. Why decompose it

`torch.ops.trtllm.mla_custom_op_inplace` looks like one op — a single
registered custom op, one node in an FX graph — but its body is a **macro**:
it looks up a registered `MLA` module by `layer_idx` and runs
`MLA.forward_impl`, which is a full Python-level pipeline:

- module calls: `kv_a_proj_with_mqa`, `q_a_layernorm` / `kv_a_layernorm`
  (dual-stream), `q_b_proj`;
- tensor glue: `split`, `concat`, slicing, scratch allocation;
- a **Python-level phase dispatch** (context vs. absorption generation;
  context further splits into fresh / cached-KV / chunked-prefill);
- further registered ops inside each branch (`bmm_out`,
  `mla_rope_generation`, cache load/merge ops, the FMHA).

The catalog's admission rule is that every entry must be a **Python-level
indivisible op**: one callable whose interior is C++/CUDA only. Multiple
kernel launches inside one call are fine; Python-level composition is not.
Vertical customization depends on this granularity — a black-box macro
cannot have interior steps swapped, measured, or certified independently,
and its hidden control flow cannot be specialized per target. "FX-graph
atomic" is therefore not an admission criterion; "Python-level atomic" is.

The macro itself was consequently rejected as an entry (an early onboarding
run produced a working contract + test for it, which was deliberately
reverted). Its decomposition map was kept and each constituent was
onboarded individually.

## 2. Granularity and sourcing policy

Moved to `../decomposition.md`, which is where the policy belongs — this file
is the worked example, not the rule. Read it there before judging a boundary.

## 3. The resulting vocabulary

Shared (all batch modes):

| role | entry |
| --- | --- |
| fused `q_a`/`ckv`/`k_pe` projection, `q_b` projection, `kv_b` expansion, `o_proj` | `gemm/cublas_mm` |
| `q_a` / `kv_a` RMS norm | `norm/flashinfer_rmsnorm` |
| projection output split, latent `[ckv \| k_pe]` concat, views | `torch/split`, `torch/concat`, `torch/reshape` |
| `k_pe` head broadcast, `k_nope` slice fill | `torch/expand`, `torch/copy_` |
| scratch / output allocation | `torch/empty`, `attention/create_mla_outputs` |

Phase-specific:

| role | entry |
| --- | --- |
| decode: absorption BMMs (`q_nope x k_b_trans`, `attn x v_b`) | `gemm/bmm_out` |
| decode: q_pe RoPE + latent append + FMHA scheduler-buffer fill | `attention/mla_rope_generation` |
| cached/chunked prefill: new-token RoPE + latent append + q assign | `attention/mla_rope_append_paged_kv_assign_q` |
| cached prefill: full `[past + new]` latent gather | `attention/load_paged_kv_cache_for_mla` |
| chunked prefill: per-loop latent slice gather | `attention/load_chunked_kv_cache_for_mla` |
| chunked prefill: log-sum-exp fold of partial passes | `attention/merge_chunked_attention_for_mla` |
| attention core, all phases (5 certified configurations) | `attention/thop_attention` |

<!-- PORT: verify — three entries named in this section are not in the in-tree
     catalog: `attention/create_mla_outputs`,
     `attention/load_chunked_kv_cache_for_mla` and
     `attention/merge_chunked_attention_for_mla` (the chunked-prefill trio).
     The rows are kept because this document is the record of the completed
     decomposition; confirm whether those entries were dropped in the move or
     are still to be ported, and whether §4.4 is therefore currently
     unreachable. -->

Entry paths in this document are relative to
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/`.

## 4. Assembly by batch mode

Shared front section (every mode):

```
[q_a | ckv | k_pe] = cublas_mm(hidden, W_a.t())        # then torch/split
q_a = flashinfer_rmsnorm(q_a, w_q_ln);  ckv = flashinfer_rmsnorm(ckv, w_kv_ln)
q    = cublas_mm(q_a, W_qb.t())
latent = concat([ckv, k_pe])
```

Phase dispatch (context vs. generation, and which context flavor) is plain
Python over metadata reads — permitted outside entries by catalog rules.
`o_proj` (`cublas_mm`) closes every mode.

### 4.1 Plain prefill (fresh context)

```
kv = cublas_mm(ckv, W_kvb.t()); k_nope, v = split(kv)
k  = empty(...); copy_(k[..., :nope], k_nope); copy_(k[..., nope:], expand(k_pe))
thop_attention(context_only: separate q/k/v, in-kernel GPT-J RoPE on q/k_pe,
               latent-cache append, causal FMHA)   -> output
```

### 4.2 Decode (absorption generation)

```
q_nope, q_pe = split(q);  fused_q, scratch = empty(...)
bmm_out(q_nope, k_b_proj_trans, fused_q[..., :C])        # absorbed q
mla_rope_generation(fused_q, q_pe, latent, ...)          # RoPE + append +
                                                         # scheduler buffers
thop_attention(generation_only: latent MQA over the pool,
               consumes the scheduler buffers)           -> attn latent
bmm_out(attn_latent, v_b_proj, out)                      # v_b absorption
```

**Whether the two producers may run concurrently depends on the pool's
dtype, and getting it wrong is a race rather than an error.**

On a **bf16** latent pool they are compatible by design: `fused_q` is
written disjointly (`[..., :C]` by the BMM, `[..., C:]` by the RoPE op),
so the two can overlap.

On an **fp8-e4m3** latent pool that is false. `mla_rope_generation` does
not write `fused_q` at all there — it *reads* `fused_q[..., :C]` to build
`quant_q_buffer`, which is what the decode FMHA then takes its query from.
So the absorbed-q BMM must have **finished** before the RoPE call starts.
A decode assembled from the bf16 reading and then switched to an fp8 KV
cache builds its quantized query out of uninitialized memory, silently.
Both halves are certified in `catalog/attention/mla_rope_generation.md`
(*Signature*, the `fused_q` row).

### 4.3 Cached-KV prefill

```
mla_rope_append_paged_kv_assign_q(q, latent, ...)   # RoPE q/k_pe in place,
                                                    # append new tokens
ckv_full, k_pe_full = load_paged_kv_cache_for_mla(...)   # gather [past+new]
kv = cublas_mm(ckv_full, W_kvb.t()); glue as in 4.1
thop_attention(context_only, latent_cache=None: no in-kernel RoPE/append,
               KV range > q range, bottom-right causal)  -> output
```

### 4.4 Chunked prefill

```
mla_rope_append_paged_kv_assign_q(...)                   # as in 4.3
for each chunk loop:                                     # caller's plan
    ckv_c, k_pe_c = load_chunked_kv_cache_for_mla(...)   # one slice
    kv_c = cublas_mm(...); glue
    thop_attention(context_only, latent_cache=None, padding mask,
                   softmax_stats_tensor=stats_c)         # partial pass
    merge_chunked_attention_for_mla(merged, partial, ..., merge_op)
final causal pass over the new tokens (thop_attention) + final merge
```

The merge is exact: folding normalized-attention triples over disjoint KV
sets reproduces full-range attention (log-sum-exp rebase; proven in the
merge entry's contract and closed-loop tested against a single-pass fp32
reference).

### 4.5 Assembler-side obligations (post-onboarding knowledge)

Everything in this subsection was established by certification runs; a
pre-onboarding decomposition report cannot contain it.

Outside the entries, target code owns: the phase dispatch and chunk
planning (`cu_*` tensors, `merge_op` codes: first pass copy, later passes
merge, sit-outs skip); load-time weight preparation (`k_b_proj_trans
[H, C, nope]`, `v_b_proj [H, v, C]` derived from `kv_b_proj`); and buffer
allocation. Hard constraints established during certification:

- **V must stay the split view of the `kv_b` output.** The context FMHA
  kernel hard-codes V's row stride (`num_kv_heads * (head_size - 64 +
  v_head_dim)`) and ignores the tensor's actual stride; a contiguous V
  yields silent garbage and out-of-bounds reads.
- `mla_rope_append_paged_kv_assign_q` is **not idempotent** (in-place
  self-reading rotation); re-prepare inputs before any retry. The append
  and gather ops are idempotent.
- Generation takes three scheduler buffers
  (`cu_q_seqlens`/`cu_kv_seqlens`/`fmha_scheduler_counter`), but only
  `cu_q_seqlens` and the counter are checked — `cu_kv_seqlens` is neither
  checked nor read on this path (the kernel takes each sequence's KV length
  from `sequence_length`). Fill all three anyway; that is the production
  shape and the certified one. The softmax scale is
  `1 / (q_scaling * sqrt(nope + rope))` — not `sqrt(C + rope)`.
- Context and generation must be **separate calls** (`context_only` /
  `generation_only`). The rejection of a mixed MLA batch lives *upstream*,
  in the Python attention backend — the op itself accepts
  `attention_input_type=0` without complaint, so a target driving the op
  directly gets no error, just a batch in which at most one phase's rows
  are right.

## 5. Certification status (post-onboarding knowledge)

All entries above are GPU-certified on sm_100 in the exact configurations
the assembly invokes, including the two producer/consumer seams that cross
entry boundaries: chunked partial passes were folded by the real merge op
in-test (stats interop proven), and the generation call consumes the
scheduler-buffer layout `mla_rope_generation` certifies producing.

One deliberate remainder — the **end-to-end parity run** of the assembled
pipeline — was left to the first MLA target. **It has since been
executed** (deepseek-v3-lite-nvfp4/sm_100/tp1): against a pure-torch
reference built from HF `DeepseekV3Attention` semantics at that
checkpoint's geometry, **0.93–1.02 bf16 ulp** across 1-seq prefill, 2-seq
prefill, mixed-batch context and mixed-batch decode, identical at
`tokens_per_block` 32 and 64. As predicted, parity is tolerance-based
rather than bit-exact: the assembly legally substitutes implementations
(e.g. `cublas_mm` where the macro's `Linear` takes `F.linear`).

What that target had to derive beyond this document — the
`rotary_cos_sin` byte layout, the `kv_b_proj` row regrouping the context
FMHA's `v` packing forces, `create_mla_outputs` being unreachable from a
target, and the cached-prefix context path that block reuse turns on by
default — is written up in `.claude/skills/staircase-modeling-core/references/latent-kv-cache.md`.

## 6. Out of scope (current envelope)

- **Quantization.** Everything is bf16: no fp8 weights (the
  `fp8_batched_quantize_1x128_permute102` / `fp8_block_scaling_bmm_out`
  family is not onboarded), no fp8/fp4 KV cache (every entry's
  `quant_mode`/scale arguments are certified inert-only), no NVFP4 output.
- **DeepSeek-V4 and DSA sparse variants.** Their extra ops
  (`deepseek_v4_q_norm*`, `mla_rope_inplace`,
  `fused_inv_rope_fp8_quant_vllm_port`, `mla_dsa_proj` /
  `mla_dsa_attn_inplace`) are not onboarded; the DSA pair are themselves
  macros needing this same decomposition treatment first.
- **Blackwell cute_dsl alternates** (`cute_dsl_bf16_gemm/bmm_blackwell`)
  are selection-dependent substitutes for `cublas_mm`/`bmm_out`; onboard
  only if selected.
- **Parallelism and decoding features.** TP=1, beam width 1, no helix CP.
  `predicted_tokens_per_seq > 1` — the multi-token query a linear-tree MTP
  draft produces — **is** certified on the decode path now, for both
  `mla_rope_generation` and `thop_attention`'s MLA generation call: each of
  the `P` query rows takes its own rope position `L_g - P + t` and attends
  causally, bottom-right aligned, to `[0, L_g - P + t]`. The two entries'
  contracts carry the layout; `.claude/skills/staircase-modeling-speculator/references/multi-token-prediction.md` has
  what the drafting layer computes.
- **Geometry.** Certified on DeepSeek-V3 per-head shapes
  (`kv_lora=512`, `rope=64`, `nope=128`, `v=128`); the decode kernel is
  source-listed for one other geometry (320/256), untested.
- **Operational notes.** `thop_attention`'s decode kernels are
  NVRTC-JIT-compiled per process (pip must resolve `tensorrt_llm` at
  runtime, so the run's interpreter needs pip present) with no on-disk cache —
  exclude first calls from any timing. Several upstream schema quirks
  (mutability annotations, misleading parameter names) are corrected
  per-entry in the contracts.
