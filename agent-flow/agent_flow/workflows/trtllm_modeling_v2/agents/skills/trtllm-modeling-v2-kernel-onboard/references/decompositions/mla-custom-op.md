# Decomposing `mla_custom_op_inplace`

The worked example of a macro decomposition: how TRT-LLM's MLA block macro
breaks into Python-level atomic entries, and how each batch mode reassembles
from them. When an onboarding request turns out to target a macro, onboard
nothing — write a report shaped like this one, to the checklist in
`../decomposition.md`.

Derived on tensorrt_llm 1.3.0rc21, torch 2.11.0+cu130, B200 (sm_100) — a stamp
on the observations, not a pin. Each entry carries its own `ARCHS`.

## 1. Why decompose it

`torch.ops.trtllm.mla_custom_op_inplace` is one registered custom op, but its
body looks up a registered `MLA` module by `layer_idx` and runs
`MLA.forward_impl`, a Python-level pipeline:

- module calls: `kv_a_proj_with_mqa`, `q_a_layernorm` / `kv_a_layernorm`
  (dual-stream), `q_b_proj`;
- tensor glue: `split`, `concat`, slicing, scratch allocation;
- a **Python-level phase dispatch** (context vs. absorption generation;
  context further splits into fresh / cached-KV / chunked-prefill);
- further registered ops inside each branch (`bmm_out`,
  `mla_rope_generation`, cache load/merge ops, the FMHA).

An entry must be **Python-level indivisible**: one callable whose interior is
C++/CUDA only. "FX-graph atomic" is not the criterion. A macro's interior
cannot be swapped, measured or certified step by step, and its hidden control
flow cannot be specialized per target, so the macro is not an entry and each
constituent is onboarded instead.

## 2. The resulting vocabulary

Shared catalog entries (paths relative to
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/`):

| role | entry |
| --- | --- |
| fused `q_a`/`ckv`/`k_pe` projection, `q_b` projection, `kv_b` expansion, `o_proj` | `gemm/cublas_mm` |
| `q_a` / `kv_a` RMS norm | `norm/flashinfer_rmsnorm` |
| attention core, all phases | `attention/thop_attention` |

Ops the DeepSeek dep4 target keeps in its own `ops/`
(`tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/ops/`);
a second MLA target promotes them to the catalog:

| role | op |
| --- | --- |
| decode: absorption BMMs (`q_nope x k_b_trans`, `attn x v_b`) | `gemm/bmm_out` |
| decode: q_pe RoPE + latent append + FMHA scheduler-buffer fill | `attention/mla_rope_generation` |
| cached/chunked prefill: new-token RoPE + latent append + q assign | `attention/mla_rope_append_paged_kv_assign_q` |
| cached prefill: full `[past + new]` latent gather | `attention/load_paged_kv_cache_for_mla` |

Split, concat, views, `expand`, slice fills and scratch allocation are native
torch calls in the target, not entries. Chunked prefill's per-loop latent
gather and log-sum-exp merge have no entry: §3.4 is a gap until a target needs
it.

## 3. Assembly by batch mode

Shared front section (every mode):

```
[q_a | ckv | k_pe] = cublas_mm(hidden, W_a.t())        # then torch.split
q_a = flashinfer_rmsnorm(q_a, w_q_ln);  ckv = flashinfer_rmsnorm(ckv, w_kv_ln)
q    = cublas_mm(q_a, W_qb.t())
latent = torch.cat([ckv, k_pe])
```

Phase dispatch (context vs. generation, and which context flavor) is plain
Python over metadata. `o_proj` (`cublas_mm`) closes every mode.

### 3.1 Plain prefill (fresh context)

```
kv = cublas_mm(ckv, W_kvb.t()); k_nope, v = split(kv)
k  = empty(...); k[..., :nope] = k_nope; k[..., nope:] = k_pe.expand(...)
thop_attention(context_only: separate q/k/v, in-kernel GPT-J RoPE on q/k_pe,
               latent-cache append, causal FMHA)   -> output
```

### 3.2 Decode (absorption generation)

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
dtype, and getting it wrong is a race rather than an error.** On a **bf16**
latent pool `fused_q` is written disjointly (`[..., :C]` by the BMM,
`[..., C:]` by the RoPE op), so they can overlap. On an **fp8-e4m3** pool
`mla_rope_generation` *reads* `fused_q[..., :C]` to build `quant_q_buffer`,
which the decode FMHA takes its query from, so the absorbed-q BMM must have
**finished** first. A decode assembled from the bf16 reading and switched to
an fp8 KV cache builds its quantized query from uninitialized memory,
silently.

### 3.3 Cached-KV prefill

```
mla_rope_append_paged_kv_assign_q(q, latent, ...)   # RoPE q/k_pe in place,
                                                    # append new tokens
ckv_full, k_pe_full = load_paged_kv_cache_for_mla(...)   # gather [past+new]
kv = cublas_mm(ckv_full, W_kvb.t()); glue as in 3.1
thop_attention(context_only, latent_cache=None: no in-kernel RoPE/append,
               KV range > q range, bottom-right causal)  -> output
```

### 3.4 Chunked prefill

As 3.3, but the past is gathered one slice per loop, each slice gets a
partial `thop_attention` pass with `softmax_stats_tensor`, and the partial
results are folded by a log-sum-exp merge, which is exact over disjoint KV
sets. The slice gather and the merge have no entry today.

## 4. Target-side obligations

Established by certification runs. Outside the entries, target code owns the
phase dispatch and chunk planning, load-time weight preparation
(`k_b_proj_trans [H, C, nope]`, `v_b_proj [H, v, C]` derived from
`kv_b_proj`), and buffer allocation. Hard constraints:

- **V must stay the split view of the `kv_b` output.** The context FMHA
  kernel hard-codes V's row stride (`num_kv_heads * (head_size - 64 +
  v_head_dim)`) and ignores the tensor's actual stride; a contiguous V
  yields silent garbage and out-of-bounds reads.
- `mla_rope_append_paged_kv_assign_q` is **not idempotent** (in-place
  self-reading rotation); re-prepare inputs before any retry. The append
  and gather ops are idempotent.
- Generation takes three scheduler buffers
  (`cu_q_seqlens`/`cu_kv_seqlens`/`fmha_scheduler_counter`); only
  `cu_q_seqlens` and the counter are read on this path. Fill all three
  anyway. The softmax scale is `1 / (q_scaling * sqrt(nope + rope))`, not
  `sqrt(C + rope)`.
- Context and generation must be **separate calls** (`context_only` /
  `generation_only`). The op accepts a mixed batch
  (`attention_input_type=0`) without complaint; at most one phase's rows are
  then right.

`.claude/skills/trtllm-modeling-v2-core/references/latent-kv-cache.md` has
what a target must derive beyond this: the `rotary_cos_sin` layout, the
`kv_b_proj` row regrouping, and the cached-prefix path block reuse turns on.

## 5. Out of scope

- **Quantized weights and KV beyond what each entry's cells state.** The fp8
  block-scaled BMM family is not onboarded.
- **DeepSeek-V4 and DSA sparse variants.** Their extra ops are not onboarded,
  and the DSA pair (`mla_dsa_proj` / `mla_dsa_attn_inplace`) are themselves
  macros.
- **Geometry.** Certified on DeepSeek-V3 per-head shapes (`kv_lora=512`,
  `rope=64`, `nope=128`, `v=128`).
- `thop_attention`'s decode kernels are NVRTC-JIT-compiled per process with
  no on-disk cache: exclude first calls from any timing.
