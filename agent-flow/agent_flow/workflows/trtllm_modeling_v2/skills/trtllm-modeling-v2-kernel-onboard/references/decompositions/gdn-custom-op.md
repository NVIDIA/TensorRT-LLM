# Decomposing `gdn_custom_op_inplace`

Why TRT-LLM's Gated DeltaNet block op cannot be a catalog entry, what its
Python-level atoms are, and which of them already have a sourcing answer.

This is a **pre-onboarding** decomposition report, written to the checklist in
`../decomposition.md`. It records what was read and what was measured on this
machine. It deliberately contains **no receipts and no caller-facing numerical
guarantees** — those exist only after each constituent is onboarded and tested
on GPU.

Derived on: tensorrt_llm 1.3.0rc26 | torch 2.12.0a0+5aff3928d8.nv26.05 |
flashinfer 0.6.18 | NVIDIA GB300 (sm_103) — the stamp on the observations
below, not a pin to assert. Receipts are keyed by GPU architecture and nothing
else; each constituent will carry its own once onboarded.

Geometry throughout is Qwen3.8-27B-NVFP4's GDN layer as the run's
`vocabulary.yaml` states it: `num_k_heads=16`, `head_k_dim=128`,
`num_v_heads=48`, `head_v_dim=128`, `mixed_qkv` width 10240,
`linear_conv_kernel_dim=4`, 48 `linear_attention` layers, and — from the run
spec's `--mamba_ssm_cache_dtype=float32` — an **fp32 recurrent-state pool**.
That last one is not a detail; it selects implementations. See §5.

## 1. Why it is a macro

The registered schema is the whole argument:

```
trtllm::gdn_custom_op_inplace(Tensor mixed_qkv, Tensor a, Tensor b,
                              str layer_idx, Tensor(a4!) output) -> ()
```

Five arguments, one of them a **string**. No weights, no convolution state, no
recurrent state, no sequence-length metadata, no cache indices. Everything the
computation actually needs is fetched from process-global state inside the
body (`tensorrt_llm/_torch/modules/mamba/gdn_mixer.py`, lines 94-112):

```python
@torch.library.custom_op("trtllm::gdn_custom_op_inplace", mutates_args=("output",))
def gdn_custom_op_inplace(mixed_qkv, a, b, layer_idx, output) -> None:
    attn_metadata, gdn_layer, spec_metadata = _extract_gdn_extra_attrs(layer_idx)
    num_tokens = attn_metadata.num_tokens
    gdn_layer.forward_core(...)
```

`_extract_gdn_extra_attrs` is a **registry lookup**: it reads
`get_model_extra_attrs()`, pulls a weakref to the live `AttentionMetadata`, and
pulls a weakref to the `Qwen3NextGatedDeltaNet` **nn.Module** out of an
`extra_attrs["gdn_layers"]` dict keyed by the `layer_idx` string. It then calls
that module's `forward_core` — a method, not a kernel.

That is the macro signature named in `../decomposition.md` verbatim: *"a
registry lookup delegating to a module `forward`"*. `forward_core` is in turn a
**Python-level phase dispatch** (`num_prefills > 0` selects `forward_extend`,
else `forward_decode`; each branch splits further on `is_target_verify` and
`use_replay`) over a pipeline of module calls, tensor glue, several distinct
kernel wrappers, and indexed state reads and writes.

**Measured, not inferred.** Driving the op the only way a catalog wrapper
could — real tensors, a `layer_idx` string, a real output buffer, nothing else,
because nothing else is in the schema:

```
torch.ops.trtllm.gdn_custom_op_inplace(mixed_qkv, a, b, "0", out)
  RAISED AssertionError: Model extra attrs is not set
```

A catalog wrapper cannot satisfy that precondition without constructing the
whole model's module tree and registering it in `model_config.extra_attrs`,
which is precisely the "black box whose interior cannot be swapped, measured or
certified independently" the catalog exists to prevent. **Onboard nothing.**

For completeness, what the body reaches for beyond its five arguments:

| source | fields consumed |
| --- | --- |
| `get_model_extra_attrs()` | `attention_metadata` (weakref), `gdn_layers[layer_idx]` (weakref to the module), `spec_metadata` |
| `AttentionMetadata` | `num_tokens`, `num_contexts`, `num_ctx_tokens`, `seq_lens`, `mamba_metadata`, `kv_cache_manager` |
| `kv_cache_manager` | `mamba_layer_cache(layer_idx)`, `is_speculative()`, `use_replay_state_update`, `get_replay_state_update_metadata()`, `use_gdn_cached_replay_all_layer_commit` |
| `Mamba2Metadata` | `has_initial_states`, `state_indices`, `query_start_loc`, `query_start_loc_long`, `replay_work_items`, `replay_n_writes` |
| layer cache | `.conv`, `.temporal`, `.intermediate_conv_window`, `.intermediate_ssm`, `.old_x`, `.old_B`, `.old_dt`, `.old_dA_cumsum`, `.cache_buf_idx`, `.prev_num_accepted_tokens` |
| the module itself | `conv1d.weight`, `conv1d.bias`, `A_log`, `dt_bias`, `activation`, and every geometry field |
| `SpecMetadata` | `runtime_draft_len` |

## 2. Scope of this map

`gdn_custom_op_inplace` covers the GDN **core** only. Two neighbours are
adjacent in the layer but **outside** the op body, and must not be
double-counted by a child:

- the `in_proj_qkvz` / `in_proj_ba` projections and the `[Q|K|V|Z]`, `[b|a]`
  column slices happen in `_compute_tokenwise_inputs`, before the op;
- the gated RMS norm and `out_proj` happen in `_postprocess_gdn_output`, after
  the op. That norm is `torch.ops.trtllm.rms_norm_gated_token_major`, which is
  already a separate item in this run's queue.

## 3. Symbol verification

Every symbol named in this document was resolved in the installed engine on
this machine before being cited (spool logs `smith1-gdn-macro-probe`,
`smith1-gdn-constituent-reach`).

`torch.ops.trtllm` ops **present**: `gdn_custom_op_inplace`,
`causal_conv1d_fwd`, `causal_conv1d_update`, `rms_norm_gated_token_major`,
`fused_gated_rmsnorm_quant`, `mamba2_mtp_ssm_cache_update`, `mamba_conv1d`,
`selective_scan`.

`torch.ops.trtllm` ops **absent** — probed, not assumed, because the absence is
the load-bearing fact for §4: `chunk_gated_delta_rule`, `gated_delta_rule`,
`fused_recurrent_gated_delta_rule`, `fused_sigmoid_gating_delta_rule_update`,
`fused_gdn_post_conv`, `l2norm`, `gdn_gating`. **The delta-rule core — the
actual recurrence this layer exists to compute — has no `torch.ops.trtllm`
surface at all.** Only the two convolution ops do.

Two schemas have drifted from
`.claude/skills/trtllm-modeling-v2-kernel-feasibility/references/trtllm-ops.md`; the
generated inventory is stale on both. Observed here:

```
causal_conv1d_fwd(..., int pad_slot_id, Tensor($3! -> )? out=None) -> ()      # inventory omits `out`
rms_norm_gated_token_major(..., Tensor? fp8_scale=None,
                           str gate_activation="silu") -> Tensor              # inventory omits `gate_activation`
```

All 24 Python-callable constituents resolved (`True` for every entry in
section C of `smith1-gdn-macro-probe.log`). `flashinfer.chunk_gated_delta_rule`
exists in flashinfer 0.6.18; `flashinfer.gated_delta_rule` does **not**.

## 4. The constituents

Labels follow `../decomposition.md`: **existing entry** / **not onboarded**
(symbol + proposed category) / **gap** (no Python-callable atomic surface in
the preferred tiers; what is reachable instead).

Entry paths are relative to
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/`.

### 4.1 Tier 1 — a `torch.ops.trtllm` op exists

| role | symbol | state |
| --- | --- | --- |
| prefill / varlen causal depthwise conv1d + silu, with conv-state read and write | `torch.ops.trtllm.causal_conv1d_fwd` (Python surface `..modules.mamba.causal_conv1d.causal_conv1d_fn`) | not onboarded → `ssm` |
| decode conv1d step + conv-state ring update | `torch.ops.trtllm.causal_conv1d_update` (Python surface `..modules.mamba.causal_conv1d.causal_conv1d_update`) | not onboarded → `ssm` |

Both Python surfaces are one op invocation plus allocation/`unsqueeze` glue —
compliant wrapper shape as-is. `causal_conv1d_fn` allocates a channel-last
`out` when `x` is channel-last; on this run's prefill path `x` arrives as a
plain contiguous `[10240, T]` and `out` stays `None`, so the op writes `x` in
place.

### 4.2 Tier 2 — no torch-op boundary; a single-kernel Python callable exists

Each of these is one Triton kernel launch plus allocation. Wrapping a
non-`torch.ops.trtllm` callable is an **approved-exception** shape per
`entry-spec.md` — it needs the run's explicit authorization, and the import
path must be recorded in the wrapper's class docstring (the precedent is
`attention/thop_attention`).

| role | symbol | proposed category |
| --- | --- | --- |
| post-conv fusion: q/k/v split of the conv output, **Q/K L2 norm**, `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)` — five outputs, one kernel | `..modules.mamba.fuse_elementwise_ops.fused_gdn_post_conv` | `ssm` |
| decode delta-rule core: in-kernel sigmoid gating + recurrent gated delta rule, initial state gathered from the pool by index, state written back | `..modules.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update` | `ssm` |
| zero the recurrent and conv rows of fresh (no-initial-state) requests | `..modules.mamba.recurrent_state_cache.reset_recurrent_state_rows` | `ssm` |
| MTP verify: conv1d step writing per-draft-token intermediate conv windows | `..modules.mamba.causal_conv1d_triton.causal_conv1d_update` | `ssm` |
| MTP verify: pack decode Q/K/V from the conv output, **without** norm or gates | `..modules.mamba.fuse_elementwise_ops.pack_gdn_decode_qkv` | `ssm` |
| MTP verify: `g` alone, when the gating is not fused into the recurrence | `..modules.mamba.gdn_mixer.fused_gdn_gating` | `ssm` |
| MTP verify: recurrent gated delta rule over `P` draft tokens, writing per-step intermediate SSM states, live pool untouched | `..modules.fla.fused_recurrent.fused_recurrent_gated_delta_rule_update` | `ssm` |
| MTP replay: cached-history replay verify (only when `kv_cache_manager.use_replay_state_update`) | `..modules.fla.cached_replay.fused_recurrent_gated_delta_rule_cached_replay_update` | `ssm` |

**One boundary question a child must settle, not assume.**
`fused_sigmoid_gating_delta_rule_update` is not unconditionally a single call:
its body dispatches to a FlashInfer CuTe-DSL kernel when
`_can_use_flashinfer_gdn_decode` passes, and to its own Triton kernel
otherwise. On this run the FlashInfer branch is statically ineligible (§5), so
the function reduces to one Triton launch — but the entry must say which
implementation its receipt certifies, and must not claim the other.

### 4.3 Movement-only — placement is the orchestrator's call

| role | symbol |
| --- | --- |
| extract a column slice of the token-major projection and transpose it to `[C, T]` for the prefill conv | `..modules.mamba.fuse_elementwise_ops.extract_transpose_prefill_slice` |

Measured equal to `src[:T, s:s+w].t()` elementwise (`True` in the probe). It
launches no math — it is layout and movement, which is what `catalog/torch/` is
for, but it is a trtllm Triton kernel rather than a torch callable. **Reported,
not placed:** `catalog/torch/` is never written by this role.

Ordinary torch glue the assembly also needs, already in the catalog:
`torch/split` (state indices, `mixed_qkv` prefill/decode halves), `torch/empty`,
`torch/reshape`, `torch/transpose`, `torch/concat` (the MTP branch's
prefill+decode output join), `torch/copy_`.

Two more are **computation**, not glue, and have no entry: `torch.exp` (inside
the FlashInfer prefill adapter) and `Tensor.sigmoid` (the MTP fallback's
`beta`). A computation mirror widens what `catalog/torch/` is for, so both are
reported for an owner's decision rather than assumed.

### 4.4 Gap — the prefill delta-rule core

This is the one place where no preferred-tier surface exists, and it is the
most important step in the layer.

`gdn_mixer._resolve_chunk_gated_delta_rule()` picks one of two implementations
at first call. **On sm_103 it returned
`tensorrt_llm._torch.modules.fla.flashinfer_chunk.chunk_gated_delta_rule`** —
measured, not read: `TLLM_USE_FLASHINFER_GDN_PREFILL` defaults to `1` and
`is_flashinfer_gdn_supported_arch()` is `True` here.

**The selected implementation is itself a sub-macro.** `flashinfer_chunk.py` is
an adapter whose body is Python-level composition:

```
g2 = torch.exp(g.squeeze(0))                                  # log-space -> linear alpha
q3, k3 = l2norm_fwd(q3), l2norm_fwd(k3)                       # only if use_qk_l2norm_in_kernel
gathered = gather_cast_vk_to_fp32_vk(state, indices, dtype)   # triton: indexed gather (+cast)
out, st  = flashinfer.chunk_gated_delta_rule(...)             # FlashInfer, tier 3
cast_scatter_fp32_vk_to_vk(st, state, indices)                # triton: indexed scatter (+cast)
```

Its atomic interior is `flashinfer.chunk_gated_delta_rule` — neither a
`torch.ops.trtllm` op (probed absent, §3) nor a torch callable, but a function
of **FlashInfer**, a pinned dependency of the engine. That is the sourcing
ladder's tier 3 (`../decomposition.md`, "FlashInfer's public API"): the child
is an `ssm` entry whose `__call__` calls `flashinfer.chunk_gated_delta_rule`
directly, and it needs no ruling. What is reachable, stated plainly:

- `flashinfer.chunk_gated_delta_rule` — one call, explicit state, no registry.
  Shaped like a legal entry; sourced from outside tensorrt_llm entirely.
- `..modules.fla.fused_state_io.gather_cast_vk_to_fp32_vk` and
  `cast_scatter_fp32_vk_to_vk` — one Triton kernel each, tier 2, `ssm`.
- `..modules.fla.l2norm.l2norm_fwd` — one Triton kernel, tier 2. **Not on this
  run's path**: GDN prefill calls the adapter with
  `use_qk_l2norm_in_kernel=False` because `fused_gdn_post_conv` has already
  normalized Q/K.

The alternate implementation (`TLLM_USE_FLASHINFER_GDN_PREFILL=0`, or any
unsupported arch) is the vendored Triton `..modules.fla.chunk`, which is **also
a sub-macro** — seven Python-level steps, each its own Triton wrapper:
`l2norm_fwd` → `chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` →
`solve_tril` → `recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` →
`chunk_fwd_o`. All seven resolve. Decomposing to that level buys seven entries
for a path this arch does not select; recorded as the fallback's shape, not
proposed as work.

## 5. What this run's configuration selects (measured)

The run spec passes `--mamba_ssm_cache_dtype=float32`. Both FlashInfer
decode-side kernels gate on a **bfloat16** state pool, so that one flag decides
which kernels this target will need. Measured on sm_103, with an otherwise
eligible geometry (`K = V = 128`):

| predicate | fp32 pool (this run) | bf16 pool |
| --- | --- | --- |
| `_can_use_flashinfer_gdn_decode(..., T=N=2)` | **False** | True |
| `_can_use_flashinfer_gdn_verify(..., draft=4)` | **False** | True |

So on this run: prefill takes the **FlashInfer** chunk path, while decode and
MTP verify take the **Triton** kernels. A decomposition derived from a bf16-KV
run would have named three different constituents. Supporting facts measured
alongside: `is_flashinfer_gdn_supported_arch() == True`,
`is_sm_100f() == True`, `_FLASHINFER_GDN_BF16_STATE_AVAILABLE == True`,
`_FI_GDN_MAX_MTP_T == 8`.

## 6. Assembly by batch mode

Phase dispatch over metadata reads is plain Python and is permitted outside
entries. `[Q|K|V|Z]` / `[b|a]` slicing precedes all of this and the gated norm
plus `out_proj` follow it (§2).

Shared preamble, every mode:

```
state_indices_p, state_indices_d = split(state_indices, [num_prefills, num_decodes])
if num_prefills > 0:
    reset_recurrent_state_rows(ssm_states, state_indices_p, has_initial_states_p, conv_states)
```

### 6.1 Prefill only (`num_prefills > 0`, `num_decode_tokens == 0`)

```
xt          = extract_transpose_prefill_slice(mixed_qkv, T, 0, 10240)   # [10240, T]
xt          = causal_conv1d_fn(xt, conv_w, None, conv_states=..., has_initial_state=...,
                               cache_indices=..., query_start_loc=...)  # causal_conv1d_fwd + silu
q,k,v,g,beta= fused_gdn_post_conv(xt, None, a, b, A_log, dt_bias, 16,128, 48,128)
o, _        = chunk_gated_delta_rule(q,k,v,g,beta, initial_state=ssm_states,
                                     initial_state_indices=cache_indices,
                                     inplace_indexed_state_update=True,
                                     cu_seqlens=query_start_loc_long,
                                     use_qk_l2norm_in_kernel=False, output=output)
```

`inplace_indexed_state_update=True` writes the new recurrent state straight
back into the shared pool; the caller owns the guarantee that the indices do
not alias live slots (the source says so at the call site, and this report does
not upgrade that to a certified constraint).

### 6.2 Decode only (`num_prefills == 0`)

```
mixed_qkv = causal_conv1d_update(mixed_qkv, conv_states, conv_w, None, "silu",
                                 conv_state_indices=cache_indices)
q,k,v     = views over mixed_qkv           # column slices, no kernel
o         = fused_sigmoid_gating_delta_rule_update(A_log, a, dt_bias, 1.0, 20.0,
                                                   q,k,v,b, ssm_states, cache_indices,
                                                   use_qk_l2norm_in_kernel=True,
                                                   cu_seqlens=query_start_loc_long,
                                                   output=output)
```

Note the asymmetry with 6.1 and do not flatten it: **on the decode path the L2
norm and the gating are inside the recurrence kernel** (`a`/`b` go in raw), so
there is no `fused_gdn_post_conv` call and no separate `g`/`beta`. On the
prefill path they are inside `fused_gdn_post_conv` instead, and the recurrence
is told `use_qk_l2norm_in_kernel=False`. Same math, different fusion boundary.

### 6.3 Mixed prefill + decode

Both convolutions run (`causal_conv1d_fn` on the prefill rows,
`causal_conv1d_update` on the decode rows), then a **single**
`fused_gdn_post_conv(prefill, decode, ...)` call fuses both halves into one
`q/k/v/g/beta` set, and one `chunk_gated_delta_rule` covers the batch.

### 6.4 MTP target verification (`is_target_verify`)

Reached when there are decode rows, `spec_metadata` is present and the KV cache
manager is speculative — i.e. the `mtp` feature this run enables. Extra
constituents over 6.2, all listed in §4.2:

```
mixed_qkv_d = causal_conv1d_update_triton(..., intermediate_conv_window=...,
                                          intermediate_state_indices=arange(num_decodes))
q,k,v       = pack_gdn_decode_qkv(mixed_qkv_d, 16,128, 48,128)     # no norm, no gates
beta        = b.sigmoid();  g = fused_gdn_gating(A_log, a, dt_bias)
o           = fused_recurrent_gated_delta_rule_update(q,k,v,g,beta,
                  initial_state_source=ssm_states[state_indices_d],
                  initial_state_indices=arange(num_decodes),
                  use_qk_l2norm_in_kernel=True, disable_state_update=True,
                  intermediate_states_buffer=intermediate_ssm_states,
                  cache_steps=draft_token_num)
```

Final state selection is deferred to the cache manager's
`update_mamba_states()`, which is runtime, not modeling. When
`kv_cache_manager.use_replay_state_update` is on, this is replaced by
`fused_recurrent_gated_delta_rule_cached_replay_update` (source-asserted limits
`draft_token_num <= 8`, `replay_history_size <= 16`). When the pool is bf16 —
**not this run** — `_flashinfer_gdn_verify` replaces it instead.

## 7. Reachability sweep (this geometry, this machine)

Every constituent below was called once at the checkpoint's geometry, to
establish that it is reachable and that the decomposition is not proposing work
against a dead path. **These are not receipts**: each says "ran, returned the
expected shape, output finite", nothing about numerical correctness. Log
`smith1-gdn-constituent-reach`.

| call | result |
| --- | --- |
| `causal_conv1d_fn`, `[10240, 37]` bf16, 2 seqs | `(10240, 37)` bf16, finite, conv state written |
| `causal_conv1d_update`, `[4, 10240]` bf16 | `(4, 10240)` bf16, finite |
| `fused_gdn_post_conv`, prefill-only, T=37 | q `(1,37,16,128)`, v `(1,37,48,128)`, g `(1,37,48)` fp32, beta fp32, finite |
| `extract_transpose_prefill_slice` | `(10240, 37)`, elementwise equal to `src[:T,:C].t()` |
| `pack_gdn_decode_qkv`, T=16 | q `(1,16,16,128)`, v `(1,16,48,128)` |
| `flashinfer_chunk`, **fp32** pool, T=64, 2 seqs | `(1,64,48,128)` bf16, finite; slots 0/1 updated, slots 2+ bitwise untouched |
| `flashinfer_chunk`, bf16 pool | same shape, finite, same slot behaviour |
| `fused_sigmoid_gating_delta_rule_update`, **fp32** pool, N=4 | `(1,4,48,128)` bf16, finite, state updated |
| `fused_sigmoid_gating_delta_rule_update`, bf16 pool | same |
| `fused_recurrent_gated_delta_rule_update`, N=2 P=4, fp32 pool | `(2,4,48,128)` bf16, finite, intermediates written |
| `reset_recurrent_state_rows` | slot 0 zeroed, slot 1 kept, conv slot 0 zeroed |

**The fp32 pool is not a blocker for the prefill path.** That was the open risk
in §5 — `flashinfer_chunk` passes the pool dtype straight through on
`is_sm_100f()` archs — and it ran.

### 7.1 A domain fact worth carrying forward

The first sweep ran the vendored Triton chunk on raw `randn` q/k and got
all-NaN. That is **not** a kernel defect, and a child must not record it as
one. Re-probed deliberately:

| input | vendored triton chunk | flashinfer chunk |
| --- | --- | --- |
| raw `randn` q/k, `use_qk_l2norm_in_kernel=False` | **NaN** (393216 elements) | finite but absmax `1.02392e+13` |
| caller-normalized q/k, `in_kernel=False` (production shape) | finite, absmax `0.0742188` | finite, absmax `0.074707` |
| raw `randn` q/k, `use_qk_l2norm_in_kernel=True` | finite, absmax `0.0742188` | — |

The gated delta rule is only stable on **L2-normalized** q/k, which is exactly
why the production prefill path normalizes inside `fused_gdn_post_conv` and
then passes `use_qk_l2norm_in_kernel=False`. Out of that domain the two
implementations diverge in different directions — one to NaN, one to 1e13 —
so neither can be used to judge the other there. In domain they agree to
`maxabs = 0.000488281` (`maxrel = 91.4489`, a tiny-denominator artifact on
near-zero bf16 outputs, not a disagreement in magnitude).

Any test written for either prefill core must build its q/k normalized, or it
will be measuring divergence rather than the kernel.

## 8. Proposed child dispatches

One entry each, no dependencies between them. Categories proposed, not settled;
`ssm` is an existing catalog category with no directory yet.

| # | target | tier | note |
| --- | --- | --- | --- |
| 1 | `torch.ops.trtllm.causal_conv1d_fwd` | 1 | prefill/varlen conv + silu + conv-state write |
| 2 | `torch.ops.trtllm.causal_conv1d_update` | 1 | decode conv step + conv-state update |
| 3 | `fuse_elementwise_ops.fused_gdn_post_conv` | 2 | needs tier-2 authorization |
| 4 | `fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update` | 2 | decode core; must name which branch the receipt certifies (§4.2) |
| 5 | `flashinfer.chunk_gated_delta_rule` | 3 | prefill delta-rule core (§4.4) |
| 6 | `recurrent_state_cache.reset_recurrent_state_rows` | 2 | |
| 7 | `fuse_elementwise_ops.extract_transpose_prefill_slice` | 2 | placement question, §4.3 |
| 8 | `causal_conv1d_triton.causal_conv1d_update` | 2 | MTP only |
| 9 | `fuse_elementwise_ops.pack_gdn_decode_qkv` | 2 | MTP only |
| 10 | `gdn_mixer.fused_gdn_gating` | 2 | MTP only |
| 11 | `fused_recurrent.fused_recurrent_gated_delta_rule_update` | 2 | MTP only |

Deliberately **not** proposed: the seven vendored-Triton chunk sub-steps (§4.4,
alternate path this arch does not select), `l2norm_fwd` (not on this run's
path), `_flashinfer_gdn_verify` and `..fla.cached_replay` (both gated off by
the fp32 pool / the replay flag), and `torch.exp` / `Tensor.sigmoid` (owner's
decision on widening `catalog/torch/`).

## 9. Out of scope / unknown

- **Every numerical claim.** Nothing here certifies accuracy, tolerance, dtype
  limits or arch behaviour. §7 establishes reachability and shapes only.
- **Tensor parallelism.** Read at tp1. The module divides `num_k_heads` and
  `num_v_heads` by `attn_tp_size` and asserts divisibility; nothing here was
  measured above tp1.
- **The vision tower.** Out of scope for this target by the run's own rule.
- **`mamba_conv1d` / `selective_scan`.** Present as trtllm ops and conceptually
  adjacent, but not called from this macro — they serve the Mamba2 mixer, not
  GDN. Named only so a later reader does not mistake them for the GDN path.
