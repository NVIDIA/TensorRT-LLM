# Design: Avoid `TrtllmAttentionArgs` by sourcing thop kwargs from existing objects

> **Branch:** `attn-avoid-unpack` &nbsp;·&nbsp; **6 commits** &nbsp;·&nbsp; rooted at `main`

## 1. Background

[yihwang-nv@d7d000b](https://github.com/yihwang-nv/TensorRT-LLM/commit/d7d000bcc255626e0a8f0239959a8c441b9a1e56)
proposes introducing a new aggregator type `TrtllmAttentionArgs` so the ~70-arg
`thop.attention(...)` and `trtllm_gen_attention(...)` calls can be replaced by
a single struct passed through both Python and C++. The struct is bound
between sides via nanobind (`+237` lines in
`cpp/tensorrt_llm/nanobind/thop/bindings.cpp`) and constructed in Python by a
new `build_trtllm_attention_args()` helper, with `TrtllmAttention._run` taking
the struct.

The maintenance cost of that design is high:

- A field is declared three times (C++ struct, nanobind `def_rw`/builder,
  Python dataclass), so adding or renaming a field requires coordinated
  changes in three places.
- The struct exists only to be unpacked at the C++ boundary — nothing in the
  codebase needs it as a *value object* (no caching, equality, replay,
  pickling).
- The struct duplicates information that already lives on
  `TrtllmAttention` / `TrtllmAttentionMetadata` / `AttentionForwardArgs`.

This design takes the opposite direction: **eliminate the intermediate type
entirely** and let the thop call read its kwargs directly from the rich
objects that already own those fields.

## 2. Goals

1. Replace the two ~75-arg positional call sites
   (`thop.attention(...)`, `trtllm_gen.trtllm_gen_attention(...)`) inside
   `TrtllmAttention._run` with **explicit-kwarg** call sites, without
   introducing a new aggregator type.
2. Every kwarg passed to the kernels is sourced as `source.attribute` from
   exactly one of four existing rich objects: the attention module
   (`self`), `TrtllmAttentionMetadata`, `AttentionForwardArgs`, and one new
   small dataclass `AttentionSparseArgs`.
3. Cheap to keep in sync: a unit test parses the wrapper's AST and fails
   any drift between (a) the wrapper, (b) the C++ binding, (c) the rich
   classes — at CI time, with a message that names the exact offending
   kwarg.
4. Preserve behaviour end-to-end. No CUDA kernel changes; no
   nanobind-binding changes.

## 3. Architecture

### 3.1 The four source classes

| Source | Type | What it owns | Lifetime |
|---|---|---|---|
| `self` | `TrtllmAttention` (nn.Module) | Module config: `num_heads`, `head_dim`, `q_scaling`, `position_embedding_type`, `rope_params`, `is_mla_enable`, `q_lora_rank`, … | Constructed once, lives as long as the module |
| `metadata` | `TrtllmAttentionMetadata` (`@dataclass`) | Per-step batch state: `sequence_length`, `context_lengths`, `kv_cache_block_offsets`, `cu_q_seqlens`, spec-decoding tensors, helix tensors, … | Refreshed per forward step |
| `fwd` | `AttentionForwardArgs` (`@dataclass(slots=True)`) | Per-call tensors: `q_pe`, `latent_cache`, `attention_sinks`, `output`, `output_sf`, `mla_bmm1_scale`, sage args, … | Constructed per attention call |
| `sparse` | `AttentionSparseArgs` (new, `@dataclass(slots=True)`) | Sparse-attention args that don't already live on the three above: five `sparse_*` fields | Constructed per attention call, default-empty when no sparse attention is configured |

> `AttentionSparseArgs` deliberately holds only the fields that **don't**
> already exist elsewhere. `num_sparse_topk` was promoted to base
> `TrtllmAttentionMetadata`; `sparse_mla_topk_lens` is passed as literal
> `None`; `compressed_kv_cache_pool_ptr` is passed as literal `None`.

### 3.2 Derived views (properties)

Some thop kwargs are simple translations or compositions of the underlying
fields. They're exposed as `@property` on the owner class, so the wrapper
can still source them as `source.attribute`:

| Property | Owner | What it returns |
|---|---|---|
| `mask_type` | `AttentionForwardArgs` | `int(AttentionMaskType)` translated from the `attention_mask` enum |
| `effective_out_scale` | `AttentionForwardArgs` | Picks `out_scale_sf` for NVFP4 output, else `out_scale` |
| `use_paged_context_fmha` | `TrtllmAttentionMetadata` | Initialized from `runtime_features` in `__post_init__`; overridden in `forward()` for SM90 / sparse-mqa-gqa / MLA |
| `effective_workspace` | `TrtllmAttentionMetadata` | Picks `cuda_graph_workspace` when capturing, else `workspace` |
| `effective_flash_mla_tile_scheduler_metadata` / `effective_flash_mla_num_splits` | `TrtllmAttentionMetadata` | Gated by `enable_flash_mla` |
| `helix_tensor_params`, `spec_decoding_bool_params`, `spec_decoding_position_offsets_for_cpp` | `TrtllmAttentionMetadata` | Compose multiple metadata fields into the positional list/tensor the kernel wants |
| `max_context_length` | `TrtllmAttentionMetadata` | `min(max_seq_len - 1, max_num_tokens)` |
| `rotary_embedding_dim/base/scale_type/scales/max_position_info` | `TrtllmAttention` | Project `self.rope_params` into the scalars/lists the kernel expects |
| `skip_softmax_threshold_scale_factor_prefill/decode` | `TrtllmAttention` | Read off `self.sparse_attention_config` when it's a `SkipSoftmaxAttentionConfig`, else `None` |
| `spec_decoding_tensor_params(metadata)` | `TrtllmAttention` | SM-version-dependent — *method*, not property, because the SM check lives on the backend |

### 3.3 The wrappers

One explicit-kwarg method on `TrtllmAttention` (for `thop.attention`), and
two public functions on `trtllm_gen` (for the trtllm-gen path) — all four
sharing the same shape:

```python
def _call_thop_attention(
    self,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: TrtllmAttentionMetadata,
    fwd: AttentionForwardArgs,
    sparse: AttentionSparseArgs,
) -> None:
    # Per-call locals — value depends on the in-flight ``k`` tensor, mixes
    # self+fwd, or requires a method call. Cannot be a pure ``source.attr``.
    is_fused_qkv = not metadata.is_cross and k is None
    update_kv_cache = not metadata.is_cross or k is not None
    layer_idx = self.get_local_layer_idx(metadata)
    attention_window_size = fwd.attention_window_size or metadata.max_seq_len
    kv_scale_orig_quant = (self.kv_scale_orig_quant
                           if fwd.kv_scales_sf_inv is None
                           else fwd.kv_scales_sf_inv)
    kv_scale_quant_orig = (self.kv_scale_quant_orig
                           if fwd.kv_scales_sf is None
                           else fwd.kv_scales_sf)
    spec_decoding_tensor_params = self.spec_decoding_tensor_params(metadata)

    thop.attention(
        # --- Inputs (per-call tensors) ---
        q=q, k=k, v=v,
        output=fwd.output,
        output_sf=fwd.output_sf,
        workspace_=metadata.effective_workspace,

        # --- TrtllmAttentionMetadata ---
        sequence_length=metadata.kv_lens_cuda_runtime,
        ...
        helix_tensor_params=metadata.helix_tensor_params,
        spec_decoding_bool_params=metadata.spec_decoding_bool_params,
        num_sparse_topk=metadata.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=metadata.effective_flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=metadata.effective_flash_mla_num_splits,
        max_context_length=metadata.max_context_length,
        use_paged_context_fmha=metadata.use_paged_context_fmha,
        ...

        # --- AttentionForwardArgs ---
        mask_type=fwd.mask_type,
        out_scale=fwd.effective_out_scale,
        mrope_rotary_cos_sin=fwd.mrope_rotary_cos_sin,
        mrope_position_deltas=fwd.mrope_position_deltas,
        ...

        # --- TrtllmAttention ---
        num_heads=self.num_heads,
        head_size=self.head_dim,
        rotary_embedding_dim=self.rotary_embedding_dim,
        skip_softmax_threshold_scale_factor_prefill=self.skip_softmax_threshold_scale_factor_prefill,
        ...

        # --- AttentionSparseArgs ---
        sparse_kv_indices=sparse.sparse_kv_indices,
        sparse_attn_indices_block_size=sparse.sparse_attn_indices_block_size,
        ...

        # --- Per-call locals (computed above) ---
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=update_kv_cache,
        layer_idx=layer_idx,
        attention_window_size=attention_window_size,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        spec_decoding_tensor_params=spec_decoding_tensor_params,

        # --- Literals intentionally None/0 (see _THOP_LITERAL_NONE) ---
        sink_token_length=0,
        sparse_mla_topk_lens=None,
        compressed_kv_cache_pool_ptr=None,
    )
```

The trtllm-gen pair lives in `trtllm_gen.py` directly:

```python
# tensorrt_llm/_torch/attention_backend/trtllm_gen.py
def is_supported(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],  # noqa: ARG001 — kept for signature parity
    metadata: "TrtllmAttentionMetadata",
    fwd: AttentionForwardArgs,
    sparse: AttentionSparseArgs,
) -> tuple[bool, str]:
    """Reads every check input from the four rich objects. ``phase`` is
    derived from ``fwd.attention_input_type``
    (``context_only``→``"context"``, ``generation_only``→``"generation"``,
    ``mixed``→``"both"``). Delegates to
    :class:`FlashInferTrtllmGenAttention.is_supported`."""
    ...

def trtllm_gen_attention(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: "TrtllmAttentionMetadata",
    fwd: AttentionForwardArgs,
    sparse: AttentionSparseArgs,
) -> None:
    """Reads every kernel input from the four rich objects. The flashinfer
    body (~210 lines) is inlined — no separate kwargs-style impl, no
    translation layer."""
    ...
```

Notes:

- Both functions share the same `(attn, q, k, v, metadata, fwd, sparse)`
  signature, so the dispatcher can use identical argument lists for the
  support check and the kernel call. `is_supported` doesn't actually use
  `v`; it's accepted for symmetry.
- `phase` was a parameter on the old kwarg-style `is_supported` and is now
  derived inside the function from `fwd.attention_input_type`. This also
  tightens the check: a context-only call no longer demands the generation
  kernel be supported.
- There is no private `_*_impl` companion — the body of
  `trtllm_gen_attention` reads `attn.X` / `metadata.Y` / `fwd.Z` / `sparse.W`
  directly throughout, eliminating the kwargs translation step. About half
  of the old impl parameters (`q_pe`, `host_total_kv_lens`,
  `chunked_prefill_buffer_batch_size`, `helix_tensor_params`, the spec-
  decoding kwargs, the sparse_mla_*, the skip_softmax_*, etc.) were never
  actually used by the body and just drop out.

The thop wrapper still calls `thop.attention(...)`. The kernel function
signatures themselves are unchanged.

### 3.4 The dispatcher

After the cleanup, `_run` is ~85 lines (was ~250):

```python
def _run(self, q, k, v, metadata, forward_args, sparse_args) -> None:
    # 1. Shape / hidden-size assertions for both fused-QKV and MLA paths
    ...
    # 2. Side-effecting setup before the kernel launch
    self._ensure_rope_table_size(metadata.max_seq_len)
    layer_idx = self.get_local_layer_idx(metadata)
    if metadata.spec_decoding_bl_tree_mask is not None and layer_idx == 0:
        metadata.spec_decoding_bl_tree_mask.zero_()
    if self.print_skip_softmax_stat:
        self.skip_softmax_stat.zero_()

    # 3. Dispatch
    helix_active = metadata.helix_position_offsets is not None
    use_sage_attn = (forward_args.sage_attn_num_elts_per_blk_q > 0 or ...)
    if (_TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION and not helix_active
            and not use_sage_attn
            and trtllm_gen.is_supported(self, q, k, metadata,
                                         forward_args, sparse_args)[0]):
        trtllm_gen_attention(self, q, k, v, metadata,
                              forward_args, sparse_args)
    else:
        self._call_thop_attention(q, k, v, metadata,
                                   forward_args, sparse_args)

    # 4. Optional skip-softmax stat printout
    ...
```

### 3.5 `AttentionForwardArgs.output` is "in-out"

`TrtllmAttention.forward()` previously created `output` / `output_sf` as
locals and passed them through `_run` and into the kernel. Now it stores them
back onto `forward_args.output` / `forward_args.output_sf` immediately after
`create_output`, so the wrapper can source them as `fwd.output`. The return
clause reads off `forward_args` directly:

```python
if forward_args.output is None:
    outputs = self.create_output(q, ...)
    forward_args.output    = outputs[0]
    forward_args.output_sf = outputs[1] if len(outputs) == 2 else None
...
self._run(q, k, v, metadata, forward_args, sparse_args)
return (forward_args.output if forward_args.output_sf is None
        else (forward_args.output, forward_args.output_sf))
```

### 3.6 `mrope_config` flattening

`AttentionForwardArgs.mrope_config: Optional[dict]` was a dict built and
immediately destructured along the call chain
(`Attention.forward` → `forward_impl` destructures → `_attn_impl` reconstructs
→ `AttentionForwardArgs(mrope_config=…)`). The dict roundtrip collapses to
two flat tensor fields:

```python
@dataclass(kw_only=True, slots=True)
class AttentionForwardArgs:
    ...
    mrope_rotary_cos_sin:  Optional[torch.Tensor] = None
    mrope_position_deltas: Optional[torch.Tensor] = None
```

The upstream `Attention.forward(mrope_config: Optional[dict])` and
decoder-layer APIs are unchanged — `forward_impl`'s existing destructure
feeds the two new fields. The multimodal-pipeline contract
(`multimodal_data["mrope_config"]` as a dict in `runtime/`, `llmapi/`,
`inputs/`) is untouched.

Four stale `Optional[Tuple[torch.Tensor, int]]` annotations in
`modeling_qwen.py` (3 sites) and `modeling_seedoss.py` (2 sites) were
pre-existing type lies and got corrected to `Optional[dict]` in the same
commit.

## 4. The sync test

`tests/unittest/_torch/attention_backend/test_attention_op_sync.py` parses
the AST of `TrtllmAttention._call_thop_attention` and enforces six invariants:

| Test | What it checks |
|---|---|
| `test_wrapper_kwargs_match_binding_kwargs` | Wrapper's kwarg set equals the C++ binding's kwarg set (via `inspect.signature(thop.attention)`). |
| `test_each_source_attr_kwarg_resolves_uniquely` | Every `kwarg=source.attr` resolves to `attr` on exactly the named source class — no ambiguity. |
| `test_literal_none_kwargs_are_allowlisted` | `kwarg=None` only allowed when name is in `_THOP_LITERAL_NONE`. |
| `test_every_forward_args_field_is_consumed` | Every `AttentionForwardArgs` dataclass field is either accessed in the wrapper (direct or via a `@property` of the dataclass) or listed in `_THOP_EXCLUDED_FIELDS`. |
| `test_every_sparse_args_field_is_consumed` | Every `AttentionSparseArgs` field is sourced in the wrapper. No exclusion list. |
| `test_no_unexpected_other_kwargs` | Kwargs whose value is neither `source.attr` nor `None` must be one of a small, named set of expected locals. |

The tests do NOT run the kernel — they fail purely on shape mismatches.

### Allowlists

```python
_THOP_EXCLUDED_FIELDS = frozenset({
    "topk_indices",          # consumed only by DSA
    "attention_mask",       # input to ``mask_type`` @property
    "attention_mask_data",  # not used by thop (separate code path)
    "out_scale",            # input to ``effective_out_scale`` @property
    "out_scale_sf",         # input to ``effective_out_scale`` @property
})

_THOP_LITERAL_NONE = frozenset({
    "sparse_mla_topk_lens",        # always None in the current trtllm path
    "compressed_kv_cache_pool_ptr",  # always None in the current trtllm path
})
```

## 5. File-level change summary

| File | Net change | Role |
|---|---|---|
| `tensorrt_llm/_torch/attention_backend/interface.py` | +~50 / -~5 | New `AttentionSparseArgs`, `_THOP_EXCLUDED_FIELDS`, `_THOP_LITERAL_NONE`, two `@property` (`mask_type`, `effective_out_scale`), `mrope_config` flatten, dead-field removal |
| `tensorrt_llm/_torch/attention_backend/trtllm.py` | +~170 / -~250 | New `_call_thop_attention`, 12 derived `@property` methods, `_run` rewrite, `forward()` store-back |
| `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` | net -~165 | Rich-object public `is_supported` and `trtllm_gen_attention` — both functions read `attn.X` / `metadata.Y` / `fwd.Z` / `sparse.W` directly throughout. No separate `_*_impl`; ~half of the old impl's kwargs were unused-for-interface-parity and just disappear. |
| `tensorrt_llm/_torch/modules/attention.py` | +0 / -8 | Delete the dict-reconstruct (lines 721-727); change two `AttentionForwardArgs(mrope_config=…)` to flat-field form |
| `tensorrt_llm/_torch/models/modeling_qwen.py` | 2 type fixes | `Tuple[torch.Tensor, int]` → `Optional[dict]` (lines 159, 220) |
| `tensorrt_llm/_torch/models/modeling_seedoss.py` | 2 type fixes | same (lines 90, 154) |
| `tests/unittest/_torch/attention_backend/test_attention_op_sync.py` | new, ~230 lines | The sync test |

## 6. Commit-by-commit walk

1. `6de7ec28dc` — Phases 1, 2, 3, and partial 4: stale annotations, dead-field removal, `mrope_config` flatten, `AttentionSparseArgs`, `mask_type` property, allowlists, base `num_sparse_topk`, `_run` takes `sparse_args`.
2. `12763ee215` — `use_paged_context_fmha` migrated onto `TrtllmAttentionMetadata`. Derived in `__post_init__` from `runtime_features`; per-call overrides become direct field mutations in `forward()`.
3. `2132e9732a` — Last two loose params resolved: `compressed_kv_cache_pool_ptr` → `_THOP_LITERAL_NONE`; `skip_softmax_threshold_scale_factor_*` → `@property` on `TrtllmAttention`.
4. `3d768c7c88` — `_call_thop_attention` wrapper. 89/89 kwarg match with the C++ binding verified by standalone AST cross-check.
5. `70dc3b233b` — initially added the wrappers as `TrtllmAttention` methods. `_run` shrinks 250 → 85 lines. (Superseded by the follow-up below — the wrappers later moved into `trtllm_gen.py` itself.)
6. `c11612f61e` — AST-based sync test + `_THOP_EXCLUDED_FIELDS` extension.
7. `d80a6e7a0e` — move the trtllm-gen wrappers from `TrtllmAttention`
   methods into `trtllm_gen.py` itself as the new public `is_supported` /
   `trtllm_gen_attention`. Eliminates two methods on `TrtllmAttention`;
   `_run` calls the module functions directly.
8. `401d5f96c5` — final cleanup: inline `_trtllm_gen_attention_impl` into
   `trtllm_gen_attention` (body now reads `attn.X`/`metadata.Y`/`fwd.Z`/
   `sparse.W` directly throughout, ~half the impl's old kwargs were
   unused and drop out); unify `is_supported`'s signature with
   `trtllm_gen_attention`'s; derive `phase` from `fwd.attention_input_type`;
   delete `AttentionForwardArgs.is_generation` and update the four call
   sites in `attention.py` / `sparse/dsa.py`.

## 7. Tradeoffs and explicit non-goals

| Tradeoff | Choice | Why |
|---|---|---|
| Single C++ struct vs. four rich objects | Four rich objects | The struct is unused as a value; eliminating it removes a sync surface. |
| `**dict` unpack vs. explicit kwargs | Explicit | Avoid `CALL_FUNCTION_EX` overhead on the hot path. |
| Registry of (kwarg → owner) vs. wrapper-is-the-registry | Wrapper-is-the-registry | The wrapper has to exist anyway; a registry would be a duplicated source of truth. |
| Protocol for `trtllm_gen`'s view of `TrtllmAttention` vs. concrete type | Concrete `TrtllmAttention` | Single internal codepath — Protocol adds maintenance for no gain. |
| Rewrite `trtllm_gen.is_supported` / `trtllm_gen_attention` signatures vs. wrap them in `TrtllmAttention` | Rewrite the public signatures **and** inline the body to read rich-object accessors directly | Final form: `trtllm_gen.py`'s public API takes rich objects, and the body uses `attn.X` / `metadata.Y` / `fwd.Z` / `sparse.W` throughout — no intermediate kwargs translation. About half of the old impl's parameters were unused-for-interface-parity and drop out entirely. |
| Cached vs. uncached `@property` for derived views | Uncached | The properties are cheap (slicing, list comprehensions). `slots=True` on the dataclass blocks `cached_property` without extra machinery. |

### Explicit non-goals

- The upstream `Attention.forward(mrope_config: Optional[dict])` and
  decoder-layer kwarg-style API are intentionally untouched. A future PR may
  flatten them too; this one stops at the `AttentionForwardArgs` boundary.
- ~~`trtllm_gen.is_supported` and `trtllm_gen.trtllm_gen_attention` keep their
  kwarg-style internal signatures.~~ *Done as a follow-up in this PR.* Both
  public functions take rich objects, and the body reads them directly —
  no `_*_impl` companion remains.
- Type-name checking (e.g. `int64_t ↔ int`, `Optional[float] ↔
  std::optional<double>`) is not in the sync test. The test catches all
  name-level drift; type-level coercion bugs would have to surface via
  runtime errors. Adding type checking requires a Python↔C++ type map that
  itself becomes a sync surface — deferred until a real type-coercion bug
  motivates it.

## 8. Validation plan

| Check | Status |
|---|---|
| AST parse on all modified `.py` files | passing locally |
| Standalone AST cross-check between wrapper and binding (89/89 match) | passing locally |
| Pre-commit hooks (codespell, formatter, type-check, AST validators) | passing on every commit |
| `pytest tests/unittest/_torch/attention_backend/test_attention_op_sync.py -v` | pending — needs a working `tensorrt_llm` install on a B200 node |
| `pytest tests/unittest/_torch/attention/` | pending — same |
| Multimodal smoke tests (qwen2vl / seedoss / hunyuan_dense) for the `mrope_config` flatten path | pending |
