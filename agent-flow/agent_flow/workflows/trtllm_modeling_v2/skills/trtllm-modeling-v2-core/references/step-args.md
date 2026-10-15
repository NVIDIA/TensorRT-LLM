# thop_attention runtime argument sourcing

How a target's modeling code fills
`catalog/attention/thop_attention`'s
arguments when running under the TensorRT-LLM engine. The entry's executable
contract
(`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/thop_attention.py`:
`__call__`, `reference`, `is_valid`, `CELLS`) defines what the op computes and
which inputs it is certified over; this reference defines **where each value
comes from at runtime**. Together they are sufficient — no TensorRT-LLM
source reading is needed to integrate the op.

Provenance: the mapping mirrors the in-tree `FallbackFmha.forward`
argument sourcing (`tensorrt_llm/_torch/attention/backends/fmha/fallback.py`)
and is verified end to end by the qwen3-8b/sm_100/tp1 gates (boot gate; full
MMLU 74.98, bit-identical across eager and CUDA-graph configs). Whenever the
engine's attention backend moves, re-derive this table against that function
and re-run the gates.

## The three-source split

Every argument of one attention call belongs to exactly one source, and each
source has its own place in the code. The entry is stateless, so all three
meet at the call, every call:

```python
# module level -- the op's inert arguments for this kind of call
_CALL_CONSTANTS = dict(is_fused_qkv=True, k=None, v=None, ...)  # 3. per-target constants

# forward -- once per step, before the layer loop
step = self.step_args(attn_metadata)                    # 2. per-step batch state
attn_out = torch.empty([num_tokens, Hq * D], ...)       # one buffer, every layer

# the layer loop
attn_out = thop_attention(
    q=qkv, output=attn_out, local_layer_idx=i,          # 1. per-layer data tensors
    attention_sinks=L.sinks,                            #    + per-layer operands
    num_heads=Hq, num_kv_heads=Hkv, head_size=D,        # target geometry
    **step, **_CALL_CONSTANTS,
)
```

`step_args(md)` is a method of each target, not a shared helper: it projects
the engine-prepared metadata onto the op's batch-state arguments, and the two
targets' projections differ. `PrefillTarget` reads `num_contexts` and
`num_ctx_tokens` off the metadata because it holds the mixed batch;
`DecodeTarget` states them as `0` because routing guarantees it — and nothing
else in `DecodeTarget` reads either field (the claims suite gates it, and so
does delivery). The values describe the batch, not the layer, so the
projection is computed once per forward and never inside the loop. Per layer,
`q` is that layer's packed-QKV activation and `output` a caller-allocated
`[num_tokens, Hq*D]` buffer: rows `[:num_tokens]` are overwritten in place,
so one buffer allocated per forward serves every layer, and the call still
states it — `attn_out = thop_attention(..., output=attn_out, ...)`.

`step_args` is also the probe: each target's `_check_step_contract` calls it
on the first forward under `TRTLLM_MODELING_V2_VALIDATE=1`, so every field it
reads is checked to exist. Read a metadata field anywhere else in the forward
and it is unchecked.

## Per-step arguments (engine metadata)

The engine constructs one `TrtllmAttentionMetadata` (the default TRTLLM
attention backend), fills it every iteration, and calls `prepare()` before
the model forward; every field below is valid only after that. The model
receives the object as the `attn_metadata` forward argument.

Graph classes: **ref** = engine-owned persistent buffer, refreshed in
place each step (CUDA-graph replay reads fresh contents automatically);
**host** = Python scalar, baked into a capture (safe: decode-only graphs
always see `num_contexts == 0`); **cfg** = engine-construction constant.

| thop argument | metadata source | class | notes |
|---|---|---|---|
| `sequence_length` | `kv_lens_cuda_runtime` | ref | per-seq total KV (cached + new), CUDA int32 view |
| `host_past_key_value_lengths` | `kv_lens_runtime` | ref | same values, CPU |
| `host_total_kv_lens` | `host_total_kv_lens` | ref | `[2]`: context sum, generation sum |
| `context_lengths` | `prompt_lens_cuda_runtime` | ref | |
| `host_context_lengths` | `prompt_lens_cpu_runtime` | ref | |
| `host_request_types` | `host_request_types_runtime` | ref | 0 = context, 1 = generation |
| `kv_cache_block_offsets` | `kv_cache_block_offsets` | ref | GPU page table, layer-agnostic |
| `host_kv_cache_pool_pointers` | `host_kv_cache_pool_pointers` | cfg | pool base addresses |
| `host_kv_cache_pool_mapping` | `host_kv_cache_pool_mapping` | cfg | row per layer: (pool, layer-in-pool) |
| `workspace_` | `effective_workspace` | ref | engine-owned scratch; the op resizes it in place |
| `tokens_per_block` | `tokens_per_block` | cfg | |
| `max_num_requests` | `max_num_requests` | cfg | |
| `max_context_length` | `max_context_length` | cfg | |
| `max_seq_len` | `max_seq_len` | cfg | |
| `attention_window_size` | `max_seq_len` | cfg | no sliding window: window = full length. Per-layer windows are chosen per call against the full length read at the top of `forward` (known only once the KV manager exists) |
| `num_contexts` | `num_contexts` | host | `DecodeTarget`: the literal `0` |
| `num_ctx_tokens` | `num_ctx_tokens` | host | `DecodeTarget`: the literal `0` |
| `trtllm_gen_jit_warmup` | `trtllm_gen_jit_warmup` | host | engine warmup flag, pass through |
| `use_paged_context_fmha` | `use_paged_context_fmha` | cfg | cached-context path selector (block reuse / chunked prefill) |
| `beam_width` | `effective_beam_width` | cfg | |
| `cache_indirection` | `cache_indirection` | cfg | None unless beam search |
| `block_ids_per_seq` | `block_ids_per_seq` | cfg | None unless FlashMLA |
| `max_context_q_len_override` | `max_context_q_len_override` | cfg | |
| `is_cross` | `is_cross` | cfg | |
| `is_spec_decoding_enabled` | `is_spec_decoding_enabled` | cfg | |
| `use_spec_decoding` | `use_spec_decoding` | host | |
| `is_spec_dec_tree` | `is_spec_dec_tree` | cfg | |
| `spec_decoding_generation_lengths` | same name | ref | None with spec-dec off |
| `spec_decoding_position_offsets_for_cpp` | same name | ref | " |
| `spec_decoding_packed_mask` | same name | ref | " |
| `spec_decoding_bl_tree_mask_offset` | same name | ref | " |
| `spec_decoding_bl_tree_mask` | same name | ref | " |
| `spec_decoding_target_max_draft_tokens` | `max_total_draft_tokens` | cfg | note the rename |
| `spec_bl_tree_first_sparse_mask_offset_kv` | same name | ref | None with spec-dec off |
| `num_sparse_topk` | `num_sparse_topk` | cfg | |
| `flash_mla_tile_scheduler_metadata` | same name | ref | None outside FlashMLA |
| `flash_mla_num_splits` | same name | ref | " |

Pass runtime-owned values through exactly as prepared — feature behavior
(block reuse, CUDA graphs, speculative decoding plumbing) is the engine's
responsibility and is validated by the target's gates, not second-guessed
in modeling code.

## Per-target constants

Everything the in-tree path reads from the attention module or its
forward args is, for one target, a compile-time constant: it encodes
*which kind of attention call this target makes*. For a bf16 packed-QKV
causal GQA target with RoPE applied outside the op (the
`fused_qk_norm_rope` pairing), the constants are exactly the standard cell
of the entry's `CELLS`, held as one module-level dict and passed to every call:

- `is_fused_qkv=True`, `k=None`, `v=None`, `update_kv_cache=True`,
  `predicted_tokens_per_seq=1`, `attention_input_type=0` (one call serves
  pure-prefill / pure-decode / mixed batches), `mask_type=1` (causal),
  `is_mla_enable=False`, `quant_mode=0`, `q_scaling=1.0`.
- In-kernel RoPE disabled: `position_embedding_type=0` and every `rope_*`
  argument at the entry's inert placeholder. **The placeholders are not
  the model's rope config** (`rope_base=10000.0` even when the model's
  theta is 1e6) — rotation happens outside, the values are dead.
- All quantization scales, MLA arguments, mRoPE / cross / relative-bias /
  sparse / helix arguments at their inert values.

MLA targets take their per-phase constants from the entry's MLA cells the
same way (separate context/generation call shapes, `attention_input_type`
1/2), stated at each call on the target that makes it: `PrefillTarget` makes
the context call and the generation one (a mixed batch splits its rows),
`DecodeTarget` only the generation one. The per-step
table above applies unchanged — both consume the same prepared metadata.

Two forward-looking notes: `helix_*` are metadata-sourced in-tree but are
None whenever helix CP is off — a non-CP target may hold them as
constants. Quantized-KV targets turn `kv_scale_orig_quant` /
`kv_scale_quant_orig` from `None` constants into small fp32 tensors built
in `post_load_weights` (checkpoint-calibrated; leaving them `None` under
fp8 KV does **not** fault — measured, it is read as a silent `1.0`, so a
miscalibrated target is a wrong-number bug rather than a crash) — see the
entry's `CELLS` and `note` for what a certification extension must cover
first.

## Fail-fast discipline

The metadata attribute names above are private trtllm surface with no
stability promise. A target consuming them must:

1. Read them in `step_args` and nowhere else on the hot path, so the opt-in
   step-contract check — which calls `step_args` — covers every one. There is
   no `_STEP_FIELDS` tuple to keep in sync: the projection is the
   declaration. **Exception: MLA's cached-KV fields are conditional.**
   `enable_context_mla_with_cached_kv`,
   `ctx_cached_token_indptr`, `ctx_kv_indptr`,
   `ctx_uncached_token_indptr`, `max_ctx_seq_len`, `max_ctx_kv_len` and
   `num_ctx_cached_tokens` exist only when block reuse is on — with
   `enable_block_reuse: false` they are absent, not False. Reading them
   unconditionally turns a legal config into a hard failure; select the
   context flavor once by their presence (on the core, unconditionally) and
   pass them only on that flavor.
2. Expect `TrtllmAttentionMetadata` — this table is meaningless for other
   backends' metadata. `route` does not read the attention backend, so a
   different backend's metadata fails in `step_args` on its first missing
   field — not an assert in the target.
3. Expect `position_ids` int32 and flatten with
   `reshape(position_ids, [-1])` before ops that consume positions (the
   engine passes an int32 `[1, T]` view of a persistent buffer).

   **Not on an mRoPE config, where that flatten is the bug.** If the config
   declares mRoPE, the engine builds the `[3, 1, T]` layout **even for a
   text-only batch** — the rank is keyed on `rope_scaling['type'] == 'mrope'`
   alone and is held constant between warmup and serving — and broadcasts the
   scalar positions across all three axes, so row 0 is the text position.
   Flattening the whole buffer yields a `[3T]` **copy**, which the op rejects,
   and the error names contiguity rather than shape: `position_ids must be
   contiguous`. Take row 0 for a text-only path, or pass the `[3, T]` view the
   op documents for mRoPE. A text-only target built from a multimodal
   checkpoint hits this even though nothing about its workload is multimodal.
4. Declare nothing else. `REQUIRED_TRTLLM_OPS` is gone: a renamed custom op
   fails at its call, which names the place, and
   `tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py`
   checks only the pybind attention entry point. There is no version to pin
   against — the target moves with the trunk. An entry's certification,
   likewise, is keyed by GPU architecture (`ARCHS`) and re-proved by its own
   test on every commit; absent means unknown.
