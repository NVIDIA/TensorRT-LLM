# Worklog: Gemma4 Multimodal Custom Attention Mask

## Problem Statement

Gemma4 VLM requires bidirectional attention within media token blobs and
causal attention for text tokens. The `token_type_ids` tensor (0 = text,
non-zero = media blob ID) drives the mask construction.

## Architecture Decisions

### VLM Export Pattern (TextModelExportInfo)

Switched `Gemma4ForConditionalGeneration` from `FullModelExportInfo` to
`TextModelExportInfo`. The `Gemma4ForCausalLM` (including `lm_head` +
softcapping) is the export target. The outer wrapper stays un-exported
and handles `token_type_ids` plumbing.

### Input Processor (Gemma4ADInputProcessor)

Every request gets `token_type_ids` in `py_multimodal_data` — text-only
requests get all-zeros, multimodal requests get blob IDs from the HF
processor. This solves mixed-batch scenarios where `_store_extra_arg`
concatenates per-request tensors.

## Issue: In-Graph Mask Computation Fails During Warmup

### Background

The `InjectCustomAttentionMask` transform inserts mask computation nodes
directly into the FX graph. These nodes take `token_type_ids` as input
and produce a `[batch, 1, seq, seq]` bool mask wired to each
`torch_attention` node.

After the KV cache transform, `torch_attention` nodes become
`triton_paged_mha_with_cache` calls. The mask flows as the
`custom_attn_mask` argument. When present (not `None`), the cached op
uses a per-sequence fallback path (`triton_paged_context_with_custom_mask`)
that loops over sequences and calls `torch_attention` individually.

### The Warmup Problem

Several initialization-time forward passes happen before any real
request arrives:

1. **`resize_kv_cache`** — dummy forward to discover output shapes and
   resize the KV cache buffer.
2. **CUDA graph warmup** — forward passes at each `cuda_graph_batch_size`
   to capture graphs.
3. **`set_example_sequence`** — initial example to set up the
   `SequenceInfo` input buffer layout.

During these passes:

- `token_type_ids` comes from the default factory (all-zeros, matching
  `input_ids` shape).
- The in-graph mask computation runs and produces a valid causal-only
  mask tensor — it is **not** `None`.
- `triton_paged_mha_with_cache` sees a non-`None` mask and enters the
  per-sequence fallback path.
- The fallback path calls `torch_attention` per sequence, but during
  warmup the KV cache may be empty / uninitialized, leading to
  **`IndexError: index 0 is out of bounds for dimension 0 with size 0`**.

### Why Band-Aid Fixes Don't Work

Adding `numel == 0` checks or `seq_len == 0` skips in the triton paged
op are treating symptoms. The fundamental issue is:

- **The mask is always computed inside the graph** — there is no way to
  conditionally produce `None` in an FX graph (no control flow).
- **The fallback path is entered on every forward** even when the mask is
  pure causal (all-zeros `token_type_ids`), wasting compute and hitting
  edge cases.

### Additional Issues Encountered with In-Graph Approach

1. **Symbolic shape error** — `torch.zeros(batch_size, seq_len, ...)` at
   Python level fails when `batch_size`/`seq_len` are symbolic ints from
   `torch.export`.
2. **Device mismatch** — `arange(seq_len)` defaults to CPU but mask
   operands are on meta device during shape propagation.
3. **Undefined symbolic variable** — `seq_len - 1` as a Python expression
   becomes an unresolvable `s70` reference after graph recompile.

Each of these required a separate fix, making the in-graph approach fragile.

## Proposed Solution: Compute Mask Outside the Graph

Move mask computation from FX graph nodes to the **wrapper's `forward()`**
(Python-level). The graph receives the finished `custom_attn_mask` tensor
(or `None`) as an input, not `token_type_ids`.

### How It Works

1. **Mask provider** adds `custom_attn_mask` as an **optional graph input**
   (default `None`) and wires it to `torch_attention` nodes. No
   computation nodes in the graph.

2. **Wrapper `forward()`** receives `token_type_ids` from `extra_args`,
   computes the mask using plain PyTorch ops, and passes
   `custom_attn_mask=mask` to the exported graph.

3. **During warmup / text-only / decode**, the wrapper passes
   `custom_attn_mask=None`. The cached attention op sees `None` and uses
   the fast `triton_paged_context` kernel (no per-sequence fallback).

### Runtime Flow

**Prefill with images:**
1. `Gemma4ADInputProcessor` produces `token_type_ids` with blob IDs.
2. Executor concatenates per-request tensors → `[1, total_tokens]`.
3. Wrapper receives `token_type_ids` via `**kwargs`.
4. `token_type_ids.any()` is `True` → wrapper computes
   `custom_attn_mask` `[batch, 1, seq, seq]` bool mask using plain
   PyTorch ops (blob detection, bidirectional-within-blob + causal).
5. Passes `custom_attn_mask=mask` to exported graph.
6. `triton_paged_mha_with_cache` uses per-sequence fallback with the
   mask — bidirectional for media blobs, causal for text.

**Text-only prefill:**
1. `token_type_ids` all zeros → `any()` is `False`.
2. Wrapper passes `custom_attn_mask=None`.
3. `triton_paged_context` fast kernel — standard causal attention.

**Mixed batch (image + text-only requests):**
1. `token_type_ids` has non-zero for image requests, zeros for text-only.
2. `any()` is `True` → compute mask.
3. Mask is causal for text-only sequences (zeros → no blobs) and
   bidirectional for image sequences. Correct per-sequence behavior.

**Decode:**
1. No `token_type_ids` in `extra_args` (decode has no `py_multimodal_data`).
2. Wrapper defaults to `custom_attn_mask=None`.
3. Fast decode kernel (ignores mask entirely).

**Warmup / CUDA graph capture / resize_kv_cache:**
1. `token_type_ids` absent or all zeros.
2. Wrapper passes `custom_attn_mask=None`.
3. Standard causal path — no per-sequence fallback, no empty KV indexing.

### Injector Role (Simplified)

The mask provider no longer computes anything inside the graph. It:
1. Adds `custom_attn_mask` as an optional graph input placeholder
   (default `None`).
2. Wires the placeholder to each `torch_attention` node's `attn_mask`.
3. The KV cache transform sees `attn_mask` is a graph node → extracts
   via `get_dynamic_inputs` → passes to `triton_paged_mha_with_cache`.

### Mask Computation (Moves to Wrapper)

The same logic currently in `_build_gemma4_token_type_mask` (graph nodes)
moves to a Python method on `Gemma4ForConditionalGeneration`:

1. **Blob detection**: `ne(token_type_ids, 0)` → shift + compare for type
   changes → `cumsum` for blob IDs.
2. **Bidirectional within blobs**: unsqueeze q/k blob IDs → `eq` (same
   blob) → `bitwise_and` with media query.
3. **Causal mask**: `arange(seq_len)` → `le(pos_q, pos_k)`.
4. **Combine**: `bitwise_or(causal, bidirectional)` → unsqueeze for head dim.

Result: `[batch, 1, seq, seq]` bool mask — same output, no graph nodes.

### Benefits

- No symbolic shape issues (plain PyTorch ops, not FX graph nodes).
- No device mismatch (tensors on the actual runtime device).
- No warmup breakage (`None` mask → standard causal path).
- Pure causal (text-only) requests use the optimized kernel, not the
  per-sequence fallback.
- Model-specific logic stays in the wrapper, not in the graph.
