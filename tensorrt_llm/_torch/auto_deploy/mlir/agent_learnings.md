# MLIR Elementwise Fusion — Issues & Fixes Log

This document records bugs encountered while developing and deploying the
`mlir_elementwise_fusion` pass, their root causes, and the fixes applied.
It serves as a reference for future development to avoid reintroducing
similar issues.

______________________________________________________________________

## E2E Testing

The MLIR elementwise fusion pass is enabled by default in
`tensorrt_llm/_torch/auto_deploy/config/default.yaml`. To test it end-to-end,
run AutoDeploy's `build_and_run_ad.py` script with any supported model:

```bash
# General pattern (uses default.yaml which has mlir_elementwise_fusion enabled)
python examples/auto_deploy/build_and_run_ad.py \
    --args.model=<HF_MODEL_PATH_OR_NAME>

# Example with a specific config overlay (e.g. for Nano/Mamba models)
python examples/auto_deploy/build_and_run_ad.py \
    --args.model=$MODEL \
    --args.yaml-extra=examples/auto_deploy/nano_v3.yaml
```

**What to verify in the logs:**

- `[APPLY] Decomposed N high-level ops into primitives` — FX ops lowered to MLIR
- `[APPLY] Discovered N fusible subgraphs (total ops: M)` — subgraph discovery succeeded
- No `RuntimeError` or `KeyError` during kernel generation
- Successful model inference output at the end

**Models available on this cluster** (under `$LLM_MODELS_ROOT`):

- `Qwen2.5-3B-Instruct` — small, fast sanity check
- `Qwen3/Qwen3-0.6B` — smallest Qwen3 model
- `DeepSeek-R1/DeepSeek-R1` — full 671B MoE (requires multi-GPU)

______________________________________________________________________

## 1. Missing FX↔MLIR converter handlers for new ops

**Symptom:** `ValueError: FX node not found for MLIR value` during
MLIR-to-FX back-conversion on Qwen3.5-35B.

**Root cause:** `AdSigmoid`, `AdExp`, and `AdSoftplus` were added to the
MLIR dialect (`dialect.py`) and Triton emitter (`triton_emitter.py`) but
not to the FX-to-MLIR converter (`fx_to_mlir.py`) or the MLIR-to-FX
converter (`mlir_to_fx.py`). Unfused instances of these ops were silently
skipped during back-conversion, leaving their output SSA values unregistered
in `_value_map`. Downstream fused kernels that referenced those values then
crashed.

**Fix:** Added `AdSigmoid`, `AdExp`, `AdSoftplus` handlers to both
converters. (commit `86ffd84a43`)

**Lesson:** When adding a new op to the dialect, update ALL four touchpoints:
`dialect.py`, `fx_to_mlir.py`, `mlir_to_fx.py`, and
`codegen/triton_emitter.py` (the `_EMIT` table).

______________________________________________________________________

## 2. Fused op insertion ordering in multi-GPU graphs

**Symptom:** `ValueError: FX node not found for MLIR value produced by 'ad.opaque' (trtllm_dist_all_reduce)` on Qwen3.5-35B with TP=4.

**Root cause:** `subgraph_replace.py` always inserted the fused op before
`subgraph.ops[0]`. In multi-GPU graphs, a fusible subgraph can span across
non-fusible ops (e.g. `trtllm_dist_all_reduce`). If the subgraph's first op
(e.g. a sigmoid in the shared-expert gating path) appears *before* the
all-reduce, but the subgraph also consumes the all-reduce's output (via a
downstream add), the fused op was placed before the all-reduce — a
topological ordering violation. During back-conversion the all-reduce hadn't
been processed yet, so its output was missing from `_value_map`.

**Example pattern:**

```
sigmoid (fusible, first subgraph op) ← fused op was inserted here
  ...
all_reduce (non-fusible, NOT in subgraph)
  ...
add(all_reduce_out, sigmoid_chain_out) (fusible, in subgraph)
```

**Fix:** In `subgraph_replace.py`, scan the block for the latest input
producer. If it appears after `subgraph.ops[0]`, insert the fused op after
that producer instead of before the first subgraph op.
(commit `d6d4e1f892`)

**Lesson:** Fusible subgraphs can span across non-fusible ops in the MLIR
block. The fused op must be placed after ALL its input producers, not just
before the first subgraph op.

______________________________________________________________________

## 3. `_get_ncols` returning wrong value for mixed-width subgraphs

**Symptom:** `RuntimeError: shape '[1, 4096, 2048]' is invalid for input of size 4096` (reshape failure after MLIR fusion on Qwen3.5).

**Root cause:** `_get_ncols()` in `triton_emitter.py` returned the last-dim
of the *first* 2D input, not the *maximum*. For subgraphs mixing a narrow
gating input `(-1, 1)` with wide hidden-state inputs `(-1, 2048)`, the
kernel used `N_COLS=1` instead of `N_COLS=2048`, producing a 1-element
output per row instead of 2048.

**Fix:** Changed `_get_ncols` to return the maximum last-dim across all
highest-rank inputs. Also fixed `ref_input_idx` to prefer the widest input
for shape reference. (commit `7a4752df2e`)

**Lesson:** When a subgraph fuses ops with different input widths (e.g.
scalar gating × wide hidden states), `N_COLS` must be the widest dimension.
The reference input for the launcher must also be the widest.

______________________________________________________________________

## 4. Narrow same-rank inputs loaded with wrong stride

**Symptom:** `RuntimeError: Triton Error [CUDA]: an illegal memory access`
in MLIR-generated kernel on Qwen3.5 (after fixing N_COLS).

**Root cause:** Inputs with the same rank but smaller last-dim (e.g.
`(-1, 1)` gating scalar in a subgraph with `N_COLS=2048`) were loaded with
`row_off + offs` where `offs = arange(0, 2048)`, reading 2048 elements from
a tensor with only 1 element per row — a massive out-of-bounds read.

**Fix:** Added `narrow_flags` detection. Narrow inputs (same rank, last dim
\< N_COLS) are loaded as scalars: `tl.load(ptr + pid * inp_last_dim)`. Triton
broadcasts the scalar when used with full-width tensors.
(commit `7a4752df2e`)

**Lesson:** The emitter must distinguish three input categories:

1. **Broadcast** (lower rank, e.g. 1D weight): load with `offs` only
1. **Narrow** (same rank, last dim \< N_COLS): load as scalar
1. **Full-row** (same rank, last dim = N_COLS): load with `row_off + offs`

______________________________________________________________________

## 5. Non-contiguous input stride used for contiguous output stores

**Symptom:** `CUDA error: illegal memory access` in MLIR-generated RMSNorm
kernel on Qwen3.5 layer 3 Q-norm. The input comes from `aten.chunk` and is
non-contiguous.

**Root cause:** The Triton kernel used a single `row_stride` (from
`input.stride(-2)`) for both loads and stores. For non-contiguous inputs
(e.g. from `aten.chunk` on the last dim), `stride(-2)` can be larger than
`N_COLS`. Example: shape `(B, S, 4, 256)` from chunk of 512 has
`stride(-2) = 512`. Loading with stride 512 is correct (reads every other
256-element block). But the output is allocated contiguous via
`torch.empty_like` with `stride(-2) = 256`. Storing with stride 512 into a
contiguous buffer writes out of bounds.

**Fix:** Added separate `out_row_off = pid * N_COLS` in the kernel preamble.
Stores use `out_row_off + offs` (contiguous output stride), while loads
continue using `row_off + offs` (input stride from `stride(-2)`).
(commit `7a4752df2e`)

**Lesson:** Input and output strides can differ when inputs are
non-contiguous views. The emitter must use separate offsets: `row_off` for
loads (from `stride(-2)`) and `out_row_off` for stores (always `N_COLS`
since outputs are contiguous).

______________________________________________________________________

## 6. `run_shape_prop` concretizes symbolic dimensions

**Symptom:** Downstream transforms (e.g. FLA gated delta rule) see concrete
tensor shapes (`2x4x...`) instead of symbolic (`s44xs70x...`) after MLIR
fusion.

**Root cause:** `mlir_elementwise_fusion` had `run_shape_prop: true` in
`default.yaml`. Shape propagation replaces symbolic dimensions with their
concrete profiling-time values. This is harmless for most downstream
transforms but can cause issues for transforms that rely on symbolic shapes.

**Fix:** Set `run_shape_prop: false` in `default.yaml` for
`mlir_elementwise_fusion`. The fused kernels' `register_fake` implementations
already handle shapes correctly via the reference input mechanism.
(commit `7a4752df2e`)

**Lesson:** Prefer `run_shape_prop: false` unless the transform specifically
needs re-propagated shapes. Shape propagation loses symbolic dimension info.

______________________________________________________________________

## 7. Interaction with multi_stream_moe transform

**Symptom:** `multi_stream_moe` produces 0 matches when
`mlir_elementwise_fusion` is enabled.

**Root cause:** MLIR fusion absorbs the merge `add` nodes at MOE layer
boundaries into fused kernels. The `_find_merge_add` function in
`multi_stream_moe.py` only looked for `aten.add.Tensor`, which no longer
exists. Additionally, `aten.sym_size.int` nodes (scalar shape computations)
falsely triggered the merge detection, and the fused merge node has 3+
inputs (shared, routed, residual) instead of the expected 2.

**Fix:** (in `multi_stream_moe.py`, commit `3380840692`)

- Replaced `_find_merge_add` with `_find_merge_node`: tracks a "MOE forward
  cone" and finds the first node with a tensor-producing `call_function`
  input outside the cone. Uses `has_shape()` to filter out SymInt nodes.
- Generalized input classification from binary (routed/shared) to three-way
  (routed/pass-through/shared) using `moe_ancestors`.

**Lesson:** Graph transforms that pattern-match specific op types
(`aten.add.Tensor`) will break when upstream fusion replaces those ops.
Design pattern matchers to be op-type-agnostic where possible.

______________________________________________________________________

## 8. Fusion scope > individual kernel speed

**Context:** Investigated whether to preserve `flashinfer_rms_norm` as an
opaque node (leveraging its ~1.7x faster standalone kernel) vs decomposing
it for MLIR fusion.

**Finding:** Under CUDA graph replay (production), MLIR Triton kernels match
FlashInfer speed (1.00x standalone, 1.14-1.85x fused). The ~9µs gap in
standalone benchmarks is pure Python dispatch overhead eliminated by CUDA
graphs. Preserving FlashInfer broke the `add + add + rmsnorm` fusion into
2 separate kernels with a memory round-trip, making e2e *slower*.

**Decision:** Decompose all rmsnorm variants. Fusion benefit (keeping
intermediates in registers) outweighs any standalone kernel speed advantage.

**Lesson:** Always benchmark e2e under CUDA graphs, not standalone kernel
speed. Fusion scope (fewer kernels, fewer memory round-trips) matters more
than individual kernel optimization.

______________________________________________________________________

## 9. PyTorch 64-argument limit for custom op schemas

**Symptom:** `RuntimeError: The function schema has 257 arguments but this PyTorch build only supports 64` on DeepSeek-R1-Distill-Qwen-32B (64 layers)
during AutoDeploy benchmark on H100.

**Root cause:** The greedy subgraph discovery merges all connected fusible ops
into maximal subgraphs. For models with many layers, shared weights (e.g. a
single rmsnorm weight used across layers) connect elementwise ops from all
layers into one subgraph. DeepSeek-R1-Distill-Qwen-32B had 4 subgraphs with
1035 total ops, one subgraph having 257 external inputs. Since each input
becomes a positional `torch.Tensor` argument in the `torch.library.custom_op`
registration, this exceeded PyTorch's hard limit of 64 schema arguments.

**Fix:** Added a post-discovery splitting step in `subgraph_discovery.py`.
After building maximal subgraphs, any subgraph with more than 64 inputs is
split into smaller partitions via a placement-aware greedy walk. Each
partition respects two constraints: (1) external input count ≤ 64, and
(2) valid block placement — the latest input producer must come before the
earliest output consumer in the MLIR block. Values produced in one partition
and consumed in another become cross-partition I/O automatically.

**Lesson:** `torch.library.custom_op` has a 64-argument schema limit.
Subgraph discovery must enforce this as a hard constraint. Models with many
layers and shared weights are particularly prone to creating oversized
subgraphs.

______________________________________________________________________

## 10. Stale subgraph inputs after sequential replacement

**Symptom:** `KeyError` in `triton_emitter.py` (`val_names` missing an
operand) when the second or later subgraph is processed. Diagnostic shows
`producer=ad.opaque, producer_in_subgraph=False, operand_in_inputs=False`.

**Root cause:** Subgraphs are discovered in one pass but replaced
sequentially. When subgraph A is replaced with a fused `ad.opaque`,
`SSAValue.replace_by()` redirects downstream operands to the fused op's
outputs. Subgraph B, discovered before the replacement, still holds OLD
SSAValues in its `inputs` list. The ops' operands now point to NEW
SSAValues (from the fused op), but `subgraph.inputs` was never updated.
The triton emitter seeds `val_names` from the stale inputs, so the ops'
actual operands don't match.

**Fix:** Added `FusibleSubgraph.refresh_inputs()` which recomputes external
inputs from the ops' current operands. Called in
`mlir_elementwise_fusion.py` before `generate_kernel_from_subgraph()`.

**Lesson:** When subgraphs are discovered ahead-of-time but replaced
iteratively, earlier replacements invalidate later subgraphs' metadata.
Always refresh I/O from the current IR state before operating on a subgraph.

______________________________________________________________________

## 11. Fused op placement violation from cross-layer partitions

**Symptom:** `ValueError: FX node not found for MLIR value produced by 'ad.opaque'` during MLIR-to-FX back-conversion. The fused op producing the
needed value was never converted — diagnostic shows
`any_producer_result_mapped=False`.

**Root cause:** Subgraph splitting (issue #9) only considered the input
count limit, not block placement. When a subgraph spans the entire model
(e.g. rmsnorm ops from all 36 layers of Qwen2.5-3B), a partition could
contain ops from many layers. The partition's output consumers (SwiGLU ops)
are interspersed with its input producers (attention ops) throughout the
block:

```
attn_0 → rmsnorm_0 → swiglu_0 → attn_1 → rmsnorm_1 → swiglu_1 → ...
```

The fused op must be placed after ALL input producers (e.g. `attn_35`) but
also before ALL output consumers (e.g. `swiglu_0`). When `attn_35` appears
after `swiglu_0` in the block, no valid position exists. The replacement
inserted the fused op after the latest input — i.e. after `attn_35` — which
is after most SwiGLU consumers. The MLIR-to-FX converter walks the block in
order, encounters the SwiGLU before the fused op, and fails.

**Fix:** Made `_split_subgraph` placement-aware. For each candidate
partition it tracks both `max_input_pos` (latest input producer in the
block) and `min_consumer_pos` (earliest output consumer). When adding an op
would make `max_input_pos >= min_consumer_pos`, the partition is flushed and
a new one started. This ensures every partition can be placed at a valid
block position.

**Lesson:** Subgraph splitting must respect block topology, not just input
count. A partition is only viable if all its input producers precede all its
output consumers in the MLIR block. For models where fusible ops are
interleaved with non-fusible consumers (the common transformer pattern),
this naturally produces per-layer partitions.

______________________________________________________________________

## 12. Topological ordering of ops within subgraphs

**Symptom:** Intermittent `KeyError` in `triton_emitter.py` where an op's
operand (produced by another op in the same subgraph) hasn't been added to
`val_names` yet.

**Root cause:** The original code sorted subgraph ops using block-level
position (`topo_order.get(o, 0)`) which can produce incorrect ordering when
group merging changes the ops' relative positions, or when the fallback
value 0 places an op at the front incorrectly.

**Fix:** Replaced `sort(key=topo_order.get(o,0))` with Kahn's algorithm
(`_topo_sort_subgraph`) that builds a proper topological sort from the
subgraph's internal dependency edges. Ties are broken by block position for
determinism.

______________________________________________________________________

## 13. Unsupported dtype crashes FX-to-MLIR conversion

**Symptom:** `ValueError: Unsupported torch dtype for MLIR conversion: torch.complex64` on Llama-4-Scout during `FXToMLIRConverter.convert()`.

**Root cause:** The FX graph contains tensors with `complex64` dtype (from
rotary position embeddings). `torch_dtype_to_mlir()` in `dialect.py` has no
mapping for complex types. Since the MLIR pass is enabled by default, any
model with unsupported dtypes in its FX graph would crash.

**Fix:** Wrapped the `converter.convert()` call in
`mlir_elementwise_fusion.py` with a `try/except ValueError` that logs a
warning and returns the original graph unchanged (`skipped=True`). The model
runs correctly without MLIR fusion.

**Lesson:** The MLIR pass must be fail-safe — unsupported ops or dtypes
should cause a graceful skip, not a crash. Any new dtype encountered in the
wild should be added to `_TORCH_TO_MLIR_DTYPE` if MLIR/Triton can handle it,
or left to trigger the graceful skip.

______________________________________________________________________

## 14. MLIR-to-FX back-conversion drops GraphModule methods

**Symptom:** `AttributeError: 'GraphModule' object has no attribute 'get_input_embeddings'` on Mistral-Small-3.1 (multimodal model) during
CUDA graph capture, after MLIR fusion succeeds.

**Root cause:** `MLIRToFXConverter.convert()` created a fresh
`GraphModule(original_gm, graph)`. The `GraphModule` constructor copies
submodules and parameters but does NOT copy class methods defined on the
original model (e.g. `get_input_embeddings`). Downstream code
(`modeling_mistral3.py`) calls `self.language_model.get_input_embeddings()`
on the returned GM, which no longer has that method.

**Fix:** After constructing the new `GraphModule`, iterate over the original
GM's public callable attributes and copy any that are missing on the new GM
via `setattr`. This preserves methods like `get_input_embeddings` without
modifying the graph structure.

Note: an in-place approach (`original_gm.graph = graph; recompile()`) was
tried first but broke DeepSeek-V3 due to lost shape/export metadata.

**Lesson:** When round-tripping through an IR (FX → MLIR → FX), the
reconstructed `GraphModule` must preserve not just the graph and parameters
but also any callable methods that downstream code relies on. Test with
multimodal models that have extra methods on their graph modules.
