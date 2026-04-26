<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepSeek V4 Triton Kernel Strategy

This strategy targets the AutoDeploy DeepSeek V4 graph emitted by the current branch. It is based on
`ad_dsv4_ramp_up.md`, `deepseek_v4_ad_plan.md`, `deepseek_v4_sparse_attention_next_steps.md`,
`dsv4_sharding_guide.md`, the `dsv4-graphs/` dumps, and the existing AutoDeploy custom ops under
`tensorrt_llm/_torch/auto_deploy/custom_ops/`.

The short version: make DeepSeek V4 attention the primary Triton workstream, keep the existing
FineGrained FP8 and MXFP4 MoE paths as the initial production baseline, and add narrow DSV4-specific
kernels only where the graph still exposes reference or Python-heavy behavior.

## Current Graph Shape

The current fixed validation fixture is:

```text
ad_run_logs/graph_dumps/deepseek_v4_flash_5layer_sampling_workspace_20260425_131939/084_compile_compile_model.txt
```

W4R-05 also attempted a fresh current-branch reduced run with graph dumping enabled. That run stopped
before final compile because executor initialization OOMed, so its dump is partial:

```text
ad_run_logs/graph_dumps/deepseek_v4_flash_5layer_w4r05_current_20260425_2358/077_cache_init_initialize_cache.txt
```

It is a reduced five-layer DSV4 graph, but it contains the key sharded surfaces needed for kernel
planning:

- The completed compile fixture lowers cached sparse attention to
  `torch_deepseek_v4_sparse_attention_v2_with_cache` in all five layers. The fresh partial graph selects
  `triton_deepseek_v4_sparse_attention_v2_with_cache` in all five layers. NB-04 adds a real ratio-0 SWA
  device path for prefill and mixed batches; ratio-4 and ratio-128 compressed prefill/mixed batches
  still need the active-token packing plus local/compressed attention workstream.
- Observed local-rank attention classes are layers 0/1 ratio-0 SWA with `topk_idxs` width `128`, layers
  2/4 ratio-4 indexer with width `640`, and layer 3 ratio-128 with width `192`.
- Ratio-4 layers still contain inlined indexer construction and elementwise FP-style quantization before
  `aten.topk(..., 512)`, plus an NCCL all-reduce over `s72xs70x2048` indexer scores.
- Attention and shared-expert dense projections are FineGrained FP8 with E4M3 weights and E8M0-backed
  `weight_scale_inv` buffers.
- `wo_a` is a TP-sharded DSV4 grouped FineGrained FP8 projection in this fixture:
  `[B, S, 1, 4096] -> [B, S, 1, 1024]`.
- Routed experts lower to `triton_deepseek_v4_mxfp4_moe_from_routing` with packed uint8 FP4 blocks,
  raw uint8 E8M0 scale bytes, `top_k=6`, and `swiglu_limit=10.0`; this is true in both the completed
  compile fixture and the fresh partial graph.
- NCCL all-reduces are visible after `wo_b`, after ratio-4 indexer score reduction, after routed MoE,
  and after shared expert down projection.

## Priority 1: Cached Sparse Attention

The first production Triton target should be a cached DSV4 sparse attention op, not a rework of classic
MLA. Existing `triton_mla.py` is useful as a reference for online softmax, block scheduling, and CUDA
graph discipline, but it does not match DSV4's local window, compressed-memory rows, sink logit, or
ratio-specific compression semantics.

Target op family:

```text
torch_deepseek_v4_sparse_attention_v2
  -> triton_deepseek_v4_sparse_attention_with_cache
```

Core contract:

- Inputs: local-rank `q [T, H_local, 512]`, current-token KV, attention sink, sequence metadata, page
  tables, SWA cache, MHC cache, compressor/indexer state, and layer constants.
- Outputs: BF16 attention output `[T, H_local, 512]`, with optional `out=` for CUDA graph replay.
- Dtypes: BF16 for query/RoPE dims and initial values; FP8 E4M3 plus E8M0 scale bytes for NoPE cache
  once the BF16 path is correct.
- Scope: ratio-0 SWA now has CUDA BF16 decode, prefill, and mixed-batch coverage without the Torch
  cached/source attention references. Ratio-4 and ratio-128 currently have CUDA BF16 decode coverage;
  their prefill/mixed path remains the next sparse-attention blocker.

Kernel structure:

1. One program owns a tile of tokens and heads. For decode, start with `BLOCK_T=1..4`,
   `BLOCK_H=1..4`, `BLOCK_D=64/128` vector lanes over head dim.
2. Gather SWA rows from paged cache for the last `window_size=128` positions.
3. Gather visible compressed rows from MHC cache:
   - ratio-128 layers can initially scan all emitted compressed rows.
   - ratio-4 layers need the indexer top-k path before they are performance representative.
4. Compute dot products in FP32 accumulation over `D=512`.
5. Add the per-head sink as an extra softmax logit that contributes probability mass but no value.
6. Use online softmax over local-window and compressed chunks so K does not require a giant logits
   scratch tensor.
7. Accumulate values and write BF16 output.

Initial Triton specializations:

| Layer class | K-select shape | First kernel |
| --- | ---: | --- |
| Ratio 0 | `<=128` | SWA-only cached sparse attention for decode/prefill/mixed |
| Ratio 128 | `128 + <=ceil(seq/128)` | SWA + all visible compressed rows |
| Ratio 4 | `128 + selected compressed rows` | SWA + precomputed compressed index list |

Do not start with a single mega-kernel that also performs Q projection, compressor projection, and
output projection. That would make correctness and cache semantics harder to isolate. The first Triton
attention kernel should consume already-formed Q/KV/cache tensors and exactly match the source op's sink
and duplicate-index semantics.

## Priority 2: Cache Update And Compression Kernels

The next Triton target should replace the reference helper ops in `deepseek_v4_kernels.py` and the
Python-style paged loops in `deepseek_v4_attention.py`.

Implement these as separate kernels with stable intermediate contracts:

1. `triton_deepseek_v4_q_rmsnorm_rope`
   - Input after `wq_b`: `[T, H_local, 512]`.
   - Per-head RMSNorm over 512 dims, then RoPE on trailing 64 dims.
   - BF16 output.

2. `triton_deepseek_v4_kv_norm_rope_cache_insert`
   - Input after `wkv`: `[T, 512]`.
   - RMSNorm, RoPE on trailing 64 dims, optional NoPE FP8 quant on first 448 dims.
   - Writes SWA pages. Use raw E8M0 bytes for scales; do not numerically cast scale tensors to uint8.

3. `triton_deepseek_v4_compressor_pool_norm_rope`
   - Ratio 128 path: simple `[128, 512]` softmax-pool to one row.
   - Ratio 4 path: overlapping two-channel transform, then softmax-pool to one row.
   - Writes emitted MHC rows only when a compression block closes.

4. `triton_deepseek_v4_indexer_q_rope_quant`
   - First version should be FP8 E4M3 + E8M0 scales.
   - Defer FP4 indexer cache until the FP8 path is validated.

5. `triton_deepseek_v4_inverse_rope_output_quant`
   - Applies inverse RoPE to attention output and prepares for `wo_a`.
   - Keep a BF16-output version first; add FP8 output quant only when the downstream grouped `wo_a`
     kernel consumes it directly.

These kernels should all accept caller-owned output/cache buffers and must not allocate during replay.

## Priority 3: Router And MXFP4 MoE

The current routed MoE path already uses `triton_kernels.matmul_ogs` through
`triton_deepseek_v4_mxfp4_moe_from_routing`. Treat it as the baseline unless profiling proves it is the
dominant bottleneck after attention is fixed.

The most useful Triton additions are around the router and routing metadata:

1. `triton_deepseek_v4_router_hash`
   - For the first three hash-routed layers.
   - Gather `tid2eid[input_ids]`, compute `sqrt(softplus(router_logits))`, gather selected scores,
     normalize, and scale.

2. `triton_deepseek_v4_router_topk`
   - For non-hash layers.
   - Compute BF16/FP32 router matmul output, `sqrt(softplus)`, add bias for selection only, top-k `k=6`,
     gather unbiased scores, normalize, and scale.
   - Keep router weight BF16. Router is small enough that correctness and launch overhead matter as much
     as raw GEMM speed.

3. `triton_deepseek_v4_route_metadata`
   - Build per-expert histograms, sorted token/expert pairs, gather indices, scatter indices, and EP-local
     masks.
   - This may be more valuable than fusing router math because `triton_mxfp4_moe_from_routing` currently
     prepares routing data around sorted expert ids.

Keep the packed expert GEMMs in `triton_kernels.matmul_ogs` for the first production path:

- It already accepts `RoutingData`, `GatherIndx`, and `ScatterIndx`.
- It already supports fused SWiGLU with `(alpha, limit)`.
- The DSV4 loader already produces the expected packed uint8 block layout:
  `gate_up_blocks [E, 2I, H/32, 16]`, `down_blocks [E, H, I/32, 16]`.

Only write a custom DSV4 MXFP4 matmul kernel if profiling shows the generic `matmul_ogs` path loses
substantial time to layout conversion, routing metadata preparation, or small per-expert occupancy.

## Priority 4: FineGrained FP8 And Grouped `wo_a`

Existing FP8 linears are viable first. The strategy is to validate and specialize the DSV4-only grouped
case rather than rewrite all FP8 GEMMs.

Keep:

- `trtllm_finegrained_fp8_linear` for standard `wq_a`, `wq_b`, `wkv`, `wo_b`, and shared-expert
  projections when block sizes are exactly `128x128`.
- BF16 dequant + cuBLAS fallback for small or non-standard shapes.

Add or specialize:

- A production `triton_deepseek_v4_wo_a_grouped_fp8_linear` if the current reference grouped op survives
  into the compiled graph or becomes a throughput issue.
- Shape specialization for `G=8`, `Dg=4096`, `R=1024`.
- TP specialization for `tp=8`, where each rank owns one output group:
  `[T, 1, 4096] x [1024, 4096] -> [T, 1, 1024]`.

Preferred implementation path:

1. For unsharded/single GPU, execute eight independent grouped matmuls inside one launch.
2. For `tp=8`, compile the single-group case and avoid carrying a group loop.
3. Reuse the same activation quantization and E8M0 scale decode rules as FineGrained FP8.

## Sharding-Aware Kernel Boundaries

Use the sharding plan in `dsv4_sharding_guide.md` as the first distributed kernel target:

```text
tp = 8
moe_ep = 8
moe_tp = 1
attention_dp = false
```

Attention:

- Shard query heads and `attn_sink` by TP rank.
- Keep compressed KV generation replicated initially, then revisit cache sharing only after correctness.
- The sparse attention kernel should consume local heads and produce local heads; the collective belongs
  after row-sharded `wo_b`.

Grouped `wo_a`:

- Split by output group boundaries, not by arbitrary flattened rows.
- With `G=8` and `tp=8`, each rank owns one group.

Routed MoE:

- Keep router replicated.
- Slice packed MXFP4 expert tensors along expert dimension.
- Convert global expert ids to local ids, mask nonlocal weights to zero, run local MoE, then all-reduce
  the partial routed output.

This makes kernel behavior match the existing `DeepSeekV4GroupedWoAShardableNode`,
`DeepSeekV4SparseAttentionShardableNode`, and `DeepSeekV4MXFP4FromRoutingShardableNode` contracts.

## CUDA Graph Rules

Every new Triton op should be graph-safe from day one:

- No allocation in the op.
- No Python-side sorting, `torch.arange`, host sync, or shape-dependent buffer creation during replay.
- Fixed output buffers supplied by the caller or allocated by AutoDeploy outside capture.
- Metadata tensor shapes bucketed by CUDA graph batch size.
- Workspace sizes determined from static bucket configuration.
- Optional `out=` support for attention-like dynamic ops, matching current source-op conventions.

Piecewise CUDA graph support already recognizes DSV4 sparse attention and placeholder DSV4 metadata prep
op names. The kernel plan should use that boundary rather than forcing all sparse/index/cache behavior
inside the static captured graph immediately.

## Bring-Up Order

1. Profile and validate the existing five-layer graph to rank attention, MoE, router, grouped `wo_a`, and
   FP8 linear time. Use this only after correctness is already established.
2. Implement SWA-only cached sparse attention for ratio-0 layers.
3. Add KV RMSNorm/RoPE/cache insert with BF16 cache rows.
4. Add ratio-128 compressor update and sparse attention over visible compressed rows.
5. Add ratio-4 overlap compressor and indexer top-k path.
6. Add FP8 NoPE cache with raw E8M0 scale-byte handling.
7. Add router metadata kernels if MoE routing overhead is visible.
8. Specialize grouped `wo_a` for the sharded one-group-per-rank path if it remains in the graph as a
   reference or dequant fallback.
9. Validate `tp=8, moe_ep=8, moe_tp=1`, including the all-reduce placements after `wo_b` and routed MoE.

## Validation Targets

Correctness targets:

- Source sparse attention parity for duplicate indices, negative masks, and sink probability mass.
- Full prefill vs chunked prefill for ratio-0, ratio-128, ratio-4, and mixed layer subsets.
- Prefix prefill plus decode vs full prefill logits.
- E8M0 raw-byte preservation tests for cache scales and MXFP4 expert scales.
- EP sharding parity against unsharded routed MoE.

Performance targets:

- Decode attention latency by ratio class.
- Cache insert bandwidth for SWA and MHC.
- Compressor update latency at block-close steps.
- Router metadata overhead before the MXFP4 MoE matmuls.
- Grouped `wo_a` latency in single-GPU and `tp=8` shapes.
- CUDA graph replay overhead and dynamic submodule count.

## Wave4 Profile/Validation Status

W4-03 added a CPU graph-dump inventory check for the fixed five-layer fixture under
`tests/unittest/_torch/auto_deploy/unit/compile/test_deepseek_v4_graph_dump_inventory.py`.

W4R-05 reduced evidence update, 2026-04-25:

- Completed fixture/e2e artifact:
  `ad_run_logs/graph_dumps/deepseek_v4_flash_5layer_sampling_workspace_20260425_131939/084_compile_compile_model.txt`.
  Its raw log reached `Running example prompts`, processed `1/1` request, and destroyed all eight
  process groups.
- Fresh current-branch artifact:
  `ad_run_logs/graph_dumps/deepseek_v4_flash_5layer_w4r05_current_20260425_2358/077_cache_init_initialize_cache.txt`.
  No `*compile_model.txt` was produced because the run failed during executor initialization.
- Focused inventory tests:
  - Default fixed fixture: `3 passed, 2 xfailed` in `0.16s`; the two expected xfails are the old
    torch attention/router gates.
  - Fresh partial graph via `DEEPSEEK_V4_GRAPH_DUMP=.../077_cache_init_initialize_cache.txt`: `5 passed`
    in `0.16s`.
- Fresh e2e blocker:
  `ad_run_logs/ad_run_deepseek_v4_flash_5layer_w4r05_current_20260425_2358.raw.log` shows repeated
  `triton_deepseek_v4_sparse_attention_v2_with_cache` fallback warnings for prefill/mixed batches, then
  CUDA OOM while allocating a `3.19 GiB` unmanaged attention cache after `7.96 GiB` KV cache allocation
  per rank.
- One invalid fresh attempt is also logged at
  `ad_run_logs/ad_run_deepseek_v4_flash_5layer_w4r05_current_20260425_2355.raw.log`: it used
  `--args.yaml-extra` for a full experiment YAML and failed Pydantic validation with
  `extra_forbidden`; the corrected command used `--yaml-extra`.

Graph inventory:

| Surface | Completed compile fixture | Fresh partial graph | Evidence-backed status |
| --- | ---: | ---: | --- |
| Cached sparse attention wrapper | `torch_deepseek_v4_sparse_attention_v2_with_cache`: 5 | `triton_deepseek_v4_sparse_attention_v2_with_cache`: 5 | Graph-selected Triton in current branch, but fresh runtime observed reference fallback for prefill/mixed batches. Real Triton remains decode-only/guarded by the op. |
| Router wrapper | `torch_deepseek_v4_router`: 5 | `triton_deepseek_v4_router`: 5 | Graph-selected guarded Triton in current branch. Runtime branch coverage is not closed by this run. |
| MXFP4 routed MoE | `triton_deepseek_v4_mxfp4_moe_from_routing`: 5 | `triton_deepseek_v4_mxfp4_moe_from_routing`: 5 | Real Triton baseline: implementation uses `triton_kernels.matmul_ogs` and a Triton DeepSeek V4 SwiGLU activation. |
| MXFP4 route metadata | Not present in old torch-wrapper inventory | `torch_deepseek_v4_mxfp4_route_metadata`: 5 | Torch wrapper remains before MoE. Treat metadata preparation as unresolved until profiled or replaced. |
| Grouped `wo_a` FP8 | `torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear`: 5 | Same: 5 | Torch-named wrapper remains. Implementation has a guarded Triton grouped FP8 path and BF16 dequant fallback; graph evidence alone does not prove which branch executes. |
| Generic FineGrained FP8 linears | `torch_fake_quant_finegrained_fp8_linear`: 37 | Same: 37 | Torch-named wrapper remains, but implementation calls HF `w8a8_block_fp8_matmul_triton`. |
| Simple linears | `torch_linear_simple`: 24 | Same: 24 | Torch wrappers remain; not proven as DSV4 Triton kernels. |
| RoPE helper | `torch_rope_with_complex_freqs`: 2 | Same: 2 | Torch wrapper remains. |
| Ratio-4 indexer top-k | `aten.topk.default`: 2 | Same: 2 | Inlined graph path remains; no standalone DSV4 Triton indexer kernel selected. |
| Distributed reductions | `trtllm_dist_all_reduce`: 17 | Same: 17 | Graph-selected collective path present after attention/indexer/MoE/shared-expert boundaries. |

Current evidence does not close per-layer runtime latency. It does close which wrappers are selected by
the completed fixture versus the fresh current-branch graph, and it shows the fresh reduced e2e blocker
is memory during executor/cache initialization rather than a graph-inventory test failure.

## Recommended Ownership Split

- Attention kernel owner: sparse attention, cache insert, compressor update, indexer path, and inverse RoPE
  output handling.
- MoE kernel owner: router kernels, routing metadata, MXFP4 MoE profiling, and EP-local masking.
- Quantization kernel owner: grouped `wo_a`, FP8 cache quantization, E8M0 byte handling, and scale decode
  utilities.
- Runtime owner: DSV4 metadata prep, page-table advancement, fixed workspace sizing, and piecewise CUDA
  graph boundaries.

Keeping these boundaries separate is important. The DSV4 attention path is already complex enough that
combining cache semantics, routing, and all quantization into one kernel effort would slow down validation.
