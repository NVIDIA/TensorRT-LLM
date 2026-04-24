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

# DeepSeek V4 AutoDeploy Sharding Guide

This guide describes the recommended sharding design for DeepSeek V4 in
AutoDeploy. It is based on the current AutoDeploy DeepSeek V4 model, the
DeepSeek V4 custom ops, the dumped graphs in `dsv4-graphs/`, and the vLLM
DeepSeek V4 reference PR:

```text
https://github.com/vllm-project/vllm/pull/40760
```

The main recommendation is to make DeepSeek V4 sharding explicit in the model
IR and in a small number of DeepSeek V4-specific shardable ops. Avoid relying on
late graph heuristics for the MLA shapes, grouped output projection, and
pre-routed MXFP4 MoE.

## Current State

The current DeepSeek V4 AutoDeploy path has the main pieces required for a
single-rank graph:

- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek_v4.py`
  emits the prefill model.
- `torch_deepseek_v4_sparse_attention` implements the source sparse attention
  contract.
- `torch_deepseek_v4_router` implements DeepSeek V4 routing, including hash
  layers.
- `torch_deepseek_v4_moe` and
  `triton_deepseek_v4_mxfp4_moe_from_routing` cover the routed expert path.
- `lower_deepseek_v4_moe` lowers the source MoE into router, MXFP4 routed MoE,
  and shared FP8 expert linears.
- `deepseek_v4_quant.py` and the quantization transform identify DeepSeek V4
  mixed FP8/MXFP4 checkpoint tensors.

The dumped graphs show the gap. After `apply_sharding_hints`, the graph still
contains full DeepSeek V4 dimensions:

- attention uses full `64` heads;
- `wo_a` uses all `8` output groups;
- `triton_deepseek_v4_mxfp4_moe_from_routing` owns all `256` routed experts;
- the DeepSeek V4 sparse attention, grouped `wo_a`, and pre-routed MXFP4 MoE
  ops are not registered as shardable nodes.

The generic sharding pass already supports normal linears, fine-grained FP8
linears, NVFP4 linears, `auto_deploy.view`, `auto_deploy.all_reduce`, and some
MoE forms. DeepSeek V4 needs a thin layer on top of that machinery rather than a
separate distributed system.

## Recommended Parallel Layout

Use tensor parallelism for dense attention and expert parallelism for routed
experts. The first production target should keep the two groups aligned:

```yaml
world_size: 8
transforms:
  apply_sharding_hints:
    enabled: true
    dist_mapping:
      tp: 8
      moe_ep: 8
      moe_tp: 1
      moe_cluster: 1
    enable_attention_dp: false
```

This matches the clean shape used by the vLLM DeepSeek V4 implementation:
attention heads/groups are local to each tensor-parallel rank, routed experts
are partitioned across the same ranks, and the partial hidden output is
all-reduced.

Keep `moe_tp > 1` out of the first implementation. The routed experts use
packed MXFP4 blocks and E8M0 scales, so intermediate-dimension tensor
parallelism requires careful block-aligned slicing in the kernel and loader.
Expert parallelism over the expert dimension is much simpler and should be the
first target.

## Model IR Strategy

Prefer a DeepSeek V4 sharding-aware model file, for example
`modeling_deepseek_v4_sharding_ir.py`, over post-export pattern surgery. The
existing `modeling_*_ir.py` files show the local pattern: emit standard
AutoDeploy custom ops with sharding hints, use `auto_deploy.view` for shape
changes that depend on parallel dimensions, and let `apply_sharding_hints`
materialize local weights and collectives.

DeepSeek V4 should follow that pattern for:

- MLA query and output projections;
- local-head sparse attention;
- grouped `wo_a`;
- shared expert FP8 linears;
- routed MXFP4 expert tensors.

Plain `aten.view` is fine for rank-local reshapes whose dimensions are already
local. Any reshape that encodes `num_heads`, `o_groups`, or a sharded hidden
dimension should use `auto_deploy.view` so the sharding pass can rewrite the
shape safely.

## Attention Sharding

Shard attention by head and output group:

| Component | Sharding |
| --- | --- |
| `wq_a` | replicated |
| `wq_b` | column/head-sharded |
| `q_norm`, RoPE | local heads |
| `wkv` | replicated |
| compressor | replicated |
| indexer | replicated |
| `attn_sink` | sliced by local head range |
| sparse attention | local heads, no collective |
| inverse RoPE | local heads |
| `wo_a` | group-sharded |
| `wo_b` | row-sharded, followed by all-reduce |

This preserves the DeepSeek V4 dataflow: compressed KV is shared, while each
rank computes only its local query heads and local output groups.

The initial constraints should be explicit:

```text
num_heads % tp_size == 0
o_groups % tp_size == 0
```

For the current DeepSeek V4 Flash shape, `num_heads = 64` and `o_groups = 8`,
so `tp_size = 8` is the natural first target. Larger TP values need a different
`wo_a` strategy because there are fewer output groups than ranks.

### Required Attention Shardable Ops

Add a shardable rule for `torch_deepseek_v4_sparse_attention` that validates
local head inputs and slices head-owned parameters such as `attn_sink`. The op
itself should not insert a collective; the collective belongs after the
row-sharded `wo_b`.

Add a shardable rule for
`torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear`. It should
split by output group, not by an arbitrary flattened row range. For each rank it
should produce:

```text
input:  [batch, sequence, local_o_groups, group_hidden]
weight: [local_o_groups * o_lora_rank, group_hidden]
scale:  matching local output rows
output: [batch, sequence, local_o_groups, o_lora_rank]
```

Then flatten the local groups and pass the result to row-sharded `wo_b`.

## MoE Sharding

Keep the router replicated. It is cheap, and its output is a global expert id
contract used by both normal and hash-routed layers.

Shard routed expert tensors over `moe_ep_size`:

- `gate_up_blocks`;
- `gate_up_scales`;
- `down_blocks`;
- `down_scales`;
- expert biases, when present.

Use the same contiguous expert partitioning as
`expert_parallel_slice` in
`tensorrt_llm/_torch/auto_deploy/transform/library/deepseek_v4_mxfp4.py`.
The first implementation can require:

```text
n_routed_experts % moe_ep_size == 0
```

The DeepSeek V4 Flash shape has `256` routed experts, so this is fine for
common sizes such as `2`, `4`, and `8`.

### Required Routed MoE Shardable Op

Add a DeepSeek V4-specific shardable rule for
`triton_deepseek_v4_mxfp4_moe_from_routing`.

The rule should:

1. Slice all routed expert tensors along expert dimension `0`.
2. Convert global selected expert ids to local expert ids for experts owned by
   this rank.
3. Mask or zero routing weights for experts not owned by this rank.
4. Run the local MXFP4 MoE op.
5. All-reduce the local partial routed output across the MoE parallel group.

This mirrors the existing idea in generic MoE sharding, but it must be applied
to the precomputed-routing DeepSeek V4 op. The current helper inside
`mxfp4_moe.py` computes token grouping from `selected_experts`, so the selected
expert ids must be localized before the local kernel sees them.

### Shared Expert Sharding

The shared expert should be tensor-parallel like a normal SwiGLU MLP:

| Component | Sharding |
| --- | --- |
| shared `w1` | column-sharded |
| shared `w3` | column-sharded |
| activation | local |
| shared `w2` | row-sharded |

The cleanest collective placement is one all-reduce after adding the local
routed expert output and the local shared expert output. This requires the first
implementation to keep `tp` and `moe_ep` aligned. If future configs split those
groups differently, the collectives need to become group-specific.

## Weight Loading

Sharding should happen before weight loading. That lets load hooks allocate and
load only local expert tensors.

For routed MXFP4 experts, prefer passing local `expert_indices` into
`load_deepseek_v4_mxfp4_experts` instead of loading all `256` experts and
slicing afterward. Load-all-then-slice is acceptable for bring-up tests, but it
defeats one of the main memory benefits of expert parallelism.

For FP8 linears, reuse the existing fine-grained FP8 scale sharding. DeepSeek V4
checkpoint aliases `.scale` to `weight_scale_inv`, and the E8M0 scale bytes must
continue to be handled by the DeepSeek V4 quantization path rather than generic
floating-point conversion.

## Transform Order

The preferred order is:

```text
export
quantize DeepSeek V4 FP8/MXFP4
lower DeepSeek V4 MoE
apply sharding hints
strip sharding hints
load weights
```

This preserves the current lowering structure while giving the sharding pass
the final ops it needs to slice: grouped `wo_a`, shared FP8 linears, sparse
attention, and pre-routed MXFP4 MoE.

## Validation Checklist

Add graph-level tests for `tp_size = 2`, `4`, and `8` with small synthetic
configs where possible:

- `wq_b` weights and FP8 scales are sharded by output/head dimension.
- head views are rewritten to local `num_heads / tp_size`.
- `attn_sink` is local.
- sparse attention consumes local query heads.
- grouped `wo_a` owns local output groups and local scale rows.
- `wo_b` is row-sharded and followed by an all-reduce.
- routed MXFP4 tensors have `num_experts / moe_ep_size` experts locally.
- global expert ids are localized or masked before the MXFP4 op.
- shared expert `w1`, `w3`, and `w2` are sharded.
- the initial `tp == moe_ep` MoE path has a single final all-reduce after
  combining routed and shared outputs.

Add numeric tests that compare unsharded and sharded execution for one small
DeepSeek V4 layer. These tests should cover both ordinary routed layers and hash
layers, because hash routing bypasses top-k selection but still produces global
expert ids.

Add loader tests for local expert slices. A minimal case with `4` experts and
`moe_ep_size = 2` should prove that rank `0` receives experts `[0, 1]` and rank
`1` receives experts `[2, 3]`, with MXFP4 blocks and E8M0 scales kept in the
checkpoint-native layout.

## Open Constraints

- `tp_size` must divide `num_heads`.
- The first implementation should require `tp_size` to divide `o_groups`.
- The first implementation should require `moe_ep_size` to divide
  `n_routed_experts`.
- `moe_tp > 1` should wait until the MXFP4 kernel and loader support
  block-aligned intermediate-dimension slicing.
- Attention DP should stay disabled for this path. DeepSeek V4 attention is
  naturally head-parallel, and enabling attention DP would skip the dense
  sharding that this model needs.

With these constraints, DeepSeek V4 sharding stays close to the existing
AutoDeploy sharding IR design: model code declares the semantic parallel axes,
DeepSeek V4-specific shardable nodes handle the few unusual ops, and the generic
pass still owns weight slicing, local shape rewrites, and collectives.
