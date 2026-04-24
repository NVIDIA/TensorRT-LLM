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

# Plan 05: DeepSeek V4 Router

## Goal

Implement and test DeepSeek V4 routing independently from expert execution.
The router must support hash-routed layers and normal top-k layers with
`sqrtsoftplus` scoring.

This work is independent because it only consumes hidden states, router weights,
optional bias, `tid2eid`, and input IDs.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/deepseek_v4_router.py`
- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek_v4.py` only
  if replacing local router logic with the canonical op is in scope
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_router.py`

Do not edit MXFP4 expert loading or MoE matmul kernels.

## Router Contract

Inputs:

```text
hidden_states: [T, H]
input_ids: [T] or [B, S]
router_weight: [E, H]
router_bias: [E] or None
tid2eid: [vocab, top_k] or None
top_k: int
route_scale: float
is_hash_layer: bool
```

Outputs:

```text
selected_experts: [T, top_k] int64 or int32
routing_weights: [T, top_k] same floating dtype as hidden compute
```

## Math

Shared score path:

```text
logits = hidden_states @ router_weight.T
scores = sqrt(softplus(logits))
```

Hash-routed layers:

```text
selected_experts = tid2eid[input_ids]
weights = gather(scores, selected_experts)
weights = weights / sum(weights, dim=-1)
weights = weights * route_scale
```

Top-k layers:

```text
biased_scores = scores + router_bias
selected_experts = topk(biased_scores, top_k)
weights = gather(scores, selected_experts)
weights = weights / sum(weights, dim=-1)
weights = weights * route_scale
```

Bias influences selection only; weights are gathered from original scores.

## Deliverables

- A reference PyTorch router implementation.
- Optional custom op:
  - `torch_deepseek_v4_router`
- Optional fused CUDA/Triton top-k kernel:
  - `triton_deepseek_v4_topk_sqrtsoftplus`
- Export-safe fake implementation if registered as a custom op.

## Standalone Tests

- Hash routing:
  - exact `tid2eid[input_ids]` selected experts
  - exact normalization and scaling
- Top-k routing:
  - bias affects selected experts
  - weights come from un-biased scores
- Numerical stability:
  - large positive and negative logits
  - no NaNs for BF16/FP32 inputs
- Shape tests:
  - `[T, H]`
  - `[B, S, H]` flattened boundary
- Export test if a canonical op is added.

## Done Criteria

- CPU reference tests pass.
- GPU fused kernel is optional for this feature plan.
- Later MoE work can consume selected experts and routing weights without
  duplicating router math.
