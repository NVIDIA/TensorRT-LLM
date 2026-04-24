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

# Plan 06: DeepSeek V4 MoE Op

## Goal

Introduce a canonical DeepSeek V4 MoE op and a production lowering path that
combines:

- DeepSeek V4 router output
- packed MXFP4 routed experts
- FineGrained FP8 shared expert
- `swiglu_limit`

This feature can start with synthetic selected experts and packed expert
tensors. It should not wait for the full model.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/deepseek_v4_moe.py`
- `tensorrt_llm/_torch/auto_deploy/transform/library/deepseek_v4_moe.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/mxfp4_moe.py` only for
  minimal activation or argument support
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_moe.py`

Coordinate with Plans 04 and 05 but do not edit their owned files unless
necessary and agreed.

## Canonical Op

Recommended source op:

```text
torch_deepseek_v4_moe(
    hidden_states,
    input_ids,
    router_weight,
    router_bias_or_none,
    tid2eid_or_none,
    routed_expert_packed_weights,
    routed_expert_scales,
    shared_expert_weights,
    shared_expert_scales,
    top_k,
    route_scale,
    swiglu_limit,
    is_hash_layer,
)
```

An alternative is two ops:

```text
torch_deepseek_v4_router
torch_deepseek_v4_moe_from_routing
```

The two-op form may be easier to test and compose.

## Production Lowering

Target graph:

```text
hidden -> router -> selected_experts, routing_weights
hidden + selected_experts + routing_weights + MXFP4 weights -> routed output
hidden + FP8 shared expert weights -> shared output
routed + shared -> output
```

The routed expert activation must implement:

```text
up = clamp(up, -limit, limit)
gate = clamp(gate, max=limit)
hidden = silu(gate) * up
```

## Mock Boundary

Until Plan 04 lands, use synthetic packed expert tensors generated in test
helpers.

Until Plan 05 lands, allow tests to pass precomputed `selected_experts` and
`routing_weights` into a lower-level op.

## Deliverables

- Canonical op and fake implementation for export.
- Reference PyTorch implementation for tiny tests.
- Transform that lowers to an available MXFP4 backend where possible.
- Shared expert path using FineGrained FP8 linears or a mocked BF16 reference in
  early tests.
- Clear error if the selected backend cannot support `swiglu_limit`.

## Standalone Tests

- Tiny BF16 reference MoE with `E=4`, `top_k=2`, small hidden/intermediate.
- `swiglu_limit=0` and `swiglu_limit=10` cases.
- Precomputed routing path independent of router.
- Hash-router integrated path once Plan 05 is available.
- Synthetic packed MXFP4 path once Plan 04 is available.
- Export test for the canonical op.
- CUDA graph replay test for fixed token count if a GPU is available.

## Done Criteria

- The dense expert fallback is not used in production lowering.
- Tests can run without the full DeepSeek V4 checkpoint.
- The op exposes a stable interface for model integration.
