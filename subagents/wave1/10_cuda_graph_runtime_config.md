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

# Plan 10: CUDA Graph Runtime Config

## Goal

Prepare AutoDeploy's compile/runtime path for DeepSeek V4 dynamic sparse
attention and MoE operations. This feature should make the graph compiler aware
of new dynamic ops and provide a model registry/runtime config skeleton.

This work is independent because it can use dummy custom ops and small toy
graphs.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/compile/piecewise_utils.py`
- `tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py`
- `examples/auto_deploy/model_registry/configs/deepseek_v4_flash.yaml`
- `tests/unittest/_torch/auto_deploy/unit/compile/test_deepseek_v4_piecewise_cuda_graph.py`

Do not implement attention or MoE kernels here.

## Dynamic Ops To Register

Reserve names for:

```text
auto_deploy::torch_deepseek_v4_sparse_attention
auto_deploy::triton_deepseek_v4_sparse_attention_with_cache
auto_deploy::torch_deepseek_v4_moe
auto_deploy::triton_mxfp4_moe
auto_deploy::triton_mxfp4_moe_ep
deepseek_v4 cache metadata prep ops
```

The exact list should be updated as Plans 06-09 land.

## Config Skeleton

Initial config direction:

```yaml
compile_backend: torch-cudagraph

runtime:
  enable_chunked_prefill: true

cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]

transforms:
  compile_model:
    piecewise_enabled: true
  multi_stream_moe:
    enabled: false
```

Avoid enabling multi-stream MoE by default until the V4 MoE path is stable.

## Deliverables

- Piecewise CUDA graph dynamic-op registration.
- Toy graph tests showing static regions are captured around V4-like dynamic
  ops.
- Decode-only padding behavior test for configured batch sizes.
- YAML skeleton for DeepSeek V4 AutoDeploy bring-up.

## Standalone Tests

- Construct a toy graph:
  - static linear
  - dummy V4 dynamic op
  - static linear
- Verify piecewise splitting.
- Verify fallback behavior when piecewise capture is disabled.
- Verify configured CUDA graph batch sizes are parsed.
- Verify no full checkpoint/model is required.

## Done Criteria

- DeepSeek V4 op names can be added without changing compile architecture.
- The config skeleton is usable by integration agents.
- Existing CUDA graph tests still pass.
