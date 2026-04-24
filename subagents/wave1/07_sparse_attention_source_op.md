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

# Plan 07: Sparse Attention Source Op

## Goal

Create a canonical AutoDeploy source op for DeepSeek V4 sparse/HMA attention.
This op should replace opaque Python gather/einsum attention logic with a
semantic graph node that later transforms can lower to cached kernels.

This work is independent because it can use tiny Q/K/V tensors and synthetic
top-k indices.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/deepseek_v4_attention.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py` only for
  descriptor registration if needed
- `tests/unittest/_torch/auto_deploy/unit/custom_ops/test_deepseek_v4_sparse_attention.py`

Do not implement paged cache resources or production kernels here.

## Source Op Contract

Initial op:

```text
torch_deepseek_v4_sparse_attention(
    q,
    kv,
    attn_sink,
    topk_idxs,
    softmax_scale,
)
```

Shapes:

```text
q:          [B, S, H, D]
kv:         [B, K, D]
attn_sink:  [H]
topk_idxs:  [B, S, K_select]
output:     [B, S, H, D]
```

This minimal contract keeps the source op independent. Higher-level model code
can compute Q/KV/compressor/indexer tensors separately until production kernels
are added.

## Reference Math

For each batch, token, and head:

1. Gather selected KV rows using `topk_idxs`.
2. Compute logits:

```text
logits = dot(q, selected_kv) * softmax_scale
```

3. Add attention sink as an extra logit.
4. Softmax over selected positions plus sink.
5. Weighted sum selected values.
6. Sink contributes probability mass but no value vector unless the reference
   implementation defines an explicit sink value.

## Deliverables

- Custom op registration.
- Meta/fake implementation for export.
- PyTorch reference implementation.
- Optional descriptor object that later cache transforms can recognize.

## Standalone Tests

- Output shape and dtype.
- Compare op output against plain PyTorch reference.
- Masking/index behavior:
  - local sliding-window indices
  - synthetic compressed indices
  - duplicate indices
- Attention sink behavior.
- Export test with dynamic batch and sequence.

## Done Criteria

- The op is stable and can be used by the model scaffold.
- No paged cache or Triton dependency is required.
- Later kernel/cache agents can pattern-match this op by name.
