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

# Plan 09: Attention Kernel Microfeatures

## Goal

Develop and test standalone DeepSeek V4 attention kernel building blocks before
integrating them into the full cached attention backend.

This work is independent because each kernel can be tested with synthetic
tensors and compared against a PyTorch reference.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/deepseek_v4_kernels.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_deepseek_v4_*.py`
- `tests/unittest/_torch/auto_deploy/singlegpu/custom_ops/attention/test_deepseek_v4_kernels.py`

Do not own the source op contract or paged cache metadata. Consume interfaces
from Plans 07 and 08 when available.

## Kernel Work Items

Implement independently testable kernels:

1. Q RMSNorm + RoPE
   - input: Q after `wq_b`
   - output: per-head normalized and RoPE-applied Q

2. KV RMSNorm + RoPE + optional FP8 cache insert
   - input: KV after `wkv`
   - output: BF16 RoPE dims and FP8 NoPE dims with E8M0 scales

3. Compressor pooling + RMSNorm + RoPE
   - supports ratio 4 with overlap
   - supports ratio 128 without overlap

4. Indexer Q RoPE + quant
   - initial FP8 path
   - optional MXFP4 path after FP8 correctness

5. Inverse RoPE + FP8 output quant
   - prepares attention output for quantized output projection

6. Sparse attention microkernel
   - local window plus selected compressed indices
   - attention sink

## Deliverables

- Kernel wrappers with fake/meta implementations if registered as custom ops.
- PyTorch references for each kernel.
- Clear dtype contracts:
  - BF16 compute where required
  - FP8 E4M3 NoPE cache where enabled
  - E8M0 scales preserved or decoded as required
- Shape contracts for hidden size 4096, head dim 512, RoPE dim 64, plus tiny
  shapes for tests.

## Standalone Tests

GPU tests where kernels require CUDA:

- Q RMSNorm + RoPE vs reference.
- KV norm/RoPE/cache insert vs reference.
- Compressor ratio-4 and ratio-128 vs reference.
- E8M0 quant error bound tests.
- Sparse attention vs `torch_deepseek_v4_sparse_attention`.
- CUDA graph replay for fixed shapes.

CPU tests:

- reference math helpers
- shape/meta/fake implementations

## Done Criteria

- Each kernel has an isolated test.
- Kernel wrappers do not allocate inside CUDA graph replay paths.
- The full attention backend can assemble kernels without rewriting reference
  math.
