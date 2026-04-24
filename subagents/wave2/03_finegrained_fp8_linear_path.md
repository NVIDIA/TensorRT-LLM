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

# Plan 03: FineGrained FP8 Linear Path

## Goal

Make DeepSeek V4's FP8 dense linears load and lower through AutoDeploy's
FineGrained FP8 linear path. The feature must handle checkpoint sibling
`.scale` tensors and E8M0 scale values.

This work can be tested independently using a tiny module with one or two
linear layers and synthetic FP8 weights.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/transform/library/quantization.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py`
- `tensorrt_llm/_torch/auto_deploy/utils/e8m0.py` only if Plan 02 has not
  landed yet
- `tests/unittest/_torch/auto_deploy/unit/quantization/test_deepseek_v4_fp8_linear.py`

Do not modify routed MoE, attention kernels, or cache resources.

## Inputs And Mock Boundary

Use a tiny module:

```python
class TinyV4FP8Module(nn.Module):
    def __init__(self):
        self.wq_a = DeepseekV4Linear(...)
        self.shared_w1 = DeepseekV4Linear(...)
```

Use a synthetic state dict with:

```text
wq_a.weight  torch.float8_e4m3fn
wq_a.scale   torch.float8_e8m0fnu
```

If `torch.float8_e8m0fnu` is not available in a test environment, create a
small skip with a clear reason.

## Deliverables

- A load hook or transform alias that maps:

```text
<module>.scale -> <module>.weight_scale_inv
```

for FineGrained FP8 linears.

- E8M0 scale support:
  - Use FP32 decoded scales for `trtllm_finegrained_fp8_linear` if required.
  - Preserve raw E8M0 only if a backend asks for raw bytes.

- Exclusion behavior for DeepSeek V4:
  - Do not quantize router, compressor, head, embed, norms, HC tensors, or
    `attn_sink`.

## Implementation Steps

1. Add a helper to recognize DeepSeek V4 FP8 linear module paths.
2. Extend FineGrained FP8 load logic to accept `.scale` as an alias.
3. Verify loaded scale shapes match `[ceil(N/128), ceil(K/128)]`.
4. Add a path for `F8_E8M0` scales.
5. Ensure the transform still skips generic `quantize_fp8_linear_from_config`
   when `weight_block_size` is present.
6. Ensure non-DeepSeek models keep existing behavior.

## Standalone Tests

- Load synthetic state dict using `.scale` and assert internal buffer is set.
- Compare dequantized FP8 matmul against `trtllm_finegrained_fp8_linear` or the
  fake/reference op on small shapes.
- Test shape validation for:
  - `[8, 32]` scale for `[1024, 4096]`
  - smaller non-multiple dimensions using ceil behavior
- Test exclusions:
  - `ffn.gate.weight` remains BF16
  - `attn.compressor.wkv.weight` remains BF16
- Test no regression for a generic FineGrained FP8 model using
  `weight_scale_inv`.

## Done Criteria

- CPU tests cover load hooks and shape classification.
- GPU test is optional but should be added if available for the optimized op.
- The transform never treats packed routed expert `I8` weights as FP8.
