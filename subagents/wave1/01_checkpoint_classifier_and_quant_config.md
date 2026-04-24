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

# Plan 01: Checkpoint Classifier And Quant Config

## Goal

Add a DeepSeek V4-aware checkpoint classification and quantization config
override. This feature must prove from metadata that DeepSeek V4 Flash is mixed
FP8 plus packed FP4, and it must expose enough structured information for later
transforms to avoid generic `fp8` misclassification.

This work is independent because it only needs model config files,
`model.safetensors.index.json`, and safetensors headers. It does not need full
model execution.

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/models/quant_config_reader.py`
- `tensorrt_llm/_torch/auto_deploy/models/deepseek_v4_quant.py` if a new helper
  module is cleaner
- `tests/unittest/_torch/auto_deploy/unit/models/test_deepseek_v4_quant_config.py`

Do not edit MoE kernels, attention kernels, cache handlers, or model code except
for minimal imports needed by the quant reader.

## Inputs

Use small metadata fixtures in tests. Do not require downloading the full model.

Minimum fixture content:

- `config.json` with:
  - `model_type: deepseek_v4`
  - `quantization_config.quant_method: fp8`
  - `quantization_config.scale_fmt: ue8m0`
  - `quantization_config.weight_block_size: [128, 128]`
- safetensors header snippets containing:
  - `layers.0.attn.wq_a.weight` as `F8_E4M3`
  - `layers.0.attn.wq_a.scale` as `F8_E8M0`
  - `layers.0.ffn.experts.0.w1.weight` as `I8`
  - `layers.0.ffn.experts.0.w1.scale` as `F8_E8M0`
  - `layers.0.ffn.gate.weight` as `BF16`
  - `layers.0.ffn.gate.tid2eid` as `I64`

## Deliverables

- A structured classifier output with categories:
  - `finegrained_fp8_linear`
  - `packed_mxfp4_expert`
  - `bf16_or_f32`
  - `integer_metadata`
  - `skipped_mtp`
  - `unknown`
- A DeepSeek V4 quantization config override, for example:

```python
{
    "quant_method": "deepseek_v4_fp8",
    "linear_quant_method": "finegrained_fp8",
    "expert_quant_method": "mxfp4",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
    "expert_block_size": 32,
}
```

- Clear excluded module patterns:
  - `embed`
  - `head`
  - `*.ffn.gate`
  - `*.attn.compressor`
  - `*.norm`
  - `*.hc_*`
  - `*.attn_sink`
  - `mtp.*` until MTP support lands

## Implementation Steps

1. Add a classifier function that accepts parsed config and tensor metadata.
2. Add path-pattern rules for DeepSeek V4 tensor families.
3. Add dtype/shape sanity checks for representative tensors.
4. Add a quant reader path that recognizes `model_type == "deepseek_v4"` and
   HF `quant_method == "fp8"`.
5. Preserve existing generic HF quant reader behavior for all other models.
6. Return a normalized config that downstream transforms can inspect.

## Standalone Tests

Add tests that use tiny JSON/header fixtures:

- DeepSeek V4 config returns `quant_method: deepseek_v4_fp8`.
- FP8 attention/shared-expert linears are classified correctly.
- Routed experts are classified as packed MXFP4.
- Router, compressor, head, embed, norms, and HC tensors are not quantized.
- `mtp.0.*` keys are classified as skipped.
- Unknown keys fail loudly unless explicitly waived.
- Non-DeepSeek `fp8` config still uses existing behavior.

## Done Criteria

- Tests pass without GPU.
- No full checkpoint download is required.
- Later agents can consume the normalized quant config without parsing HF files
  again.
- The implementation does not change runtime behavior for other model families.
