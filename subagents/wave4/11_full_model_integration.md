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

# Plan 11: Full Model Integration

## Goal

Integrate the independently developed DeepSeek V4 AutoDeploy features and run
reduced-layer, full-layer, and serving validation.

This plan should start only after enough independent features have landed:

- checkpoint classifier and quant override
- E8M0 helpers
- FP8 linear path
- packed MXFP4 expert loader
- router
- V4 MoE op
- sparse attention source op
- V4 cache resources or a temporary correctness fallback
- CUDA graph config support

## Owner Scope

Suggested files:

- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek_v4.py`
- `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py`
- `examples/auto_deploy/model_registry/models.yaml`
- `examples/auto_deploy/model_registry/configs/deepseek_v4_flash.yaml`
- integration tests under `tests/unittest/_torch/auto_deploy/` as appropriate

This agent must not rewrite independent feature implementations unless fixing
integration bugs with focused patches.

## Integration Steps

1. Update the model scaffold to emit canonical V4 ops:
   - `torch_deepseek_v4_sparse_attention`
   - `torch_deepseek_v4_moe`
2. Ensure checkpoint loading:
   - loads normal layers
   - loads `tid2eid`
   - skips `mtp.0.*` explicitly
   - treats skipped MTP as expected
3. Add model registry entry and standalone YAML.
4. Run reduced-layer AutoDeploy flow.
5. Run full 43-layer load and graph transform.
6. Enable decode with CUDA graph batch padding.
7. Enable chunked prefill.
8. Run serving sanity check.

## Initial Runtime Config

Start conservative:

```yaml
compile_backend: torch-cudagraph

model_kwargs:
  skip_mtp: true
  ad_rope_cache_len: 8192

runtime:
  enable_chunked_prefill: true

cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]

transforms:
  quantize_deepseek_v4_from_config:
    enabled: true
  compile_model:
    piecewise_enabled: true
  multi_stream_moe:
    enabled: false
```

## Validation Sequence

### Reduced Layer

- 2 to 4 layers.
- Include at least:
  - one uncompressed attention layer
  - one ratio-4 layer
  - one ratio-128 layer
  - one hash-routed MoE layer
  - one top-k MoE layer if possible

### Full Layer

- Load all 43 layers.
- Skip MTP.
- Confirm no accidental BF16 fallback for routed experts.
- Confirm FP8 linears use loaded E8M0 scales.
- Confirm no unexpected missing checkpoint keys except waived MTP keys.

### Serving

- Short prompt sanity.
- Chunked prefill.
- Decode with CUDA graphs.
- Batch sizes 1, 2, 4, 8 first.
- Larger batch sizes only after memory is characterized.

## Integration Tests

- Full checkpoint key loading test using metadata fixtures.
- Reduced model export test.
- Reduced model transform test.
- Reduced model CUDA graph replay test.
- Optional GPU generation smoke test.

## Performance Bring-Up

Record:

- model load time
- peak memory
- prefill tokens/sec
- decode tokens/sec
- TTFT
- TPOT
- CUDA graph replay hit rate
- routed expert load balance
- all-to-all time if EP is enabled

Compare against vLLM public numbers only as a directional sanity target.

## Done Criteria

- Reduced-layer run succeeds.
- Full model loads and transforms.
- Serving path works with CUDA graph enabled.
- Known limitations are documented:
  - MTP skipped
  - maximum supported context length
  - enabled MoE backend
  - enabled attention backend
  - any eager fallback regions
