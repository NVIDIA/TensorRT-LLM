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

# DeepSeek V4 AutoDeploy Branch Summary

Generated from branch `bala/dsv4` on 2026-04-24.

## Branch Snapshot

| Item | Value |
| --- | --- |
| Branch | `bala/dsv4` |
| Compared against | `upstream/main` |
| Merge base | `43cd23f5c9e2d63f982898702dfe9f4e9f6f5b2f` |
| Local HEAD | `8fa8790e71` |
| `origin/bala/dsv4` | `97ad658fbe` |
| Local delta vs origin | `0` behind, `1` ahead |
| Commits in branch | `17` |
| Diff size | `47 files changed, 10677 insertions(+), 189 deletions(-)` |

The latest local commit is the sharding IR commit. Everything below is based on
the committed branch range `upstream/main..HEAD`.

## Current Support Stage

DeepSeek V4 AutoDeploy support is now in a code-complete bring-up state for the
core model path. The branch contains model onboarding, checkpoint quantization
classification, FP8 and MXFP4 loading/lowering support, DeepSeek V4 custom
attention and MoE ops, cache resource integration, CUDA graph configuration,
registry wiring, sharding IR, and focused unit/single-GPU tests.

The next stage should be full-model integration validation: load the downloaded
DeepSeek V4 Flash checkpoint, run the configured AutoDeploy path end to end,
exercise CUDA graph capture, validate multi-rank sharding, and collect serving
and performance results. The commits show extensive implementation and focused
tests, but they do not by themselves establish completed full-checkpoint
serving validation.

## Added Capabilities

### Model Onboarding And Registry

- Added `DeepseekV4Config`, `DeepseekV4ForCausalLM`, and
  `DeepseekV4AutoModelForCausalLMFactory` in
  `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek_v4.py`.
- Registered the custom model through
  `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py` and
  `examples/auto_deploy/model_registry/models.yaml`.
- Added the deployment config
  `examples/auto_deploy/model_registry/configs/deepseek_v4_flash.yaml`.
- The registry config uses:
  - `runtime: trtllm`
  - `model_factory: DeepseekV4AutoModelForCausalLM`
  - `compile_backend: torch-cudagraph`
  - `max_batch_size: 64`
  - `max_seq_len: 8192`
  - `max_num_tokens: 8192`
  - `enable_chunked_prefill: true`
  - `skip_mtp: true`
  - CUDA graph batch sizes `[1, 2, 4, 8, 16, 32, 64]`

### Checkpoint Quantization Support

- Added DeepSeek V4 quantization metadata handling in
  `tensorrt_llm/_torch/auto_deploy/models/deepseek_v4_quant.py`.
- Introduced the branch-specific quant method
  `deepseek_v4_fp8`, with checkpoint categories for fine-grained FP8 linear
  tensors and packed MXFP4 expert tensors.
- Added config-reader integration in
  `tensorrt_llm/_torch/auto_deploy/models/quant_config_reader.py`.
- Added E8M0 helper utilities in
  `tensorrt_llm/_torch/auto_deploy/utils/e8m0.py`.
- Extended FP8 dequant utilities in
  `tensorrt_llm/_torch/auto_deploy/utils/fp8_dequant.py`.
- Added support for DeepSeek V4 `wo_a` grouped fine-grained FP8 quantization in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py`.
- Added MXFP4 expert checkpoint loading and layout handling in
  `tensorrt_llm/_torch/auto_deploy/transform/library/deepseek_v4_mxfp4.py`.

### Attention And KV Cache

- Added the DeepSeek V4 sparse attention source op in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/deepseek_v4_attention.py`.
- Added DeepSeek V4 attention microkernels in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/deepseek_v4_kernels.py`.
- Added masked sparse-attention index support.
- Extended the AutoDeploy attention interface with DeepSeek V4 paged resources:
  `DeepSeekV4PagedResourceHandler` and `DeepSeekV4CacheResourceDescriptor`.
- The cache work covers named paged resources such as SWA, MHC, indexer, and
  compressor state, while preserving paged-resource behavior instead of falling
  back to large unpaged allocations.

### Router And MoE

- Added the DeepSeek V4 router op in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/deepseek_v4_router.py`.
- Added the DeepSeek V4 source MoE op in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/deepseek_v4_moe.py`.
- Extended MXFP4 MoE support in
  `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/mxfp4_moe.py`.
- Added the `lower_deepseek_v4_moe` transform in
  `tensorrt_llm/_torch/auto_deploy/transform/library/deepseek_v4_moe.py`.
- The model registry config enables:
  - `lower_deepseek_v4_moe`
  - `quantize_mxfp4_moe`
  - `quantize_finegrained_fp8_linear_from_config`
- The config keeps `multi_stream_moe` disabled for this bring-up path.

### CUDA Graph Runtime Path

- Added DeepSeek V4 CUDA graph configuration in the model registry config.
- Extended `tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py`
  and `tensorrt_llm/_torch/auto_deploy/compile/piecewise_utils.py`.
- The final config intentionally sets
  `transforms.compile_model.piecewise_enabled: false`, so the current default
  path is monolithic CUDA graph capture rather than piecewise prefill capture.

### Sharding And Distributed IR

- Added a sharding-aware DeepSeek V4 model IR file:
  `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek_v4_ir.py`.
- Added DeepSeek V4 shardable-node support in
  `tensorrt_llm/_torch/auto_deploy/transform/library/sharding_ir.py`,
  including rules for:
  - grouped `wo_a` FP8 quantized linear
  - DeepSeek V4 sparse attention
  - DeepSeek V4 MXFP4 MoE from routing
- Added `dsv4_sharding_guide.md`, which documents the recommended first
  distributed target:
  - `world_size: 8`
  - `tp: 8`
  - `moe_ep: 8`
  - `moe_tp: 1`
  - `enable_attention_dp: false`
- The guide positions expert parallelism over the expert dimension as the first
  target, with `moe_tp > 1` deferred because packed MXFP4 block slicing requires
  more careful kernel and loader work.

### Demos, Tests, And Validation Assets

- Added `examples/deepseek_v4_5layer_forward.py` as a focused forward/demo
  harness.
- Added unit and single-GPU coverage for:
  - DeepSeek V4 modeling
  - quant config classification
  - E8M0 helpers
  - FP8 linear handling
  - MXFP4 loader
  - router
  - MoE lowering
  - sparse attention and attention kernels
  - paged cache resources
  - piecewise CUDA graph behavior
  - sparse attention sharding
- Existing RoPE and MRoPE delta cache tests were adjusted for the new behavior.

## Commit Inventory

| Commit | Area | Summary |
| --- | --- | --- |
| `7b4019c82b` | Model | Onboarded the initial DeepSeek V4 AutoDeploy model. |
| `c8618e15bb` | Quantization | Added E8M0 scale helpers. |
| `9efb63a2a4` | Attention | Added DeepSeek V4 sparse attention source op. |
| `da05c3a1bd` | MoE/router | Added DeepSeek V4 router op. |
| `dbb376a092` | Runtime | Added DeepSeek V4 CUDA graph config. |
| `86378f94ed` | Quantization | Classified DeepSeek V4 quant metadata. |
| `c85d53ee80` | KV cache | Added DeepSeek V4 paged cache resources. |
| `7fcc9f6111` | Attention | Supported masked sparse attention indices. |
| `fdef542542` | Quantization/MoE | Added DeepSeek V4 MXFP4 expert loader. |
| `08c13798de` | Attention | Added attention microkernels. |
| `b3fc1e52b3` | Quantization | Supported DeepSeek V4 FP8 linear scales. |
| `328fda50ab` | MoE | Added DeepSeek V4 MoE source op. |
| `bb47b6ad83` | Demo/test | Added 5-layer forward demo. |
| `58fc7f9c7b` | Quantization | Supported `wo_a` FP8 quantization. |
| `530dc6eca2` | Integration | Wired DeepSeek V4 AutoDeploy integration. |
| `97ad658fbe` | Runtime | Disabled piecewise CUDA graph by default. |
| `8fa8790e71` | Sharding | Added DeepSeek V4 AutoDeploy sharding IR. |

## Focused Test Files Added

- `tests/unittest/_torch/auto_deploy/unit/attention/test_deepseek_v4_sparse_attention_sharding.py`
- `tests/unittest/_torch/auto_deploy/unit/compile/test_deepseek_v4_piecewise_cuda_graph.py`
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_moe.py`
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_mxfp4_loader.py`
- `tests/unittest/_torch/auto_deploy/unit/fused_moe/test_deepseek_v4_router.py`
- `tests/unittest/_torch/auto_deploy/unit/models/test_deepseek_v4_quant_config.py`
- `tests/unittest/_torch/auto_deploy/unit/quantization/test_deepseek_v4_fp8_linear.py`
- `tests/unittest/_torch/auto_deploy/unit/runtime/test_deepseek_v4_cache_resources.py`
- `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_deepseek_v4_modeling.py`
- `tests/unittest/_torch/auto_deploy/unit/utils/test_e8m0.py`
- `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_deepseek_v4_kernels.py`
- `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_deepseek_v4_sparse_attention.py`

## Current Configuration Choices

- MTP is skipped with `model_kwargs.skip_mtp: true`.
- Chunked prefill is enabled.
- CUDA graph support is selected through `compile_backend: torch-cudagraph`.
- CUDA graph capture sizes are configured up to batch size 64.
- Piecewise CUDA graph capture is disabled by default.
- Fine-grained FP8 quantization from checkpoint config is enabled.
- Fine-grained FP8 linear fusion is disabled in the current YAML.
- MXFP4 MoE quantization/lowering is enabled.
- Multi-stream MoE is disabled in the current YAML.

## Recommended Next Validation Stage

1. Run the focused unit tests added by the branch.
2. Run the 5-layer forward demo against the downloaded checkpoint path:
   `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home/manual/deepseek-ai__DeepSeek-V4-Flash`.
3. Run full-checkpoint AutoDeploy graph construction using
   `examples/auto_deploy/model_registry/configs/deepseek_v4_flash.yaml`.
4. Validate CUDA graph capture for the configured batch sizes.
5. Validate full serving through `trtllm-serve` or the AutoDeploy model registry
   flow.
6. Validate multi-rank sharding, starting with the documented `tp=8, moe_ep=8,
   moe_tp=1` layout.
7. Collect correctness, memory, throughput, and latency data before declaring
   production-ready DeepSeek V4 support.

All commands for this workspace should be run through the requested wrapper:

```bash
bash -ic "f9 && <command>"
```
