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

# DeepSeek V4 AutoDeploy Parallel Work Plans

This directory splits `deepseek_v4_ad_plan.md` into feature waves. Files inside
the same wave can be implemented and tested by separate agents in parallel.
Later waves should wait for the earlier wave contracts to land, or use the
mock boundaries described in the individual plan files.

Each feature plan is written to be useful even before the full DeepSeek V4 model
works end to end. Agents should use synthetic tensors, reduced configs, or
checkpoint metadata where possible. Full-model integration is intentionally kept
in `wave4/11_full_model_integration.md`.

## Wave Map

| Wave | Plan | Feature | Can Test Without Full Model |
| - | - | - |
| `wave1` | `01_checkpoint_classifier_and_quant_config.md` | Checkpoint classification and V4 quant config override | yes |
| `wave1` | `02_e8m0_scale_utils.md` | E8M0 upcast/raw-byte scale helpers | yes |
| `wave1` | `05_deepseek_v4_router.md` | Hash/top-k `sqrtsoftplus` router | yes |
| `wave1` | `07_sparse_attention_source_op.md` | Canonical sparse attention source op | yes |
| `wave1` | `08_paged_v4_cache_resources.md` | V4 named/composite paged cache resources | yes |
| `wave1` | `10_cuda_graph_runtime_config.md` | CUDA graph dynamic-op/runtime config support | yes |
| `wave2` | `03_finegrained_fp8_linear_path.md` | FP8 linears and `.scale` aliasing | yes |
| `wave2` | `04_packed_mxfp4_expert_loader.md` | Packed FP4 expert loading/layout | yes |
| `wave2` | `09_attention_kernel_microfeatures.md` | Standalone attention kernel microfeatures | yes |
| `wave3` | `06_deepseek_v4_moe_op.md` | Canonical V4 MoE op and lowering skeleton | yes, with synthetic expert tensors |
| `wave4` | `11_full_model_integration.md` | Full model integration and serving validation | no |

## Dependency Graph

```text
wave1/
  01 checkpoint classifier / quant config
  02 E8M0 scale utilities
  05 router
  07 sparse attention source op
  08 paged V4 cache resources
  10 CUDA graph/runtime config

wave2/
  03 FP8 linear path              depends on 01, 02
  04 packed MXFP4 expert loader   depends on 01, 02
  09 attention kernel features    depends on 02, 07, 08

wave3/
  06 DeepSeek V4 MoE op           depends on 03, 04, 05

wave4/
  11 full model integration       depends on all prior waves
```

## Coordination Rules

- Each agent owns only the files listed in its plan.
- Agents must not revert edits made by other agents.
- Agents should prefer new focused unit tests over broad end-to-end tests.
- Agents should keep public interfaces small and documented.
- Shared names should use the same vocabulary across plans:
  - `deepseek_v4_fp8` for the mixed quantization plan.
  - `torch_deepseek_v4_moe` for the canonical MoE op.
  - `torch_deepseek_v4_sparse_attention` for the canonical attention op.
  - `DeepSeekV4CacheResource` or equivalent for the V4 cache resource family.
- Full-model behavior should not be assumed until
  `wave4/11_full_model_integration.md`.
