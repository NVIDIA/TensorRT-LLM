<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

# PR #17495 — Disagg 兼容设置与后续讨论

[PR #17495](https://github.com/NVIDIA/TensorRT-LLM/pull/17495) 将 KVCM V2 设为默认。为保留现有 disagg 功能和回归覆盖，PR 对部分文件显式指定了 CPP runtime 或 KVCM V1。

本 PR 以 CI 通过为合入前提，保留当前必要的 WAR（兼容设置），包括 AutoDeploy 的 CPP/V1 指定。AutoDeploy 已确认继续使用 V1，无需额外 fix。

本文按文件列出当前处理和待讨论项，供 transceiver team review。第 2–3 节涉及的迁移和功能扩展取决于实际需求与已有覆盖，放在后续 PR 处理。已有等价 V2 coverage 的重复 V1 测试已在本 PR 删除，不列入后续跟进项。

配置与支持范围：

- **Transport：**团队支持 NIXL-based 方案，NIXL+CPP 在这一范围内。直接 UCX/MPI backend、NIXL 使用的 UCX plugin 和 MPI 启动设施属于不同层面的配置。
- **Manager 与 runtime：**下文 V1/V2 指 KVCM 版本，CPP/Python 指 transceiver runtime。
- **Helix：**目前按具体需求提供支持，已有实现对应 Kimi K3。是否扩展到 DeepSeek/Qwen 等模型取决于后续需求。

## 1. AutoDeploy：保留 V1 及本 PR 的 CI WAR

AutoDeploy 继续使用 V1 manager 和 CPP runtime。本 PR 保留显式 CPP/V1 WAR，使默认切到 V2 后仍走现有执行路径，并由本 PR 的 CI 验证。这部分无需修复或迁移。

| 文件 | 本 PR 保留的 CI WAR |
|---|---|
| [examples/auto_deploy/model_registry/configs/disagg_ctx.yaml][ad_ctx] | `backend: DEFAULT` + `transceiver_runtime: CPP`；manager 由 AutoDeploy factory 创建为 V1。 |
| [examples/auto_deploy/model_registry/configs/disagg_gen.yaml][ad_gen] | `backend: DEFAULT` + `transceiver_runtime: CPP`；manager 由 AutoDeploy factory 创建为 V1。 |
| [tests/integration/defs/disaggregated/test_ad_disagg.py][ad_test] | 显式 V1+CPP。 |
| [tests/integration/defs/disaggregated/test_ad_disagg_trtllm_serve.py][ad_serve] | 显式 V1+CPP。 |
| [tests/unittest/auto_deploy/singlegpu/smoke/test_disagg.py][ad_smoke] | 显式 V1+CPP。 |

## 2. Helix 与 Nemotron：支持范围及测试取舍

Helix 的实现针对 Kimi K3 的具体需求。下列 DeepSeek/Qwen case 在本 PR 保留 CPP/V1 WAR；其他模型的支持扩展由后续需求决定。

| 文件 | 本 PR 保留的设置与覆盖 | 后续讨论点 |
|---|---|---|
| [tests/integration/defs/accuracy/test_disaggregated_serving.py][accuracy] | DeepSeek/Qwen Helix 的 V1+CPP；launcher 通过环境变量选择 UCX | 若出现对应模型的支持需求，以同拓扑的精度及 overlap 测试评估 NIXL/V2 方案。 |
| [tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp1cp2_deepseek_v3_lite_bf16_tllm_gen.yaml][helix_yaml] | 为既有 UCX+CPP 补显式 V1；保留 CTX TP2 → GEN TP1/CP2 | NIXL/V2 支持按后续需求评估，当前保留原有拓扑覆盖。 |
| [tests/scripts/perf-sanity/disaggregated/h200_nemotron-super-fp8_8k1k_con64_ctx1_tp2_gen1_tp2_eplb0_mtp0_ccb-UCX.yaml][nemotron_ucx] | 为既有 UCX+CPP 补显式 V1；保留 hybrid 模型性能场景 | 删除或迁移到 NIXL，取决于这条 Hopper workload 的维护需求与覆盖价值。若保留，再确定 NIXL 下的 manager/runtime 组合。 |

Nemotron 这份配置仍由 [perf 测试自动收集][perf_collector]。目前未找到同 H200、TP2、8k/1k、concurrency 64 的等价 NIXL 配置，删除或迁移尚未确定。本 PR 保留现有 WAR。

迁移方案的覆盖比较以原模型、拓扑、workload 和断言为基准。`fifo_v2` 表示通信实现；KVCM V2 的覆盖取决于 manager 配置和实际运行路径。

## 3. C++ 专属接口与指标：NIXL 下的可选方案

| 文件 | 当前依赖 / 限制 | 后续讨论点 |
|---|---|---|
| [tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py][single_gpu] | arbitrary-transfer 测试使用 C++ serialized `DataTransceiverState`；Python `get_context_state()` 尚未实现。本 PR 为既有 CPP case 补 V1。 | NIXL+CPP 是候选保留方案。若已有 Python 等价接口计划，对应覆盖包括成功传输和缺块错误。 |
| [tests/integration/defs/disaggregated/test_configs/disagg_config_llama4_kv_cache_overflow.yaml][llama4] | 本 PR 显式指定 CPP/V1，保留 128k input 与 2048-token C++ transfer buffer 的溢出回归。 | NIXL+CPP 是候选迁移方案，覆盖比较关注原 buffer 路径及溢出触发条件。 |
| [tests/integration/defs/disaggregated/test_configs/disagg_config_metrics.yaml][metrics] | 本 PR 使用 runtime/manager `auto`；调用方 UCX 环境变量使其选择 CPP/V1。现有 timing metrics 与 send/recv CSV 缺少等价 Python 验证。 | 待确定保留现有指标所需的 NIXL runtime。迁移涉及调用方和配置；Python 方案的覆盖以现有指标及 CSV 断言为基准。 |

上述候选方案尚未验证。覆盖是否等价取决于原断言是否通过，以及接口、buffer 和指标路径是否仍实际执行。

## 4. 保留的 C++ 专项回归

以下测试验证 C++ 本身的行为，本 PR 通过显式 CPP 指定固定测试对象。这部分保留为专项回归，无 Python 功能补齐事项。

| 文件 | 保留原因 |
|---|---|
| [tests/unittest/others/test_kv_cache_transceiver.py][binding_tests] | 五处 CPP 指定分别覆盖 C++ cancel 状态、shared_ptr/GC 生命周期、timeout warning 去重、warning 开关和 bounded polling；无 legacy selector 时使用 NIXL+CPP。 |
| [tests/unittest/_torch/disaggregation/test_disagg_inflight_cancel_gate.py][cancel_gate] | 四处 CPP 指定固定 backend/配置选择行为，包含 NIXL+LIBFABRIC 和 legacy selector 优先级。legacy selector 测试随对应配置接口退役再清理。 |

当前待讨论项集中在 Nemotron 的覆盖取舍，以及第 3 节的 NIXL 接口与指标方案。后续若确认等价覆盖，可据此评估撤回对应 WAR，并用 CI 验证。本 PR 保留当前必要的 WAR，以 CI 通过后合入为目标。

代码核查版本：`5eed11ca57`，下方文件链接固定到该提交。AutoDeploy 和 Helix 的处理已按最新沟通更新；候选迁移方案尚待运行验证。

[ad_ctx]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/examples/auto_deploy/model_registry/configs/disagg_ctx.yaml#L8
[ad_gen]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/examples/auto_deploy/model_registry/configs/disagg_gen.yaml#L8
[ad_test]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_ad_disagg.py#L187
[ad_serve]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_ad_disagg_trtllm_serve.py#L81
[ad_smoke]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/unittest/auto_deploy/singlegpu/smoke/test_disagg.py#L46
[accuracy]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/accuracy/test_disaggregated_serving.py#L836
[helix_yaml]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_configs/disagg_config_ctxtp2_gentp1cp2_deepseek_v3_lite_bf16_tllm_gen.yaml#L13
[nemotron_ucx]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/scripts/perf-sanity/disaggregated/h200_nemotron-super-fp8_8k1k_con64_ctx1_tp2_gen1_tp2_eplb0_mtp0_ccb-UCX.yaml#L62
[llama4]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_configs/disagg_config_llama4_kv_cache_overflow.yaml#L19
[metrics]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_configs/disagg_config_metrics.yaml#L16
[single_gpu]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py#L1008
[binding_tests]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/unittest/others/test_kv_cache_transceiver.py#L418
[cancel_gate]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/unittest/_torch/disaggregation/test_disagg_inflight_cancel_gate.py#L542
[perf_collector]: https://github.com/yizhang-nv/TensorRT-LLM/blob/5eed11ca57b4224a3896d6ab30209305ea1c7e4e/tests/integration/defs/perf/test_perf_sanity.py#L4402
