<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# CBTS unhandled files 审计与规则实验

## 1. 结论摘要

本轮实验不支持“给所有未处理后缀加一个 out-of-scope 规则”。最近样本中的 unhandled files
至少来自五种完全不同的风险域：文档、CI 编排、C++/CUDA、依赖与 vendor、Python import-time。
将它们按文件后缀统一放行会漏掉真实 L0 影响。

建议分四类处理：

1. **可以增加精确规则**：`.github/CODEOWNERS`、独立的 security-scanning lock/metadata；
2. **文档必须运行专属 stage**：`docs/**`、Markdown 和 RST 应由 DocsRule 映射到
   `CPU-Build_Docs`，不能作为 noop；
3. **需要传播或消费者规则，不能 noop**：test helper、example entry point、生成的配置数据库、
   telemetry manifest、core Python import-time、新 Python module；
4. **继续 fallback**：Jenkins stage Groovy、C++/CUDA/CMake、产品依赖、vendor patch/source lock、
   未知 GitHub workflow 和无法闭合的动态加载。

规则扩展只解决第一层文件所有权问题。最近 100 个 main 提交中，58 个 fallback 里有 33 个先被
non-core-Python 文件挡住；移除 RST blocker 后，三个混合 PR 立即暴露出 import-time 或
class/signature blocker。因此，unhandled-file 规则和 import-execution 方案必须组合评估，不能用
Tier 1 noop 掩盖 Tier 2 的未知边。

## 2. 实验环境与方法

- 实验分支：`cbts-unhandled-files-experiments`；
- 实验 worktree：`worktrees/cbts-unhandled-files-experiments`；
- 样本：`upstream/main` 最近 100 个提交，按单提交历史快照回放；
- coverage DB：L0 PostMerge build 2961，x86_64 与 SBSA 合并 DB；
- DB commit：`0aede9d2a45d56334e2c5223413a0debd1f7537f`；
- 每个提交使用当时的 test-db YAML 与 `jenkins/L0_Test.groovy`；
- 为隔离 selector policy，dry-run metadata 中将 drift 固定为 0。

最后一点意味着本报告回答的是“如果 coverage freshness gate 通过，selector 能否形成安全上界”，
不是对历史 PR 当时真实 DB freshness 的复原。线上仍必须保留 30 commits drift 上限和架构 artifact
完整性检查。

本轮按单提交回放，适合发现文件类别和第二层 blocker。一个多提交 PR 的最终验收仍应使用 PR
merge-base 到 head 的累计 diff。

## 3. 最近 100 个提交的基线

scope 分布如下：

| Scope | 数量 |
|---|---:|
| fallback (`scope=None`) | 58 |
| `waiveonly` | 20 |
| `testdefonly` | 11 |
| `coverage` | 4 |
| `testsonly` | 4 |
| `noop` | 2 |
| `agentflowonly` | 1 |

58 个 fallback 的第一原因：

| 第一原因 | 数量 | 含义 |
|---|---:|---|
| non-core-Python residual | 33 | Tier 1 未认领，Tier 2 只接受 core `.py` |
| import-time effectful/unresolved | 13 | module/class/signature 在 import 阶段执行，当前无法闭合 |
| zero-touch/new file | 4 | DB 没有 row，不能把空集合解释成无影响 |
| class/signature import-time | 4 | class body、decorator、signature 等 import-executed scope |
| rule-forced fallback | 3 | 规则明确拒绝或找不到可执行 stage |
| 其他 | 1 | test-list 规则强制 fallback（#18941） |

fallback residual 中共出现 375 个文件，顶层目录分布：

| 顶层目录 | 文件出现次数 |
|---|---:|
| `tensorrt_llm/` | 223 |
| `cpp/` | 57 |
| `security_scanning/` | 26 |
| `docs/` | 15 |
| `jenkins/` | 13 |
| `examples/` | 7 |
| `docker/` | 7 |
| `3rdparty/` | 6 |
| `tests/` | 4 |

这些数字是文件在 fallback 决策中的出现次数，不是去重文件数，也不是每类规则可新增的 hit 数。

## 4. 规则实验

### 4.1 精确 metadata/security-scanning 探针

临时实验认领以下文件，实验后已恢复代码：

- `.github/CODEOWNERS`；
- `security_scanning/metadata.json`；
- `security_scanning/**/poetry.lock`。

100 提交回放因时间预算在第 78 个样本停止；已完成部分中有 4 个原 fallback 转为 `noop`：

- #18861：仅修改 `.github/CODEOWNERS`；
- 三个只修改 security-scanning lock/metadata 的 nightly 提交。

另两个 nightly 提交仍 fallback，因为同时修改 `security_scanning/pyproject.toml`。其 diff 改变了
OpenAI、Torch、NCCL、Diffusers、Safetensors、Click 等依赖约束，而且
`getMultiGpuFileChanged()` 明确认领该路径。结论是：

- lock/metadata-only 可以作为独立的 L0 noop 候选；
- 不得把整个 `security_scanning/` 前缀设为 noop；
- `security_scanning/pyproject.toml` 继续 fallback，除非后续建立 diff-aware dependency rule，且保留
  Groovy 的 multi-GPU 强制语义。

### 4.2 RST 收益上限探针

临时将 `.rst` 作为 out-of-scope，回放 5 个由 RST 首先阻塞的 PR，结果为 **0/5 直接 hit**：

| PR | 移除 RST blocker 后的下一原因 |
|---|---|
| #19022 | `commands/eval.py` unresolved import replacement |
| #18684 | `llmapi/disagg_utils.py` class/signature import change |
| #13872 | `executor_request_queue.py` class/signature import change |
| #19023 | `docs/source/conf.py` 仍为 non-core residual |
| #19050 | `docs/source/conf.py` / `helper.py` 仍为 non-core residual |

这说明 RST 规则本身收益有限，但能解除 import-execution 方案之前的假 blocker。实验中的临时
out-of-scope 只用于暴露第二层 blocker，不应成为生产行为。仓库已有 `CPU-Build_Docs`，会执行
Doxygen 和 Sphinx `make html`；生产实现应让 `.md`、`.rst` 和 `docs/**` 都选择该 stage。当前
`.md → noop` 会让 CBTS Layer 2 覆盖旧的 docs-only 选择并丢掉文档构建，应由 DocsRule 修复。

### 4.3 文档构建配置精确路径

临时额外认领：

- `docs/source/conf.py`；
- `docs/source/helper.py`。

结果：临时 noop 探针令 #19023 和 #19050 从 fallback 转为 `noop`，但这不是推荐的生产结果。
这两个文件是 Sphinx 构建配置，应由 DocsRule 转为 `docsonly` 并运行 `CPU-Build_Docs`。

不能由此扩大为 `docs/source/**/*.py`：

- `docs/source/_ext/llmapi_config_telemetry.py` 被
  `tests/unittest/usage/test_llmapi_config_telemetry_docs.py` 直接加载；
- `docs/source/_static/config_db.json` 被 config selector/sync tests 读取；
- docs 下的 JS、JSON、Python extension 可能有显式消费者。

因此 docs 路径应统一保留文档构建；有显式单测消费者的特殊文件还应额外映射对应 test anchor。

## 5. 候选规则分级

### 5.1 高置信、可进入实现

| 候选 | 处理方式 | 安全边界 |
|---|---|---|
| `.github/CODEOWNERS` | exact-path noop | 只影响 ownership/review；不扩大到 `.github/` |
| security lock/metadata | prefix + exact suffix/name noop | 排除 `pyproject.toml` 和其他可执行配置 |
| `docs/**`、`*.md`、`*.rst` | DocsRule → `CPU-Build_Docs` | 文档改动不允许 noop；混合 PR 与其他规则取 stage 并集 |
| AutoDeploy standalone | 补 `_AD_LEAKER_PATTERNS` | 找不到 AD blocks/entries 时仍 fallback |
| Ruff exclude | TOML semantic diff rule | 只允许 `[tool.ruff].exclude` 增加当前 PR 路径 |

上述规则必须有负向测试，证明相邻但有风险的路径不被认领。

### 5.2 需要消费者传播，不能直接 noop

| 类别 | 推荐方案 |
|---|---|
| `tests/test_common/*.py`、integration helper | 建 test reference/import graph，映射到消费它的 registered tests |
| `examples/**/*.py` entry point | 建显式 entry-point/stage inventory；找不到消费者时 fallback |
| `docs/source/_static/config_db.json` | 映射 config selector/sync tests |
| `docs/source/_ext/*.py` | repository references 转直接 unit-test anchors |
| `llm_args_golden_manifest.json` | 映射 telemetry/API tests，不作为生成文件 noop |
| core Python module/class/signature | 使用 importer closure/import-execution impact bound |
| 新 Python module | reverse import graph + SCC + test anchors；图不闭合时 fallback |

### 5.3 明确保留 fallback

- `jenkins/L0_Test.groovy`：它定义 stage inventory，也是 CBTS 自己解析的输入；
- `jenkins/Build*.groovy`、Docker/toolchain 安装脚本：可改变构建和执行环境；
- `cpp/**`、CUDA、CMake、Nanobind：当前 Python coverage DB 不是完整影响上界；
- `requirements*.txt`、产品 constraints、`security_scanning/pyproject.toml`；
- `3rdparty/vendor_sources.lock.yaml` 与 vendor patches；
- 未建专属语义规则的 YAML/JSON/TOML/TXT；
- 未知 `.github/workflows/**`：workflow 可能改变 CI 触发或 gate；
- VisualGen rule-forced fallback：本轮只记录，不在该实验中改变其公共或内部 API 边界。

## 6. 与 import-execution 方案的合并关系

unhandled-file 规则解决的是 Tier 1 ownership；以下能力仍应由 selector evidence graph 提供：

1. external binding reference 返回引用位置，而不是只返回 binding name；
2. test-side reference 转 test-db anchor；
3. production reference 继续进入 importer/consumer propagation；
4. 新 module 通过 reverse import graph 与 SCC 找到完整 importer closure；
5. 新增静态 import、可界定的 module-level deletion/replacement 使用 importer bound；
6. dynamic import、star import、module escape、解析失败、图预算耗尽时 fail closed。

本轮 RST 探针证明了两层是串联关系：#19022、#18684、#13872 只有在文档文件被正确认领后，才会进入
import-time 分析。最终 replay 必须检查完整 residual，不能以“第一条 fallback reason 消失”为完成标准。

## 7. 建议的 PR 划分

不建议把全部 E/F/G 和 import graph 合成一个 PR。建议按安全模型拆成三组：

1. **低风险 Tier 1 rule gaps**：DocsRule、AutoDeploy standalone、Ruff diff-aware、精确 metadata 规则；
2. **launcher/stage widening**：`trtllm-llmapi-launch` 与 multi-GPU forced stages，单独评审；
3. **import/reference selector**：test-side external reference、importer bound、新 module graph；其中新 module
   graph 复杂度最高，可再独立一个 PR。

如果希望减少 PR 数，AutoDeploy、Ruff、DocsRule/metadata 可以合并为一个“bounded Tier 1 rule gaps”PR；
launcher 不应混入，因为它不是 noop 或精确 test mapping，而是 stage-level 保守扩大。import/reference graph
也不应与这些静态规则混合，否则 reviewers 很难同时验证 AST completeness 和路径规则边界。

## 8. 验收标准

- 每条正向规则至少有一个对应的相邻负向 case；
- 使用 build 2961 合并 DB 回放目标 PR，并报告 DB commit 与 drift；
- 使用 PR 累计 diff，而不是只回放单 commit；
- 原 fallback 转 hit 时，reason 中能看见 exact rule、test anchor 或 widening evidence；
- 任意 `docs/**`、Markdown 或 RST 改动都保留 `CPU-Build_Docs`，包括与 Tier 2 core Python residual
  混合的 PR；
- #18808 依赖/toolchain PR 继续 fallback；
- Jenkins Groovy、C++/CUDA、vendor/dependency canary 继续 fallback；
- shadow 模式比较 selected-away cases 与完整基线，出现任何 escape 即停止上线；
- 没有兼容 coverage DB、DB 超过 30 commits、任一架构 artifact 缺失时继续 fail closed。
