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

# CBTS 命中率提升方案：import-executed、新文件与保守扩大

## 1. 背景与目标

CBTS 当前采用两层选择：Tier 1 静态规则先认领明确的文件和测试块，Tier 2 coverage selector
再处理剩余的 core Python 文件。任意 residual 文件无法得到可信影响上界时，CBTS fail closed，
返回 `scope=null` 并交回完整基线流水线。

本方案以 2026-09-15 的以下依赖和数据为基线：

- [PR #19124](https://github.com/NVIDIA/TensorRT-LLM/pull/19124)：共享 Python change analysis
  与 repository reference analysis；
- [PR #18802](https://github.com/NVIDIA/TensorRT-LLM/pull/18802)：按 PR merge base 选择兼容的
  coverage DB，并继续执行 30 commits drift 的 fail-closed 检查；
- L0 PostMerge build 2961 的 x86_64 与 SBSA 合并 DB，commit
  `0aede9d2a45d56334e2c5223413a0debd1f7537f`。

目标如下：

1. 保持“被跳过的 case 不受本次改动影响”这一安全约束；
2. 接受 [PR #18808](https://github.com/NVIDIA/TensorRT-LLM/pull/18808) 继续 fallback，
   因为它同时改变 dependency、container、vendor source 和未采集的 kernel 文件；
3. 提高其余样本 PR 的命中率，目标样本为 #19153、#19035、#17408、#18357、#19034、#18182；
4. #19148 已由 Tier 1 `outofscope` 规则命中，作为回归样本保持结果不变；
5. 不通过“返回非空 scope 但实际运行完整基线”美化 hit 指标。确需保留全部相关 stages 时，
   必须在 telemetry 中明确记录为保守扩大，而不是 case-level narrowing。

## 2. 安全定义

本方案中的“安全”指 CBTS 最终保留集合是实际受影响 L0 case 集合的上界：

```text
actual_impacted_cases ⊆ selected_cases
```

静态分析不需要证明一个 case 一定受影响，只需要证明被移除的 case 不受影响。任何无法证明完整性的
边、解析错误、动态行为或 coverage 不可信信号都必须导致扩大选择范围；无法构造有限且可信上界时继续
fallback。

以下情况不得为了 hit rate 静默放行：

- 变量形式的 `importlib.import_module()` 或 `__import__()`；
- `from module import *` 且无法解析 `__all__`；
- module object 被传给未知函数、写入容器或暴露给无法跟踪的反射代码；
- entry point、plugin registry、字符串配置、C++/shell/subprocess 发起的动态加载；
- 无法解析的 Python/TOML/diff，或 forge 没有提供可用 patch；
- coverage DB stale、缺少某个架构、case capture 不完整或 stage family 没有被采集；
- 新增/修改文件的 importer closure 无法闭合。

## 3. 当前 import-executed 方案

### 3.1 为什么不能直接使用 `<module>` coverage rows

模块体、class body、函数 signature 和 decorator 在 import 阶段执行。当前 coverage 收集没有完整覆盖
MPI worker 等进程的 import 阶段，因此 `<module>`、class qualname，甚至整个 file row set 都不是所有
importers 的可靠上界。现有 selector 不使用这些 rows 直接缩小测试。

### 3.2 changed line 到 qualname

当前分析先把 diff 中的 post-image 行映射到 AST scope：

| 位置 | 归属 |
|---|---|
| 普通函数或方法 body | `Class.method` / `func` |
| module statement | `<module>` |
| class body | `ClassName` |
| signature/decorator | enclosing import-executed scope |
| closure/comprehension | 最近的可记录 enclosing scope |

注释和空行由 `strip_noop_diff_lines` 移除；删除行锚定到后继 post-image 行，并由
`iter_diff_deleted_post_lines` 保存 old statement，供 replacement 分析使用。

### 3.3 当前低风险 allowlist

当前方案仅把以下 import-executed 改动转换成 local consumers：

- literal scalar/container 的 module assignment；
- old/new 都是安全 literal 且绑定名不变的 replacement；
- `from __future__ import annotations` 下的 annotated literal；
- 新增 builtin-module import；
- 没有 decorator、default expression 和 eager annotation 的新增普通函数声明。

分析结果包含：

- `changed_bindings`：变化的 module bindings；
- `binding_consumers`：读取这些 bindings 的本地函数/方法；
- `callers`：可静态证明的本地直接调用边；
- `callable_escapes`：函数作为值逃逸、进入嵌套 scope 或 import-time code 的情况；
- `limitation`：无法安全解释变化的原因。

当前 worktree 中相同概念暂时位于 `coverage_selection/qualname_map.py` 和
`coverage_selection/reference_map.py`；在 #19124 依赖落地后，最终实现应复用
`python_change_analysis.py` 和 `repository_reference.py`，不要保留两套并行 AST/reference
实现。

### 3.4 当前 impact 计算

对于已支持的 module binding：

1. repository reference analysis 检查 binding 是否被其他文件引用；
2. 有外部引用时当前直接 decline；
3. 没有外部引用时，把本地 consumers 当成 changed qualnames；
4. consumer 有 DB rows 时使用精确 rows；
5. 新增 consumer 没有 rows 时，沿本地 caller graph 向上寻找全部有 rows 的 callers；
6. caller 分支不完整、成环或发生 callable escape 时，退到配置的 no-data policy；
7. 默认 no-data policy 使用该文件的 whole-file test set。

该方案已经覆盖类似“module literal 改变，但实际只被少量函数消费”的常见改动，同时避免把所有
import-executed change 都误认为普通函数变更。

### 3.5 当前方案的命中缺口

当前实现仍是 file-level all-or-nothing gate：任意 residual 是非 `.py`、不在 DB，或存在不支持的
import-executed statement，整个 Tier 2 都 decline。repository reference 也只返回“哪些 binding 被
外部引用”，不返回引用位置，所以 selector 无法把一个安全可定位的 test-side reference 转成 impacted
test。

## 4. 样本 PR 诊断

| PR | 第一层 blocker | 移除第一层后的 blocker | 目标策略 |
|---|---|---|---|
| #19153 | 18 个新增 `trtllm_gen` Python 文件没有 DB rows | 现有文件的 module binding 被判定有外部引用，例如 `_K3_ROUTED_EXPERT_MODULE_PREFIXES` | 新文件 importer closure + 外部引用保守扩大 |
| #19035 | 新增 `from tensorrt_llm.logger import logger` 被判为 effectful module statement | 无更早 blocker | import-time importer bound |
| #17408 | AutoDeploy source 被认领，但规则找不到 AD block，返回 `scope=None` | SpecDec/TestDef 已有有效选择 | 补齐 AutoDeploy standalone entry 识别 |
| #18357 | `pyproject.toml` 是非 core-Python residual | 三个新增 CuTeDSL Python 文件没有 DB rows | 严格 TOML diff rule + 新文件 importer closure |
| #19034 | `trtllm-llmapi-launch` 是 Bash residual | `mpi_session.py` 删除 module-level env 设置，被判 unresolved replacement | launcher 静态规则 + deletion-only importer bound |
| #18182 | `_MEGA_MOE_SYMM_BUFFER_CACHE` 有跨文件引用 | 直接引用来自已登记的 `test_kimi_k3_situ_moe.py` | test-side reference 转 impacted test |
| #19148 | 已命中 | 无 | 仅做回归保护 |
| #18808 | dependency/container/vendor + zero-touch kernel | 明确接受 fallback | 不纳入本方案 |

## 5. 目标模型：从二元拒绝改为证据图上的保守扩大

将当前“发现外部关系就 decline”改为构造 impact evidence graph：

```text
changed statement / binding / new module
    ├── local consumer ──> local caller ──> covered qualname rows
    ├── external binding reference
    │       ├── test file ──> test-db anchor
    │       └── production file ──> its covered qualnames/importers
    └── module import edge ──> importer module ──> importer tests
```

图遍历只有三种结果：

- `complete_bound`：所有分支都到达可信的 DB rows 或明确 test-db anchors，可以 hit；
- `conservative_bound`：可以闭合，但必须扩大到 file/importer/test-family/stage-family 集合，可以 hit并记录
  widening；
- `unresolved`：存在动态边、解析失败或未覆盖终点，继续 fallback。

不得把 `unresolved` 自动降级成 whole-file rows，因为 whole-file rows 对 import-time execution 不是上界。

## 6. 策略一：外部 binding reference 位置化

### 6.1 API 变化

扩展 #19124 的 repository reference API，不再只返回 binding name set，而是返回结构化结果：

```text
BindingReferenceFacts
  direct_test_references: binding -> test paths
  direct_source_references: binding -> production paths
  opaque_escapes: binding -> paths/reasons
  complete: bool
```

AST 必须区分：

- `from target import NAME`；
- `alias.NAME`；
- 常量参数形式的 `getattr(alias, "NAME")`；
- `import *`；
- module alias 的简单赋值传播；
- module object 的未知传递/存储；
- 常量与非常量 dynamic import。

### 6.2 选择规则

- 直接测试引用通过 `YAMLIndex.find_match_for_path` 转成 test-db anchor，并 union 到 Tier 1 的
  `block_filters`；
- production reference 递归进入该文件的 qualname/importer impact；
- opaque escape 不再假装成“所有 binding 有直接引用”，而是携带明确 limitation；只有能扩大到完整
  importer family 时才继续，否则 fallback；
- 任一候选引用文件无法读取或解析时 `complete=False`，fail closed。

### 6.3 覆盖样本

#18182 中 `_MEGA_MOE_SYMM_BUFFER_CACHE` 的直接外部 Python 引用来自
`tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`，该测试已登记在 B200/B300 test DB，可以安全
转成必须保留的 test anchors。#19153 的 production/module reference 则需要继续做 importer widening，
不能套用测试引用的精确路径。

## 7. 策略二：新增 Python 模块 importer closure

### 7.1 为什么新文件不能使用 no-data file fallback

新增文件在历史 DB 中必然没有 rows。空 rows 表示“未知谁会执行它”，不是“没有测试执行它”。因此必须
从 post-image repository graph 反向寻找 importers，而不是把空集合当成可跳过。

### 7.2 算法

1. 从 diff 明确识别 added file；优先使用 forge change type，并以 `new file mode`/`/dev/null` 校验；
2. 将 repo path 规范化成 Python module，正确处理 package `__init__.py` 和 relative import；
3. 在 PR head 构建 reverse import graph；
4. 对多个相互引用的新模块先求 strongly connected components，避免递归环；
5. 从新增模块/SCC 向外遍历所有静态 importers；
6. 遇到测试文件时转成 test-db anchors；
7. 遇到历史文件时，使用该 importer 中实际 import statement 所属 scope：
   - module import：继续向 importer closure 传播；
   - lazy function import：使用该函数的 DB rows 和 caller bound；
8. 所有分支都到达可信 rows/anchors 后才返回 bound；
9. 动态 import、ambiguous short import、无法解析 source、C++/shell loader 或 graph 终点没有 DB/test-db
   覆盖时 fallback。

### 7.3 限制与扩大

- `__init__.py` 必须同时处理 re-export 和 package import semantics；
- `from x import *` 只有在静态 `__all__` 可解析时继续；
- changed tests 对新增模块的直接引用必须保留，即使 production importer graph 为空；
- graph 深度不应作为安全截断条件。可以设置计算预算，但超预算必须 fallback；
- 新文件 import chain 最终落到某个 DB 已知 module 时，不能使用不完整 `<module>` rows作为终点，仍需
  继续到 test anchor、函数 rows 或完整 importer family。

该策略覆盖 #19153 的 18 个新 `trtllm_gen` leaf，以及 #18357 的三个新 CuTeDSL 文件。两者都必须
在实现后重新跑完整 residual；只消除第一个 zero-touch 报错不足以证明命中。

## 8. 策略三：import-time importer bound

### 8.1 支持范围

在现有 local-consumer allowlist 之外，新增一个更宽但仍 fail-closed 的 importer-bound 分支：

- 静态、常量 module path 的新增 `import` / `from ... import ...`；
- 删除 module-level statement；
- old/new 无法转换成本地 binding consumer、但 AST 能完整界定 statement 边界的 replacement。

以下继续不支持：

- class body、decorator 和 eager default/annotation 的任意 effectful expression；
- 非常量 dynamic import；
- import 位于无法静态求值的条件/循环/exception control flow，除非直接扩大到完整 importer family；
- 同一个 patch 中跨多个 statement 的模糊删除锚点；
- 无法取得 old source 或无法解析 old/new AST。

### 8.2 影响上界

import-time statement 的上界不是本地 consumer rows，而是所有导入当前模块的测试：

```text
bound = direct test importers
      ∪ production importer closure 的终点测试
      ∪ changed test definitions 已要求保留的 tests
      ∪ untrusted / incomplete-capture tests
```

对于新增静态 import，变化的是“导入当前模块时新增执行另一个模块”这条边，因此必须保留当前模块的
完整 importer closure；不能只选择使用 imported binding 的函数。对于删除环境变量设置等副作用，也必须
使用同样的 importer closure。

### 8.3 覆盖样本

- #19035：新增 `tensorrt_llm.logger` 静态 import 后，以
  `fused_moe_triton.py` 的 importer closure 作为上界；`TritonFusedMoE.__init__` 的普通函数改动继续
  union 精确 function rows；
- #19034：`mpi_session.py` 删除 module-level `FLASHINFER_CUBIN_DIR` 设置，以
  `mpi_session.py` 的 importer closure 作为上界，不能把删除后的注释视为无执行影响。

## 9. 策略四：小而明确的 Tier 1 规则补齐

### 9.1 AutoDeploy standalone entry

`AutoDeployRule` 当前只识别 `test_llm_api_autodeploy.py` 和 `_autodeploy-`，而 #17408 对应 test DB
中存在：

```text
unittest/auto_deploy/standalone/test_standalone_package.py::
    TestStandalonePackage::test_run_unit_tests
```

将 `unittest/auto_deploy/standalone/` 加入 AD leaker pattern。规则必须 union：

- 所有 `backend: autodeploy` blocks；
- PyTorch block 中明确的 AD leaker entries；
- standalone package entry。

若 AD source 被修改但三类 entry 都不存在，仍返回 `scope=None`。

### 9.2 Ruff exclude 的 diff-aware noop

#18357 的 `pyproject.toml` 只在 `[tool.ruff].exclude` 中新增两个 source path。这不会改变 L0 test
执行结果，但不能把所有 `pyproject.toml` 变化列为 out-of-scope。

新增受限规则：

1. 解析 base/head TOML；
2. 证明唯一语义变化是 `[tool.ruff].exclude` 新增字符串；
3. 不允许删除/替换 exclude、不允许修改其他 key；
4. 新增路径必须与本 PR 新增/修改文件对应；
5. 满足时将该文件作为 `noop` contribution；否则不认领并 fallback。

该规则只处理 test selection。Build、pre-commit 和 lint gate 仍照常运行。

### 9.3 LLMAPI launcher 规则

`trtllm-llmapi-launch` 是 Bash，无法进入 Python coverage selector。它又被多 GPU、perf、disagg 和
serving 流程使用，不能仅根据本 PR 新增的 unit tests 缩小。

新增明确的 launcher rule：

- 认领 `tensorrt_llm/llmapi/trtllm-llmapi-launch`；
- 保留所有 baseline-eligible multi-GPU/launcher-consuming stages；
- union TestDef/Waives 对本 PR 测试变化给出的 block filters；
- 不对这些 multi-GPU blocks 做 case-level coverage pruning；
- 继续沿用 Groovy `MULTI_GPU_FILE_CHANGED` 对该路径的强制保护；
- 若 stage inventory 无法识别 launcher consumers，则 fallback。

这会让 #19034 成为保守命中，但预期 skip rate 低于普通 Python-only PR。Telemetry 必须标记
`widening=launcher_stages`。

### 9.4 明确不增加 dependency/toolchain rule

#18808 同时改变 `requirements.txt`、`constraints.txt`、Dockerfile、vendor lock 和未采集 kernel。
此类变化可能影响 import、build、code generation 和所有 kernel consumers，本方案明确保留 fallback，
不增加会掩盖风险的宽泛 dependency rule。

## 10. 与 Tier 1/Tier 2 的组合方式

当前 Tier 2 只返回 coverage narrowing，外部 test references、新文件 test anchors 和 launcher stages
会引入规则式证据。建议不要把这些逻辑分散成互相看不到的临时集合，而是引入统一的 impact bound：

```text
ImpactBound
  impacted_tests_by_family
  block_filters
  forced_stages
  no_data_functions
  widening_reasons
  unresolved_reasons
```

组合规则：

1. Tier 1 block filters、coverage function rows、reference-derived anchors 取并集；
2. `forced_stages` 只能增加 stage，不能被 coverage pruning 删除；
3. `untrusted`、`no_data`、coarse `-k` entries 保持现有强制运行语义；
4. 任意 `unresolved_reasons` 使该 residual analysis decline；
5. 只有实际删除 case 或 stage 时记录 `outcome=narrowed`；
6. 仅成功构造保守上界但没有删除任何 case 时记录 `outcome=conservative_no_skip`，不得计入有效
   skip-rate 分子。

## 11. Telemetry 与可解释性

新增或扩展 detail 字段：

| 字段 | 含义 |
|---|---|
| `import_bound_files` | 通过 importer closure 处理的文件 |
| `new_module_bound_files` | 通过新增模块 graph 处理的文件 |
| `external_reference_tests` | 外部 test references 转换出的 test 数量 |
| `external_reference_sources` | 进入 production propagation 的文件数量 |
| `widening_reasons` | `importers`、`test_reference`、`launcher_stages` 等 |
| `graph_nodes` / `graph_edges` | 本次证据图规模 |
| `graph_complete` | 所有分支是否闭合 |
| `unresolved_reasons` | fallback 的结构化原因，而非只保留首个字符串 |
| `outcome` | `narrowed`、`conservative_no_skip` 或 `fallback` |

诊断应累计所有能发现的 blocker，但任何 blocker 都不能阻止其余只读分析继续进行。这样一次 dry-run
可以看到修复第一层后是否还会遇到下一层问题。

## 12. 实施顺序

### 阶段 A：依赖整合与现有逻辑收敛

1. 以 #18802 和 #19124 为依赖重放当前 selector policy；
2. 删除/避免重复的 AST 与 repository reference 实现；
3. coverage selector 只消费 #19124 的共享 facts/API；
4. 保留当前 literal binding、local consumer、caller bound 和 caller escape tests；
5. 用 build 2961 重跑现有基线，确保 #19148 仍 hit、#18808 仍 fallback。

### 阶段 B：低风险高收益项

1. AutoDeploy standalone pattern，目标 #17408；
2. test-side external reference anchors，目标 #18182；
3. Ruff exclude diff-aware noop，作为 #18357 的一个前置条件。

这三项边界最清楚，应单独提交和验收。

### 阶段 C：import-time importer bound

1. 扩展 repository importer facts 的 completeness/limitation；
2. 支持新增静态 import，目标 #19035；
3. 支持可完整解析的 deletion-only module statement，目标 #19034 的 Python 部分；
4. 增加 launcher forced-stage rule，完成 #19034；
5. 对 dynamic/conditional/ambiguous cases 保持 fallback。

### 阶段 D：新增模块与 production reference graph

1. 新增文件识别和 module normalization；
2. reverse import graph 与 SCC；
3. test anchors、lazy import function rows 和 importer closure 终点；
4. production external reference propagation；
5. 目标 #18357、#19153；
6. 设置计算预算，超时/超预算 fail closed。

阶段 D 风险和复杂度最高，不应与阶段 B/C 合并成一个不可审查的大提交。

## 13. 测试计划

### 13.1 单元测试

每条允许路径都需要正向和反向用例：

- literal assignment、replacement、builtin import、plain function 的现有用例保持；
- direct test import、module alias attribute、constant `getattr`；
- production reference 的多层传播；
- module object escape、dynamic import、star import、unparsable source 必须 fallback；
- 新 module、package `__init__`、relative import、re-export、import cycle/SCC；
- lazy function import 与 module-level import 的不同 bound；
- static import addition 和 deletion-only statement；
- old/new AST 不一致或删除跨多个 statement 时 fallback；
- Ruff exclude 纯新增命中，删除/替换/其他 TOML key 变化 fallback；
- AutoDeploy standalone 与 AD blocks union；
- launcher rule 强制保留 multi-GPU stages；
- graph 超预算/超时 fallback；
- telemetry 字段稳定且 deterministic。

### 13.2 回放验收

使用与 CI 相同的 x86_64+SBSA 合并 DB，并同时检查 DB/PR base drift。目标矩阵：

| PR | 预期 |
|---|---|
| #19148 | 继续由 Tier 1 hit，skip rate 不退化 |
| #17408 | hit，保留 AD standalone + AD blocks + SpecDec/TestDef union |
| #18182 | hit，显式保留 `test_kimi_k3_situ_moe.py` 对应 blocks |
| #19035 | hit，reason 显示 `importers` widening |
| #19034 | hit，launcher/multi-GPU forced stages 保留 |
| #18357 | 在 Ruff rule 与 new-module graph 均成功时 hit |
| #19153 | 只有全部 18 个新增模块和 production references 闭合时 hit；否则保持 fallback |
| #18808 | 继续 fallback |

除这些目标样本外，还要回放最近至少 200 个 core-Python PR，并比较：

- 原 hit 不得变成更窄但缺失原 impacted case 的选择；
- 原 fallback 变 hit 的每一项都必须有结构化 evidence；
- 对 dynamic/reflection/new-file 边界的负向 canary 必须继续 fallback；
- 统计 stage skip、case skip、untrusted forced-run 和 selector timeout。

### 13.3 Shadow 与上线门槛

1. 先以 shadow 模式生成选择，不实际过滤；
2. 对 shadow-selected-away cases 与完整基线结果做差异检查；
3. 至少覆盖多个连续 PostMerge DB revisions 和两个 CPU 架构；
4. 任何“被 selector 跳过但完整基线失败”的 escape 都阻止上线；
5. 先上线阶段 B，再上线阶段 C，最后单独评审阶段 D；
6. 保留 `--disable-cbts` kill switch 和现有 artifact/render 失败回退。

## 14. 完成标准

本方案完成不以 dashboard hit 数字单独判断，而以以下条件共同判断：

- #18808 明确且稳定地 fallback；
- #19148 保持原有 Tier 1 命中；
- #17408、#18182、#19035、#19034 获得可解释且 fail-closed 的 hit；
- #18357、#19153 仅在完整 importer/reference graph 闭合后 hit；
- 所有新增 widening 都能在 decision JSON 和 OpenSearch 中查询；
- 负向动态/反射/解析失败测试继续 fallback；
- shadow 验证没有 selector escape；
- 无新 coverage DB、DB drift 超限、架构 artifact 不完整时继续完整回退。

最终期望不是无条件把样本从 `fallback` 改名为 `hit`，而是让 CBTS 对可证明的变化形成安全上界，
对无法证明的变化继续明确拒绝。
