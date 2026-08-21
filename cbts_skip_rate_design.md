<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS Skip Rate 设计

**状态：** 实现方案

**日期：** 2026-08-21

**范围：** 标准 `/bot run` 的 expected baseline

## 1. 目标

在 CBTS decision 产生时计算并上报：

```text
case_skip_rate = 1 - P / B
```

其中：

- `P`：CBTS 最终选择、并且在 expected scheduling gates 下会运行的 test-db entries；
- `B`：不应用 CBTS 时，标准 pre-merge CI 在同一时刻预计运行的 test-db entries。

Case 的单位是一个 stage family 下的一条 test-db entry。同一 family 的 pytest-split shards 只计算
一次；同一 entry 在不同 hardware stage families 中分别计算。

## 2. 最小 baseline 模型

标准 single-GPU baseline 对同一 repository commit 视为固定。每次 pipeline 只需要额外判断是否将
standard multi-GPU baseline 加入 `B`：

```text
include_multi_gpu = non_cbts_multi_gpu_required && multi_gpu_label_gate_open

B = standard_single_gpu_cases
    + (include_multi_gpu ? standard_multi_gpu_cases : 0)
```

CI 只新增两个原始事实：

```json
{
  "b_non_cbts_multi_gpu_required": true,
  "b_multi_gpu_label_gate_open": true
}
```

不收集完整 `parallelJobsFiltered`、baseline stage-family ledger 或实际运行结果。

## 3. 两个布尔值的来源

### 3.1 `b_non_cbts_multi_gpu_required`

直接使用 `L0_MergeRequest.groovy` 已经计算的：

```groovy
testFilter[MULTI_GPU_FILE_CHANGED]
```

该值来自 `getMultiGpuFileChanged()`，表示普通 non-CBTS policy 是否要求 multi-GPU。

### 3.2 `b_multi_gpu_label_gate_open`

复用 multi-GPU dispatch 已有的 `ci: full pre-merge approved` validation policy：

- label 存在且由 active approver 添加：gate open；
- label 缺失或添加者未授权：gate closed；
- GitLab MR 和 post-merge exemption：gate open；
- label API validation 未完成：与现有 dispatch policy 一致，fail open。

这个字段表示有效 gate outcome，不等同于“同名 label string 是否存在”。

Telemetry 在 CBTS decision 时做 read-only snapshot。真正 dispatch multi-GPU 前仍重新检查，因此
telemetry 不能改变 CI 行为。用户在 decision 之后补加 label，不追溯修改此前的 expected-baseline
sample。

当 `b_non_cbts_multi_gpu_required=false` 时，label gate 与 denominator 无关。Telemetry 不发起额外
GitHub API 请求，并将 gate 视为 effectively open；只有 requirement 为 true 时才执行 label
validation。

## 4. Case count

Reporter 继续使用当前 commit 中的：

- `L0_Test.groovy` stage definitions；
- source test-db YAMLs；
- CBTS `affected_stage_test_counts`；
- sanity/perfsanity force-keep information。

规则：

- OnDemand 永远不进入标准 baseline；
- pre-merge 不包含 post-merge-only stages；
- multi-GPU 只有在两个布尔值同时为 true 时进入 `B` 和 expected `P`；
- shard suffix 去重，一个 stage family 只计算一次完整 test list；
- CBTS narrowed family 使用 `affected_stage_test_counts`；
- force-kept test-db family 使用完整 entry count。

## 5. OpenSearch fields

有效 CBTS hit document 写入：

```json
{
  "l_total_cases": 1401,
  "l_cbts_cases": 120,
  "d_case_skip_rate": 0.9143,
  "b_case_skip_rate_valid": true,
  "b_non_cbts_multi_gpu_required": true,
  "b_multi_gpu_label_gate_open": true
}
```

Dashboard 展示单次 pipeline 时可直接使用 `d_case_skip_rate`。跨 pipeline 聚合必须使用：

```text
1 - sum(l_cbts_cases) / sum(l_total_cases)
```

不能平均每条 document 的 rate。

## 6. Fallback

CBTS 整体 fallback、disabled 或 deferred 时不形成有效 skip-rate sample：

```text
b_case_skip_rate_valid = false
```

Dashboard 必须按 validity 过滤，不能把占位的 `0.0` 解释为真实 0% skip rate。

Tier 1 和白名单 Tier 2 都只消费最终 combined selection `P`。非白名单用户不执行 Tier 2 shadow
evaluation。

## 7. 明确不做

- 不收集真正启动、rendered 或完成的 testcase 数；
- 不建立 per-case ledger；
- 不从 `L0_Test` child 回传 baseline plan；
- 不复制所有 Groovy gates 到 reporter；
- 不精确重建 auto-trigger、only-one-group 或 backend filter 后的 counterfactual stage set；
- 不让 telemetry 或 label lookup failure 阻塞 CI。

因此该指标是 standard-baseline expected skip rate，不是实际执行完整率，也不是所有特殊调度模式下
的精确 counterfactual rate。

## 8. CI 稳定性

- rate 计算只包含本地 test-db 读取和简单算术；
- label snapshot 是 read-only、best-effort；
- interruption 继续向上传播，普通 telemetry exception fail open；
- label snapshot 不写 build description，也不改变真正 dispatch gate；
- reporter failure 只写日志，不让 pipeline 失败。
