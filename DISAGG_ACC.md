# DISAGG_ACC：本地 Disagg Perf Pipeline 集成 Accuracy 测试设计（v2）

## 背景

当前 `jenkins/scripts/perf/local/submit.py` 支持通过 pytest 驱动的 disagg E2E benchmark 流程
（`test_perf_sanity.py::test_e2e[disagg-e2e-{config}]`）。`examples/disaggregated/slurm/benchmark/submit.py`
（下称"examples submit"）已有 accuracy 测试的完整实现，但走的是直接 shell 路径。本文件描述如何将
accuracy **运行**能力移植到本地 perf pipeline，**验证逻辑留给 CI**。

---

## 设计原则

1. **开关**：只用 YAML 里的 `accuracy.enable_accuracy_test`，无 CLI flag
2. **只跑不验证**：`test_perf_sanity.py` 只负责运行 lm_eval 并保存结果 JSON，不做任何 assert；threshold 字段保留在 YAML 里供 CI 读取
3. **数据集**：使用已有的 parquet + 本地 task YAML 方案，零网络依赖

---

## 现有架构

```
submit.py
  ↓ 生成 slurm_launch.sh（env vars + srunArgs）
sbatch slurm_launch.sh
  └─ 多节点，DISAGG_SERVING_TYPE 不同
       slurm_run.sh → eval $pytestCommand
         → test_perf_sanity.py::test_e2e[disagg-e2e-{config}]
              ├── CTX  role → 启动 CTX server
              ├── GEN  role → 启动 GEN server
              ├── DISAGG_SERVER role → 启动 disagg server，写 server_config.{idx}.yaml
              └── BENCHMARK role
                    → wait_for_endpoint_ready(http://{host}:{port}/health)
                    → [NEW] 运行 lm_eval，保存结果 JSON
                    → 运行 trtllm-bench（原有）
```

---

## Dataset 方案（现有基础设施已支持）

### 现状

代码库里已有：
- `tests/integration/lm_eval_configs/gsm8k_local.yaml` — 本地 parquet task 定义
- `tests/integration/lm_eval_configs/gpqa_diamond_local.yaml` — 同类

`gsm8k_local.yaml` 关键片段：
```yaml
task: gsm8k_local
dataset_path: parquet
dataset_kwargs:
  data_files:
    test: LLM_MODELS_ROOT/datasets/openai/gsm8k/main/test-00000-of-00001.parquet
```

其中 `LLM_MODELS_ROOT` 是一个**占位符**，运行时由 `replace_env_in_file()` 替换为实际路径。
parquet 文件极小（GSM8K 测试集约 1.3k 条，几 MB），已存放在 `$LLM_MODELS_ROOT/datasets/` 下。

### 使用方式

```
lm_eval \
  --model local-completions \
  --tasks gsm8k_local \
  --model_args model=<path>,base_url=http://{host}:{port}/v1/completions,... \
  --include_path {work_dir}/lm_eval_configs/ \   ← 指向替换后的 task YAML 目录
  --log_samples \
  --output_path {output_dir}/accuracy_eval_gsm8k/
```

`replace_env_in_file()` 已在 `examples/disaggregated/slurm/benchmark/submit.py` 实现，
直接移植到 `jenkins/scripts/perf/local/submit.py`。

### 零依赖清单

| 所需内容 | 来源 | 大小 |
|----------|------|------|
| `gsm8k_local.yaml` | `tests/integration/lm_eval_configs/`（已有） | < 2 KB |
| `test-00000-of-00001.parquet` | `$LLM_MODELS_ROOT/datasets/openai/gsm8k/main/`（预下载） | ~2 MB |
| lm_eval CLI | 容器内已安装 | — |

**不需要**：HuggingFace Hub 网络访问、完整 HF cache 目录、`HF_DATASETS_OFFLINE` 设置。

---

## YAML 配置格式

在各 disagg test config YAML 里扩展 `accuracy` 段（目前仅有 `enable_accuracy_test: false`）：

```yaml
accuracy:
  enable_accuracy_test: true          # 总开关
  env_var: {}                         # 可选，lm_eval 进程的额外环境变量
  tasks:
    gsm8k_local:                      # lm_eval task 名（与 task YAML 中的 task: 字段对应）
      model: local-completions        # local-completions → /v1/completions
                                      # local-chat-completions → /v1/chat/completions
      model_args_extra: "num_concurrent=512,max_retries=3,tokenized_requests=false,timeout=1200,max_gen_toks=256,max_length=4096"
      threshold: 84.0                 # 供 CI 消费，test_perf_sanity.py 不使用此字段
      extra_kwargs:
        custom_config: tests/integration/lm_eval_configs/gsm8k_local.yaml
        # limit: 500                  # 可选，限制样本数
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `enable_accuracy_test` | bool | 总开关，`false` 时整个 accuracy 流程跳过 |
| `tasks` | dict | key 是 lm_eval task 名 |
| `model` | str | lm_eval model backend |
| `model_args_extra` | str | 追加到 `--model_args` |
| `threshold` | float | CI 用，pytest 不使用 |
| `extra_kwargs.custom_config` | str | 指向本地 task YAML；submit.py 会替换其中的 `LLM_MODELS_ROOT` 并写到 work_dir |

---

## 涉及改动的文件

### 1. `jenkins/scripts/perf/local/submit.py`

**新增**：`replace_env_in_file()` 函数（从 examples submit 移植，约 20 行）

**改动逻辑**（在生成 disagg 模式 `script_prefix_lines` 时追加）：

```python
# 读 accuracy 配置
acc_cfg = config.get("accuracy", {})
if acc_cfg.get("enable_accuracy_test"):
    import json

    # 处理 custom_config：替换 LLM_MODELS_ROOT 占位符，写到 work_dir/lm_eval_configs/
    env_sub = {"LLM_MODELS_ROOT": args.llm_models_root}
    processed_acc_cfg = copy.deepcopy(acc_cfg)
    for task_name, task_cfg in processed_acc_cfg.get("tasks", {}).items():
        extra_kwargs = task_cfg.get("extra_kwargs", {})
        if "custom_config" in extra_kwargs:
            config_path = extra_kwargs["custom_config"]
            if not os.path.isabs(config_path):
                config_path = os.path.join(llm_src, config_path)
            # replace_env_in_file 写到 work_dir/lm_eval_configs/，返回该目录
            lm_eval_configs_dir = replace_env_in_file(work_dir, config_path, env_sub)
            extra_kwargs["include_path"] = lm_eval_configs_dir
            del extra_kwargs["custom_config"]

    # 序列化后 export 给 BENCHMARK 节点
    script_prefix_lines.append(
        f"export ACCURACY_CONFIG_JSON='{json.dumps(processed_acc_cfg)}'"
    )
    srun_args_lines.append("--container-env=ACCURACY_CONFIG_JSON")
```

**改动量**：约 +30 行（含 `replace_env_in_file` 移植）

---

### 2. `tests/integration/defs/perf/test_perf_sanity.py`

**改动位置**：`DisaggTestCmds.run()` 的 `BENCHMARK` 分支，`wait_for_endpoint_ready` 之后、
trtllm-bench 循环之前。

**新增辅助函数 `_run_accuracy_tests()`**（**不含任何 assert，只跑并保存**）：

```python
def _run_accuracy_tests(accuracy_cfg, server_hostname, server_port,
                        model_path, output_dir, server_idx):
    """Run lm_eval against the running disagg server. No validation — just collect results."""
    endpoint_map = {
        'local-completions':      'v1/completions',
        'local-chat-completions': 'v1/chat/completions',
    }
    env_var = accuracy_cfg.get("env_var") or {}

    for task_name, task_cfg in accuracy_cfg.get("tasks", {}).items():
        model_type      = task_cfg.get("model", "local-completions")
        model_args_extra = task_cfg.get("model_args_extra", "")
        extra_kwargs    = task_cfg.get("extra_kwargs", {})

        endpoint    = endpoint_map.get(model_type, "v1/completions")
        base_url    = f"http://{server_hostname}:{server_port}/{endpoint}"
        model_args  = f"model={model_path},base_url={base_url},{model_args_extra}"

        acc_output_dir = os.path.join(output_dir,
                                      f"accuracy_eval_{task_name}.{server_idx}")
        log_file = os.path.join(output_dir,
                                f"accuracy_eval_{task_name}.{server_idx}.log")
        os.makedirs(acc_output_dir, exist_ok=True)

        cmd = ["lm_eval", "--model", model_type,
               "--tasks", task_name,
               "--model_args", model_args,
               "--log_samples",
               "--output_path", acc_output_dir]

        include_path = extra_kwargs.get("include_path")
        if include_path:
            cmd += ["--include_path", include_path]

        for k, v in extra_kwargs.items():
            if k in ("include_path",):
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")

        run_env = copy.deepcopy(os.environ)
        run_env.update(env_var)

        print_info(f"[Accuracy] Running {task_name} → {log_file}")
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd, env=run_env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        print_info(f"[Accuracy] {task_name} done, exit_code={ret.returncode}")
        # 不 raise，让 benchmark 继续跑；CI 读结果 JSON 做验证

    print_info("[Accuracy] All accuracy tasks completed.")
```

**在 BENCHMARK 分支调用处**（约 +5 行）：

```python
elif self.disagg_serving_type == "BENCHMARK":
    try:
        disagg_server_hostname, disagg_server_port = ...
        wait_for_endpoint_ready(...)

        # === 新增 ===
        acc_cfg_json = os.environ.get("ACCURACY_CONFIG_JSON")
        if acc_cfg_json:
            import json as _json
            acc_cfg = _json.loads(acc_cfg_json)
            if acc_cfg.get("enable_accuracy_test"):
                _run_accuracy_tests(
                    accuracy_cfg=acc_cfg,
                    server_hostname=disagg_server_hostname,
                    server_port=disagg_server_port,
                    model_path=os.environ.get("LLM_MODELS_ROOT", ""),
                    output_dir=self.test_output_dir,
                    server_idx=server_idx,
                )
        # === 原有：trtllm-bench ===
        for client_idx, client_cmd in enumerate(self.client_cmds[server_idx]):
            ...
```

**改动量**：约 +50 行（1 个函数 + 调用处）

---

## 输出文件（CI 消费）

```
{test_output_dir}/                            ← test_perf_sanity 的 output_dir/test_param_labels/
├── accuracy_eval_gsm8k_local.0.log           # lm_eval stdout/stderr
├── accuracy_eval_gsm8k_local.0/              # --output_path 目录
│   └── gsm8k_local/                          # lm_eval 按 task 组织
│       └── results_YYYYMMDD_HHMMSS.json      # 结果 JSON，含 acc,none / acc_norm,none
├── trtllm-benchmark.0.0.log                  # 原有
└── ...
```

**CI 读取示例（Python）**：
```python
import json, glob

result_files = glob.glob("**/results_*.json", recursive=True)
data = json.load(open(result_files[0]))
score = data["results"]["gsm8k_local"]["acc_norm,none"]  # 0-1 之间
threshold = 0.84  # 从 YAML 的 threshold 字段读取
assert score >= threshold, f"GSM8K: {score:.3f} < {threshold}"
```

---

## 数据流总结

```
submit.py
  → 读 YAML accuracy.enable_accuracy_test
  → replace_env_in_file(gsm8k_local.yaml)  ← 替换 LLM_MODELS_ROOT
      → 写 {work_dir}/lm_eval_configs/gsm8k_local.yaml
  → export ACCURACY_CONFIG_JSON='{"enable_accuracy_test": true, "tasks": {..., "include_path": ...}}'
  → srunArgs: --container-env=ACCURACY_CONFIG_JSON

SLURM BENCHMARK 节点
  slurm_run.sh → eval $pytestCommandBenchmark
    → test_perf_sanity.py BENCHMARK role
      → wait_for_endpoint_ready            (已有)
      → _run_accuracy_tests(...)           (新增，无 assert)
          → lm_eval --tasks gsm8k_local
                    --include_path {work_dir}/lm_eval_configs/
                    → 读 parquet: $LLM_MODELS_ROOT/datasets/openai/gsm8k/.../test.parquet
                    → 写结果到 test_output_dir/accuracy_eval_gsm8k_local.0/
      → trtllm-bench ...                   (已有)

CI
  → 读 accuracy_eval_gsm8k_local.0/results_*.json
  → 比较 score 与 YAML 里的 threshold
  → pass / fail
```

---

## 不需要改动的文件

| 文件 | 原因 |
|------|------|
| `jenkins/scripts/perf/local/slurm_run.sh` | 纯 `eval $pytestCommand`，逻辑在 pytest 内 |
| `tests/integration/lm_eval_configs/gsm8k_local.yaml` | 已有，直接用 |
| `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` | 角色调度逻辑不变 |
| `tests/integration/defs/perf/disagg/reporting/accuracy_validator.py` | 不引用，CI 自己做验证 |
