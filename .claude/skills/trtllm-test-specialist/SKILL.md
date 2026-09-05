---
name: trtllm-test-specialist
description: >
  Runs model-level and module-level tests for TensorRT-LLM. First classifies
  the test scope (module test or model test), then dispatches to the appropriate
  workflow. Model tests are further classified by type (functionality/smoke test,
  benchmark, or evaluation). Prompts the user for parallelism parameters
  (tp, ep, dp), dataset paths, device type, or a config file as needed. All
  test execution is delegated to trtllm-case-executor.
tags:
  - tensorrt-llm
  - testing
  - modeling
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Model Test Specialist

> **Execution mode:** After the Step 0a parameter-confirmation pause (the one mandatory stop — see Step 0a), run all remaining steps end-to-end. Invoke sub-skills (`trtllm-case-executor`, `exec-env-check`, `trtllm-test-script-builder`, etc.) sequentially and proceed immediately to the next step as soon as each returns. Do not stop to summarize intermediate results or ask the user for additional input unless a required parameter is genuinely missing.

This skill handles perf-sanity, module-level, and model-level testing. It classifies the test scope, builds the appropriate test commands, and delegates execution to `trtllm-case-executor`.

**Three test scopes:**

- **Perf-sanity benchmark test** — `test_cmd` targets `test_perf_sanity.py` (contains `test_perf_sanity` in the path or `test_e2e[` in the selector). Routed to the Benchmark Test Workflow's pre-built path. No ≤8-GPU limit. `device_type` and `model_name` auto-derived from the test ID.
- **Module test** — run a caller-provided test command for a module. The skill does not generate or author test scripts.
- **Model test** — end-to-end model-level tests:
  - **Functionality test (smoke test)** — `pytest examples/pytorch/quickstart_advanced.py` with pytorch backend.
  - **Benchmark test** — performance benchmarking. Single-node uses `trtllm-bench` by default; multi-node falls back to the perf-sanity config-YAML path.
  - **Evaluation test** — accuracy evaluation. Single-node uses `trtllm-eval` by default; multi-node falls back to the perf-sanity config-YAML path (`accuracy.enable_accuracy_test: true`).

---

## Config File

`config_file` is a standalone super parameter. **The file must be a valid YAML file with a `.yaml` or `.yml` extension.** When provided, it is the **sole** source of parameters — all values specified directly in the prompt are dropped.

If the user provided other parameters alongside `config_file`, issue this warning before doing anything else:

> ⚠️ **Config file provided — all directly specified parameters will be dropped.**
> The following parameters you specified will be ignored: `<param1>`, `<param2>`, ...
> Only values from `<config_file>` will be used. To override individual settings, edit the config file directly.

Then proceed:

1. Parse the YAML file using `scripts/parse_config.py`.
2. Use the parsed values as the **sole** authoritative parameter set. Discard all parameters supplied directly in the prompt.
3. All subsequent steps operate on the config-file parameter set only.

Valid keys are the input names listed in the [Inputs](#inputs) section. Unrecognized keys are warned about and ignored. A full template with all supported fields and inline documentation is at `references/test_config_template.yaml`.

---

## Inputs

### Common Inputs

| Input | Description | Required | Example |
|-------|-------------|----------|---------|
| `model_name` | Short model name | **Yes** (not required for pre-built perf-sanity — derived from test ID) | `llama_3_1_8b` |
| `checkpoint_path` | Path to the HuggingFace checkpoint directory or model name | **Yes** (not required for pre-built perf-sanity) | `/models/Llama-3.1-8B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct` |
| `repo_path` | Root of the TensorRT-LLM repo (default: current working directory) | No | `/home/user/TensorRT-LLM` |
| `report_file` | Path to write the test report (markdown). Parent directories are created automatically. Default: `./<MODEL_NAME>-auto-test-report.md` | No | `bring_up/llama/test_results/single_gpu_summary.md` |
| `config_file` | Path to a YAML super-parameter file. See [Config File](#config-file). | No | `configs/benchmark_llama_8b.yaml` |

### Module Test Inputs

| Input | Description | Required | Example |
|-------|-------------|----------|---------|
| `test_cmd` | The exact test command to run (e.g. a `pytest` invocation). The skill runs this verbatim via `trtllm-case-executor` and does not generate test scripts. | **Yes** (module test) | `pytest tests/unittest/_torch/attention/test_attention.py -v` |
| `class_name` | Name of the class under test (used for reporting only) | No | `Attention` or `LlamaAttention` |
| `required_devices` | Number of GPUs required to run `test_cmd` | No (default: `1`) | `1`, `4` |

### Model Test Inputs

| Input | Description | Required | Example |
|-------|-------------|----------|---------|
| `test_type` | Type of model test: `functionality`, `benchmark`, or `evaluation` | No (auto-classified) | `functionality` |
| `mtp_layers` | Comma-separated MTP speculation depths to benchmark (one run per value). Triggered only by explicit "mtp layers" or "speculative decoding layers" mentions. | No | `1,2,3,4,5,6` |
| `tp_size` | Tensor parallelism size | No (default: `1`) | `4` |
| `ep_size` | Expert parallelism size (MoE models) | No (default: `0`) | `8` |
| `dp_size` | Data parallelism size | No (default: `1`) | `2` |
| `pp_size` | Pipeline parallelism size | No (default: `1`) | `2` |
| `device_type` | Target GPU device type | No | `GB200`, `B200` |
| `dataset_path` | Path to evaluation or benchmark dataset file | No (benchmark/eval) | `/data/datasets/gsm8k.json` |
| `benchmark_config_yaml` | Path to an existing benchmark config YAML file (perf-sanity fallback) | No (benchmark) | `examples/disaggregated/slurm/benchmark/config.yaml` |
| `perf_config_yaml` | Path to a pre-built perf-sanity config YAML. Used with pre-built perf-sanity `test_cmd` to pass node/GPU sizing to `trtllm-case-executor`. | No (perf-sanity) | `/work/deepseek_v3_disagg.yaml` |
| `bench_subcommand` | `trtllm-bench` subcommand: `throughput` or `latency` | No (default: `throughput`) | `latency` |
| `backend` | Backend for `trtllm-bench`: `pytorch`, `tensorrt`, or `_autodeploy` | No (default: `pytorch`) | `pytorch` |
| `num_requests` | Synthetic dataset size when `dataset_path` is not provided | No (default: `100`) | `512` |
| `input_mean` / `output_mean` | Mean input / output token lengths for synthetic dataset | No (defaults: `128` / `128`) | `1024` / `1024` |
| `concurrency` | `--concurrency` flag for `trtllm-bench` | No | `32` |
| `max_batch_size` / `max_num_tokens` | Runtime limits for `trtllm-bench` | No | `64` / `8192` |
| `extra_llm_api_options_yaml` | Path to a YAML file passed via `--config` / `--extra_llm_api_options` to override LLM API settings. Accepted by both `trtllm-bench` (all subcommands) and `trtllm-eval` (all tasks). | No | `/tmp/pytorch_extra.yaml` |
| `eval_tasks` | Evaluation benchmark tasks (comma-separated) | No (evaluation) | `gsm8k,gpqa` |

---

## Step 0: Load Config File

If `config_file` is provided:

1. If any other parameters were also supplied in the prompt, output the warning described in [Config File](#config-file) before proceeding. List every extra parameter the user specified.
2. Run the parser script via Bash:
   ```bash
   python3 <skill_dir>/scripts/parse_config.py --config-file <config_file>
   ```
   The script exits non-zero on any error (file not found, wrong extension, invalid YAML). If it fails, stop and report the error to the user before proceeding.
3. Parse the JSON output. The output has three top-level keys:
   - `params` — recognised parameters found in the config file plus any scope-specific defaults that were filled in.
   - `missing_required` — required parameters still absent after defaults were applied (list of strings).
   - `defaults_applied` — parameters whose values were filled in from built-in defaults, not from the config file (list of strings; informational only).
4. If `missing_required` is non-empty, **stop immediately**. Report to the user which required parameters are missing, and instruct them to add those parameters to the config file before retrying. Example message:
   > The following required parameters are missing from the config file. Please add them to `<config_file>` and re-run:
   > - `checkpoint_path` — Path to the HuggingFace checkpoint directory or model name
   > - `tp_size` — Tensor parallelism size
   Do **not** proceed to any subsequent step.
5. Use the values in `params` as the **sole** parameter set. All parameters supplied directly in the prompt are discarded.

All subsequent steps operate on the config-file parameter set only.

---

## Step 0a: Log Resolved Parameters

After all parameters are resolved — from `config_file` (Step 0) or directly from user input — output a table of every resolved parameter before proceeding to Step 1. This makes the active configuration visible and auditable.

**Table format:**

| Parameter | Value | Source |
|-----------|-------|--------|
| `model_name` | `llama_3_1_8b` | `user_input` |
| `checkpoint_path` | `/models/Llama-3.1-8B-Instruct` | `config_file` |
| `tp_size` | `4` | `config_file` |
| `ep_size` | `0` | `default` |
| `pp_size` | `1` | `default` |

**Source** column values:

| Source | Meaning |
|--------|---------|
| `config_file` | Value was read from the YAML config file |
| `user_input` | Value was provided directly in the prompt |
| `default` | Value was auto-filled from built-in defaults (corresponds to `defaults_applied` entries when a config file was parsed) |
| `derived` | Value was auto-derived (e.g., `model_name` inferred from `checkpoint_path`) |

Include all parameters that have a resolved value. Omit parameters that are not set and have no applicable default.

After printing the table, **pause and ask the user to confirm before proceeding**:

> The above parameters will be used for this test run. Do you want to continue?

Wait for an explicit confirmation (`yes`, `y`, `ok`, `proceed`, or equivalent) before advancing to Step 1. If the user says no or requests changes, apply the requested changes and re-display the updated table before asking again.

---

## Step 1: Classify Test Scope

Evaluate in this order (first match wins):

1. **Perf-sanity benchmark test** — `test_cmd` is provided AND (`test_cmd` contains `test_perf_sanity` in the file path OR `test_e2e[` in the test selector). → **Benchmark Test Workflow** (pre-built perf-sanity path, see Step 0 of that workflow)
2. **Module test** — `test_cmd` is provided. → **Module Test Workflow**
3. **Model test** — no `test_cmd`. Classify further in Step 1a.
4. **Ambiguous** — ask the user to clarify.

### Step 1a: Classify Model Test Type

**Layer disambiguation:** "mtp layers N,M,…" or "speculative decoding layers N,M,…" are interpreted as MTP speculation depths (`mtp_layers`) and route to the **Benchmark/Evaluation test**.

1. **Evaluation test** — user requests accuracy evaluation, or `eval_tasks` is provided.
2. **Benchmark test** — user requests benchmarking, `mtp_layers` is provided, or `benchmark_config_yaml`/`benchmark_model_name` is provided.
3. **Functionality test** — all other model-level test requests (default).

---

## Module Test Workflow

Runs a caller-provided test command against a TRT-LLM module. The skill does
**not** search for existing tests, generate test scripts, or modify source
files — it only executes the given `test_cmd` via `trtllm-case-executor`.

### Step 1: Validate Input

- `test_cmd` is required. If missing, ask the user for the exact command to run.

### Step 1a: Derive `required_devices` and `device_type` from Test Markers

When `required_devices` or `device_type` are **not** provided by the user, run:

```bash
python3 <skill_dir>/scripts/extract_test_markers.py --test-cmd "<test_cmd>"
```

The script parses the target class and function in the test file and returns JSON:

```json
{
  "required_devices": 4,
  "device_type": "Blackwell (B200/GB200)",
  "sources": {
    "required_devices": "derived",
    "device_type": "derived"
  }
}
```

- If the user already provided `required_devices` or `device_type`, **skip** running the script for those fields — keep the user values with source = `user_input`.
- Use the `sources` field from the JSON to set the source column in the parameter table (`derived` or `default`).
- If `device_type` is an empty string, the field is unconstrained — omit it from the parameter table.

Update the parameter table in Step 0a with the values before proceeding.

### Step 1b: Validate Resource Limits (pytest / module test)

After `required_devices` is resolved, enforce these hard limits before proceeding:

- **Max GPUs:** `required_devices` must be `<= 8`. If it exceeds 8, **stop immediately** and report:
  > `required_devices = N` exceeds the maximum of 8 GPUs allowed for pytest / module tests. Reduce `tp_size`/`pp_size` or split the test.
- **Max nodes:** `node_count` for pytest is always `1` (single-node). If the derived `required_devices` would require more than one node given the available `gpus_per_node`, **stop immediately** and report:
  > `required_devices = N` exceeds the GPUs available on a single node (`gpus_per_node = M`). Pytest does not support multi-node execution.

Do not delegate to `trtllm-case-executor` if either limit is violated.

### Step 2: Run

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `test_cmd`: the user-provided command (run verbatim)
- `model_name`: `<class_name>` if provided, else `<MODEL_NAME>`
- `required_devices`: derived from markers (Step 1a), user-provided value, or `1`
- `device_type`: derived `device_type` if set (e.g. `B200`, `GB200`) — passed through to case-executor

### Step 3: Report

[Parse the results](#report-generation) and [write the report file](#write-report-file).

The report must include these fields:

| Field | Description |
|-------|-------------|
| `module_test_status` | **passed**, **failed**, or **error** |
| `module_test_results` | Per-test results with test ID, status, duration |
| `module_test_recommendations` | Fix recommendations for failures |

---

## Functionality Test Workflow

Smoke test using `pytest examples/pytorch/quickstart_advanced.py` with the pytorch backend.

### Step 1: Gather Parallelism Parameters

If tp/ep/dp values not provided, ask the user:
- `tp_size` (default `1`), `ep_size` (default `0`), `dp_size` (default `1`), `pp_size` (default `1`)

### Step 2: Run

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `test_cmd`: `pytest examples/pytorch/quickstart_advanced.py --model_dir <checkpoint_path> --backend pytorch --tp_size <N> --pp_size <N> [--ep_size <N>] [--dp_size <N>] -v` (include `--ep_size` only when > 0, `--dp_size` only when > 1)
- `model_name`: `<MODEL_NAME>`
- `required_devices`: `tp_size * pp_size * dp_size`

### Step 3: Report

[Parse the results](#report-generation). Generate a markdown report containing: overall status table (passed/failed/skipped counts, duration), test command, per-test results, failure details with tracebacks, and fix recommendations. [Write the report file](#write-report-file).

---

## Benchmark Test Workflow

Performance benchmarking. Three paths, selected in this order:

- **Pre-built perf-sanity** — `test_cmd` is already a perf-sanity pytest command (classified via Step 1 of Classify Test Scope). Skip YAML generation; delegate directly to `trtllm-case-executor` (Step 0 below).
- **Single-node → `trtllm-bench`** (default). Used whenever the requested world size fits on one node and no `benchmark_config_yaml` / `benchmark_model_name` is provided.
- **Multi-node → perf-sanity config YAML** (fallback). Used when the world size exceeds one node, or when the user explicitly provides `benchmark_config_yaml` / `benchmark_model_name`. `trtllm-bench` itself is treated as single-node only — multi-node invocations (via `trtllm-llmapi-launch`) are intentionally not generated here.

### Step 0: Pre-Built Perf-Sanity Path (when routed from Step 1 classification)

Applies only when `test_cmd` was provided and identified as a perf-sanity test in Step 1. No YAML generation or `trtllm-bench` invocation is performed.

**Step 0a: Derive `device_type` from test ID**

If `device_type` is not provided by the user, extract it from the bracketed test ID in `test_cmd`. Scan the test ID tokens (split on `-`) for known device strings and map them:

| Token in test ID | `device_type` |
|-----------------|---------------|
| `gb200` | `GB200` |
| `gb300` | `GB300` |
| `b200` | `B200` |
| `b300` | `B300` |
| `h100` | `H100` |
| `h200` | `H200` |
| `a100` | `A100` |

If no known token is found, leave `device_type` unset (unconstrained).

**Step 0b: Derive `model_name` from test ID**

If `model_name` is not provided, extract it from the test ID: take the token immediately after the device token (e.g., `gb200_deepseek-v32` → `deepseek_v32`). Convert hyphens to underscores.

**Step 0c: Delegate to `trtllm-case-executor`**

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `test_cmd`: the pre-built pytest command (verbatim)
- `model_name`: derived in Step 0b or user-provided
- `device_type`: derived in Step 0a or user-provided (omit if unset)
- `perf_config_yaml`: `perf_config_yaml` input (omit if not provided)
- `required_devices`: `1` (the pytest orchestrator process; actual compute is allocated by the test internally)

Skip Steps 1–3b entirely. Proceed to Step 4 (Report) once the executor returns.

### Step 1: Gather Parameters

Required: `model_name` (HuggingFace id, e.g. `meta-llama/Llama-3.1-8B-Instruct`), `checkpoint_path`, `tp_size`, `pp_size`.

Optional: `ep_size`, `mtp_layers` (comma-separated MTP depths to sweep, e.g. `1,2,3`), `backend` (default `pytorch`), `bench_subcommand` (default `throughput`), `dataset_path`, `num_requests`, `input_mean`, `output_mean`, `concurrency`, `max_batch_size`, `max_num_tokens`, `extra_llm_api_options_yaml`.

Parameters are already merged from `config_file` by Step 0. If any required parameter is still missing, ask the user.

### Step 2: Select Path

Compute `required_devices = tp_size * pp_size` (ignore `dp_size` — `trtllm-bench` has no `--dp` flag; if the user sets `dp_size > 1` treat the run as multi-node).

Query the target environment's `gpus_per_node` from `exec-env-check` (or the cluster config when `slurm_cluster` is set).

- If `benchmark_config_yaml` **or** `benchmark_model_name` is provided → **perf-sanity fallback** (Step 3b).
- Else if `required_devices > gpus_per_node` (or `dp_size > 1`) → **perf-sanity fallback** (Step 3b). Tell the user: "trtllm-bench is single-node only; falling back to perf-sanity for multi-node."
- Else → **trtllm-bench** (Step 3a).

### Step 3a: Single-Node trtllm-bench

Build the `bench_cmd` using `scripts/build_test_command.py`:

```bash
python3 <skill_dir>/scripts/build_test_command.py --type bench \
  --model <model_name> \
  --model-path <checkpoint_path> \
  --subcommand <bench_subcommand> \
  --backend <backend> \
  --tp <tp_size> [--pp <pp_size>] [--ep <ep_size>] \
  [--num-requests <N>] [--input-mean <N>] [--input-stdev <N>] \
  [--output-mean <N>] [--output-stdev <N>] \
  [--concurrency <N>] [--max-batch-size <N>] [--max-num-tokens <N>] \
  [--workspace <workspace>] \
  [--kv-cache-free-gpu-mem-fraction <kv_cache_free_gpu_mem_fraction>] \
  [--max-seq-len <max_seq_len>] [--beam-width <beam_width>] [--warmup <warmup>] \
  [--streaming] [--no-chunked-context] [--scheduler-policy <scheduler_policy>] \
  [--cluster-size <cluster_size>] [--iteration-log <iteration_log>] \
  [--config <extra_llm_api_options_yaml>] \
  --work-dir <WORK_DIR>
```

The script prints the fully-constructed `prepare-dataset && <subcommand>` command string.
Defaults: `--subcommand throughput`, `--backend pytorch`, `--tp 1`, `--num-requests 100`, `--input-mean 128`, `--output-mean 128`.
Dataset output: `<WORK_DIR>/bench_dataset.txt`; report: `<WORK_DIR>/bench_report.json`.
Note: `--streaming`, `--no-chunked-context`, and `--scheduler-policy` are throughput-subcommand-only flags and are silently ignored for latency.

Note: `trtllm-bench` uses `--tp` / `--pp` / `--ep` (not `--tp_size` / `--pp_size` / `--ep_size`), and takes the subcommand positionally (`throughput` / `latency`) after the global `--model` / `--model_path`.

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `bench_cmd`: the output of `build_test_command.py --type bench` above. The case-executor's Bench workflow auto-parses `--tp`/`--pp`/`--ep` from the run step for device count and enforces the single-node check.
- `model_name`, `checkpoint_path`, `device_type`
- `required_devices`: `tp_size * pp_size` (must be `<= gpus_per_node`, otherwise Step 2 already routed to the perf-sanity fallback)

### Step 3b: Multi-Node Perf-Sanity Fallback (Benchmark)

**Step 3b-1: Generate the perf-sanity config YAML**

Use `<skill_dir>/scripts/generate_benchmark_config.py`:

**MTP sweep:** When `mtp_layers` is provided (e.g. `1,2,3,4,5,6`), generate one config file per MTP value and run Steps 3b-2 through 3b-4 separately for each. Name each file `<benchmark_model_name>_mtp<N>.yaml` so the configs don't overwrite each other:

```bash
# For each MTP value in mtp_layers:
for mtp in <mtp_layers_values>; do
  python3 <skill_dir>/scripts/generate_benchmark_config.py \
    --config-type aggr \
    --model-name <benchmark_model_name> \
    [--from-config <base_config_yaml>] \
    --tp <tp_size> --ep <ep_size> --pp <pp_size> \
    --mtp-layers $mtp \
    [--max-batch-size <N>] [--max-num-tokens <N>] \
    [--concurrency <N>] [--iterations <N>] \
    [--input-length <N>] [--output-length <N>] \
    [--kv-cache-dtype fp8|fp16] \
    --output <WORK_DIR>/<benchmark_model_name>_mtp${mtp}.yaml
done

# Without MTP sweep (single config):
python3 <skill_dir>/scripts/generate_benchmark_config.py \
  --config-type aggr \
  --model-name <benchmark_model_name> \
  [--from-config <base_config_yaml>] \
  --tp <tp_size> --ep <ep_size> --pp <pp_size> \
  [--max-batch-size <N>] [--max-num-tokens <N>] \
  --output <WORK_DIR>/<benchmark_model_name>.yaml

# Disaggregated (separate ctx/gen workers):
python3 <skill_dir>/scripts/generate_benchmark_config.py \
  --model-name <benchmark_model_name> \
  [--from-config <base_config_yaml>] \
  --gen-tp <N> --ctx-tp <N> \
  [--benchmark-mode e2e|gen_only|ctx_only] \
  --output <WORK_DIR>/<benchmark_model_name>.yaml
```

**Step 3b-2: Identify available server config names** (aggregated only)

Run for each generated YAML:

```bash
python3 <skill_dir>/scripts/build_test_command.py --type perf_sanity \
  --config-file <WORK_DIR>/<benchmark_model_name>[_mtp<N>].yaml \
  --list-test-names
```

The output lists all `server_configs[].name` values in the YAML, e.g.:
```
gpt_oss_tp4_ep4_pp2_mtp1
gpt_oss_tp4_ep4_pp2_mtp2
```

**Step 3b-3: Build the perf_sanity test command**

Run for each YAML / server config name:

```bash
# Aggregated — one command per server config entry to test:
python3 <skill_dir>/scripts/build_test_command.py --type perf_sanity \
  --serving-type aggr \
  --config-file <WORK_DIR>/<benchmark_model_name>[_mtp<N>].yaml \
  --test-name <server_config_name> \
  --repo-root <repo_root>

# Disaggregated:
python3 <skill_dir>/scripts/build_test_command.py --type perf_sanity \
  --serving-type disagg \
  --config-file <WORK_DIR>/<benchmark_model_name>.yaml \
  --benchmark-mode <e2e|gen_only|ctx_only> \
  --repo-root <repo_root>
```

**Step 3b-4: Delegate to `trtllm-case-executor`**

For each test command (one per MTP value when `mtp_layers` is set), [delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `test_cmd`: the pytest command from Step 3b-3
- `perf_config_yaml`: `<WORK_DIR>/<benchmark_model_name>[_mtp<N>].yaml` (the generated YAML, so the executor uses it instead of auto-deriving the path from the test ID)
- `model_name`, `device_type`, `benchmark_model_name`

When running a multi-value MTP sweep, submit all jobs before monitoring any of them so they run concurrently on the cluster.

### Step 4: Report

Parse `<WORK_DIR>/bench_report.json` (single-node path) or the perf-sanity summary (fallback). Report: subcommand, backend, parallelism (`tp`/`pp`/`ep`), throughput / latency metrics, concurrency, dataset path and generation params (`num_requests`, `input_mean`, `output_mean`), work directory, path used (`trtllm-bench` vs perf-sanity), and any errors. [Write the report file](#write-report-file).

---

## Evaluation Test Workflow

Accuracy evaluation. Two paths, selected in this order:

- **Single-node → `trtllm-eval`** (default). Used whenever the requested world size fits on one node and no `benchmark_config_yaml` / `benchmark_model_name` is provided.
- **Multi-node → perf-sanity config YAML** (fallback). Used when the world size exceeds one node, or when the user explicitly provides `benchmark_config_yaml` / `benchmark_model_name`. The perf-sanity run enables accuracy mode via `accuracy.enable_accuracy_test: true`. `trtllm-eval` itself is treated as single-node only.

### Step 1: Gather Parameters

Required: `checkpoint_path`, `eval_tasks`, `tp_size` (default `1`), `pp_size` (default `1`).

Optional: `ep_size`, `dataset_path`, `device_type`. Note: `trtllm-eval` has no `--dp_size` flag — `dp_size` is ignored for this workflow.

For the multi-node (perf-sanity) path, additional inputs apply: `benchmark_config_yaml`, `benchmark_model_name`, and any perf-sanity overrides (see the Benchmark workflow Step 3b inputs).

Parameters are already merged from `config_file` by Step 0. If any required parameter is still missing, ask the user.

### Step 2: Select Path

Compute `required_devices = tp_size * pp_size`.

Query the target environment's `gpus_per_node` from `exec-env-check` (or the cluster config when `slurm_cluster` is set).

- If `benchmark_config_yaml` **or** `benchmark_model_name` is provided → **perf-sanity fallback** (Step 3b).
- Else if `required_devices > gpus_per_node` → **perf-sanity fallback** (Step 3b). Tell the user: "trtllm-eval is single-node only; falling back to perf-sanity for multi-node accuracy."
- Else → **trtllm-eval** (Step 3a).

### Step 3a: Single-Node trtllm-eval

Build the `eval_cmd` using `scripts/build_test_command.py`:

```bash
python3 <skill_dir>/scripts/build_test_command.py --type eval \
  --model <model_name_or_checkpoint_path> [--model-path <checkpoint_path>] \
  --tp-size <tp_size> [--pp-size <pp_size>] [--ep-size <ep_size>] \
  [--tokenizer <eval_tokenizer>] \
  [--trust-remote-code] [--disable-kv-cache-reuse] \
  [--kv-cache-free-gpu-memory-fraction <eval_kv_cache_free_gpu_memory_fraction>] \
  [--max-batch-size <N>] [--max-num-tokens <N>] [--max-seq-len <N>] \
  [--config <extra_llm_api_options_yaml>] \
  [--num-samples <eval_num_samples>] \
  [--apply-chat-template] [--chat-template-kwargs '<eval_chat_template_kwargs>'] \
  [--system-prompt '<eval_system_prompt>'] \
  [--max-input-length <eval_max_input_length>] [--max-output-length <eval_max_output_length>] \
  [--log-samples] [--output-path <eval_output_path>] \
  [--check-accuracy] [--accuracy-threshold <eval_accuracy_threshold>] [--num-fewshot <eval_num_fewshot>] \
  --tasks <eval_tasks>
```

The script prints one `trtllm-eval` command per task, joined with `&&` — tasks are Click subcommands and cannot be chained in a single invocation. `--tasks` accepts a comma-separated list (e.g. `gsm8k,mmlu`).

Model-path resolution: `trtllm-eval`'s `--model` is also used to load the tokenizer, so it must be a real local path or a valid HuggingFace ID — a short label like `gpt_oss_120b` will fail tokenizer lookup. Pass either (a) the local `<checkpoint_path>` as `--model`, or (b) the short name as `--model` and the local path as `--model-path` — when `--model-path` is set, the builder uses it as the `--model` value in the rendered `trtllm-eval` command so the tokenizer resolves locally.

Option placement rules applied by the script:
- Global options (`--tp_size`, `--config`, `--kv_cache_free_gpu_memory_fraction`, etc.) are placed before the task name.
- Common per-task options (`--num_samples`, `--apply_chat_template`, `--max_output_length`, etc.) are placed after each task name.
- `--check-accuracy`, `--accuracy-threshold`, and `--num-fewshot` are mmlu-specific — appended only to the `mmlu` task invocation, not to any other task.

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `eval_cmd`: the output of `build_test_command.py --type eval` above.
- `model_name`, `checkpoint_path`, `device_type`, plus additional eval parameters.
- `required_devices`: `tp_size * pp_size` (must be `<= gpus_per_node`, otherwise Step 2 already routed to the perf-sanity fallback).

### Step 3b: Multi-Node Perf-Sanity Fallback (Evaluation)

**Step 3b-1: Generate the perf-sanity config YAML with accuracy enabled**

Use `<skill_dir>/scripts/generate_benchmark_config.py`:
The key difference from the benchmark path is `--enable-accuracy-test` and `--accuracy-task`:

```bash
# Aggregated accuracy evaluation:
python3 <skill_dir>/scripts/generate_benchmark_config.py \
  --config-type aggr \
  --model-name <benchmark_model_name> \
  [--from-config <base_config_yaml>] \
  --tp <tp_size> --ep <ep_size> --pp <pp_size> \
  --enable-accuracy-test \
  --accuracy-task <task_name> \
  [--accuracy-model <lm_eval_model_name>] \
  [--accuracy-model-args-extra <extra_args>] \
  [--accuracy-trust-remote-code] \
  --output <WORK_DIR>/<benchmark_model_name>.yaml

# Disaggregated accuracy evaluation:
python3 <skill_dir>/scripts/generate_benchmark_config.py \
  --model-name <benchmark_model_name> \
  --gen-tp <N> --ctx-tp <N> \
  --enable-accuracy-test \
  --accuracy-task <task_name> \
  --output <WORK_DIR>/<benchmark_model_name>.yaml
```

**Step 3b-2 and 3b-3** are identical to the Benchmark fallback — list server config names, then build the perf_sanity test command:

```bash
# List available names:
python3 <skill_dir>/scripts/build_test_command.py --type perf_sanity \
  --config-file <WORK_DIR>/<benchmark_model_name>.yaml \
  --list-test-names

# Build the test command:
python3 <skill_dir>/scripts/build_test_command.py --type perf_sanity \
  --serving-type aggr \
  --config-file <WORK_DIR>/<benchmark_model_name>.yaml \
  --test-name <server_config_name> \
  --repo-root <repo_root>
```

**Step 3b-4: Delegate to `trtllm-case-executor`**

[Delegate to `trtllm-case-executor`](#delegating-to-trtllm-case-executor) with:

- `test_cmd`: the pytest command from Step 3b-3
- `perf_config_yaml`: `<WORK_DIR>/<benchmark_model_name>.yaml` (accuracy config with `enable_accuracy_test: true`)
- `model_name`, `device_type`, `benchmark_model_name`

### Step 4: Report

Parse the single-node eval log (Step 3a) or the perf-sanity accuracy summary (Step 3b). Report: accuracy scores per task, pass/fail status, parallelism (`tp`/`pp`/`ep`), configuration, work directory, path used (`trtllm-eval` vs perf-sanity), and any errors. [Write the report file](#write-report-file).

---

## Common Procedures

### MODEL_NAME Derivation

If `model_name` is not provided, derive from `checkpoint_path`:
- HF model name `meta-llama/Llama-3.1-8B-Instruct` → `llama_3_1_8b_instruct`
- Checkpoint directory `/models/Llama-3.1-8B-Instruct/` → `llama_3_1_8b_instruct`

### Delegating to `trtllm-case-executor`

All test execution is delegated to the `trtllm-case-executor` skill.

```
Skill(skill="trtllm-agent-toolkit:trtllm-case-executor", args="""
- <key>: <value>
- <key>: <value>
...
""")
```

The skill runs to completion (classifying the workflow, detecting the environment, building the script, dispatching to its own executor subagents) and returns the full result — status, exit code, log file paths, and any error output.

### Monitoring Test Results

While `trtllm-case-executor` runs the test, monitor for completion before advancing to report generation.

**Slurm jobs (`local_slurm`, `remote_slurm`):** poll the queue for the jobs `trtllm-case-executor` submitted (it returns `job_id`s in its result; for remote runs, poll over SSH). Stop monitoring as soon as **all of those jobs have left the queue** — that means every job has finished and exited (completed, failed, cancelled, timed out, or preempted). Do not wait, re-poll, or resubmit after the queue shows none of them.

```bash
# Local Slurm — check every 30s until no tracked jobs remain
while squeue --jobs=<job_id_csv> --noheader --states=all 2>/dev/null | grep -q .; do
    sleep 30
done
# All jobs are gone → stop monitoring, collect final state via sacct, proceed to Report
sacct -j <job_id_csv> --format=JobID,State,ExitCode,Elapsed --parsable2
```

For remote Slurm, run the same loop over SSH using the executor's `ssh_host` / `slurm_user`. If `squeue` returns a transient error (e.g. SSH hiccup), retry the poll on the next tick — do not assume the jobs are gone.

**Local direct / local Docker:** the executor (or this skill, for `local_direct`) already blocks until the process exits; no separate queue polling is required. Use the exit code it reports.

Once monitoring stops, pass the final log/exit code to [Report Generation](#report-generation).

### Report Generation

For pytest-based workflows (Core Module, Modeling Module, Functionality), parse test output with:

```bash
python3 <skill_dir>/scripts/generate_report.py \
  --test-type <test_type> \
  --output-file <log_file> \
  --exit-code <exit_code> \
  --format markdown
```

Map the active workflow to `--test-type`:

| Workflow | `--test-type` value |
|----------|---------------------|
| Module test, Functionality test | `functional` |
| Benchmark test | `benchmark` |
| Feature-matrix test | `feature-matrix` |

For all workflows, analyze failures and produce fix recommendations. Reference `references/trtllm_test_fix_recommendations.md` for common error patterns. Each recommendation should include: error type, root cause, concrete fix actions, and files to modify (`<file_path>:<line_number>`).

### Write Report File

After generating the report content, write it to the output path:
- If `report_file` is provided → `mkdir -p $(dirname <report_file>)` then write to `<report_file>`
- Otherwise → write to `./<MODEL_NAME>-auto-test-report.md`

Confirm to the user: `Test report written to: <path>`

---

## Resources

| File | Purpose |
|------|---------|
| `scripts/extract_test_markers.py` | Extract `required_devices` and `device_type` from pytest markers in a test file. Pass `--test-cmd "<pytest_cmd>"` or `--test-file <path> [--class-name X] [--test-name Y]`. Outputs JSON with `required_devices`, `device_type`, and `sources`. |
| `scripts/build_test_command.py` | Build CLI commands for `trtllm-bench`, `trtllm-eval`, and `test_perf_sanity`. Accepts `--type {bench,eval,perf_sanity}` plus all relevant flags; prints the fully-constructed command string to stdout. Non-perf pytest, python, and custom commands are passed directly to `trtllm-case-executor` without using this script. Run `python3 build_test_command.py --help` for all options. |
| `scripts/generate_benchmark_config.py` | Generates perf-sanity config YAMLs for aggregated and disaggregated benchmark/evaluation runs. Pass `--from-config` for a base config or `--config-type aggr/disagg` to build from a built-in template. |
| `scripts/parse_config.py` | Parses and validates a YAML config file; prints recognised parameters as JSON |
| `scripts/generate_report.py` | Parses pytest output into structured markdown |
| `references/test_config_template.yaml` | Full config file template with all supported fields and inline documentation |
| `references/agg_config_template.yaml` | Base aggregated benchmark config template (used with `generate_benchmark_config.py --from-config`) |
| `references/benchmark_config_template.yaml` | Minimal benchmark config template for simple single-server runs |
| `references/disagg_config_template.yaml` | Base disaggregated benchmark config template |
| `references/trtllm_test_fix_recommendations.md` | Common error patterns and fix recommendations |

## Notes

- Tests requiring multiple GPUs (TP > 1) are skipped when only 1 GPU is available — expected, not a failure.
- This skill reads definition files only for root-cause analysis — it does not modify source files. All recommendations are advisory.
