---
name: trtllm-test-script-builder
description: >-
  Build Slurm scripts or Docker commands for TensorRT-LLM workloads. Resolves
  all parameters (docker image, mounts, parallelism,
  MPI mode), generates the complete script from Category templates, and writes
  both the script and a job_spec.json manifest to the work directory.
license: Apache-2.0
---

# TensorRT-LLM Script Builder

Resolve parameters and generate execution scripts for all workflow types.

## Input (from orchestrator prompt)

The orchestrator passes these fields. Not all fields are present for every workflow.

| Field | Description | Required |
|-------|-------------|----------|
| `workflow_type` | `pytest`, `eval`, `bench`, `custom`, or `perf_sanity` | Always |
| `work_dir` | Absolute path to local work directory (already created) | Always |
| `skill_dir` | Absolute path to the test-script-builder skill directory (`.claude/skills/trtllm-test-script-builder`) | Always |
| `repo_root` | Absolute path to the TensorRT-LLM repo root | Always |
| `test_cmd` | Pytest command string | pytest |
| `eval_cmd` | `trtllm-eval` command string, or `evaluation`, `accuracy`, `score`, `dataset` | eval |
| `bench_cmd` | `trtllm-bench` command (may chain `prepare-dataset` and `throughput`/`latency` with `&&`). Auto-parses `--tp`/`--pp`/`--ep` from the last `trtllm-bench` invocation. | bench |
| `custom_cmd` | Custom command string, `test command`, non-pytest command | custom |
| `custom_script` | Path to custom script file | custom (optional) |
| `custom_env` | Space-separated `KEY=VALUE` pairs | custom (optional) |
| `perf_config_yaml` | Path to pre-built perf-sanity or benchmark config YAML | perf_sanity (required) |
| `perf_test_name` | Server config name for aggregated mode | pytest (optional) |
| `model_name` | Short model name (auto-derived if not provided) | Optional |
| `required_devices` | Number of GPUs needed (auto-derived if not provided) | Optional |
| `node_count` | Number of Slurm nodes (auto-derived if not provided) | Optional |
| `partition` | Slurm partition. **For slurm scenarios, this MUST be the value resolved by `trtllm-case-executor` Step 2.5 against the cluster (validated to exist in `slurm_env.partitions[].name`).** Do not detect or default it here. | Required (slurm) |
| `account` | Slurm account. **For slurm scenarios, this MUST be the value resolved by `trtllm-case-executor` Step 2.5 against the cluster (validated to exist in `slurm_env.accounts[]`).** Do not detect or default it here. | Required (slurm) |
| `slurm_env_file` | Absolute path to the `slurm_env.json` file written by `trtllm-case-executor` Step 2.5 (canonically `<work_dir>/slurm_env.json`). Schema: `{user, default_account, accounts[], default_partition, partitions[{name,state,time_limit,nodes,arch,gres,gpus_per_node,requires_gres}], pmix:{available[],preferred}, errors[]}`. The script builder loads it once in Step 1, uses it to select the Docker image (arch), drive `--gres` / `--mpi=` flags, and embeds the parsed object into `job_spec.json` for downstream consumers. | Required (slurm) |
| `time_limit` | Slurm time limit (default: `02:00:00`) | Optional |
| `llm_models_root` | Models root **inside the container**. Resolution order: caller input → value already resolved by `trtllm-case-executor` and forwarded in `job_spec.json` → delegate to `internal-env-info` for the `default_llm_models_root` key when that skill is installed → otherwise **ask the user**. The script builder does not invent a default. | Optional |
| `checkpoint_path` | Host path to model checkpoint | Optional |
| `models_path` | Host path to LLM models directory | Optional |
| `device_type` | Required GPU/device hardware type | Optional |
| `execution_scenario` | `local_docker`, `local_slurm`, or `remote_slurm` | Always |
| `persistent_mode` | `true` or `false`. **Only meaningful for `execution_scenario=local_slurm`** — when true, add persistent-mode fields to `job_spec.json` so `exec-local-slurm` can reuse a persistent allocation. Default: `true` for `local_slurm`. Ignored (no-op) for `local_docker` and `local_direct`. For `remote_slurm`, this field is **not used** — `exec-remote-slurm` owns its own persistence model (tmux-held `salloc` + named container; see remote-executor Recipe 6) and does not consume `persistent_mode` from `job_spec.json`. | Optional |

> Per-partition hardware (`arch`, `gres`, `gpus_per_node`, `requires_gres`) lives under `slurm_env.partitions[]`. The chosen `partition` / `account` / `container_image` are passed as top-level inputs.

## Procedure

### Step 1: Resolve Docker Image

1. If the top-level `container_image` input is set (case-executor resolved it as the single source of truth in its Step 3), use it directly — skip the remaining steps.
2. Search `<repo_root>` for an image-tags properties file (e.g., under `jenkins/` or a similar CI config directory). Extract the relevant key based on arch (see step 3).
3. Determine compute node architecture:

   **Local Docker** (`execution_scenario == local_docker`):
   - Run `uname -m` via Bash tool on the local machine.

   **Local Slurm / Remote Slurm**:
   - Load the SLURM environment from the file at `slurm_env_file` (e.g. `python3 -c 'import json,sys; print(json.load(open(sys.argv[1])))' "$slurm_env_file"` or `jq` — pick whatever's available). The path is canonically `<work_dir>/slurm_env.json`, written by `trtllm-case-executor` Step 2.5. Do **not** invoke any SLURM detection command (`sinfo`, `scontrol`, `srun --mpi=list`) — case-executor already ran the one-pass probe. If `slurm_env_file` is missing or unreadable for a slurm scenario, stop and report that case-executor did not run Step 2.5 before invoking the script builder. The loaded shape is:
     ```json
     {
       "partitions": [
         {"name": "<part>", "state": "up|down",
          "time_limit": "...", "nodes": "...",
          "arch": "x86_64|aarch64|null",
          "gres": "gpu:8|(null)", "gpus_per_node": 8|null,
          "requires_gres": true|false}
       ],
       "pmix": {"available": ["pmix","pmix_v3","pmix_v4","pmix_v5"], "preferred": "pmix_v5"},
       "errors": ["..."]
     }
     ```
   - Look up the entry where `name == partition` (the top-level `partition` input, resolved by case-executor Step 2.5); use its `arch` for the image selection below and pass `gres`, `gpus_per_node`, `requires_gres`, and the top-level `pmix.preferred` through to Step 3 / `job_spec.json` (consumed by the executors for the `--mpi=` flag).
   - If `slurm_env.errors[]` indicates `"SLURM client tools not found"` (the case-executor probe ran on a host without `sinfo`/`scontrol`) or the entry's `arch` is `null`, fall back to the `internal-env-info` skill for arch and leave `pmix.preferred` unset so Step 3 falls back to the bare `pmix` plugin name. If `internal-env-info` is also not installed, skip the lookup silently and use `uname -m` for `local_slurm` or default to `x86_64` for `remote_slurm` when otherwise unknowable. Do not report the missing skill as an error.

   **Arch → properties key mapping** (apply to the lookup result):

   | `Arch=` value | Properties key |
   |---------------|----------------|
   | `aarch64` | `LLM_SBSA_DOCKER_IMAGE` |
   | `x86_64` | `LLM_DOCKER_IMAGE` |

   Use the arch for the chosen `partition` (the target partition for this job) to select the Docker image. If `slurm_env` is missing or its entry's `arch` is null, fall back as described above.

   **GRES interpretation** — applied in Step 3:
   - `gres` starts with `gpu:` (e.g., `gpu:4`, `gpu:4(IDX:0-3)`) → partition requires `--gres`
   - `gres` is `(null)` → partition does not use GRES scheduling; omit `--gres` from the script

### Step 2: Resolve Container Mounts

Build comma-separated `<CONTAINER_MOUNTS>`:

**Resolve models host path** — evaluate rules in priority order and stop at the first match. All mount entries below use `<llm_models_root>` as the container-side target — the resolved input from Step 2.5 of `trtllm-case-executor` (see the `llm_models_root` row in [Input](#input-from-orchestrator-prompt)).

1. **`checkpoint_path` provided** → use its **parent directory** as the models host path. Construct one mount entry: `<parent_of_checkpoint_path>:<llm_models_root>`. This takes priority over `slurm_env.mounts`.
2. **`models_path` provided** → use that as the models host path. Construct: `<models_path>:<llm_models_root>`.
3. **`slurm_env.mounts` available** → use all entries verbatim (pre-configured for the cluster by `trtllm-case-executor` Step 2.5; includes the models directory and any symlink targets). Do **not** construct a separate models mount entry.
4. **Fallback** → resolve `<default_models_repo_host>` in this order: `env_check.default_models_repo` (forwarded by `trtllm-case-executor`) → `internal-env-info`'s `default_models_repo_host` key if the skill is installed → otherwise **stop and ask the user** for the host path. Do not hard-code a fallback in this skill. Once resolved, construct: `<default_models_repo_host>:<llm_models_root>`.

**Home directory**: For local, `$HOME`. For remote, resolve from `slurm_env.remote_cwd`'s parent (the user's home on the cluster) or `ssh <host> 'echo $HOME'`.

**Project path**: For local, `<repo_root>`. For remote, `remote_repo_path` (read from `job_spec.json`; constructed by `trtllm-case-executor` as `<default_user_root_dir>/<repo_dir_name>`, where `<default_user_root_dir>` is resolved by case-executor in this order: `internal-env-info` per cluster → fall back to `internal-env-info`'s `default_user_root_dir_example` key when only documentation-level guidance is needed → otherwise case-executor asks the user. The script builder consumes the resolved `remote_repo_path` verbatim and is never hard-coded in this skill). Do **not** use `slurm_env.remote_cwd` — that is the user's root, not the cloned repo path.

**Remote work_dir**: For remote scenarios, `remote_work_dir` is read directly from `job_spec.json` — the case-executor is the single source of truth for that path (`<remote_repo_path>/work_dirs/<basename(local work_dir)>`). The script builder consumes it verbatim and never recomputes.

**Additional mounts** (custom workflow only): If `custom_script` parent directory is not covered by existing mounts, add it.

Assemble: `<home>:<home>,<project>:<project>,<models_mount or slurm_env.mounts entries>[,<extra>:<extra>]`

### Step 3: Resolve Parallelism & Slurm Parameters

**By workflow type:**

**Pytest**:
- Detect sub-type: perf-sanity vs non-perf-sanity (check if `test_cmd` contains `perf-sanity` or a perf sanity test file path, or if `perf_config_yaml` is provided)
- Non-perf: inspect test file for `skip_less_device(N)`, `@pytest.mark.parametrize` GPU markers → `required_devices`. If not derivable, use `required_devices` input (default: 1).
- Non-perf: pytest is always single-node. If `required_devices` > `slurm_env.partitions[name=<partition>].gpus_per_node` (or `available_gpus` from env-check for non-slurm scenarios), **stop and report an error** — pytest does not support multi-node execution.
- Non-perf: **Hard limit — `required_devices` must be `<= 8`.** If `required_devices > 8`, stop and report an error: `required_devices = N exceeds the maximum of 8 GPUs for pytest.`
- Non-perf: **Hard limit — `node_count` must be `<= 1`.** Pytest is always single-node; `node_count` is always set to `1`. This is a redundant safety check — if any derivation produces `node_count > 1`, stop and report an error.
- Perf-sanity: all Slurm parameters (nodes, ntasks-per-node, gpus_per_node) are derived by `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` exclusively from the config YAML hardware section — do not use `slurm_env.partitions[].gpus_per_node`, user input, or any default value

**Eval**:
- Parse from `eval_cmd`: `--tp_size N` (default: 1), `--pp_size N` (default: 1), `--gpus_per_node N` (overrides cluster config)
- `world_size` = `tp_size * pp_size`; `required_devices` = `world_size`
- **Single-node only**: `trtllm-eval` does not support multi-node execution. If `world_size > gpus_per_node` (i.e., the job would need more than one node), **do not generate a Category 5 script** — fall back to the perf-sanity path (Category 4) by generating a benchmark config with `accuracy.enable_accuracy_test: true` and submitting via `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py`. Same fallback applies to disaggregated accuracy testing.
- Auto-derive `model_name` from last component of `--model` arg (e.g., `Llama-3.1-8B-Instruct` → `llama_3_1_8b_instruct`)
- Eval task subcommand (e.g., `gsm8k`, `mmlu`) passes through as-is in `eval_cmd`
- **Model path resolution**: If `--model` is a HuggingFace model ID (contains `/` but does NOT start with `/`), resolve it to a local path under `<llm_models_root>` (the resolved input — see the `llm_models_root` row in [Input](#input-from-orchestrator-prompt) for the priority chain). Search subdirectories (max depth 3) for a directory matching the model name component (case-insensitive). If found, rewrite `--model` in `eval_cmd` to the local absolute path. If not found, keep the original and warn that HF download will be attempted.

**Bench** (trtllm-bench):
- Parse from the last `trtllm-bench` invocation in `bench_cmd`: `--tp N` (default: 1), `--pp N` (default: 1), `--ep N` (default: 0)
- `world_size = tp * pp`; `required_devices = world_size`
- Split `bench_cmd` on `&&` into `PREP_CMD` (the `prepare-dataset` step) and `BENCH_RUN_CMD` (the `throughput`/`latency` step). If no `&&` is present (single command only), set `PREP_CMD` to empty and `BENCH_RUN_CMD` to the full command.
- Single-node only: if `world_size > gpus_per_node`, stop and report an error — multi-node `trtllm-bench` is not supported.
- Auto-derive `model_name` from the `--model` or `--model_path` argument
- `NTASKS = world_size` (used for the benchmark run srun; the `prepare-dataset` srun always overrides to `--ntasks=1`)

**Benchmark**:
- Derived from config YAML hardware section by `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` — skip manual derivation

**Custom**:
- Use `required_devices` input (default: 1)
- **Hard limit — `required_devices` must be `<= 8`.** If `required_devices > 8`, stop and report an error: `required_devices = N exceeds the maximum of 8 GPUs for custom commands.`
- **Hard limit — `node_count` must be `<= 1`.** Compute `node_count = ceil(required_devices / gpus_per_node)`. If `node_count > 1`, stop and report an error: `required_devices = N requires N nodes (gpus_per_node = M), but custom commands are limited to 1 node. Reduce required_devices to at most M.`

**Common Slurm parameters:**
- `node_count` = 1 for pytest (non-perf only, always single-node); derived from config YAML by `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` for perf-sanity and benchmark; 1 for eval and bench (single-node only); `ceil(required_devices / gpus_per_node)` for custom, where `gpus_per_node` = `slurm_env.partitions[name=<partition>].gpus_per_node` for slurm scenarios, or `env_check.gpus_per_node` for local_docker; if neither resolves, stop and ask the user.
- `NTASKS` = 1 for pytest; `world_size` for eval and bench; `required_devices` for custom
- `NTASKS_PER_NODE` = 1 for pytest; `NTASKS / node_count` for other workflows
- `PARTITION` = the `partition` input — the value resolved and validated by `trtllm-case-executor` Step 2.5 against the cluster's `slurm_env.partitions[].name`. **Do not fall back to any default** for slurm scenarios; if `partition` is not provided, stop and report that case-executor did not run Step 2.5 before invoking the script builder.
- `ACCOUNT` = the `account` input — the value resolved and validated by `trtllm-case-executor` Step 2.5 against the cluster's `slurm_env.accounts[]`. **Do not fall back to any hard-coded default or to the `internal-env-info` skill's `default_account`** for slurm scenarios; if `account` is not provided, stop and report that case-executor did not run Step 2.5 before invoking the script builder.
- `TIME_LIMIT` = `time_limit` input or `02:00:00`
- `GRES_FLAG`: look up the `gres` value for `<PARTITION>` in the partition → {arch, gres} mapping from Step 1:
  - `gres` starts with `gpu:` → `GRES_FLAG = --gres=gpu:<GPUS_PER_NODE>`
  - `gres` is `(null)` or mapping unavailable → `GRES_FLAG = ` (empty string; omit the `#SBATCH --gres` line entirely from the generated script)

**MPI mode:**

Use `pmix.preferred` from `slurm_env` (case-executor Step 2.5's output, also embedded in `job_spec.json`) as the resolved `<MPI_PLUGIN>` (e.g., `pmix_v5`). If the field is unset (detection unavailable or fallback path taken), use the bare plugin name `pmix`. Never hardcode `pmix` — always read the resolved value.

- Pytest: empty (no `--mpi`), clear MPI env vars in script (always single-node)
- Eval: always `--mpi=pmix` on the eval run srun (including single-task). `trtllm-eval` imports `mpi4py` at module load, which auto-inits MPI; OpenMPI in the container requires srun to provide a PMI/PMIx runtime, otherwise `MPI_Init` aborts on a NULL communicator.
- Bench (trtllm-bench): `world_size > 1` → `--mpi=<MPI_PLUGIN>` on the benchmark run srun only; `world_size == 1` → empty; `prepare-dataset` srun never uses `--mpi`
- Custom single-node: empty; multi-node: `--mpi=<MPI_PLUGIN>`
- Benchmark/perf-sanity: handled by `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py`

### Step 4: Generate Script

Select the Category template from `<skill_dir>/references/trtllm_test_template.md` and generate:

> **Note on `local_direct`:** There is intentionally no Category for `execution_scenario == local_direct`. That scenario runs the command directly on the host without a container or Slurm wrapper, and `trtllm-case-executor` handles it inline (skipping this skill entirely). The Categories below cover only the three scenarios this skill supports: `local_docker`, `local_slurm`, and `remote_slurm`.

**Category 1 (Local Docker)** — `execution_scenario == local_docker`:
- Read the Category 1 template
- Substitute placeholders
- Write Docker command to `<work_dir>/docker_cmd.sh`

**Category 2 (Non-perf pytest)** — `workflow_type == pytest` and non-perf-sanity:

Run `build_slurm_script.py` to generate the script:
```bash
python3 <skill_dir>/scripts/build_slurm_script.py pytest \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes <NODES_NUM> --ntasks <NTASKS> --ntasks-per-node <NTASKS_PER_NODE> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --llm-models-root <LLM_MODELS_ROOT> \
  --test-cmd "<TEST_CMD>" \
  --output <work_dir>/<model_name>_auto_test.slurm
```
- Omit `--gres` entirely when `GRES_FLAG` is empty (script omits the `#SBATCH --gres` line when `--gres` is not provided or empty)
- For remote: use remote paths for `--project-path`, `--log-dir`, `--container-mounts`. Also pass `--local-log-dir <work_dir>` so the script's local `mkdir -p` runs against a path on the orchestrating host (`--log-dir` is cluster-only and not visible locally).

**Category 3 (Custom script)** — `workflow_type == custom`:

Resolve `CUSTOM_CMD` first: `bash <custom_script>` if `custom_script` was given, otherwise the literal `custom_cmd`. Then run:
```bash
python3 <skill_dir>/scripts/build_slurm_script.py custom \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes <NODES_NUM> --ntasks <NTASKS> --ntasks-per-node <NTASKS_PER_NODE> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --custom-cmd "<CUSTOM_CMD>" \
  --custom-env "<CUSTOM_ENV>" \
  --custom-workdir <CUSTOM_WORKDIR> \
  --run-script <skill_dir>/scripts/slurm_run_custom.sh \
  --output <work_dir>/<model_name>_custom.slurm
```
- For remote: also transfer `slurm_run_custom.sh` to the cluster; use remote path for `--run-script`. Also pass `--local-log-dir <work_dir>` so the local `mkdir -p` lands on the orchestrating host's filesystem.

**Category 4 (Perf-sanity / Benchmark)** — `workflow_type == perf_sanity` or pytest perf-sanity:

> Benchmark requires Slurm — local Docker is not supported.

The caller (trtllm-case-executor) always provides a pre-built `perf_config_yaml`. Config YAML generation is not done here. Use `perf_config_yaml` directly with the repo's `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py`.

Generate the launch script via the repo's `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py`. This script is self-contained — it resolves its own helper scripts (run/install/draft) internally, so no `--run-sh`/`--install-sh`/`--draft-launch-sh` flags are passed:
```bash
python3 <repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py \
  <--test-list "<pytest_test_string>" | --config-file <yaml> [--test-name <name>] [--benchmark-mode <mode>]> \
  --partition <PARTITION> --account <ACCOUNT> \
  --job-name <model_name>_<perf|benchmark> \
  --image <DOCKER_IMAGE> --mounts <CONTAINER_MOUNTS> \
  --work-dir <work_dir_or_remote_work_dir> \
  --llm-src <repo_root_or_remote_repo_path> \
  --llm-models-root <llm_models_root> \
  --launch-sh <work_dir>/slurm_launch.sh \
  --time <TIME_LIMIT>
```
- `--test-list`: perf-sanity with bracket string; `--config-file` + optional `--test-name`/`--benchmark-mode`: explicit config file
- Aggregated vs disaggregated execution is selected internally by `submit.py` based on the config.
- Use `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` (it ships with the cloned repo, so nothing extra is transferred); `--work-dir`, `--llm-src` use **remote paths**, `--launch-sh` uses a **local path**

**Category 6 (Bench)** — `workflow_type == bench`:

```bash
python3 <skill_dir>/scripts/build_slurm_script.py bench \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes 1 --ntasks <NTASKS> --ntasks-per-node <NTASKS> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --bench-cmd "<bench_cmd>" \
  --output <work_dir>/<model_name>_bench.slurm
```
- `<NTASKS>` = `world_size` (tp * pp); single node so `--ntasks` == `--ntasks-per-node`
- `bench_cmd` passed verbatim — the script splits on `&&` internally; if no `&&`, prep srun runs install only
- For remote_slurm: pass `--log-dir <remote_work_dir>` (baked into `#SBATCH --output/--error`) and `--local-log-dir <work_dir>` (used for the local `mkdir -p`).

**Category 5 (Eval)** — `workflow_type == eval`:

> **Single-node only.** `trtllm-eval` does not support multi-node execution. If Step 3 computes `node_count > 1` (i.e., `world_size > gpus_per_node`), **do not generate a Category 5 script** — fall back to Category 4 (perf-sanity) with `accuracy.enable_accuracy_test: true` in the benchmark config. Same fallback applies to disaggregated accuracy testing.

```bash
python3 <skill_dir>/scripts/build_slurm_script.py eval \
  --model-name <MODEL_NAME> \
  --log-dir <LOG_DIR> \
  --partition <PARTITION> --account <ACCOUNT> --time-limit <TIME_LIMIT> \
  --nodes 1 --ntasks <NTASKS> --ntasks-per-node <NTASKS> \
  [--gres <GRES>] \
  --docker-image <DOCKER_IMAGE> \
  --container-mounts <CONTAINER_MOUNTS> \
  --container-name <CONTAINER_NAME> \
  --project-path <PROJECT_PATH> \
  --eval-cmd "<EVAL_CMD>" \
  --output <work_dir>/<model_name>_eval.slurm
```
- `<NTASKS>` = `world_size` (tp_size * pp_size); MPI mode is always `pmix` on the eval run srun (including single-task) because `trtllm-eval` auto-inits MPI at module load
- The script installs any required eval dependencies and, when `world_size > 1`, prepends `trtllm-llmapi-launch` to the eval command
- For remote_slurm: pass `--log-dir <remote_work_dir>` and `--local-log-dir <work_dir>` (same pattern as bench).

### Step 5: Write job_spec.json

Write `<work_dir>/job_spec.json` with all resolved values:

```json
{
  "workflow_type": "pytest|eval|custom|benchmark",
  "model_name": "<MODEL_NAME>",
  "execution_scenario": "local_docker|local_slurm|remote_slurm",
  "docker_image": "<resolved image>",
  "script_path": "<work_dir>/<script_name>",
  "script_name": "<script_name>",
  "log_file_pattern": "<MODEL_NAME>_<type>_%j.out",
  "success_patterns": ["passed", "accuracy:"],
  "failure_patterns": ["FAILED", "Error", "AssertionError"],
  "extra_files": [
    "<skill_dir>/scripts/slurm_run_custom.sh"
  ],
  "slurm_params": {
    "partition": "<PARTITION>",
    "account": "<ACCOUNT>",
    "time_limit": "<TIME_LIMIT>",
    "nodes": 1,
    "ntasks": 4,
    "ntasks_per_node": 4,
    "gpus_per_node": 4,
    "gres": "gpu:4",
    "mpi_plugin": "pmix_v5"
  },
  "slurm_env": { /* the full parsed JSON loaded from `slurm_env_file` in Step 1: accounts[], partitions[{name,state,time_limit,nodes,arch,gres,gpus_per_node,requires_gres}], default_account, default_partition, pmix.{available,preferred}, errors[]. Embedded verbatim so downstream slurm executors can read arch/gres/pmix from `job_spec.slurm_env` without opening the file or re-detecting. */ }
}
```

**Remote-slurm fields (required for `execution_scenario == remote_slurm`):** the case-executor populates these and the script builder echoes them verbatim. The builder MUST NOT recompute or substitute defaults for them.

```json
{
  "remote_repo_path": "<default_user_root_dir>/<repo_dir_name>",
  "remote_work_dir": "<remote_repo_path>/work_dirs/<basename(work_dir)>"
}
```

For remote scenarios, `--log-dir <LOG_DIR>` rendered into the slurm script uses `remote_work_dir`, and `--project-path <PROJECT_PATH>` (== `--container-workdir`) uses `remote_repo_path`. If either field is missing from `job_spec.json` for a `remote_slurm` scenario, stop and surface the error — do not fall back to `remote_cwd` or any other default.

**Persistent mode fields:** When `persistent_mode=true` and `execution_scenario=local_slurm`, add these fields to `job_spec.json` (all already resolved during script generation):

```json
{
  "persistent_mode": true,
  "container_name": "<CONTAINER_NAME>",
  "container_mounts": "<CONTAINER_MOUNTS>"
}
```

`container_name` is the resolved `<CONTAINER_NAME>` placeholder value (e.g., `llama_auto_test`, `gpt_custom`, `llama_eval`). `container_mounts` is the resolved comma-separated mount string. `docker_image` and `slurm_params` are already present in the base schema. The `.slurm` script generation is completely unchanged — persistent mode reuses the same scripts.

**Scenario scope:** These persistent-mode fields apply **only to `local_slurm`**. Do not emit them for any other scenario:

- `local_docker` / `local_direct` — no Slurm allocation exists, so `persistent_mode` has no meaning. Omit the fields even if `persistent_mode=true` was passed in.
- `remote_slurm` — `exec-remote-slurm` runs its own tmux-held `salloc` + named-container persistence flow (see that skill's Recipe 6). It does **not** read `persistent_mode`, `container_name`, or `container_mounts` from `job_spec.json` for that purpose; remote allocation reuse is controlled directly by the remote executor's inputs (`release_allocation`, `alloc_time_limit`, etc.) rather than by this field. Always omit the persistent-mode fields from `job_spec.json` when `execution_scenario=remote_slurm`.

**Pattern selection by workflow type:**

| Workflow | `log_file_pattern` | `success_patterns` | `failure_patterns` |
|----------|-------------------|-------------------|-------------------|
| pytest | `<name>_auto_test_%j.out` | `passed` | `FAILED`, `ERROR` |
| eval | `<name>_eval_%j.out` | `accuracy:`, `score:` | `AssertionError`, `Error` |
| bench | `<name>_bench_%j.out` | `token/s`, `throughput` | `Error`, `Traceback` |
| perf_sanity | `slurm-%j.out` | `8_done_` | `FAILED`, `Error` |
| custom | `<name>_custom_%j.out` | *(none)* | `Error`, `Traceback` |

For **local Docker**, use `docker_cmd` field instead of `script_path`:
```json
{
  "docker_cmd": "<complete docker run command>",
  "log_file": "<work_dir>/<model_name>_local.log"
}
```

### Step 6: Report

Print a summary of what was generated:
```
Generated: <work_dir>/job_spec.json
Script: <work_dir>/<script_name>
Workflow: <workflow_type>
Model: <model_name>
Docker image: <image>
Devices: <required_devices> GPUs across <node_count> nodes
```

## Rules

- Always use **absolute paths** in generated scripts and job_spec.json
- For remote execution, the script must contain **remote paths** for container-workdir, mounts, and log directories — but `job_spec.json` records the **local** script path (the executor skill transfers it)
- Never execute the script — only generate it. Execution is the executor skill's job.
- If `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` fails, report the error and stop
- Validate that required files exist before referencing them (docker image properties, test files, config YAMLs)
