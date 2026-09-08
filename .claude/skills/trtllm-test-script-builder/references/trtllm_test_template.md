# TensorRT-LLM Test Executor Templates

## Placeholders

All templates below use these placeholders — substitute with actual values before execution.

| Placeholder | Description |
|-------------|-------------|
| `<PWD>` | Current working directory (result of `pwd`) |
| `<HOME_DIR>` | User's home directory (value of `$HOME`) |
| `<PROJECT_PATH>` | Absolute path to the project/repo root directory |
| `<MODEL_NAME>` | Short model name from the `model_name` input |
| `<DOCKER_IMAGE>` | Docker image selected from `jenkins/current_image_tags.properties` |
| `<TEST_CMD>` | Test command(s) to run inside the container |
| `<CHECKPOINT_PATH>` | Host path to the model checkpoint (from the `checkpoint_path` input) |
| `<CHECKPOINT_PARENT_PATH>` | Parent directory of `<CHECKPOINT_PATH>` (e.g., `/data/models` if checkpoint is `/data/models/llama-7b`) |
| `<MODELS_HOST_PATH>` | Resolved host path for models mount (see Models Mount Resolution in SKILL.md) |
| `<REQUIRED_DEVICES>` | Number of GPU devices requested |
| `<NODES_NUM>` | Number of Slurm nodes to allocate (default: 1) |
| `<NTASKS>` | Total number of tasks (pytest: always 1; other workflows: `<REQUIRED_DEVICES>`) |
| `<NTASKS_PER_NODE>` | Tasks per node (pytest: always 1; other workflows: `<REQUIRED_DEVICES>` / `<NODES_NUM>`) |
| `<GPUS_PER_NODE>` | GPUs per node (always 4, regardless of `<NTASKS_PER_NODE>`) |
| `<CONTAINER_NAME>` | Container name (default: `<MODEL_NAME>_auto_test`) |
| `<CONTAINER_MOUNTS>` | Comma-separated mount mappings for `--container-mounts` |
| `<LOG_DIR>` | Directory for log output (default: current working directory) |
| `<PARTITION>` | Slurm partition name |
| `<ACCOUNT>` | Slurm account name |
| `<TIME_LIMIT>` | Slurm job time limit (default: `02:00:00`) |
| `<MPI_FLAG>` | `--mpi=pmix` for multi-node, empty for single-node |
| `<GRES_FLAG>` | `--gres=gpu:<GPUS_PER_NODE>` when the partition reports GPU GRES (e.g., `gpu:4`); empty string (entire `#SBATCH --gres` line omitted) when partition reports `(null)` |
| `<EVAL_CMD>` | Complete `trtllm-eval` command string (Category 5 only) |
| `<PREP_CMD>` | The `prepare-dataset` step from `bench_cmd` (Category 6 only). Empty string if `bench_cmd` has no `&&`. |
| `<BENCH_RUN_CMD>` | The `throughput`/`latency` step from `bench_cmd` (Category 6 only). Full `bench_cmd` if no `&&`. |

---

> **Scenario coverage:** Categories below cover only `execution_scenario` values of `local_docker`, `local_slurm`, and `remote_slurm`. The `local_direct` scenario has no Category here — it runs on the host without a container or Slurm wrapper and is handled inline by `trtllm-case-executor` (which skips this skill).

## Category 1: Local Docker Command

Use this template when the environment check returns `satisfied, local`.

### Command template

```bash
docker run -it --rm \
  --net=host \
  --runtime=nvidia \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_ADMIN \
  --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  --env REPO_DIR=tekit_sync \
  --env HOME_DIR=<PWD> \
  --env LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models/ \
  --mount type=bind,source=<HOME_DIR>,target=<HOME_DIR> \
  --mount type=bind,source=<PROJECT_PATH>,target=<PROJECT_PATH> \
  --mount type=bind,source=<MODELS_HOST_PATH>,target=/scratch.trt_llm_data/llm-models/ \
  --name <MODEL_NAME>_auto_test \
  <DOCKER_IMAGE> \
  bash -c '
    set -euo pipefail
    cd <PROJECT_PATH>
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    python -c "import tensorrt_llm; print(\"tensorrt_llm version:\", tensorrt_llm.__version__)" 2>&1 \
      || { echo "ERROR: TRT-LLM installation verification failed" >&2; exit 1; }
    <TEST_CMD>
  '
```

---

## Category 2: Non-Perf Slurm Job Script

Use this template when the environment check returns `satisfied, slurm` and the test is NOT a perf-sanity test. Supports both single-node and multi-node.

### Script template

Write this to `<MODEL_NAME>_auto_test.slurm` in the current working directory:

```bash
#!/bin/bash
#SBATCH --job-name=<MODEL_NAME>_auto_test
#SBATCH --output=<LOG_DIR>/<MODEL_NAME>_auto_test_%j.out
#SBATCH --error=<LOG_DIR>/<MODEL_NAME>_auto_test_%j.err
#SBATCH --nodes=<NODES_NUM>
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH <GRES_FLAG>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES, Tasks: $SLURM_NTASKS"

# Step 1: Install TRT-LLM (one task per node) — only present for local_slurm
# (omit entirely for remote_slurm where the image already has TRT-LLM installed)
srun -l \
  --ntasks-per-node=1 \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    echo "Node ${SLURM_NODEID:-0}: Installing TRT-LLM from source..."
    cd <PROJECT_PATH>
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    echo "Node ${SLURM_NODEID:-0}: Install complete"
  '

# Step 2: Run tests
srun -l \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  <MPI_FLAG> \
  bash -c '
    nvidia-smi
    # Single-node jobs: clear MPI/SLURM env vars to prevent false MPI init
    if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
      for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
          unset "$v"
        fi
      done
    fi
    export LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models/
    cd <PROJECT_PATH>
    <TEST_CMD>
  '

exit_code=$?
echo "Job $SLURM_JOB_ID finished at $(date) with exit code $exit_code"
exit $exit_code
```

### Submission command

```bash
sbatch <MODEL_NAME>_auto_test.slurm
```

Report the Slurm job ID. The user can monitor with `squeue -j <JOB_ID>` and view logs at `<MODEL_NAME>_auto_test_<JOB_ID>.out`.

---

## Category 3: Custom Script Slurm Job Script

Use this template when running an arbitrary user script or command (not a pytest test or perf-sanity benchmark) on a Slurm cluster.

### Additional placeholders

| Placeholder | Description |
|-------------|-------------|
| `<CUSTOM_CMD>` | The user's command string. If `custom_script` is provided, this is `bash <custom_script>`; otherwise the literal `custom_cmd` value. |
| `<CUSTOM_ENV>` | Space-separated `KEY=VALUE` pairs to export inside the container (from `custom_env` input). Empty string if not provided. |
| `<CUSTOM_WORKDIR>` | Working directory for the command inside the container. Defaults to `<PROJECT_PATH>`. |
| `<RUN_SCRIPT>` | Absolute path to `scripts/slurm_run_custom.sh`. |

### Script template

Write this to `<WORK_DIR>/<MODEL_NAME>_custom.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=<MODEL_NAME>_custom
#SBATCH --output=<LOG_DIR>/<MODEL_NAME>_custom_%j.out
#SBATCH --error=<LOG_DIR>/<MODEL_NAME>_custom_%j.err
#SBATCH --nodes=<NODES_NUM>
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH <GRES_FLAG>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES, Tasks: $SLURM_NTASKS"

export CUSTOM_COMMAND="<CUSTOM_CMD>"
export CUSTOM_ENV="<CUSTOM_ENV>"
export CUSTOM_WORKDIR="<CUSTOM_WORKDIR>"
export llmSrcNode="<PROJECT_PATH>"

# Step 1: Install TRT-LLM (one task per node)
srun -l \
  --ntasks-per-node=1 \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    echo "Node ${SLURM_NODEID:-0}: Installing TRT-LLM from source..."
    cd <PROJECT_PATH>
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    echo "Node ${SLURM_NODEID:-0}: Install complete"
  '

# Step 2: Run custom command
srun -l \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<CUSTOM_WORKDIR> \
  --kill-on-bad-exit=1 \
  <MPI_FLAG> \
  <RUN_SCRIPT>

exit_code=$?
echo "Job $SLURM_JOB_ID finished at $(date) with exit code $exit_code"
exit $exit_code
```

### Submission command

```bash
sbatch <WORK_DIR>/<MODEL_NAME>_custom.slurm
```

Report the Slurm job ID. The user can monitor with `squeue -j <JOB_ID>` and view logs at `<MODEL_NAME>_custom_<JOB_ID>.out`.

---

## Category 4: Perf-Sanity Slurm Job Script

Use the repo's `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` to generate launch scripts for perf-sanity tests and benchmarks. This handles both aggregated and disaggregated modes automatically.

`submit.py` is self-contained — it bundles and resolves its own helper scripts (container run, TRT-LLM install, and the aggregated/disaggregated draft templates) internally, so none of those are passed as flags.

### Commands

**Step 1**: Generate the launch script from the pre-built `perf_config_yaml` provided by the caller:

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

### Generated output

`submit.py` produces a `slurm_launch.sh` that combines:
1. `#SBATCH` directives (nodes, gpus, partition, time)
2. Exported environment variables (pytest commands, hardware config, env vars per role)
3. `srunArgs` array (container image, mounts, workdir)
4. Draft template content (aggregated or disaggregated execution logic)

Submit with `sbatch <WORK_DIR>/slurm_launch.sh`.

---

## Category 5: Eval Slurm Job Script

Use this template for running `trtllm-eval` accuracy evaluations on a **single node**. For multi-GPU runs (`ntasks > 1`), wraps `trtllm-eval` with `trtllm-llmapi-launch` and uses `--mpi=pmix`; for single-GPU runs, neither is applied.

**Single-node only.** `trtllm-eval` does not support multi-node execution. If the requested `world_size` exceeds the node's GPUs, fall back to Category 4 (perf-sanity) with `accuracy.enable_accuracy_test: true` in the benchmark config YAML. Same fallback applies to disaggregated accuracy testing.

### Additional placeholders

| Placeholder | Description |
|-------------|-------------|
| `<EVAL_CMD>` | The complete `trtllm-eval` command string (e.g., `trtllm-eval --model /path/to/model --tp_size 4 gsm8k`). |

### Script template

Write this to `<WORK_DIR>/<MODEL_NAME>_eval.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=<MODEL_NAME>_eval
#SBATCH --output=<LOG_DIR>/<MODEL_NAME>_eval_%j.out
#SBATCH --error=<LOG_DIR>/<MODEL_NAME>_eval_%j.err
#SBATCH --nodes=<NODES_NUM>
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH <GRES_FLAG>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES, Tasks: $SLURM_NTASKS"

# Step 1: Install TRT-LLM and eval dependencies (one task per node)
srun -l \
  --ntasks-per-node=1 \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    echo "Node ${SLURM_NODEID:-0}: Installing TRT-LLM and eval dependencies..."
    cd <PROJECT_PATH>
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -3
    if [ -f examples/trtllm-eval/requirements.txt ]; then
      pip install -r examples/trtllm-eval/requirements.txt 2>&1 | tail -3
    fi
    echo "Node ${SLURM_NODEID:-0}: Install complete"
  '

# Step 2: Run eval
# <MPI_FLAG> is --mpi=pmix for ntasks > 1, empty for single-GPU
# <LAUNCHER> is "trtllm-llmapi-launch " for ntasks > 1, empty for single-GPU
srun -l \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  <MPI_FLAG> \
  bash -c '
    set -euo pipefail
    nvidia-smi
    export LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models/
    cd <PROJECT_PATH>
    <LAUNCHER><EVAL_CMD>
  '

exit_code=$?
echo "Job $SLURM_JOB_ID finished at $(date) with exit code $exit_code"
exit $exit_code
```

### Submission command

```bash
sbatch <WORK_DIR>/<MODEL_NAME>_eval.slurm
```

Report the Slurm job ID. The user can monitor with `squeue -j <JOB_ID>` and view logs at `<MODEL_NAME>_eval_<JOB_ID>.out`.

### Notes

- **Single-node constraint**: `#SBATCH --nodes` is always `1`. If the eval's `world_size` (`tp_size * pp_size`) exceeds the node's GPU count, route to Category 4 instead — do not try to schedule Category 5 across nodes.
- **Single-GPU vs multi-GPU eval**: for `ntasks > 1`, use `srun --mpi=pmix` and prefix the eval command with `trtllm-llmapi-launch`. For `ntasks == 1` (single-GPU), omit both — `trtllm-llmapi-launch` with pmix on a single task causes unnecessary MPI overhead.
- **`<GRES_FLAG>`**: Resolved dynamically in Step 1/3 via `sinfo -p <partition> -h -o "%G"`. The entire `#SBATCH --gres` line is omitted when the partition reports `(null)`.
- **Eval output**: `trtllm-eval` prints accuracy scores to stdout. Only rank 0 produces meaningful output; other ranks are workers. Look for lines containing `accuracy`, `score`, or assertion errors from `--check_accuracy`.
- **`--config` / `--extra_llm_api_options`**: The `trtllm-eval` CLI supports a YAML config file for additional LLM API options. If the user provides one, ensure it's accessible inside the container (via mounts).

---

## Category 6: Bench (trtllm-bench) Slurm Job Script

Use this template for running `trtllm-bench` throughput/latency benchmarks on a **single node**. The `prepare-dataset` step always runs as a single process (no MPI). The benchmark run step uses `srun --mpi=pmix` + `trtllm-llmapi-launch` only for multi-GPU runs (`ntasks > 1`); single-GPU runs omit both.

**Single-node only.** If `world_size > gpus_per_node`, stop and tell the caller to use the perf-sanity Benchmark (Category 4) workflow instead.

### Additional placeholders

| Placeholder | Description |
|-------------|-------------|
| `<PREP_CMD>` | The `prepare-dataset` step from `bench_cmd` (everything before `&&`). Omit the first srun block entirely if empty. |
| `<BENCH_RUN_CMD>` | The `throughput`/`latency` step from `bench_cmd` (everything after `&&`, or the full command if no `&&`). |

### Script template

Write this to `<WORK_DIR>/<MODEL_NAME>_bench.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=<MODEL_NAME>_bench
#SBATCH --output=<LOG_DIR>/<MODEL_NAME>_bench_%j.out
#SBATCH --error=<LOG_DIR>/<MODEL_NAME>_bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH <GRES_FLAG>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>

set -euo pipefail

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES, Tasks: $SLURM_NTASKS"

# Step 1: Install TRT-LLM and prepare benchmark dataset (always single process)
srun -l --ntasks=1 \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    nvidia-smi

    echo "Installing TRT-LLM from source..."
    cd <PROJECT_PATH>
    pip install -e . 2>&1 | tail -5
    pip install -r requirements-dev.txt 2>&1 | tail -5
    echo "TRT-LLM installed"

    <PREP_CMD>
  '

# Step 2: Run benchmark
# <MPI_FLAG> is --mpi=pmix for ntasks > 1, empty for single-GPU
# <LAUNCHER> is "trtllm-llmapi-launch " for ntasks > 1, empty for single-GPU
srun -l <MPI_FLAG> \
  --container-image=<DOCKER_IMAGE> \
  --container-name=<CONTAINER_NAME> \
  --container-mounts=<CONTAINER_MOUNTS> \
  --container-workdir=<PROJECT_PATH> \
  --kill-on-bad-exit=1 \
  bash -c '
    set -euo pipefail
    cd <PROJECT_PATH>
    <LAUNCHER><BENCH_RUN_CMD>
  '

exit_code=$?
echo "Job $SLURM_JOB_ID finished at $(date) with exit code $exit_code"
exit $exit_code
```

### Submission command

```bash
sbatch <WORK_DIR>/<MODEL_NAME>_bench.slurm
```

Report the Slurm job ID. The user can monitor with `squeue -j <JOB_ID>` and view logs at `<MODEL_NAME>_bench_<JOB_ID>.out`.

### Notes

- **Single-node constraint**: `#SBATCH --nodes` is always `1`. If `world_size > gpus_per_node`, route to Category 4 instead.
- **`#SBATCH --ntasks=<NTASKS>`** allocates `world_size` tasks so the second srun can use all of them. The first srun explicitly overrides with `--ntasks=1`.
- **Single-GPU vs multi-GPU bench**: for `ntasks > 1`, use `srun --mpi=pmix` and prefix `<BENCH_RUN_CMD>` with `trtllm-llmapi-launch`. For `ntasks == 1` (single-GPU), omit both — `trtllm-llmapi-launch` with pmix on a single task causes unnecessary MPI overhead.
- **Container reuse**: both srun calls share `--container-name=<CONTAINER_NAME>`, so TRT-LLM installed in Step 1 is available in Step 2 without reinstalling.
- **`<GRES_FLAG>`**: Entire `#SBATCH --gres` line is omitted when the partition reports `(null)`.
- **Bench output**: look for lines containing `token/s` or `throughput` for success; `trtllm-bench` writes a JSON summary to the `--report_json` path.

---

## job_spec.json Schema

Write `<work_dir>/job_spec.json` after generating the script. This manifest is consumed by executor subagents.

### Slurm workflows (Categories 2–5)

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
    "gpus_per_node": 4
  }
}
```

### Local Docker (Category 1)

Use `docker_cmd` and `log_file` instead of `script_path`:

```json
{
  "workflow_type": "pytest|eval|custom|benchmark",
  "model_name": "<MODEL_NAME>",
  "execution_scenario": "local_docker",
  "docker_image": "<resolved image>",
  "docker_cmd": "<complete docker run command>",
  "log_file": "<work_dir>/<model_name>_local.log",
  "success_patterns": ["passed", "accuracy:"],
  "failure_patterns": ["FAILED", "Error", "AssertionError"]
}
```

### Pattern selection by workflow type

| Workflow | `log_file_pattern` | `success_patterns` | `failure_patterns` |
|----------|-------------------|-------------------|-------------------|
| pytest | `<name>_auto_test_%j.out` | `passed` | `FAILED`, `ERROR` |
| eval | `<name>_eval_%j.out` | `accuracy:`, `score:` | `AssertionError`, `Error` |
| bench | `<name>_bench_%j.out` | `token/s`, `throughput` | `Error`, `Traceback` |
| benchmark | `slurm-%j.out` | `8_done_` | `FAILED`, `Error` |
| custom | `<name>_custom_%j.out` | *(none)* | `Error`, `Traceback` |
