---
name: exec-local-docker
description: >-
  Execute a TensorRT-LLM workload locally in Docker. Runs a fully-resolved
  Docker command in background, monitors completion, reads logs, and reports
  results. Workflow-agnostic — does not need to know if the workload is pytest,
  eval, benchmark, or a custom script.
tags: [docker, execution, infrastructure]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Local Docker Executor

Run a Docker command locally, monitor it, and report results.

## Input (from orchestrator prompt)

The orchestrator passes these fields in the skill prompt:

| Field | Description |
|-------|-------------|
| `docker_cmd` | Complete `docker run` command string, ready to execute |
| `work_dir` | Local work directory for logs and artifacts |
| `log_file` | Full path to the log file (output redirected here) |
| `model_name` | Short model name for reporting |
| `workflow_type` | `pytest`, `eval`, `custom`, or `benchmark` — for output parsing hints |
| `success_patterns` | Comma-separated patterns indicating success (e.g., `passed,accuracy:`) |
| `failure_patterns` | Comma-separated patterns indicating failure (e.g., `FAILED,Error,AssertionError`) |

## Procedure

### Step 0: Resolve Image and Build (when `build_project=true`)

This executor owns image selection and the build for the local Docker target. Skip the entire step when `build_project=false`.

1. **Detect the target GPU type.** Use `gpu_type` from `job_spec.json` if upstream env-check resolved it; otherwise probe locally with `nvidia-smi --query-gpu=name --format=csv,noheader | head -1`.
2. **Detect the host CPU arch** with `uname -m` (`x86_64` or `aarch64`).
3. **Resolve the container image.** Read `<repo_root>/jenkins/current_image_tags.properties` and pick the tag whose CPU-arch flavor matches the host. If the orchestrator already passed a `container_image` field in the job spec, use that and skip the lookup.
4. **Map GPU → build arch (`-a` flag):** `H100`/`H200` → `90-real`; `B200`/`GB200`/`B300`/`GB300` → `100-real`; `A100` → `80-real`; `L40S` → `89-real`. Default `100-real` when the GPU is unknown.
5. **Compile.** Invoke the `exec-local-compile` skill with `repo_dir=<repo_root>`, `image=<resolved tag>`, `arch=<arch>`. Wait for completion.
6. **On failure**, do not launch the workload. Report `BUILD_FAILED` with the last 100 lines of the compile log.

`build_project`, `gpu_type`, `repo_root`, and (optionally) `container_image` come from `job_spec.json`.

### Step 1: Launch

Run the Docker command in background using `run_in_background`:

```bash
<docker_cmd> 2>&1 | tee <log_file>
```

Report to the orchestrator: "Launched locally, log at `<log_file>`"

### Step 2: Monitor for Hangs

While waiting for the background process to complete, actively monitor the log
file for hang indicators. Launch a monitoring loop using `run_in_background`:

```bash
while true; do
  sleep 60
  if [ -f "<log_file>" ] && grep -qi "hang detected" "<log_file>"; then
    echo "HANG_DETECTED: Found 'hang detected' in log file"
    docker ps --filter "ancestor=<image>" -q | xargs -r docker kill 2>/dev/null
    exit 1
  fi
done
```

- If the monitor detects a hang, it kills the Docker container and exits with
  code 1. The main background process will also terminate.
- When the main process completes normally (background notification received),
  kill the monitoring loop (it is no longer needed).
- If a hang is detected, skip to Step 4 and report `HANG_DETECTED` status
  instead of proceeding to normal result collection.

### Step 3: Wait for Completion

The Bash tool's `run_in_background` will notify when the process finishes
(either normally or because the container was killed by the hang monitor).

### Step 4: Read Results

On completion:

1. **Read exit code** from the background command result.
2. **Read the last 100 lines** of `<log_file>` using the Read tool.
3. **If exit code != 0**, also read the first 50 lines to catch early errors (import failures, setup crashes).
4. **Search for patterns**:
   - Grep `<log_file>` for each `success_patterns` entry
   - Grep `<log_file>` for each `failure_patterns` entry

### Step 5: Report

Return a structured result:

```
Status: PASSED | FAILED | ERROR | HANG_DETECTED
Exit code: <N>
Log file: <log_file>
Work directory: <work_dir>
Summary: <relevant output lines — pytest summary, accuracy score, or last few lines>
Errors: <if failed, the relevant error output>
```

### Output Parsing by Workflow Type

- **pytest**: Look for `X passed, Y failed in Zs` summary line
- **eval**: Look for `accuracy:` or `score:` lines; check for `Expected accuracy >= X, but got Y` assertion
- **custom**: No specific patterns — report last 10 lines of output
- **benchmark**: Look for throughput/latency numbers

## Rules

- Never run the Docker command in foreground — always use `run_in_background`
- Never `cat` the full log file — use Read with offset/limit or tail
- If the Docker command fails immediately (exit code within seconds), check if the image exists locally
- Report results even if the log file is empty (container may have failed to start)
