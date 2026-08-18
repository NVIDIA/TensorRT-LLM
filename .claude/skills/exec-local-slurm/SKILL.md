---
name: exec-local-slurm
description: >-
  Submit and monitor a Slurm job on a local cluster. Supports two modes:
  (1) Persistent allocation (default) — allocates nodes once via nohup salloc,
  imports the container once, installs once, and reuses across runs by setting
  SLURM env vars and running the sbatch script via bash. (2) One-shot sbatch —
  submits a fully-generated Slurm script via sbatch, polls job status, reads
  logs on completion, and reports results. Workflow-agnostic — handles pytest,
  eval, benchmark, and custom scripts identically.
tags: [slurm, execution, infrastructure]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Local Slurm Executor

Submit a Slurm job locally, monitor it, and report results. Uses persistent
allocation by default to eliminate queue wait, container import, and install
overhead on repeated runs.

## Input (from orchestrator prompt)

The orchestrator passes these fields in the skill prompt:

| Field | Description |
|-------|-------------|
| `script_path` | Full local path to the generated `.slurm` script |
| `work_dir` | Local work directory for logs and artifacts |
| `model_name` | Short model name for reporting |
| `workflow_type` | `pytest`, `eval`, `custom`, or `benchmark` — for output parsing hints |
| `success_patterns` | Comma-separated patterns indicating success |
| `failure_patterns` | Comma-separated patterns indicating failure |
| `log_file_pattern` | Log filename pattern with `%j` placeholder (e.g., `llama_auto_test_%j.out`) |
| `persistent_mode` | Default: `true` for all local slurm workflows. Set to `false` to force one-shot sbatch (opt-out). |
| `release_allocation` | `true` to release the current allocation and stop. Only set when the user explicitly says no more jobs are needed. Default: `false`. Never auto-release. |
| `alloc_time_limit` | Walltime for the persistent allocation. Default: `04:00:00`. |
| `docker_image` | Container image for persistent container import. From `job_spec.json`. |
| `container_name` | Container name used in the `.slurm` script (e.g., `llama_auto_test`). Must match exactly. From `job_spec.json`. |
| `container_mounts` | Comma-separated mount mappings. From `job_spec.json`. |
| `repo_root` | Repo root path (for locating state file and project path). |
| `slurm_params` | Slurm parameters object: `partition`, `account`, `nodes`, `ntasks`, `ntasks_per_node`, `gpus_per_node`. From `job_spec.json`. |

## Procedure

### Pre-step: Resolve Image and Build (when `build_project=true`)

This executor owns image selection and the build for the local SLURM cluster. Skip the entire pre-step when `build_project=false`.

1. **Read the cluster GPU type** from `device_type` in `job_spec.json` (resolved upstream by env-check / internal-env-info). If absent, probe from any compute node via `srun -p <partition> --ntasks-per-node=1 nvidia-smi --query-gpu=name --format=csv,noheader | head -1`. If `skills/internal-env-info/` is not installed, skip the upstream lookup silently and rely on the srun probe — do not report the missing skill as an error.
2. **Determine the partition's CPU arch.** Look up `job_spec.slurm_env.partitions[]` for the entry where `name == partition` and read its `arch` (`x86_64` / `aarch64`). Case-executor's Step 2.5 already detected this — do **not** run `scontrol show node` here. If `slurm_env` is absent or the entry's `arch` is null (case-executor ran without SLURM tools), fall back to inferring from `device_type`: Grace-based parts (`GB*` / `GH*`) → `aarch64`; everything else → `x86_64`.
3. **Resolve the container image.** If `docker_image` is already set in `job_spec.json`, use it directly. Otherwise read `<repo_root>/jenkins/current_image_tags.properties` and pick the tag matching the partition's CPU arch.
4. **Map GPU → build arch (`-a` flag):** `H100`/`H200` → `90-real`; `B200`/`GB200`/`B300`/`GB300` → `100-real`; `A100` → `80-real`; `L40S` → `89-real`. Default `100-real`.
5. **Compile.** Invoke the `exec-slurm-compile` skill with `repo_dir=<repo_root>`, `partition`, `account`, `container_image=<resolved tag>`, `user_root_dir`, `arch`. Wait for completion.
6. **On failure**, do not allocate or run the workload. Report `BUILD_FAILED` with the last 100 lines of the build log.

`build_project`, `device_type`, `repo_root`, `partition`, `account`, `user_root_dir`, and (optionally) `docker_image` come from `job_spec.json`.

### Step 0: Allocation Management

This step runs before any execution. It determines whether to reuse an existing
persistent allocation, create a new one, or fall through to one-shot sbatch.

#### Step 0A — Release mode

If `release_allocation=true`:

1. Read `<repo_root>/work_dirs/.slurm_alloc.json`
2. If file exists and `job_id` is present:
   ```bash
   scancel <job_id>
   ```
3. Delete the state file and `.salloc.log`
4. Report: "Allocation <job_id> released." Stop.

#### Step 0B — One-shot mode

If `persistent_mode=false`, skip entirely to Step 1 (One-Shot Path).

#### Step 0C — Validate existing allocation

Read `<repo_root>/work_dirs/.slurm_alloc.json`.

- If state file **does not exist** → go to Step 0D.
- If state file exists, validate the allocation:
  ```bash
  squeue -j <job_id> -h -o "%T %L"
  ```
  - **RUNNING + remaining > 5 min + params compatible** → **reuse** (skip to Step 0F)
  - **RUNNING + remaining <= 5 min** → warn user ("Allocation expiring soon"), `scancel <job_id>`, delete state file, go to Step 0D
  - **RUNNING + params incompatible** → `scancel <job_id>`, delete state file, go to Step 0D
  - **Empty output / error** (job gone) → stale state file, delete it, go to Step 0D

**Params compatibility check:**
- `partition` must match
- `nodes` in state must be **>=** requested nodes (a 2-node allocation can serve 1-node jobs)
- `docker_image` must match

#### Step 0D — Justify and prepare allocation

Before allocating, ensure no orphaned allocations exist and log the reason:

1. Check for existing persistent jobs:
   ```bash
   squeue -u $(whoami) -h -o "%i %T %j" --name=<account>-trtllm.persistent
   ```
2. If found → orphan (state file was missing/corrupt). Cancel it:
   ```bash
   scancel <orphan_job_id>
   ```
   Log: "Released orphaned allocation <orphan_job_id> — state file was missing."
3. Log justification: "No reusable allocation found. Allocating <nodes> node(s) on partition <partition> for <alloc_time_limit>."

#### Step 0E — Allocate new

Convert `alloc_time_limit` from `HH:MM:SS` to seconds (e.g., `04:00:00` → `14400`).

```bash
mkdir -p <repo_root>/work_dirs
nohup salloc --partition=<partition> --account=<account> \
  --nodes=<nodes> --time=<alloc_time_limit> \
  --job-name=<account>-trtllm.persistent \
  sleep <alloc_time_limit_seconds> \
  > <repo_root>/work_dirs/.salloc.log 2>&1 &
```

Retrieve job ID:
```bash
squeue -u $(whoami) -h -o "%i" --name=<account>-trtllm.persistent
```

If no job appears after 10 seconds, read `<repo_root>/work_dirs/.salloc.log`
for errors (bad partition, invalid account, etc.) and report to user.

Otherwise, poll until RUNNING (every 10s, max 60 polls). Once RUNNING, get
the nodelist:
```bash
squeue -j <JOB_ID> -h -o "%N"
```

**Import container** on all nodes (using the **same container name** as the
`.slurm` script — critical for pyxis reuse):
```bash
srun --jobid=<JOB_ID> -N <nodes> --ntasks-per-node=1 \
  --container-image=<docker_image> \
  --container-name=<container_name> true
```

**Warm up filesystem mounts** on all nodes — `ls` each mounted path so
Lustre/NFS metadata is cached for later use:
```bash
srun --jobid=<JOB_ID> -N <nodes> --ntasks-per-node=1 \
  --container-name=<container_name> \
  --container-mounts=<container_mounts> \
  bash -c 'for p in <mount_target_1> <mount_target_2> ...; do ls "$p" > /dev/null 2>&1; done'
```
Parse mount targets from `container_mounts` — the right-hand side of each
`host:container` pair.

**Check GPU status** on all nodes (no container needed — `nvidia-smi` is on
the host):
```bash
srun --jobid=<JOB_ID> -N <nodes> --ntasks-per-node=1 \
  bash -c 'echo "=== $(hostname) ===" && nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader'
```
- If output shows no processes → GPUs are clean, proceed
- If unexpected processes found → warn user: "GPU processes found on
  <node>: <process_list>. These may interfere with the job."

**Run install** on all nodes with `--container-writable` (so packages persist
in the named container across srun calls). Install all common requirements
upfront so that subsequent jobs of any workflow type can skip install:
```bash
srun --jobid=<JOB_ID> -N <nodes> --ntasks-per-node=1 \
  --container-name=<container_name> --container-writable \
  --container-mounts=<container_mounts> \
  bash -c 'cd <repo_root> && pip install -e . && pip install -r requirements-dev.txt && \
    if [ -f examples/trtllm-eval/requirements.txt ]; then pip install -r examples/trtllm-eval/requirements.txt; fi'
```
For custom workflow with `skip_install=true`, skip this step entirely.

**Write state file** `<repo_root>/work_dirs/.slurm_alloc.json`:
```json
{
  "job_id": "<JOB_ID>",
  "container_name": "<container_name>",
  "nodelist": "<NODELIST>",
  "partition": "<partition>",
  "account": "<account>",
  "nodes": <nodes>,
  "gpus_per_node": <gpus_per_node>,
  "docker_image": "<docker_image>",
  "container_mounts": "<container_mounts>",
  "allocated_at": "<ISO timestamp>",
  "time_limit": "<alloc_time_limit>",
  "installed": true
}
```

Proceed to Step 1 (Persistent Path).

#### Step 0F — Reuse path validation

Allocation is valid. Check if container name matches current request:

- If `container_name` in state file **matches** `container_name` from
  `job_spec.json` → proceed to Step 1 (Persistent Path). Container and install
  are already set up.
- If **different** (model changed) → import the new container, warm up mounts,
  and install:
  ```bash
  srun --jobid=<job_id> -N <nodes> --ntasks-per-node=1 \
    --container-image=<docker_image> \
    --container-name=<new_container_name> true
  srun --jobid=<job_id> -N <nodes> --ntasks-per-node=1 \
    --container-name=<new_container_name> \
    --container-mounts=<container_mounts> \
    bash -c 'for p in <mount_targets>; do ls "$p" > /dev/null 2>&1; done'
  srun --jobid=<job_id> -N <nodes> --ntasks-per-node=1 \
    --container-name=<new_container_name> --container-writable \
    --container-mounts=<container_mounts> \
    bash -c 'cd <repo_root> && pip install -e . && pip install -r requirements-dev.txt'
  ```
  Update `container_name` in state file. Proceed to Step 1 (Persistent Path).

---

### Step 1: Execute

Two execution paths depending on `persistent_mode`.

#### Persistent Path (persistent_mode=true)

**Pre-flight time check** — verify the allocation has enough remaining time:
```bash
squeue -j <job_id> -h -o "%L"
```
If remaining time < job's `time_limit` from `slurm_params`, warn user:
"Allocation has <remaining> left but job expects <time_limit>. The job may
be killed early."

**Pre-flight GPU check** — verify GPUs are not occupied by leftover processes:
```bash
srun --jobid=<job_id> -N <nodes> --ntasks-per-node=1 \
  bash -c 'procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null); [ -n "$procs" ] && echo "WARNING: GPU processes on $(hostname): $procs"'
```
If unexpected processes found, warn user before proceeding.

**Run the sbatch script** with SLURM env vars — the `#SBATCH` directives are
comments when run via `bash`; the inner `srun` inherits the env vars and uses
the persistent allocation:
```bash
export SLURM_JOB_ID=<job_id>
export SLURM_JOB_NUM_NODES=<nodes>
export SLURM_NNODES=<nodes>
export SLURM_NTASKS=<ntasks>
export SLURM_NTASKS_PER_NODE=<ntasks_per_node>
export SLURM_NODELIST=<nodelist>
export SLURM_JOB_NODELIST=<nodelist>
bash <script_path> 2>&1 | tee <work_dir>/<model_name>_<workflow_type>.log
```

Run with `run_in_background`. Container reuse is automatic (pyxis skips
re-import when `--container-name` already exists on the node). The install
step (Step 1 srun) always runs to keep the container up to date.

**Step cancellation:** If a job hangs or the user wants to abort, cancel just
the srun step without killing the allocation:
```bash
# List active steps:
squeue -s -j <job_id>
# Cancel a specific step (e.g., step 0):
scancel <job_id>.0
```
The allocation stays RUNNING — new jobs can run immediately. The job name
`<account>-trtllm.persistent` uniquely identifies allocations created by this
skill via `squeue --name=<account>-trtllm.persistent`.

#### One-Shot Path (persistent_mode=false)

```bash
sbatch <script_path>
```

Parse the job ID from output: `Submitted batch job <JOB_ID>`.

If sbatch fails, report the error immediately and stop.

### Step 2: Report Submission

**One-shot mode:**
```
Job ID: <JOB_ID>
Script: <script_path>
Work directory: <work_dir>
Log files: <work_dir>/<log_file_pattern with %j replaced>
Monitor: squeue -j <JOB_ID>
```

**Persistent mode:**
```
Allocation: <job_id> (persistent, remaining: <time>)
Container: <container_name>
Script: <script_path>
Work directory: <work_dir>
Log file: <work_dir>/<model_name>_<workflow_type>.log
```

### Step 3: Poll / Wait for Completion (with Hang Monitoring)

Both polling modes must also monitor the log file for hang indicators. Every
60 seconds, check the log file for the pattern `hang detected`
(case-insensitive). If found, terminate the job and skip to Step 5 with
`HANG_DETECTED` status.

**One-shot mode:** Poll every 30 seconds, max 60 polls (30 minutes):

```bash
squeue -j <JOB_ID> -h -o "%T %M %R"
```

- Report state transitions: `PENDING` → `RUNNING` → `COMPLETING` → done
- If `PENDING` for > 5 minutes, check the reason:
  ```bash
  squeue -j <JOB_ID> -o "%i %T %r %S"
  ```
- The job is done when `squeue -j <JOB_ID>` returns empty (no rows)
- If max polls exceeded, report the job is still running and provide manual
  check commands
- **Hang check** (every other poll, i.e., every 60s): Once the job is RUNNING,
  resolve the log file path (`<work_dir>/<log_file_pattern>` with `%j` replaced
  by `<JOB_ID>`) and check:
  ```bash
  grep -qi "hang detected" <log_file> 2>/dev/null && echo "HANG_DETECTED"
  ```
  If `HANG_DETECTED` is printed, immediately cancel the job:
  ```bash
  scancel <JOB_ID>
  ```
  Then read the last 200 lines of the log file and skip to Step 5.

**Persistent mode:** While waiting for `run_in_background` completion
notification, launch a hang monitoring loop using `run_in_background`:

```bash
while true; do
  sleep 60
  LOG_FILE="<work_dir>/<model_name>_<workflow_type>.log"
  if [ -f "$LOG_FILE" ] && grep -qi "hang detected" "$LOG_FILE"; then
    echo "HANG_DETECTED: Found 'hang detected' in log file"
    squeue -s -j <job_id> -h -o "%i" | head -1 | xargs -r scancel 2>/dev/null
    exit 1
  fi
done
```

When the main process completes normally, kill the monitoring loop. If the
monitor fires first, it cancels the active Slurm step (not the allocation
itself — the allocation stays for future runs).

### Step 4: Collect Results

On completion:

1. **Get exit code:**
   - **One-shot:** `sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS -n`
   - **Persistent:** From the `run_in_background` result exit code

2. **Resolve log file paths:**
   - **One-shot:** Replace `%j` in `log_file_pattern` with `<JOB_ID>`:
     - Output: `<work_dir>/<MODEL_NAME>_..._<JOB_ID>.out`
     - Error: `<work_dir>/<MODEL_NAME>_..._<JOB_ID>.err`
   - **Persistent:** Single log at `<work_dir>/<model_name>_<workflow_type>.log`

3. **Read log files:**
   - Read the last 100 lines of the output log
   - If job failed, also read the first 50 lines to catch early errors
   - **One-shot only:** If job failed, read the last 50 lines of the `.err` file

4. **Search for patterns:**
   - Grep the log file for each `success_patterns` entry
   - Grep the log file for each `failure_patterns` entry

### Step 5: Report

Return a structured result:

```
Job ID: <JOB_ID> (or allocation <job_id> for persistent mode)
Status: COMPLETED | FAILED | TIMEOUT | CANCELLED | OUT_OF_MEMORY | HANG_DETECTED
Exit code: <exit code>
Elapsed: <time>
Log files:
  stdout: <path>
  stderr: <path> (one-shot only)
Work directory: <work_dir>
Summary: <relevant output — pytest summary, accuracy score, throughput, or last lines>
Errors: <if failed, relevant error output>
```

### Output Parsing by Workflow Type

- **pytest**: Look for `X passed, Y failed in Zs` summary line. Status maps:
  `COMPLETED` + 0 exit → `PASSED`; non-zero → `FAILED`
- **eval**: Look for `accuracy:` lines. If `--check_accuracy` was used,
  assertion error means threshold not met
- **benchmark**: Look for throughput/latency numbers. Check for
  `8_done_*.txt` completion marker
- **custom**: No specific patterns — report last 10 lines of output

### Failure Diagnosis

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `TIMEOUT` | Job exceeded time limit | Report and suggest increasing time |
| `OUT_OF_MEMORY` | OOM on node | Report and suggest reducing batch size |
| Exit `1:0` + `FAILED` | Non-zero exit from srun | Read `.err` file for details |
| `CANCELLED` | Preempted or user cancelled | Check `sacct --format=JobID,State,Reason` |
| Empty output log | Container failed to start | Check `.err` / `.salloc.log` for mount or image errors |
| `HANG_DETECTED` in log | Process/GPU hang (deadlock, NCCL timeout) | Job auto-cancelled; check last 200 log lines for root cause |
| `srun: error: Unable to create step` | Allocation expired | Delete state file, re-run (will re-allocate) |
| `srun: error: ... does not exist` | Named container missing | Delete state file, re-run (will re-import) |

## Rules

- Never use `squeue` with the `-uall` flag
- Always use `sacct` (not `squeue`) for final job status in one-shot mode
- Poll with `run_in_background` + `sleep 30 &&` to avoid blocking (one-shot)
- Never read full log files — use Read with offset/limit
- If the Slurm script path doesn't exist, report the error immediately
- For cluster hardware info (GPU types, partitions, constraints, storage
  paths), use the `trtllm-agent-toolkit:internal-env-info` skill
  when it is installed. If the skill is absent, skip the lookup silently and
  use the caller-supplied fields or the srun probe in Step 1 — do not report
  the missing skill as an error.
- Never auto-release an allocation after job completion — it should persist
  for potential future runs until walltime expires or user explicitly releases
- Must pass `-N <nodes>` to every direct `srun --jobid` call — without it,
  srun defaults to 1 node even in a multi-node allocation
- The container name used in setup steps must **exactly match** the
  `--container-name` in the `.slurm` script, otherwise pyxis will re-import
  the image and lose `--container-writable` state
