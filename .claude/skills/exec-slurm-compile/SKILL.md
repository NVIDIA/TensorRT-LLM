---
name: exec-slurm-compile
description: Compile TensorRT-LLM on a SLURM cluster. Covers submitting a batch job with a container image, monitoring the job, and verifying the build. Use when the user wants to compile TRT-LLM remotely via SLURM rather than on a local compute node.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Compile TensorRT-LLM on SLURM Cluster

Submit, monitor, and verify a TensorRT-LLM compilation job on a SLURM cluster using enroot containers.

## When to Use

| Scenario | Use This Skill? |
|----------|----------------|
| User wants to compile TRT-LLM on a SLURM cluster | Yes |
| User is already on a compute node and wants to compile | No — use `exec-local-compile` skill instead |

## Finding the Docker Image

The official Docker image tag for a given TensorRT-LLM version is recorded in the repo itself:

```
<repo_dir>/jenkins/current_image_tags.properties
```

Read this file to find the current image URL (e.g., `urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.12-py3-aarch64-ubuntu24.04-trt10.14.1.48-skip-tritondevel-202602011118-10901`).


## Pre-dumping the Container Image (enroot import)

SLURM clusters using enroot/pyxis require a `.sqsh` container image. To avoid download overhead at compile time, **pre-dump the image in advance** using the `enroot-import` companion script:

```bash
# Basic usage — submits a SLURM job on a CPU partition to import the image
enroot-import --partition cpu_datamover --debug <docker_image_url>
```

The script submits an `sbatch` job that runs `enroot import docker://<image_url>` and produces a `.sqsh` file in the current directory. The output on stdout is the SLURM job ID.

### enroot-import flags

| Flag | Description |
|------|-------------|
| `-p, --partition` | SLURM partition for the import job (use a CPU partition like `cpu_datamover`) |
| `-d, --debug` | Enable debug output and preserve the SLURM log (recommended) |
| `-o, --output` | Custom output path for the `.sqsh` file |
| `-A, --account` | SLURM account (defaults to user's first account) |
| `-t, --time` | Time limit for the import job (default: 1 hour) |
| `-n, --just-print` | Print the sbatch command without executing |
| `-J, --job-name` | Custom job name |

### enroot-import workflow

1. Read the image tag from `jenkins/current_image_tags.properties` in the TRT-LLM repo.
2. Run `enroot-import` to submit the import job:
   ```bash
   cd <directory_where_sqsh_should_be_stored>
   <path_to>/enroot-import --partition cpu_datamover --debug <image_url>
   ```
   **IMPORTANT:** Convert `urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:xxx` to `urm.nvidia.com#sw-tensorrt-docker/tensorrt-llm:xxx` to avoid credential issues.
3. Wait for the import job to complete (`squeue -j <job_id>`).
4. The resulting `.sqsh` file is the `container_image` used in the compile step.


## Prerequisites

The user must provide (or you must ask for) these values:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `container_image` | Path to `.sqsh` container image (see enroot import above) | `/path/to/pytorch.sqsh` |
| `repo_dir` | Path to the TensorRT-LLM repository | `/path/to/TensorRT-LLM` |
| `mount_dir` | Top-level directory to bind-mount into the container | `/shared/users` |
| `partition` | SLURM partition | `batch` |
| `account` | SLURM account | `my_account` |

Optional parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `jobname` | SLURM job name | `trtllm-compile.<username>` |
| `gpu_count` | Number of GPUs to request | `4` |
| `time_limit` | Job time limit | `02:00:00` |
| `arch` | GPU architecture(s) for `-a` flag | `100-real` |
| `extra_build_args` | Extra flags for `build_wheel.py` | (none) |

## Companion Scripts

This skill includes three companion scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `enroot-import` | Pre-dump a Docker image to `.sqsh` via a SLURM batch job |
| `submit_compile.sh` | Template for submitting the SLURM job — copy and customize |
| `compile.slurm` | SLURM batch script — launches the container and calls `compile.sh` |
| `compile.sh` | Runs inside the container — executes `build_wheel.py` |

Scripts directory: `skills/exec-slurm-compile/scripts/`

## Instructions

Follow these steps in order:

### Step 0: Resolve the Container Image (if needed)

If the user does not already have a `.sqsh` container image:

1. Read the Docker image tag from `<repo_dir>/jenkins/current_image_tags.properties`.
2. Use `enroot-import` to pre-dump it:
   ```bash
   cd <directory_for_sqsh_files>
   <scripts_dir>/enroot-import --partition cpu_datamover --debug <image_url>
   ```
3. Monitor the import job with `squeue -j <job_id>`.
4. Once complete, the `.sqsh` file path becomes the `container_image` parameter.

If the user already has a `.sqsh` file, skip this step.

### Step 1: Gather Information

Ask the user for any missing prerequisite values listed above. At minimum you need:
- `container_image` (or the Docker image URL — then run Step 0 first)
- `repo_dir`
- `mount_dir`
- `partition` and `account`

If the user has used this workflow before, check if previous values are stored in memory files.

### Step 2: Prepare the Scripts Directory

The compile scripts must be accessible from inside the container (i.e., under `mount_dir`). Either:

**Option A** — Copy companion scripts to a location under `mount_dir`:
```bash
scripts_dir=<mount_dir>/<username>/workspace/tensorrt_llm_scripts
mkdir -p ${scripts_dir}/log
cp skills/exec-slurm-compile/scripts/compile.sh ${scripts_dir}/
cp skills/exec-slurm-compile/scripts/compile.slurm ${scripts_dir}/
chmod +x ${scripts_dir}/compile.sh ${scripts_dir}/compile.slurm
```

**Option B** — If the user already has scripts at a known location, use those directly.

### Step 3: Submit the Job

Run `sbatch` from the login node (or a node with SLURM client access):

```bash
sbatch \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:<gpu_count> \
    --partition=<partition> \
    --account=<account> \
    --job-name=<jobname> \
    --time=<time_limit> \
    <scripts_dir>/compile.slurm \
    <container_image> <mount_dir> <scripts_dir> <repo_dir>
```

Capture and report the job ID from the `sbatch` output.

### Step 4: Monitor the Job (Proactive — Do NOT Wait for User)

**You MUST actively poll the job until it completes.** Do not submit and walk away.

```bash
# Check job status (repeat every 30-60 seconds)
squeue -j <job_id> -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

# Once running, periodically tail the log (do NOT use tail -f, use tail -30 instead)
tail -30 <scripts_dir>/log/compile_<job_id>.srun.log
```

**Monitoring loop:**
1. Poll `squeue -j <job_id>` to check state
2. If `PD` (pending) — report the reason, keep polling every 30-60s
3. If `R` (running) — tail the build log every 30-60s; look for `[XX%] Building`, errors, or completion
4. If the job disappears from `squeue`, it has finished — proceed to Step 5
5. If `F` (failed) — immediately read the full log and report the error

**Progress indicators to look for in the log:**
- `[XX%] Building CXX object...` — compilation progress
- `Linking CXX...` — link phase
- `FAILED:`, `error:`, `fatal error:` — build failure
- `Successfully built` — success

### Step 5: Verify the Build

Once the job completes, check for success:

```bash
# Check SLURM exit code
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed

# Check the build log for errors
tail -50 <scripts_dir>/log/compile_<job_id>.srun.log
```

A successful build ends with a message like `Successfully built tensorrt_llm` or completes without error.

## Common Build Flags Reference

| Flag | Description |
|------|-------------|
| `--trt_root /usr/local/tensorrt` | TensorRT installation path (standard in NVIDIA containers) |
| `--benchmarks` | Build the C++ benchmarks |
| `-a "100-real"` | Target architecture — `100` for Blackwell, `90` for Hopper, etc. |
| `--nvtx` | Enable NVTX markers for profiling |
| `--no-venv` | Skip virtual environment creation |
| `-ccache` | Use ccache to speed up recompilation |
| `--skip_building_wheel` | Build in-place without creating a wheel file |
| `-f` | Fast build — skip some kernels for faster dev compilation |
| `-c` | Clean build — wipe build directory before building |

Common architecture values:
- `"100-real"` — Blackwell (B200, GB200)
- `"90-real"` — Hopper (H100, H200)
- `"89-real"` — Ada Lovelace (L40S)
- `"80-real"` — Ampere (A100)
- `"90;100-real"` — Multiple architectures

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `sbatch: error: invalid partition` | Verify partition name with `sinfo -s` |
| `sbatch: error: invalid account` | Check available accounts with `sacctmgr show assoc user=$USER` |
| Container image not found | Verify the `.sqsh` path exists and is readable |
| Build fails with missing TensorRT | Ensure `--trt_root` points to the correct path inside the container |
| Build OOM (out of memory) | Reduce parallelism with `-j <N>` flag to `build_wheel.py` |
| `srun: error: Unable to create step` | The node may lack enroot/pyxis — check with cluster admin |
| Job stuck in `PD` state | Check `squeue -j <id> -o %R` for the reason (e.g., resource limits, priority) |
| `enroot import` fails with auth error | Check `~/.config/enroot/.credentials` has the correct registry credentials |
| `enroot import` produces empty/corrupt `.sqsh` | Re-run with `--debug` and check the SLURM log; verify the image URL has no `https://` prefix |
| Weird compile issues | Retry with a clean build (`-c` flag) |
| `QOSGrpNodeLimit` shown in `NODELIST(REASON)` | Not a blocker, just wait for the job to get scheduled |

## Example Interaction

**User**: "Compile TRT-LLM on the OCI cluster"

**Agent actions**:
1. Ask for container image path, repo path, mount dir (if not known)
2. Confirm partition/account for OCI cluster
3. Copy scripts to accessible location under mount_dir
4. Submit with `sbatch`
5. Report job ID
6. Monitor with `squeue` until complete
7. Check logs and report success/failure
