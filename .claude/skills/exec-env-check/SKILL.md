---
name: exec-env-check
description: >-
  Check the local execution environment for GPU availability, Docker support,
  and Slurm access. Returns the execution scenario (`satisfied, local, docker`,
  `satisfied, local, direct`, `satisfied, slurm, local`, or `not_satisfied`),
  the number of available GPUs, and the GPU type. On Slurm login nodes
  without local GPUs, the cluster is identified by delegating the hostname
  to internal-env-info (hostname-based mode), which owns the
  hostname → cluster_name patterns; GPU type and gpus_per_node then come
  from that skill's reference files. If internal-env-info is not
  installed, the scenario falls back to `not_satisfied` without probing
  compute nodes via srun.
tags: [infrastructure, slurm, environment]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Environment Check

Detect whether the current machine can run a GPU workload locally (Docker) or via Slurm, and report hardware details.

## Input

| Field | Description | Required |
|-------|-------------|----------|
| `required_devices` | Minimum number of GPUs needed | Yes |
| `account` | Slurm account (unused for GPU probing, kept for compatibility) | No |

## Procedure

### 1. Check local GPUs

```bash
timeout 5 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

If this succeeds:
- Count the number of GPU lines → `available_gpus`
- Extract the GPU name from the first line → `device_type` (e.g., `NVIDIA B200`, `NVIDIA H100 80GB HBM3`)
- Normalize `device_type`: strip `NVIDIA ` prefix and trailing memory info to get the short name (e.g., `B200`, `H100`, `A100`, `L40S`, `RTX 6000`)

If `nvidia-smi` fails or returns no GPUs → `available_gpus = 0`, `device_type = null`.

### 2. Check if local GPUs are sufficient

If `available_gpus >= required_devices`, continue to step 2a to determine whether Docker is available on this host.

### 2a. Check Docker availability

```bash
command -v docker >/dev/null 2>&1 && timeout 3 docker info >/dev/null 2>&1
```

- If the command succeeds (Docker CLI exists **and** the daemon responds) → **Result**: `satisfied, local, docker`
- Otherwise (Docker CLI missing, daemon not running, or permission denied) → **Result**: `satisfied, local, direct`

Include `available_gpus` in the result. Do NOT include `device_type` — local execution does not need it.

### 3. Check Slurm availability

If local GPUs are insufficient (or as additional detection), check for Slurm:

```bash
which squeue 2>/dev/null && squeue --version 2>/dev/null
```

If Slurm is NOT available → go to step 5.

### 4. Resolve GPU type

**Optional dependency.** Before doing anything else in this step, check whether `skills/internal-env-info/` exists in the toolkit. If it does **not**, skip this step entirely and return `scenario: not_satisfied` with `available_gpus = 0`, `device_type = null`, `gpus_per_node = null`, `cluster_name = null`, `default_models_repo = null`, and `default_user_root_dir = null`. Do **not** report this as an error — the skill is an internal-only dependency.

When Slurm is available but local `nvidia-smi` returned no GPUs or `device_type` is null (login nodes typically have no GPUs), capture the hostname and delegate cluster identification to `internal-env-info`:

```bash
hostname -f 2>/dev/null || hostname
```

- Pass the captured hostname to `internal-env-info` in **hostname-based mode**. That skill owns the NVIDIA-internal login-host patterns and the hostname → `<cluster>` mapping; do **not** parse the hostname or hard-code any cluster identifier here.
- It returns the standard output template (`cluster_name`, `device_type`, `gpus_per_node`, `default_models_repo`, `default_user_root_dir`) plus supplementary field (`mfa_style`).
- Set `available_gpus` = `gpus_per_node` (if resolved).
- Set `cluster_name` = the value returned (a placeholder `<cluster>` token in this skill). The orchestrator uses this to fetch per-cluster info (`mfa_style`, `default_models_repo`, `default_user_root_dir`, `gpus_per_node`) from `internal-env-info`; connection fields (`mounts`, `ssh_host`, `partition`, etc.) come from caller-supplied inputs in `job_spec.json` (with `internal-env-info` default values / ask-the-user fallbacks).
- Set `default_user_root_dir` = the user root directory returned (with `<user_name>` substituted with the actual SLURM username). Set to `null` if not found.
- If `internal-env-info` returns `null` for `cluster_name` (no pattern matched), set `device_type = null`, `cluster_name = null`, `default_user_root_dir = null` and fall through to Step 5 (`not_satisfied`).

**Result**: `satisfied, slurm, local` with `device_type`, `gpus_per_node`, `cluster_name`, and `default_user_root_dir`.

### 5. Not satisfied

If neither local GPUs nor Slurm is available:
- **Result**: `not_satisfied`

## Output

Return a single structured result:

```
scenario: <satisfied, local, docker | satisfied, local, direct | satisfied, slurm, local | not_satisfied>
available_gpus: <N>
device_type: <short GPU/device name or null>
gpus_per_node: <N or null>
cluster_name: <cluster name or null>
default_models_repo: <host path to llm-models directory, or null>
default_user_root_dir: <per-user root directory on cluster storage, or null>
```

**Examples:**

```
scenario: satisfied, local, docker
available_gpus: 4
```

```
scenario: satisfied, local, direct
available_gpus: 4
```

```
scenario: satisfied, slurm, local
available_gpus: 4
device_type: B200
gpus_per_node: 4
cluster_name: <cluster>
default_user_root_dir: <default_user_root_dir>/<user_name>
```

```
scenario: not_satisfied
available_gpus: 0
device_type: null
cluster_name: null
```

## Rules

- Never install drivers or modify the system
- If `nvidia-smi` hangs, use a 5-second timeout: `timeout 5 nvidia-smi ...`
- GPU type is derived from the hostname by matching the cluster name via the `internal-env-info` skill when it is installed — no `srun` allocation needed. If that skill is absent, the scenario falls back to `not_satisfied` (see Step 4); do not error.
- Report the GPU type exactly as found in the mapping (do not guess or fabricate)
