---
name: trtllm-case-executor
description: >
  Run TensorRT-LLM test cases, benchmarks, evaluations, or custom scripts by
  checking the environment (local GPU or Slurm), selecting the appropriate
  Docker image, and executing either locally or via Slurm job submission.
  Accepts pre-built command strings — command construction for trtllm-bench,
  trtllm-eval, and test_perf_sanity is handled upstream by the caller
  (e.g. trtllm-test-specialist using build_test_command.py).
tags:
  - tensorrt-llm
  - testing
  - execution
  - slurm
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

This skill executes TensorRT-LLM test commands by checking the environment and
dispatching to the appropriate executor. All commands are accepted as pre-built
strings — this skill does **not** build or modify commands.

**Four workflows** (classified in this order — first match wins):

1. **Custom** — `custom_cmd` or `custom_script` provided. Runs arbitrary commands via `slurm_run_custom.sh`.
2. **Bench** — `bench_cmd` provided. Accepts a pre-built `trtllm-bench` command chain. Parses `--tp`/`--pp` for `required_devices` only; does not modify the command.
3. **Eval** — `eval_cmd` provided. Accepts a pre-built `trtllm-eval` command. Parses `--tp_size`/`--pp_size` for `required_devices` only; does not modify the command.
4. **Pytest / test_cmd** — `test_cmd` provided. Handles all other commands: non-perf pytest, perf-sanity (`test_e2e[...]`), python scripts, and anything else. When `perf_config_yaml` is provided alongside a perf-sanity `test_cmd`, it is passed directly to the script builder for node/GPU sizing; no auto-derivation from the test ID is performed.

---

## Inputs

### Core Inputs

| Input | Description | Required | Example |
|-------|-------------|----------|---------|
| `test_cmd` | Test command to run inside the container (pytest, python, perf_sanity, etc.) | **Yes** (pytest/test_cmd) | `pytest tests/integration/defs/perf/test_perf_sanity.py -v "test_e2e[aggr-gpt_oss-gpt_oss_tp4ep4pp2_mtp1]"` |
| `bench_cmd` | Pre-built `trtllm-bench` command string. May chain `prepare-dataset` and `throughput`/`latency` with `&&`. Mutually exclusive with `test_cmd`/`eval_cmd`/`custom_cmd`. | **Yes** (bench) | `trtllm-bench ... prepare-dataset ... && trtllm-bench ... throughput ...` |
| `eval_cmd` | Pre-built `trtllm-eval` command string. Mutually exclusive with `test_cmd`/`bench_cmd`/`custom_cmd`. | **Yes** (eval) | `trtllm-eval --model /models/llama --tp_size 4 gsm8k` |
| `custom_cmd` | Arbitrary command to run inside the container. Mutually exclusive with `test_cmd`. | **Yes** (custom) | `python3 examples/run_inference.py --model llama` |
| `custom_script` | Host script path — takes precedence over `custom_cmd` (becomes `bash <script>`). Auto-mounts parent dir. | No | `/home/user/my_benchmark.sh` |
| `custom_env` | Space-separated `KEY=VALUE` pairs for custom workflow only | No | `BATCH_SIZE=32 NUM_WORKERS=4` |
| `perf_config_yaml` | Explicit path to a perf-sanity config YAML. When provided, passed directly to the script builder for node/GPU sizing — no auto-derivation from the test ID. | No (perf_sanity) | `/work/gpt_oss_120b_fp4.yaml` |
| `model_name` | Short name for container/job naming. Auto-derived from command if not provided. | No | `llama_7b` |
| `required_devices` | GPU count. Auto-parsed from `--tp`/`--pp` (bench) or `--tp_size`/`--pp_size` (eval) if not provided. Default: `1`. | No | `8` |
| `total_required_devices` | Total GPU processes across all nodes. If not provided, derived per the rules below. | No | `8` |
| `required_devices_per_node` | GPU processes per node. If not provided, derived per the rules below. | No | `4` |
| `skip_install` | Skip `pip install -e .` when image already has TRT-LLM. Default: `false`. | No | `true` |
| `build_project` | Build TRT-LLM from source before running the test. Default: `true`. Set to `false` to skip the build step when the image already contains a current build. | No | `false` |
| `container_image` | Override container image URI. If unset, case-executor resolves from `<REPO_ROOT>/jenkins/current_image_tags.properties` using the CPU arch derived from `device_type` (or local env-check). The resolved value is forwarded to every downstream consumer in `job_spec.json` — none of them re-resolve. For an example URI shape, delegate to `internal-env-info` for the `container_registry_example` key when that skill is installed; otherwise ask the user for a sample. | No | (delegated) |
| `monitor_timeout` | Wall-clock limit. Accepts `1h`, `30m`, `3600s`, or `HH:MM:SS`. Default: `1h`. | No | `2h` |

### Environment & Slurm Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `node_count` | Slurm nodes. Computed by this skill as `ceil(total_required_devices / required_devices_per_node)` and forwarded in `job_spec.json` to every downstream consumer (`trtllm-test-script-builder`, `exec-remote-slurm`). Caller may override only when an explicit value is required. | computed |
| `partition` | Slurm partition | *(required for Slurm)* |
| `account` | Slurm account | resolution chain (applied in [Step 2.5](#step-25-detect-slurm-account-and-valid-partitions-slurm-scenarios-only)): caller input → `default_account` from `detect_slurm_env.sh` JSON → delegate to `internal-env-info` for the `default_account` key when that skill is installed → otherwise ask the user. No NVIDIA-internal default is hard-coded in this skill. |
| `time_limit` | Slurm walltime | `02:00:00` |
| `device_type` | Required GPU type (e.g., `B200`, `GB200`). Auto-routes to remote cluster if local doesn't match. | — |
| `checkpoint_path` | Host model checkpoint path | — |
| `llm_models_root` | Models root inside container | caller input → delegate to `internal-env-info` for the `default_llm_models_root` key when that skill is installed → otherwise ask the user. No NVIDIA-internal default is hard-coded in this skill. |

### Remote Slurm Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `slurm_cluster` | Remote cluster name — skips local env-check, looks up per-cluster info (`mfa_style`, `default_models_repo`, `default_user_root_dir`, `gpus_per_node`) via `internal-env-info` skill. Connection fields (`ssh_host`, `partition`, `account`, `mounts`, etc.) come from caller-supplied inputs in `job_spec.json` (or `internal-env-info` defaults / ask-the-user fallbacks). When `internal-env-info` is not installed, the executor uses whatever explicit cluster fields the caller passed in `job_spec.json`. | — |
| `ssh_host` | Full SSH destination override. When unset, `exec-remote-slurm` Step 2.0 delegates to `internal-env-info` for the `ssh_host_pattern` key to construct the destination from `<slurm_user>` and `cluster_name`; it delegates to the same skill for `ssh_host_example` when documentation needs a sample. When `internal-env-info` is not installed and the user did not provide `ssh_host`, **ask the user** to supply `ssh_host` directly — this skill does not hard-code an NVIDIA-internal login-host pattern. `mfa_style` itself comes from `internal-env-info` Step 1b and is used by `exec-remote-slurm` Step 2 only to pick the SSH path (direct vs MFA), not to build the username; when that skill is absent, `mfa_style` defaults to `null` and `exec-remote-slurm` uses its probe-direct-first-then-MFA fall-back path. | constructed (delegated) |
| `user_name` | Raw login username forwarded to `internal-env-info` so it can build the SSH-ready `slurm_user` (applying whatever suffix that cluster's `mfa_style` requires). This skill never builds `slurm_user` itself — it only forwards `user_name` and reads the result back. | `$(whoami)` |
| `slurm_user` | SSH/Slurm username override. When set, used verbatim and the cluster-info value is discarded. When unset, this skill passes `user_name` to `internal-env-info` and copies the returned per-cluster `slurm_user` into `slurm_env` as-is. | (from cluster-info) |
| `slurm_password` | SSH password (**prompted if not provided** — never stored/logged) | *(prompted)* |
| `remote_cwd` | Remote working directory | `<REPO_ROOT>` |

### Device Count Auto-Parsing

If the caller passes `total_required_devices` and `required_devices_per_node` directly, use those values verbatim — skip all derivation below.

Otherwise derive all three fields (`required_devices`, `total_required_devices`, `required_devices_per_node`) using the rules in the table. Derivation order: `required_devices` first, then the two per-node/total fields.

| Workflow | `required_devices` | `total_required_devices` | `required_devices_per_node` |
|----------|--------------------|--------------------------|----------------------------|
| `bench_cmd` | `tp * pp` parsed from last `trtllm-bench` invocation | same as `required_devices` | same as `required_devices` (single-node only) |
| `eval_cmd` | `tp_size * pp_size` parsed from `eval_cmd` | same as `required_devices` | same as `required_devices` (single-node only) |
| `test_cmd` (non-perf pytest) | From `skip_less_device(N)` marker in test file; otherwise `1` | same as `required_devices` | same as `required_devices` (always single-node) |
| `test_cmd` (perf-sanity) with `perf_config_yaml` | `tensor_parallel_size * pipeline_parallel_size` for the matching `server_configs` entry | same as `required_devices` | `hardware.gpus_per_node` from the YAML |
| `test_cmd` (perf-sanity) without `perf_config_yaml` | default `1` | `1` | `1` |
| `custom_cmd` | `required_devices` input; default `1` | same as `required_devices` | same as `required_devices` |

**Perf-sanity YAML extraction procedure** (applies when `perf_config_yaml` is provided):

1. Read the YAML file and extract `hardware.gpus_per_node` → `required_devices_per_node`.
2. Identify the target server config name from `test_cmd` — it is the bracketed test ID suffix, e.g., `aggr-gpt_oss-gpt_oss_tp4ep4pp2_mtp1` in `test_e2e[aggr-gpt_oss-gpt_oss_tp4ep4pp2_mtp1]`. Match the trailing component against `server_configs[].name` in the YAML.
3. From the matched entry, read `tensor_parallel_size` (default `1`) and `pipeline_parallel_size` (default `1`).
4. `required_devices = tensor_parallel_size * pipeline_parallel_size`.
5. `total_required_devices = required_devices` (total GPU processes = world size).
6. If no matching entry is found, fall back to `required_devices = 1`, `total_required_devices = 1`; keep `required_devices_per_node` from step 1.

### Models Mount Resolution

Mount to `<llm_models_root>` (the resolved input — see the `llm_models_root` row above). Host-side source priority: `checkpoint_path` parent → `env_check.default_models_repo` → delegate to `internal-env-info` for the `default_models_repo_host` key when that skill is installed → otherwise stop and ask the user. No NVIDIA-specific path is hard-coded in this skill.

---

## Procedure

### Step 1: Create Work Directory and Classify

```bash
WORK_DIR="<REPO_ROOT>/work_dirs/<model_name>_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"
```

Classify workflow (custom → bench → eval → test_cmd). Derive `model_name`, `required_devices`, `total_required_devices`, and `required_devices_per_node` per the [Device Count Auto-Parsing](#device-count-auto-parsing) rules above. If the caller already passed `total_required_devices` and `required_devices_per_node`, skip derivation and use those values directly.

Then compute `node_count = ceil(total_required_devices / required_devices_per_node)` (use the caller's value only when explicitly provided). This skill is the **single source of truth for `node_count`** — every downstream consumer reads it from `job_spec.json` and must not recompute. Floor-clamp to `1` when both totals are `1`.

Build the resolved `job_name` here too:

- `detail = workflow_type` (`pytest`, `bench`, `eval`, `custom`) — or a more specific token derived from `model_name` when available.
- `job_name = f"{account}.{detail}"`.

This skill is the **single source of truth for `job_name`** — `exec-remote-slurm` (and any other downstream consumer) takes the resolved string from `job_spec.json` and substitutes it directly into `-J <job_name>`. The convention `<account>.<detail>` is documented and applied here only.

**Remote work_dir derivation (remote_slurm scenarios only):** Compute the remote-side equivalents of `WORK_DIR` so the remote layout mirrors the local layout (`<REPO_ROOT>/work_dirs/<basename>`):

```text
repo_dir_name      = basename(repo_url stripped of .git)         # e.g., "TensorRT-LLM"
remote_repo_path   = <default_user_root_dir>/<repo_dir_name>
remote_work_dir    = <remote_repo_path>/work_dirs/<basename(WORK_DIR)>
```

`default_user_root_dir` comes from the cluster config (resolved by `internal-env-info` cluster-info modes with `<user_name>` substituted). When that skill is not installed, fall back in this order: caller-supplied `remote_cwd` or explicit `remote_repo_path` in `job_spec.json` → otherwise stop and ask the user. The `default_user_root_dir_example` key under `internal-env-info`'s default-values section is documentation-only — usable only if the cluster actually matches that layout. Do not error on a missing optional skill; just route to the next source. This skill is the **single source of truth for `remote_work_dir` and `remote_repo_path`** — both are written verbatim into `job_spec.json` and every downstream consumer (`trtllm-test-script-builder`, `exec-remote-slurm`) must read them from the manifest, never recompute.

### Step 2: Detect Environment

This skill **never parses cluster connection YAML**. It only decides local vs remote and forwards inputs downstream. YAML resolution is owned by `exec-remote-slurm`.

**Optional dependency note.** If `skills/internal-env-info/` is not installed in the toolkit, the routing below still works using only env-check output and the caller's explicit inputs. Downstream executors handle the missing skill gracefully (per their own optional-dependency rubrics). Do **not** error out here when the skill is absent.

1. Run `trtllm-agent-toolkit:exec-env-check` with `required_devices`. Returns: `scenario`, `available_gpus`, `device_type`, `gpus_per_node`, `default_models_repo`, `cluster_name` (the name of the cluster this skill is running on, or `null` if not on a known cluster).
2. If `slurm_cluster` is provided, compare it to `env_check.cluster_name`:
   - **Match** (same cluster, case-insensitive) → `local_slurm`. Continue with the local env-check output; do not call `internal-env-info`.
   - **Mismatch** (or `env_check.cluster_name` is `null`) → `remote_slurm` with explicit cluster. Forward `slurm_cluster` (the name only) to `exec-remote-slurm`; that skill resolves per-cluster info via `internal-env-info` and reads connection fields from `job_spec.json`.
3. Otherwise (no `slurm_cluster`): if the `device_type` input doesn't match `env_check.device_type`, route to `remote_slurm` with **auto-select**. Forward `device_type`, `total_required_devices`, and `required_devices_per_node` to `exec-remote-slurm`, which queries `internal-env-info` in catalog mode and picks the cluster.
4. Otherwise map env-check to local scenarios: `satisfied, local, docker` → `local_docker` | `satisfied, local, direct` → `local_direct` | `satisfied, slurm, local` → `local_slurm` | `not_satisfied` → fall through to step 3's auto-select path.
5. Prompt for `slurm_password` if not provided (any `remote_slurm` path).

### Step 2.5: Detect SLURM Account and Valid Partitions (slurm scenarios only)

**MANDATORY before Step 4.** For every `local_slurm` and `remote_slurm` run, this step
**must complete successfully** before invoking `trtllm-test-script-builder`. The script
builder relies on `account`, `partition`, and the validated `slurm_env` already being
present in `job_spec.json` — it must not be invoked with caller-supplied / default
values that haven't been verified against the cluster.

**Do NOT skip Step 2.5 just because case-executor is running on a host that cannot
SSH directly**. In that case use **Option A (delegation)** below
— dispatch a `exec-remote-slurm` preflight subagent that performs the SSH
and runs the script over it. Skipping detection and falling back to defaults is
forbidden — defaults can produce a script that submits to a non-existent partition
or an account the user has no association with, which only fails after the SLURM
queue accepts it.

For `local_slurm` and `remote_slurm`, run `scripts/detect_slurm_env.sh` to populate `account`, the list of valid partitions, per-partition hardware (arch / GRES / gpus_per_node), and the cluster's PMIx plugin in **one pass**. **Skip this step entirely for `local_docker` and `local_direct`** — they don't use SLURM.

The script (shipped with this skill) wraps `sacctmgr`, `sinfo`, `scontrol show node`, and `srun --mpi=list` and emits a single JSON document of the form:

```json
{
  "user": "<whoami>",
  "default_account": "<first-account-or-null>",
  "accounts": ["acct1", "acct2", "..."],
  "default_partition": "<partition-marked-default-or-null>",
  "partitions": [
    {
      "name": "batch",
      "state": "up",
      "time_limit": "08:00:00",
      "nodes": "32",
      "arch": "x86_64",
      "gres": "gpu:8",
      "gpus_per_node": 8,
      "requires_gres": true
    },
    ...
  ],
  "pmix": {"available": ["pmix", "pmix_v3", "pmix_v5"], "preferred": "pmix_v5"},
  "errors": ["..."]
}
```

`arch`, `gres`, `gpus_per_node`, `requires_gres`, and the top-level `pmix` block are consumed downstream by `trtllm-test-script-builder` (Docker image arch selection, GRES handling, `--mpi=` flag) and the slurm executors (compile-step arch lookup). They must **not** be re-detected downstream.

#### `local_slurm`: run the script directly

Execute on the host where case-executor is running (the login or compute node):

```bash
SLURM_ENV_JSON=$(bash "<SKILL_DIR>/scripts/detect_slurm_env.sh" --format=json)
echo "$SLURM_ENV_JSON" > "<WORK_DIR>/slurm_env.json"
```

#### `remote_slurm`: reuse `exec-remote-slurm` to run via SSH

Reuse the SSH connection logic that `exec-remote-slurm` already owns — do **not** re-implement SSH host construction or preflight here. Two equivalent options:

**Option A (preferred — delegation):** Dispatch a small `exec-remote-slurm` subagent in "preflight + one-off command" mode that:
1. Resolves the cluster (its Step 1) — produces `cluster_name`, `mfa_style`, `slurm_user`, etc.
2. Constructs `<ssh_host>` per its Step 2.0 by delegating to `internal-env-info` for the `ssh_host_pattern` key (the single-line snippet that concatenates the already-suffixed `slurm_user` with `cluster_name` into `SSH_HOST`). When the skill is not installed and the user did not provide `ssh_host`, **ask the user** for it — this skill does not hard-code an NVIDIA-internal login-host pattern.
3. Runs SSH preflight per its Step 2a/2b/2c keyed on `mfa_style`.
4. Streams the local script over SSH and captures the JSON output:
   ```bash
   <ssh_cmd> <ssh_host> "bash -s -- --format=json" < "<SKILL_DIR>/scripts/detect_slurm_env.sh"
   ```
5. Returns the captured stdout to case-executor.

**Option B (inline — when an SSH session is already established by an earlier exec-remote-slurm invocation in the same conversation):** Just run the script over the existing `<ssh_cmd>` socket:
```bash
SLURM_ENV_JSON=$(<ssh_cmd> <ssh_host> "bash -s -- --format=json" < "<SKILL_DIR>/scripts/detect_slurm_env.sh")
```

In either case, save the JSON to `<WORK_DIR>/slurm_env.json`.

#### Resolve `account` and `partition` from the JSON

Apply this priority order (do not silently fall back past the user input):

| Field | Resolution |
|-------|------------|
| `account` | caller input → `default_account` from JSON → delegate to `internal-env-info` for the `default_account` key when that skill is installed → otherwise **stop and ask the user**. Do not hard-code an NVIDIA-internal account in this skill. |
| `partition` | caller input → `default_partition` from JSON → **stop and ask the user** if both are absent |

**Validation:** If the caller provided `partition` and the JSON's `partitions[].name` list does not include it, surface the mismatch (list available names) and ask the user before proceeding. Same for `account` vs `accounts[]`.

The resolved `account` and `partition` are written into `job_spec.json` so `trtllm-test-script-builder` and the executors can read them without re-detecting. Also include the full parsed JSON as the `slurm_env` object in `job_spec.json` — downstream consumers look up per-partition `arch` / `gres` / `gpus_per_node` / `requires_gres` (e.g. `slurm_env.partitions[name=<partition>].arch`) and the cluster's `slurm_env.pmix.preferred` from there. The on-disk `<WORK_DIR>/slurm_env.json` is the canonical artifact; `job_spec.slurm_env` is the parsed convenience copy.

If the script exits non-zero, surface the `errors[]` array to the user before continuing — don't silently fall back to defaults when SLURM detection itself failed. If a specific failure mode is "SLURM client tools not found" (exit 1), forward the empty `slurm_env` so downstream skills know to consult `internal-env-info` for arch and skip pmix.

#### Merge cluster YAML policy fields into `slurm_env`

After the detection JSON is on disk, enrich it with the cluster's site-policy fields so that `slurm_env` is the **single object** that downstream skills (especially `trtllm-test-script-builder`) consume.

Source of the policy fields:

- **Known cluster (env-check returned a non-null `cluster_name`, or the caller supplied `slurm_cluster`):** invoke `trtllm-agent-toolkit:internal-env-info` in single-cluster mode, passing the `user_name` input (default `$(whoami)`). The selected cluster entry returns `cluster_name`, `mfa_style`, and the already-suffixed `slurm_user` (the skill applies its `slurm_user` construction rule against the cluster's own `mfa_style`). **Copy `slurm_user` into `slurm_env` verbatim — this skill never builds, suffixes, or re-derives `slurm_user` itself.** If the caller passed an explicit `slurm_user` in `job_spec.json`, that override wins and the cluster-info value is discarded; otherwise the cluster-info value is the only source. For the remaining policy fields (`mounts`, `ssh_host`, `remote_cwd`), follow the same fallback chain documented above for `default_user_root_dir`: caller-supplied input in `job_spec.json` → delegate to `internal-env-info` when that skill is installed → otherwise stop and ask the user. There is no per-cluster connection-config file to parse. `container_image` is owned by Step 3.
- **Unknown cluster (no match in `internal-env-info`, or the skill is not installed):** leave the policy fields absent / `null`. Downstream consumers must handle the absence (e.g., script builder falls back to caller-supplied `checkpoint_path` / `models_path` for mounts).

Merge those fields into the same JSON document and rewrite `<WORK_DIR>/slurm_env.json`. The post-merge shape is:

```json
{
  "user": "<whoami>",
  "default_account": "...", "accounts": [...],
  "default_partition": "...", "partitions": [ { "name": "...", "arch": "...", "gres": "...", "gpus_per_node": <int>, "requires_gres": <bool>, ... } ],
  "pmix": {"available": [...], "preferred": "..."},
  "errors": [...],

  "cluster_name": "<cluster>|null",
  "mounts": ["host:container", "..."]      | null,
  "ssh_host": "<remote_slurm only>"        | null,
  "slurm_user": "<remote_slurm only; copied verbatim from internal-env-info (or caller override); never built here>" | null,
  "remote_cwd": "<remote_slurm only>"      | null,
  "mfa_style": <true|false|null>
}
```

Refresh `job_spec.slurm_env` to the merged object. 

### Step 3: Resolve Container Image and Forward Build Inputs

Case-executor does **not** build, but it **does** resolve the container image once so that the test-script-builder, the run-time executor, and (for `remote_slurm`) the remote build job all use the same image.

**Resolution rules (in priority order):**

1. If the caller provided `container_image` as input → use it verbatim.
2. Else read `<REPO_ROOT>/jenkins/current_image_tags.properties` and select by CPU arch:
   - `aarch64` (when `device_type` starts with `GB` / `GH`, or local env-check reports an `aarch64` host) → `LLM_SBSA_DOCKER_IMAGE`.
   - otherwise → `LLM_DOCKER_IMAGE` (x86_64).

Store the resolved URI verbatim in `job_spec.json` as `container_image`. Transport-specific URL rewrites happen at use-time inside the consumer — when an enroot-based executor needs the rewrite, it delegates to `internal-env-info` for the `enroot_uri_rewrite` key (which provides the canonical recipe and example). Case-executor itself always stores the canonical form (no `#`).

**Forwarded fields in `job_spec.json`:**

- `container_image` — resolved image URI (canonical form, no enroot rewrite).
- `build_project` (default `true`) — executors skip their build step when `false`.
- `device_type` — for `local_*` from env-check; for `remote_slurm` from the resolved cluster.
- `repo_root`.

Each executor's build step (and the remote build job) reads `container_image` directly from `job_spec.json` and applies any transport-specific transform on the way to `--container-image=`.

If an executor reports `BUILD_FAILED`, stop and surface the failure to the user without dispatching the workload.

The `local_direct` scenario has no executor SKILL — its build is inlined inside the `local_direct` block in Step 5, and it does not use a container.

### Step 4: Build Script

> **Precondition (slurm scenarios only): Step 2.5 must have completed.** Before invoking the
> script builder for `local_slurm` / `remote_slurm`, verify that `<WORK_DIR>/slurm_env.json`
> exists and that `account` + `partition` were resolved (and validated against the cluster's
> `accounts[]` / `partitions[].name`). If `slurm_env.json` is missing or `errors[]` from the
> detection script is non-empty and unresolved, **stop and go back to Step 2.5** — do not
> invoke the script builder with unvalidated SLURM parameters.

Invoke `trtllm-test-script-builder` skill with all inputs + classified workflow type + execution scenario. Pass `work_dir`, `repo_root`, `skill_dir`, `perf_config_yaml` (if provided), `container_image` (resolved in Step 3), `total_required_devices`, `required_devices_per_node`, `node_count` (already computed in Step 1), and — for slurm scenarios — the `account` and `partition` resolved in Step 2.5 along with `slurm_env_file=<WORK_DIR>/slurm_env.json` (the path to the canonical JSON written in Step 2.5, with cluster-policy fields merged in; the script builder loads it in its Step 1 and reads `mounts` / per-partition `gpus_per_node` from there). It still resolves mounts and generates:

- `<WORK_DIR>/job_spec.json` — manifest with all resolved values (including `node_count` and `container_image`)
- `<WORK_DIR>/<MODEL_NAME>_<type>.slurm` (Slurm scenarios) or `docker_cmd.sh` (for `local_docker`)

For `local_direct`, **skip the script builder entirely**. Write a minimal `<WORK_DIR>/job_spec.json`:

- `scenario: local_direct`
- `direct_cmd`: the raw `test_cmd` / `eval_cmd` / `bench_cmd` / `custom_cmd` (use `bash <custom_script>` if `custom_script` was given)
- `custom_env`: the `KEY=VALUE` pairs to export (if provided)
- `monitor_timeout_seconds`: normalized wall-clock limit (default `3600`)
- `work_dir`, `log_file`, `model_name`, `workflow_type`, `success_patterns`, `failure_patterns`

If the skill reports an error, stop and report to the user.

### Step 5: Dispatch to Executor Subagent

Read `<WORK_DIR>/job_spec.json` and dispatch by `job_spec.scenario`. The scenario was decided in Step 2 — **do not re-derive or re-map it here**; the per-scenario blocks below are dispatch *implementations* for each value of `job_spec.scenario`, not a second copy of the routing rule.

**Hang detection + task timeout:** Every executor enforces hang detection (poll log for `hang detected`, case-insensitive → `HANG_DETECTED`) and `monitor_timeout_seconds` (overall wall-clock limit → `TIMEOUT`). The policy and implementation live in each executor's SKILL; case-executor does not restate them per dispatch. Before spawning, normalize `monitor_timeout` to seconds (`1h` → `3600`, `30m` → `1800`, `HH:MM:SS` → seconds) and include the value in `job_spec.json` as `monitor_timeout_seconds`. Default `3600` (1h).

**For `local_docker`:**

Docker is available; the workload runs locally and is short-lived enough that a separate subagent context buys nothing. Do **not** spawn a subagent — load the `exec-local-docker` skill into this context and follow its procedure with the fields from `<WORK_DIR>/job_spec.json`:

```
Skill(skill="trtllm-agent-toolkit:exec-local-docker", args="""
Job spec (from <WORK_DIR>/job_spec.json):
- docker_cmd, work_dir, log_file, model_name, workflow_type
- success_patterns, failure_patterns, monitor_timeout_seconds
- build_project, gpu_type, repo_root, container_image (consumed by the skill's optional Step 0 build pre-step)
""")
```

When the skill returns, treat its result (status, exit code, log file, summary, errors) as this dispatch's outcome and proceed to Step 6.

**For `local_direct`:**

Docker is not available on this host. Do **not** spawn an executor subagent. Run the command in place:

```bash
cd "<REPO_ROOT>"
[ -n "$CUSTOM_ENV" ] && export $CUSTOM_ENV

timeout --kill-after=30s "${MONITOR_TIMEOUT_SECONDS}s" bash -c '<direct_cmd>' 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
```

Hang detection: monitor `$LOG_FILE` in the background (every 10s, case-insensitive grep for `hang detected`). If found, send `SIGTERM` to the command's process group and report `HANG_DETECTED`.

Task timeout: `timeout` enforces `monitor_timeout_seconds` (default `3600`). If `EXIT_CODE` is `124` (or `137` after the `--kill-after` grace period), report `TIMEOUT`.

Map the result: `EXIT_CODE=0` → `PASSED`; otherwise apply the workflow's `failure_patterns` / `success_patterns` to `$LOG_FILE` to pick `FAILED` / `ERROR` / `TIMEOUT` / `OUT_OF_MEMORY`, etc.

**For `local_slurm`:**
```
Agent(subagent_type="exec-local-slurm", prompt="""
Run the job described in <WORK_DIR>/job_spec.json.

Job spec (from <WORK_DIR>/job_spec.json):
- script_path, work_dir, model_name, workflow_type
- success_patterns, failure_patterns, log_file_pattern, monitor_timeout_seconds
""")
```

**For `remote_slurm`:**
7/600
Before spawning the executor, resolve the local repo URL and branch:

```bash
REPO_URL=$(git -C "<REPO_ROOT>" remote get-url origin 2>/dev/null)
REPO_BRANCH=$(git -C "<REPO_ROOT>" rev-parse --abbrev-ref HEAD 2>/dev/null)
```

Include these in `job_spec.json` as `repo_url` and `repo_branch`.

```
Agent(subagent_type="exec-remote-slurm", prompt="""
Run the job described in <WORK_DIR>/job_spec.json.

Job spec (from <WORK_DIR>/job_spec.json):
- script_path, script_name, work_dir, model_name, workflow_type
- success_patterns, failure_patterns, log_file_pattern
- slurm_cluster (cluster name only, when explicit; absent for auto-select). The remote executor resolves per-cluster info via internal-env-info; connection fields come from job_spec.json.
- ssh_host, slurm_user, remote_cwd, remote_work_dir (input overrides; absent required fields cause the remote executor to stop and ask the user)
- slurm_password, extra_files
- repo_url: Git remote URL of the local TensorRT-LLM repo (for remote environment setup)
- repo_branch: Git branch to check out on the remote cluster
- device_type, total_required_devices, required_devices_per_node: hardware constraints. When slurm_cluster is absent, the executor uses device_type and required_devices_per_node to auto-select a cluster from internal-env-info catalog mode.
- container_image: resolved by case-executor (Step 3) — the executor uses this directly for both the build job and the run job. Apply transport-specific URL rewrites (e.g., enroot's `/` → `#`) at use-time. Never re-grep `current_image_tags.properties`.
- node_count: precomputed by case-executor (Step 1) — the executor uses this directly for `--nodes`, never recomputes from totals.
- job_name: precomputed by case-executor (Step 1) — the executor substitutes this into `-J <job_name>` directly, never reconstructs from `account` / `detail`.
- monitor_timeout_seconds
""")
```

### Step 6: Report Results

Relay the subagent result to the user with: task type, Docker image, command/script path, status (`PASSED`/`FAILED`/`TIMEOUT`/`CANCELLED`/`HANG_DETECTED`/etc.), exit code, log file paths, summary, errors (if any), work directory, and Slurm details (if applicable).

---

## Failure Diagnosis

| Symptom | Cause | Action |
|---------|-------|--------|
| Exit 1 + `FAILED` | Pytest assertion failure | Show failing tests and messages |
| Exit 2 + `ERROR` | Collection/import failure | Check first 50 lines for ImportError |
| `TIMEOUT` | Exceeded time limit | Increase `time_limit` |
| `OUT_OF_MEMORY` | OOM | Reduce batch size or increase nodes |
| `CANCELLED` | Preempted or user-cancelled | `sacct --format=JobID,State,Reason` |
| Empty output file | Container failed to start | Check `.err` for mount/image errors |
| `hang detected` in log | GPU/NCCL deadlock | Auto-terminated; check log context for root cause |
| `Invalid generic resource` | Cluster doesn't support `--gres` | Omit `--gres` |
| SSH `Broken pipe` | ControlMaster died | Subagent re-establishes |

## Reference Files

| File | Purpose |
|------|---------|
| `scripts/detect_slurm_env.sh` | One-pass SLURM env probe: accounts, partitions (with per-partition `arch` / `gres` / `gpus_per_node` / `requires_gres`), and the cluster's PMIx plugin (`pmix.available` / `pmix.preferred`). JSON or text output. Used in Step 2.5 (run locally for `local_slurm`; over SSH via `exec-remote-slurm` for `remote_slurm`). The emitted JSON is the single source of truth — downstream skills (`trtllm-test-script-builder`, the slurm executors) read it via `job_spec.slurm_env` and do **not** re-detect. |
| `<repo_root_or_remote_repo_path>/jenkins/scripts/perf/local/submit.py` (in the TRT-LLM repo) | Generates perf-sanity launch scripts from config YAMLs |
| `scripts/slurm_run_custom.sh` | Runs custom commands inside Slurm container |

### Skills & Subagents

| Skill | Role | Step |
|-------|------|------|
| `exec-env-check` | Detect GPUs + Slurm availability | 2 |
| `trtllm-test-script-builder` | Resolve params + generate script + `job_spec.json` | 4 |
| `exec-local-docker` | Local Docker execution | 5 |
| `exec-local-slurm` | Local Slurm submission + monitoring | 5 |
| `exec-remote-slurm` | Remote Slurm: SSH + submit + monitor | 5 |
