# Local SLURM Launch Scripts

## Overview

This directory contains scripts for running perf sanity tests locally via SLURM. The workflow has three steps:

1. **`submit.py`** generates a complete `slurm_launch.sh` script. It reads the test config YAML, detects aggregated vs disaggregated mode, and combines SBATCH parameters + environment variables + the appropriate draft template (`jenkins/scripts/perf/aggregated/slurm_launch_draft.sh` or `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh`) into a single launch script. A `test_list.txt` is also written to the work directory.

2. **`sbatch slurm_launch.sh`** submits the job to SLURM. Inside the launch script:
   - For **aggregated** mode, a single `srun` invokes `slurm_run.sh`.
   - For **disaggregated** mode, `srun` first runs `slurm_install.sh` on all nodes, then launches separate `srun` commands for gen workers, ctx workers, the disagg server, and the benchmark client.

3. **`slurm_install.sh`** handles build and installation inside the container. It optionally builds the TensorRT-LLM wheel (when `--build-wheel` is set) and then runs `pip install -e .` plus dev requirements. A lock-file mechanism ensures only one process per node performs the install while others wait.

4. **`slurm_run.sh`** runs the pytest command. In aggregated mode, it first sources `slurm_install.sh` to run the install step, then executes the pytest command. In disaggregated mode, the install has already been done by the launch script, so `slurm_run.sh` runs pytest directly.

```
submit.py
  |
  v
slurm_launch.sh  (generated)
  |
  |-- srun --> slurm_install.sh   (build wheel + pip install)
  |-- srun --> slurm_run.sh       (run pytest)
```

## Optional Arguments

- `--test-list`: Test string, e.g., `perf/test_perf_sanity.py::test_e2e[aggr-config-test_name]`. If both `--test-list` and `--config-file` are provided, `--test-list` takes precedence.
- `--config-file`: Path to config YAML file.
- `--test-name`: Test name (only used for aggregated mode when `--config-file` is provided).
- `--benchmark-mode`: `e2e` | `gen_only` | `ctx_only` (only used for a disagg `--config-file`; with `--test-list` the mode is read off the test id).
- `--time-breakdown`: Also record the per-request lifecycle breakdown. This adds the `time_breakdown` modifier segment to the generated test id (`disagg-e2e-time_breakdown-<yaml-stem>`); the modifier is orthogonal to `--benchmark-mode` and does not change the workload.
- `--time`: SLURM time limit (default: `02:00:00`).
- `--mounts`: Container mounts.
- `--work-dir`: Work directory (used for both workdir and container-workdir).
- `--draft-launch-sh`: Path to draft-launch.sh script.
- `--launch-sh`: Path to output launch.sh script.
- `--run-sh`: Path to slurm_run.sh script.
- `--install-sh`: Path to slurm_install.sh script.
- `--llm-src`: Path to LLM source code.
- `--build-wheel`: Add this flag to build the wheel before running tests.
- `--install-mode`: Installation mode - `source` (pip install -e ., default) or `wheel` (pip install *.whl).
- `--capture-nsys`: Add this flag to capture an nsys profile during the test run.
- `--nsys-start-stop`: Nsys start-stop range (default: `1-100`).
- `--ctx-nsys-start-stop`: CTX Worker Nsys start-stop range (default: `1-100`).
- `--gen-nsys-start-stop`: GEN Worker Nsys start-stop range (default: `1-100`).

`--image` can be obtained by:

```bash
# B200
image=$(grep LLM_DOCKER_IMAGE  $trtllm/jenkins/current_image_tags.properties | head -1 | awk -F "=" '{print $2}' )
image=$(echo $image | sed 's|urm.nvidia.com/|urm.nvidia.com#|g')
# GB200
image=$(grep LLM_SBSA_DOCKER_IMAGE  $trtllm/jenkins/current_image_tags.properties | head -1 | awk -F "=" '{print $2}' )
image=$(echo $image | sed 's|urm.nvidia.com/|urm.nvidia.com#|g')
```

## Cluster Settings

| Cluster | `--partition` | `--account` |
|---------|---------------|-------------|
| OCI | `batch` | `coreai_comparch_trtllm` |
| DLCluster | `gb200nvl72_preprod` | `coreai_comparch_trtllm` |

## Examples

### Aggregated Mode

```bash
python3 submit.py --test-list "perf/test_perf_sanity.py::test_e2e[aggr-deepseek_r1_fp4_v2_2_nodes_grace_blackwell-r1_fp4_v2_tep8_mtp3]" \
    --draft-launch-sh $trtllm/jenkins/scripts/perf/aggregated/slurm_launch_draft.sh \
    --launch-sh $work_dir/slurm_launch.sh \
    --install-sh $trtllm/jenkins/scripts/perf/local/slurm_install.sh \
    --run-sh $trtllm/jenkins/scripts/perf/local/slurm_run.sh \
    --llm-src $trtllm \
    --work-dir $work_dir \
    --partition $partition \
    --account $account \
    --job-name aggr_test \
    --image $image \
    --mounts $mounts \
    --llm-models-root $llm_models_path
```

### Disaggregated Mode

```bash
python3 submit.py --test-list "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-UCX]" \
    --draft-launch-sh $trtllm/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh \
    --launch-sh $work_dir/slurm_launch.sh \
    --install-sh $trtllm/jenkins/scripts/perf/local/slurm_install.sh \
    --run-sh $trtllm/jenkins/scripts/perf/local/slurm_run.sh \
    --llm-src $trtllm \
    --work-dir $work_dir \
    --partition $partition \
    --account $account \
    --job-name disagg_test \
    --image $image \
    --mounts $mounts \
    --llm-models-root $llm_models_path
```

---

## Running the AgentX perf-sanity lane

The AgentX client replays a recorded multi-turn conversation corpus for a fixed
wall-clock window (`AGENTX_DURATION`, default 3600 s) rather than a fixed number of
fixed-shape prompts, so `isl`/`osl`/`iterations` in the config are descriptive and
the run is judged on duration coverage, not a request count. Example test id
(`_upload` is stripped for local runs, so nothing reaches OpenSearch):

```
perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL]
```

### Environment variables

`submit.py` reads `EXTRA_CONTAINER_EXPORTS` -- a `;`-separated `KEY=VALUE` list --
at **generation time** and splices it into the four per-role env prefixes
(`CTX_WORKER_ENV_VARS`, `GEN_WORKER_ENV_VARS`, `SERVER_ENV_VARS`,
`BENCHMARK_ENV_VARS`). Export it before running `submit.py`, not at `sbatch` time:

```bash
export EXTRA_CONTAINER_EXPORTS="HF_HOME=$hf_cache;HF_HUB_CACHE=$hf_cache/hub;HF_DATASETS_CACHE=$hf_cache/datasets;PIP_CACHE_DIR=$work_dir/.pip-cache"
```

Set all three HF variables, not just `HF_HOME`: `submit.py` appends its own
`HF_HOME=/tmp/hf_home` **after** the splice for the ctx and gen worker roles, and
the later assignment wins. `HF_HUB_CACHE` and `HF_DATASETS_CACHE` are never
overridden and outrank `HF_HOME` in `huggingface_hub` / `datasets`, making the
result independent of splice order.

`dataset_file` in the config is an aiperf `--public-dataset` loader name, not a
path, and is fetched from Hugging Face at runtime. Either warm the cache above
before submitting, or confirm the compute nodes can reach the HF CDN.

The `AGENTX_*` knobs are not `submit.py` flags -- they come from `client_env_var`
in the config YAML, so exporting them in your shell has no effect. Edit the YAML
to change one. `AGENTX_DURATION` is the main cost knob; keep `AGENTX_SEED`
(default 42) fixed when comparing runs.

```yaml
client_env_var: 'AGENTX_MAX_CTX=996579 AGENTX_DURATION=3600 AGENTX_WARMUP_PER_LANE=3'
```

### Generate and submit

```bash
python3 submit.py \
    --test-list "perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8_gen1_dep8_eplb0_dspark3_ccb-NIXL]" \
    --draft-launch-sh $trtllm/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh \
    --launch-sh $work_dir/slurm_launch.sh \
    --install-sh $trtllm/jenkins/scripts/perf/local/slurm_install.sh \
    --run-sh $trtllm/jenkins/scripts/perf/local/slurm_run.sh \
    --llm-src $trtllm \
    --work-dir $work_dir \
    --llm-models-root $llm_models_path \
    --partition $partition \
    --account $account \
    --job-name agentx_test \
    --image $image \
    --mounts $mounts \
    --install-mode wheel \
    --wheel-path $wheel \
    --cluster-name $cluster

cd $work_dir && sbatch slurm_launch.sh
```

`--cluster-name` selects the UCX and env rules in `cluster_env.py`; an unmatched
(cluster, GPU) pair falls through to a catch-all that pins no transport, changing
performance silently. Use a fresh `--work-dir` every run -- a reused disaggregated
work directory reads stale hostname files and hangs. This lane takes 6 nodes /
24 GPUs and roughly 2.5 h including the wheel install, so allow a 4 h limit.

### Viewing the perf results

Artifacts land under `$work_dir/<case-name>/`:

| Path | Contents |
|---|---|
| `agentx.0.0/concurrency_1156/profile_export_aiperf.json` | metrics, percentiles, `submission_valid`, duration coverage (client log alongside in `logs/aiperf.log`) |
| `trtllm-benchmark.0.0.log` | human-readable metric block |
| `{ctx,gen}_server_*.log`, `disagg_server.log` | server logs |

Gate on the export rather than on the pytest exit status:

```bash
python3 -c "
import json
d = json.load(open('$work_dir/<case>/agentx.0.0/concurrency_1156/profile_export_aiperf.json'))
m = d['metadata']
print('submission_valid', m.get('submission_valid'))
print('was_cancelled   ', m.get('was_cancelled'))
print('errors          ', m.get('error_summary'))
for p in m.get('metric_duration_coverage') or []:
    print('coverage', p)
"
```

Coverage reports the TTFT and ITL sample ratios against the requested window, so a
run that only partly covered it is detectable instead of quietly averaging a
truncated segment. A green pytest summary alone is not a passing stage.
