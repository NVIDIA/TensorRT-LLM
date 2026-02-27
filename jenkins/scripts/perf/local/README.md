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
- `--time`: SLURM time limit (default: `02:00:00`).
- `--mounts`: Container mounts.
- `--work-dir`: Work directory (used for both workdir and container-workdir).
- `--draft-launch-sh`: Path to draft-launch.sh script.
- `--launch-sh`: Path to output launch.sh script.
- `--run-sh`: Path to slurm_run.sh script.
- `--install-sh`: Path to slurm_install.sh script.
- `--llm-src`: Path to LLM source code.
- `--build-wheel`: Add this flag to build the wheel before running tests.
- `--capture-nsys`: Add this flag to capture an nsys profile during the test run.
- `--nsys-start-stop`: Nsys start-stop range (default: `1-100`).

`--image` can be obtained by:

```bash
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
