# TensorRT-LLM Perf Sanity Test (`test_perf_sanity.py`)

Performance sanity testing scripts for TensorRT-LLM with configuration-driven test cases supporting single-node, multi-node aggregated, and multi-node disaggregated architectures.

This document serves as a reference for both developers and AI agents working with the perf sanity system.

For the underlying regression pipeline architecture (three-layer design, baseline calculation, how to add new test scripts), see [README_perf_regression_system.md](README_perf_regression_system.md).

## How `test_perf_sanity.py` Uses the Pipeline

`test_perf_sanity.py` is the top layer of the [three-layer perf regression system](README_perf_regression_system.md). It is responsible for:

1. **Parsing YAML configs** into `ServerConfig` / `ClientConfig` objects
2. **Running benchmarks** and collecting metric outputs
3. **Assembling `new_data_dict`** — a `Dict[int, dict]` mapping `cmd_idx` to data dicts containing:
   - Test config fields (`s_gpu_type`, `s_runtime`, `s_model_name`, `l_tp`, `l_concurrency`, etc.)
   - Metric fields (`d_seq_throughput`, `d_mean_ttft`, etc.)
   - `s_test_case_name` for human-readable identification
4. **Building `match_keys`** — the list of fields that uniquely identify a test case
5. **Calling `process_and_upload_test_results()`** with its own metric definitions

### Metric Definitions

| List | Count | Contents |
|------|-------|----------|
| `MAXIMIZE_METRICS` | 7 | Throughputs (`d_seq_throughput`, `d_token_throughput`, `d_total_token_throughput`, `d_user_throughput`) + TPOT (`d_mean_tpot`, `d_median_tpot`, `d_p99_tpot`) |
| `MINIMIZE_METRICS` | 9 | TTFT, ITL, E2EL latencies (mean/median/P99 for each) |
| `REGRESSION_METRICS` | 2 | `d_token_throughput`, `d_total_token_throughput` — only these gate pass/fail |

### Match Keys

Match keys differ by deployment mode:

- **Aggregated**: `["s_gpu_type", "s_runtime"]` + `ServerConfig.to_match_keys()` + `ClientConfig.to_match_keys()`
- **Disaggregated**: `["s_gpu_type", "s_runtime", "s_benchmark_mode", "l_num_ctx_servers", "l_num_gen_servers"]` + prefixed ctx/gen `ServerConfig.to_match_keys()` + `ClientConfig.to_match_keys()`

## Overview

- Run performance sanity benchmarks across multiple model configs
- Support three deployment architectures: single-node, multi-node aggregated, and multi-node disaggregated
- Manage test cases through YAML config files
- Automated resource calculation and job submission via SLURM

## System Architecture

### Key Scripts

| Script | Purpose | SLURM Launch Draft |
|--------|---------|-------------------|
| `tests/integration/defs/perf/test_perf_sanity.py` | Main pytest entry point for all perf sanity tests | N/A |
| `jenkins/scripts/perf/disaggregated/submit.py` | CI submission script (disaggregated tests only) | Uses `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` |
| `jenkins/scripts/perf/local/submit.py` | Local submission script (both aggregated and disaggregated tests) | Uses `jenkins/scripts/perf/aggregated/slurm_launch_draft.sh` or `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` |
| `jenkins/L0_Test.groovy` | Jenkins CI pipeline for test orchestration | N/A |

### SLURM Launch Script Generation

The submit scripts generate `slurm_launch.sh` from draft templates:

| Submit Script | Mode | Draft Template Used |
|---------------|------|---------------------|
| `jenkins/scripts/perf/disaggregated/submit.py` | Disaggregated (CI only) | `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` |
| `jenkins/scripts/perf/local/submit.py` | Aggregated (local) | `jenkins/scripts/perf/aggregated/slurm_launch_draft.sh` |
| `jenkins/scripts/perf/local/submit.py` | Disaggregated (local) | `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` |

## Environment Variables

The config folder paths can be overridden via environment variables. Both submit scripts (`local/submit.py` and `disaggregated/submit.py`) propagate these into the pytest execution environment.

| Variable | Default | Description |
|----------|---------|-------------|
| `AGG_CONFIG_FOLDER` | `tests/scripts/perf-sanity/aggregated` | Path to aggregated config YAML files |
| `DISAGG_CONFIG_FOLDER` | `tests/scripts/perf-sanity/disaggregated` | Path to disaggregated config YAML files |

**Example**: Run with custom config folders:
```bash
AGG_CONFIG_FOLDER=my/custom/agg DISAGG_CONFIG_FOLDER=my/custom/disagg \
  python jenkins/scripts/perf/local/submit.py ...
```

## Configuration Files

There are two modes for perf sanity tests: aggregated (aggr) and disaggregated (disagg).

### Aggregated Mode Config Files

**Location**: `tests/scripts/perf-sanity/aggregated`

**File Naming**: `xxx.yaml` where words are connected by `_` (underscore), not `-` (hyphen). The suffix indicates the GPU target:
- B200 configs: `xxx_blackwell.yaml` (e.g., `deepseek_r1_fp4_v2_blackwell.yaml`)
- GB200 configs: `xxx_grace_blackwell.yaml` (e.g., `deepseek_r1_fp4_v2_grace_blackwell.yaml`)
- Multi-node: add `_2_nodes` before the GPU suffix (e.g., `deepseek_r1_fp4_v2_2_nodes_grace_blackwell.yaml`)

**Server Config Rules**:
- Each `server_config` entry should have exactly **one** `client_config`. Use separate server configs for different ISL/OSL combinations.
- Server config `name` should include ISL and OSL (e.g., `llama70b_fp4_tp4_512_32`). Do **not** include adjustable parameters like `max_batch_size` or `max_num_tokens` in the name.

**Use Cases**:
- Single-node: Performance tests on a single server with multiple GPUs
- Multi-node: Model runs across multiple nodes with unified execution

### Disaggregated Mode Config Files

**Location**: `tests/scripts/perf-sanity/disaggregated`

**File Naming**: `xxx.yaml` (can contain `-` hyphen).

**Example**: `deepseek-r1-fp4_1k1k_ctx1_gen1_dep8_bs768_eplb0_mtp0_ccb-UCX.yaml`

**Use Case**: Disaggregated architecture where model runs across multiple nodes with separate context (prefill) and generation (decode) servers.

## Test Case Formats

In each test db yml file (with keyword `perf_sanity`), there are four test types:

### 1. Normal Aggregated Test

Uses aggregated config files from `tests/scripts/perf-sanity/aggregated`.

**Format**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-{agg config file base name}-{test name}]
```

**Example**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k1k]
```

### 2. Aggregated ctx_only Test

Uses disaggregated config files but runs context (prefill) phase only in aggregated mode.

**Format**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-ctx_only-{disagg config file base name}]
```

**Example**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-ctx_only-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]
```

### 3. Disaggregated gen_only Test

Uses disaggregated config files and runs generation (decode) phase only.

**Format**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-{disagg config file base name}]
```

**Example**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]
```

### 4. Disaggregated e2e Test

Uses disaggregated config files and runs full end-to-end disaggregated flow.

**Format**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-{disagg config file base name}]
```

**Example**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]
```

## CI Test Database

Test lists are defined in `tests/integration/test_lists/test-db/`.

### YAML File Naming Convention

| Test Type | File Pattern | Example |
|-----------|--------------|---------|
| Single-node aggregated | `l0_{gpu_type}_multi_gpus_perf_sanity.yml` | `l0_b200_multi_gpus_perf_sanity.yml` |
| Multi-node aggregated | `l0_{gpu_type}_multi_nodes_perf_sanity_node{node count}_gpu{gpu count per test}.yml` | `l0_b200_multi_nodes_perf_sanity_node2_gpu16.yml` |
| Multi-node disaggregated | `l0_{gpu_type}_multi_gpus_perf_sanity_ctx{ctx worker count}node{node count per ctx worker}_gpu{gpu count per ctx worker}_gen{gen worker count}node{node count per gen worker}_gpu{gen gpus per gen worker}.yml` | `l0_b200_multi_gpus_perf_sanity_ctx1node1_gpu8_gen1node1_gpu8.yml` |

### Jenkins Pipeline Configuration

Tests are defined in `jenkins/L0_Test.groovy` under the `launchTestJobs` function:

| Config Variable | Test Type |
|-----------------|-----------|
| `x86SlurmTestConfigs` | Single-node aggregated tests |
| `SBSASlurmTestConfigs` | Multi-node aggregated tests |
| `multiNodesSBSAConfigs` | Multi-node disaggregated tests |

## CI Stage Rules

### Test Batching Rules

| Test Type | Nodes per Test | Max Tests per Stage | Notes |
|-----------|----------------|---------------------|-------|
| Normal aggregated test | 1 | 6 | Multiple tests can share a stage |
| Aggregated ctx_only test | 1 | 6 | Multiple tests can share a stage |
| Normal aggregated test | > 1 | 1 | One test per stage |
| Aggregated ctx_only test | > 1 | 1 | One test per stage |
| Disaggregated gen_only test | Any | 1 | Always one test per stage |
| Disaggregated e2e test | Any | 1 | Always one test per stage |

**Important**: Pre-merge and post-merge tests must be in separate stages.

### GPU Hours Calculation

- Each CI stage runtime is approximately **1 hour**
- GPU hours = (number of stages) x (GPUs per stage) x 1 hour

**Example**: A test configuration with 12 single-node tests (6 tests x 2 stages) using 8 GPUs each = 2 stages x 8 GPUs x 1 hour = 16 GPU hours

## Running Tests

**Important**: Do NOT add `--perf` flag when running pytest. Perf sanity tests are static test cases and do not use perf mode.

### Local Run Examples

```bash
# Run a normal aggregated test
pytest perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k1k]

# Run an aggregated ctx_only test
pytest perf/test_perf_sanity.py::test_e2e[aggr_upload-ctx_only-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]

# Run a disaggregated gen_only test
pytest perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]

# Run a disaggregated e2e test
pytest perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8]
```

### Using Local Submit Script

For local SLURM job submission (supports both aggregated and disaggregated tests):

```bash
python jenkins/scripts/perf/local/submit.py --help
```

## Disaggregated Test SLURM Execution

A disaggregated test runs **four srun steps** within a single multi-node SLURM job allocation. Each step has a different role set via `DISAGG_SERVING_TYPE`:

| Step | `DISAGG_SERVING_TYPE` | Needs MPI | Notes |
|------|-----------------------|-----------|-------|
| Context worker(s) | `CTX_0`, `CTX_1`, ... | Yes | Launched via `trtllm-llmapi-launch`, multi-GPU |
| Generation worker(s) | `GEN_0`, `GEN_1`, ... | Yes | Launched via `trtllm-llmapi-launch`, multi-GPU |
| Disagg server | `DISAGG_SERVER` | No | Runs `trtllm-serve disaggregated`, single process |
| Benchmark client | `BENCHMARK` | No | Runs benchmark pytest, single process |

All four srun steps share the same `srunArgs` array, but `--mpi=pmix` is added **only** to the CTX/GEN worker srun commands in `slurm_launch_draft.sh` (not in srunArgs). This prevents unwanted MPI initialization in the disagg server and benchmark processes. See the MPI/PMI section in `jenkins/scripts/perf/README.md` for details.

## Quick Reference for AI Agents

When working with perf sanity tests, use these paths:

| Resource | Path |
|----------|------|
| Pytest script | `tests/integration/defs/perf/test_perf_sanity.py` |
| Regression pipeline | `tests/integration/defs/perf/perf_regression_utils.py` |
| DB utilities | `tests/integration/defs/perf/open_search_db_utils.py` |
| Aggregated configs | `tests/scripts/perf-sanity/aggregated/*.yaml` |
| Disaggregated configs | `tests/scripts/perf-sanity/disaggregated/*.yaml` |
| CI submit (disagg only) | `jenkins/scripts/perf/disaggregated/submit.py` |
| Local submit (all) | `jenkins/scripts/perf/local/submit.py` |
| Jenkins pipeline | `jenkins/L0_Test.groovy` |
| Test database | `tests/integration/test_lists/test-db/` |
| Test waives | `tests/integration/test_lists/waives.txt` |

## Step-by-Step: Adding or Re-enabling Disaggregated Perf Sanity Tests

When adding a new disaggregated perf sanity test (or uncommenting an existing one), you must update **two files**: the test-db YAML and `jenkins/L0_Test.groovy`. This section describes how to locate and edit each one.

### Step 1: Identify the Disaggregated Config YAML

Config files live in `tests/scripts/perf-sanity/disaggregated/`. The filename encodes the GPU type and test parameters:

```
{gpu_type}_{model}-{precision}_{ISL}k{OSL}k_con{concurrency}_ctx{ctx_count}_tp{ctx_tp}_gen{gen_count}_{gen_parallelism}_eplb{N}_mtp{N}_ccb-{transport}.yaml
```

Example: `gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-UCX.yaml`

The **base name** (filename without `.yaml`) is used as the test case ID in the test-db.

### Step 2: Calculate Resource Requirements from Config YAML

Read the config YAML and extract these fields:

```yaml
hardware:
  gpus_per_node: 4          # GPUs per physical node
  num_ctx_servers: 1         # Number of context workers
  num_gen_servers: 1         # Number of generation workers
worker_config:
  ctx:
    tensor_parallel_size: 1  # GPUs per ctx worker
  gen:
    tensor_parallel_size: 8  # GPUs per gen worker
```

Calculate:

| Value | Formula | Example (ctx_tp=1, gen_tp=8, gpus_per_node=4) |
|-------|---------|-----------------------------------------------|
| Nodes per ctx worker | `ceil(ctx_tp / gpus_per_node)` | `ceil(1/4) = 1` |
| Nodes per gen worker | `ceil(gen_tp / gpus_per_node)` | `ceil(8/4) = 2` |
| Total nodes | `(nodes_per_ctx * num_ctx) + (nodes_per_gen * num_gen)` | `1*1 + 2*1 = 3` |
| Total GPUs | `total_nodes * gpus_per_node` | `3 * 4 = 12` |

### Step 3: Find the Test-db YAML File

The test-db file name follows this pattern (all in `tests/integration/test_lists/test-db/`):

```
l0_{gpu_type}_multi_nodes_perf_sanity_ctx{num_ctx}_node{nodes_per_ctx}_gpu{ctx_tp}_gen{num_gen}_node{nodes_per_gen}_gpu{gen_tp}.yml
```

Example: for ctx_tp=1, gen_tp=8, 1 ctx worker, 1 gen worker on GB200:
```
l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node2_gpu8.yml
```

The `system_gpu_count` in the test-db condition section equals the total GPUs calculated above.

### Step 4: Add or Uncomment the Test in the Test-db File

Each disagg test line in the test-db file follows one of these formats:

```yaml
# gen_only test:
- perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-{config_base_name}] TIMEOUT (120)

# e2e test:
- perf/test_perf_sanity.py::test_e2e[disagg_upload-e2e-{config_base_name}] TIMEOUT (120)

# ctx_only test (placed in aggregated test-db files, not disagg ones):
- perf/test_perf_sanity.py::test_e2e[aggr_upload-ctx_only-{config_base_name}] TIMEOUT (120)
```

- If the test line already exists but is **commented out** (prefixed with `# `), remove the `# ` prefix.
- If the test line does not exist, add it to the `tests` list.
- Count the total number of **active (uncommented) tests** in the file — you will need this count for Step 5.

### Step 5: Update `jenkins/L0_Test.groovy`

Open `jenkins/L0_Test.groovy` and search for the `multiNodesSBSAConfigs` section inside `launchTestJobs()`. Disaggregated perf sanity stages are added via `buildStageConfigs()`:

```groovy
def buildStageConfigs(stageName, platform, testlist, testCount, gpuCount, nodeCount, runWithSbatch=false)
```

| Parameter | Description |
|-----------|-------------|
| `stageName` | CI stage name prefix (see naming convention below) |
| `platform` | Hardware platform, e.g., `"auto:gb200-flex"` |
| `testlist` | Test-db filename **without** `.yml`, e.g., `"l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4"` |
| `testCount` | Number of **active (uncommented)** tests in the test-db file. Each disagg test gets its own stage, so `testCount` must equal the number of active tests. |
| `gpuCount` | Total GPUs from Step 2 (= `total_nodes * gpus_per_node`) |
| `nodeCount` | Total nodes from Step 2 |

**Stage naming convention:**

```
GB200-{gpuCount}_GPUs-{nodeCount}_Nodes-PyTorch-Disagg-PerfSanity-CTX{num_ctx}-NODE{nodes_per_ctx}-GPU{ctx_tp}-GEN{num_gen}-NODE{nodes_per_gen}-GPU{gen_tp}-Post-Merge
```

**If a `buildStageConfigs` entry already exists** for the test-db file: update `testCount` to match the new total number of active tests.

**If no entry exists** for the test-db file: add a new `buildStageConfigs` block. Insert it in the correct section sorted by node count (2 Nodes, 3 Nodes, 4 Nodes, etc.).

### Step 6: Check Waives

Search `tests/integration/test_lists/waives.txt` for the exact test case string. If the test is listed there with a `SKIP` directive, remove that line (otherwise the test will be skipped even if present in the test-db).

### Worked Example

Adding back `qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-UCX` as a gen_only test:

1. Config file: `tests/scripts/perf-sanity/disaggregated/gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-UCX.yaml`
2. From config: `gpus_per_node=4`, `ctx_tp=1`, `gen_tp=4`, `num_ctx=1`, `num_gen=1`
3. Nodes per ctx = `ceil(1/4)=1`, nodes per gen = `ceil(4/4)=1`, total nodes = 2, total GPUs = 8
4. Test-db file: `l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4.yml`
5. Uncomment the line: `- perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-UCX] TIMEOUT (120)`
6. Count active tests in that file (now 4)
7. In `L0_Test.groovy`, find the existing `buildStageConfigs` for `l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4`, update `testCount` from 3 to 4
8. Check `waives.txt` — no matching entry, done
