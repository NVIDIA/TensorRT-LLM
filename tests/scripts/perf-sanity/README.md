# TensorRT-LLM Perf Sanity Test System

Performance sanity testing scripts for TensorRT-LLM with configuration-driven test cases supporting single-node, multi-node aggregated, and multi-node disaggregated architectures.

This document serves as a reference for both developers and AI agents working with the perf sanity system.

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

**File Naming**: `xxx.yaml` where words are connected by `_` (underscore), not `-` (hyphen).

**Examples**:
- `deepseek_r1_fp4_v2_grace_blackwell.yaml` - Single-node aggregated test
- `deepseek_r1_fp4_v2_2_nodes_grace_blackwell.yaml` - Multi-node aggregated test

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

## Quick Reference for AI Agents

When working with perf sanity tests, use these paths:

| Resource | Path |
|----------|------|
| Pytest script | `tests/integration/defs/perf/test_perf_sanity.py` |
| Aggregated configs | `tests/scripts/perf-sanity/aggregated/*.yaml` |
| Disaggregated configs | `tests/scripts/perf-sanity/disaggregated/*.yaml` |
| CI submit (disagg only) | `jenkins/scripts/perf/disaggregated/submit.py` |
| Local submit (all) | `jenkins/scripts/perf/local/submit.py` |
| Jenkins pipeline | `jenkins/L0_Test.groovy` |
| Test database | `tests/integration/test_lists/test-db/` |
