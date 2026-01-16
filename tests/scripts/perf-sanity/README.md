# TensorRT-LLM Perf Sanity Test System

Performance sanity testing scripts for TensorRT-LLM with configuration-driven test cases supporting single-node, multi-node aggregated, and multi-node disaggregated architectures.

## Overview

- Run performance sanity benchmarks across multiple model configs
- Support three deployment architectures: single-node, multi-node aggregated, and multi-node disaggregated
- Manage test cases through YAML config files
- Automated resource calculation and job submission via SLURM

## Configuration File Types

There are two modes for perf sanity tests: aggregated (aggr) and disaggregated (disagg).

### Aggregated Mode (aggr)

**Config Location**: [`tests/scripts/perf-sanity`](./)

**File Naming**: `xxx.yaml` where words are connected by `_` (underscore), not `-` (hyphen).

**File Examples**:
- `deepseek_r1_fp4_v2_grace_blackwell.yaml` - Single-node aggregated test
- `deepseek_r1_fp4_v2_2_nodes_grace_blackwell.yaml` - Multi-node aggregated test

**Use Cases**:
- Single-node: Performance tests on a single server with multiple GPUs
- Multi-node: Model runs across multiple nodes with unified execution

**Test Case Names**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-{config yaml file base name}]
perf/test_perf_sanity.py::test_e2e[aggr_upload-{config yaml file base name}-{server_config_name}]
```

- Without server config name: runs all server configs in the YAML file
- With server config name: runs only the specified server config (the `name` field in `server_configs`)

**Examples**:
```
perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell]
perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k1k]
perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_tep4_mtp3_1k1k]
```

### Disaggregated Mode (disagg)

**Config Location**: [`tests/integration/defs/perf/disagg/test_configs/disagg/perf`](../../integration/defs/perf/disagg/test_configs/disagg/perf)

**File Naming**: `xxx.yaml` (can contain `-` hyphen).

**File Example**: `deepseek-r1-fp4_1k1k_ctx1_gen1_dep8_bs768_eplb0_mtp0_ccb-UCX.yaml`

**Use Case**: Disaggregated architecture where model runs across multiple nodes with separate context (prefill) and generation (decode) servers.

**Test Case Name**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-{config yaml file base name}]
```

**Example**:
```
perf/test_perf_sanity.py::test_e2e[disagg_upload-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8_bs768_eplb0_mtp0_ccb-UCX]
```

## Running Tests

**Important**: Do NOT add `--perf` flag when running pytest. Perf sanity tests are static test cases and do not use perf mode.

```bash
# Run all server configs in an aggregated test
pytest perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell]

# Run a specific server config in an aggregated test
pytest perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k1k]

# Run a specific disaggregated test
pytest perf/test_perf_sanity.py::test_e2e[disagg_upload-deepseek-r1-fp4_1k1k_ctx1_gen1_dep8_bs768_eplb0_mtp0_ccb-UCX]
```
