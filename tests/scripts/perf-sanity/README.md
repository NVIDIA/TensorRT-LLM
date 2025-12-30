# TensorRT-LLM Perf Sanity Test System

Performance sanity testing scripts for TensorRT-LLM with configuration-driven test cases supporting single-node, multi-node aggregated, and multi-node disaggregated architectures.

## Overview

- Run performance sanity benchmarks across multiple model configs
- Support three deployment architectures: single-node, multi-node aggregated, and multi-node disaggregated
- Manage test cases through YAML config files
- Automated resource calculation and job submission via SLURM

## Configuration File Types

There are three types of YAML config files for different deployment architectures.
Aggregated config files are in [`tests/scripts/perf-sanity`](./).
Disaggregated config files are in [`tests/integration/defs/perf/disagg/test_configs/disagg/perf`](../../integration/defs/perf/disagg/test_configs/disagg/perf).

### 1. Single-Node Aggregated Test Configuration

**File Example**: `deepseek_r1_fp4_v2_grace_blackwell.yaml`

**Use Case**: Single-node performance tests on a single server with multiple GPUs.

### 2. Multi-Node Aggregated Test Configuration

**File Example**: `deepseek_r1_fp4_v2_2_nodes_grace_blackwell.yaml`

**Use Case**: Multi-node aggregated architecture where model runs across multiple nodes with unified execution.

### 3. Multi-Node Disaggregated Test Configuration

**File Example**: `deepseek-r1-fp4_1k1k_ctx1_gen1_dep8_bs768_eplb0_mtp0_ccb-UCX.yaml`

**Use Case**: Disaggregated architecture where model runs across multiple nodes with separate context (prefill) and generation (decode) servers.
