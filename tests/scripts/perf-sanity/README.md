# TensorRT-LLM Perf Sanity Test System

Performance sanity testing scripts for TensorRT-LLM with configuration-driven test cases supporting single-node, multi-node aggregated, and multi-node disaggregated architectures.

## Overview

- Run performance sanity benchmarks across multiple model configurations
- Support three deployment architectures: single-node, multi-node aggregated, and multi-node disaggregated
- Manage test cases through YAML configuration files
- Automated resource calculation and job submission via SLURM

## Configuration File Types

There are three types of YAML configuration files for different deployment architectures:

### 1. Single-Node Aggregated Test Configuration

**File Example**: `l0_dgx_b200.yaml`

**Use Case**: Single-node performance tests on a single server with multiple GPUs.

### 2. Multi-Node Aggregated Test Configuration

**File Example**: `l0_gb200_multi_nodes.yaml`

**Use Case**: Multi-node aggregated architecture where model runs across multiple nodes with unified execution.

### 3. Multi-Node Disaggregated Test Configuration

**File Example**: `l0_gb200_multi_nodes_disagg.yaml`

**Use Case**: Disaggregated architecture where model runs across multiple nodes with separate context (prefill) and generation (decode) servers.

## Submission Scripts

### Single-Node && Multi-Node Aggregated Tests
Location: `multi-node-aggr/`

See `jenkins/scripts/perf/aggregated/README.md` for detailed usage.

### Multi-Node Disaggregated Tests
Location: `multi-node-disagg/`

See `jenkins/scripts/perf/disaggregated/README.md` for detailed usage.
