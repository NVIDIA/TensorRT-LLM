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

**Structure**:
```yaml
server_configs:
  - name: "r1_fp8_dep8_mtp1_1k1k"
    model_name: "deepseek_r1_0528_fp8"
    gpus: 8
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    pipeline_parallel_size: 1
    max_batch_size: 512
    max_num_tokens: 8192
    attention_backend: "TRTLLM"
    enable_attention_dp: true
    attention_dp_config:
      batching_wait_iters: 0
      enable_balance: true
      timeout_iters: 60
    moe_config:
      backend: 'DEEPGEMM'
    cuda_graph_config:
      enable_padding: true
      max_batch_size: 512
    kv_cache_config:
      dtype: 'fp8'
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.8
    speculative_config:
      decoding_type: 'MTP'
      num_nextn_predict_layers: 1
    client_configs:
      - name: "con4096_iter10_1k1k"
        concurrency: 4096
        iterations: 10
        isl: 1024
        osl: 1024
        random_range_ratio: 0.8
        backend: "openai"
```


### 2. Multi-Node Aggregated Test Configuration

**File Example**: `l0_gb200_multi_nodes.yaml`

**Use Case**: Multi-node aggregated architecture where model runs across multiple nodes with unified execution.

**Structure**:
```yaml
# Hardware Config
hardware:
  gpus_per_node: 4
  gpus_per_server: 8

server_configs:
  - name: "r1_fp4_v2_dep8_mtp1"
    model_name: "deepseek_r1_0528_fp4_v2"
    gpus: 8
    gpus_per_node: 4
    trust_remote_code: true
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    pipeline_parallel_size: 1
    max_batch_size: 512
    max_num_tokens: 2112
    attn_backend: "TRTLLM"
    enable_attention_dp: true
    attention_dp_config:
      batching_wait_iters: 0
      enable_balance: true
      timeout_iters: 60
    moe_config:
      backend: 'CUTLASS'
    cuda_graph_config:
      enable_padding: true
      max_batch_size: 512
    kv_cache_config:
      dtype: 'fp8'
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.5
    client_configs:
      - name: "con32_iter12_1k1k"
        concurrency: 32
        iterations: 12
        isl: 1024
        osl: 1024
        random_range_ratio: 0.8
        backend: "openai"
```
