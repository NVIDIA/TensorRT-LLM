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

**Key Features**:
- **Hardware section** specifies:
  - `gpus_per_node`: Number of GPUs per node
  - `gpus_per_server`: Total GPUs across all nodes
- Resource calculation: `nodes = gpus_per_server / gpus_per_node`
- Each server config specifies `gpus_per_node` for multi-node layout

### 3. Multi-Node Disaggregated Test Configuration

**File Example**: `l0_gb200_multi_nodes_disagg.yaml`

**Use Case**: Disaggregated architecture with separate context (prefill) and generation (decode) servers.

**Structure**:
```yaml
# Hardware Config
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  gpus_per_ctx_server: 4
  num_gen_servers: 1
  gpus_per_gen_server: 4
numa_bind: true
timeout: 7200

# Disagg Configs
disagg_configs:
  - name: "r1_fp4_v2_dep8_mtp1"
    model_name: "deepseek_r1_0528_fp4_v2"
    # Benchmark Config
    benchmark:
      iterations: 8
      random_range_ratio: 0.8
      streaming: true
      concurrency: 16
      isl: 1024
      osl: 1024
    # Gen Server Config
    gen:
      tensor_parallel_size: 4
      moe_expert_parallel_size: 4
      enable_attention_dp: true
      enable_lm_head_tp_in_adp: true
      pipeline_parallel_size: 1
      max_batch_size: 256
      max_num_tokens: 512
      max_seq_len: 4608
      cuda_graph_config:
        enable_padding: true
        max_batch_size: 256
      kv_cache_config:
        enable_block_reuse: false
        free_gpu_memory_fraction: 0.8
        dtype: fp8
      moe_config:
        backend: CUTLASS
      cache_transceiver_config:
        max_tokens_in_buffer: 4608
        backend: DEFAULT
      speculative_config:
        decoding_type: MTP
        num_nextn_predict_layers: 1
    # Ctx Server Config
    ctx:
      max_batch_size: 4
      max_num_tokens: 4608
      max_seq_len: 4608
      tensor_parallel_size: 4
      moe_expert_parallel_size: 4
      enable_attention_dp: true
      pipeline_parallel_size: 1
      cuda_graph_config: null
      disable_overlap_scheduler: true
      kv_cache_config:
        enable_block_reuse: false
        free_gpu_memory_fraction: 0.85
        dtype: fp8
      cache_transceiver_config:
        max_tokens_in_buffer: 4608
        backend: DEFAULT
      speculative_config:
        decoding_type: MTP
        num_nextn_predict_layers: 1
```

**Key Features**:
- **Hardware section** specifies:
  - `gpus_per_node`: GPUs per physical node
  - `num_ctx_servers`: Number of context servers
  - `gpus_per_ctx_server`: Total GPUs for context server
  - `num_gen_servers`: Number of generation servers
  - `gpus_per_gen_server`: Total GPUs for generation server
- **disagg_configs** instead of `server_configs`
- Three sub-sections per config:
  - `benchmark`: Client/benchmark configuration
  - `gen`: Generation server configuration
  - `ctx`: Context server configuration
- Resource calculation:
  - `nodes_per_ctx_server = (gpus_per_ctx_server + gpus_per_node - 1) / gpus_per_node`
  - `nodes_per_gen_server = (gpus_per_gen_server + gpus_per_node - 1) / gpus_per_node`
  - `total_nodes = num_ctx_servers * nodes_per_ctx_server + num_gen_servers * nodes_per_gen_server`

## Submission Scripts

### Multi-Node Aggregated Tests
Location: `multi-node-aggr/`

See `tests/scripts/perf-sanity/multi-node-aggr/README.md` for detailed usage.
