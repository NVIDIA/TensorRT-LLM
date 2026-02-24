# Test Configurations

This directory contains YAML configuration files for TensorRT-LLM disaggregated benchmark tests.

## Directory Structure

```
test_configs/
├── disagg/                    # Disaggregated architecture
│   ├── perf/                  # Performance tests (24 configs)
│   └── accuracy/              # Accuracy tests
└── wideep/                    # Wide-deep architecture  
    ├── perf/                  # Performance tests (15 configs)
    └── accuracy/              # Accuracy tests (1 config)
```

## File Naming Convention

Format: `{model}_{benchmark_type}_ctx{N}_gen{M}_{parallel_config}_bs{B}_eplb{E}_mtp{T}_ccb-{backend}.yaml`

**Components:**
- `model`: Model name (e.g., `deepseek-r1-fp4`, `Qwen3-235B-A22B-FP4`)
- `benchmark_type`: Input/output lengths (e.g., `1k1k` = 1024/1024, `8k1k` = 8192/1024)
- `ctx{N}_gen{M}`: N context servers, M generation servers
- Parallel configuration:
  - `dep{N}`: Data parallel with attention_dp, TP size = N
  - `tep{N}`: Tensor parallel only, TP size = N
- `bs{B}`: Batch size
- `eplb{E}`: Expert parallel load balancing slots (0 = disabled)
- `mtp{T}`: Multi-token prediction layers (0 = disabled)
- `backend`: Cache transceiver backend (`NIXL`, `UCX`, `DEFAULT`)

**Examples:**
```
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL.yaml
  → DeepSeek R1 FP4, 1k/1k, 1 ctx + 4 gen servers, TP=8, BS=32, no MTP, NIXL backend

deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-UCX.yaml  
  → DeepSeek R1 FP4, 8k/1k, 1 ctx + 3 gen servers, TP=8, BS=16, MTP=3, UCX backend

Qwen3-235B-A22B-FP4_1k1k_ctx2_gen1_dep16_bs128_eplb0_mtp1_ccb-NIXL.yaml
  → Qwen3 235B FP4, 1k/1k, 2 ctx + 1 gen servers, DP+TP=16, BS=128, MTP=1, NIXL backend
```

## Current Test Configurations

### Models
- **DeepSeek R1 FP4**: 16 configs (disagg) + 15 configs (wideep) = 31 configs
- **Qwen3-235B-A22B FP4**: 8 configs (disagg) + 6 configs (wideep) = 14 configs

### Benchmark Types
- **1k1k** (1024/1024): 32 configs
- **8k1k** (8192/1024): 8 configs

### Backends
- **NIXL**: 20 configs
- **UCX**: 19 configs
- **DEFAULT**: 1 config

## Configuration Structure

Each YAML file contains:

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200"]

benchmark:
  input_length: 1024
  output_length: 1024
  streaming: true
  concurrency_list: "..."

hardware:
  num_ctx_servers: 1
  num_gen_servers: 4

worker_config:
  gen:
    tensor_parallel_size: 8
    max_batch_size: 32
    max_seq_len: 2251
    speculative_config:
      num_nextn_predict_layers: 3  # MTP layers
  ctx:
    tensor_parallel_size: 4
    max_batch_size: 4
    max_seq_len: 2251
```

## Key Configuration Constraints

1. **Streaming**: Must be `true`
2. **max_seq_len**: Both ctx and gen must be > (input_length + output_length)
3. **gen_max_tokens**: Must equal `gen_max_batch_size * (mtp_size + 1)` when MTP is enabled
4. **supported_gpus**: Currently all configs use `["GB200"]`

## Configuration Mapping Reference

| Field | YAML Path |
|-------|-----------|
| Input/Output lengths | `benchmark.input_length`, `benchmark.output_length` |
| Server counts | `hardware.num_ctx_servers`, `hardware.num_gen_servers` |
| Tensor parallel size | `worker_config.gen.tensor_parallel_size` |
| Batch size | `worker_config.gen.max_batch_size` |
| Attention DP | `worker_config.gen.enable_attention_dp` |
| MTP layers | `worker_config.gen.speculative_config.num_nextn_predict_layers` |
| Backend | `worker_config.gen.cache_transceiver_config.backend` |
| Concurrency levels | `benchmark.concurrency_list` |

## Adding New Configurations

1. Copy an existing config file as a template
2. Update the filename to match your configuration
3. Modify the YAML content:
   - Update `metadata` section
   - Adjust `benchmark` parameters
   - Configure `worker_config` for ctx and gen
4. Ensure configuration constraints are met
5. Run tests - the new config will be automatically discovered

For detailed usage and test execution, see the main README.md in the parent directory.
