# TensorRT-LLM Recipe System

The TensorRT-LLM recipe system provides optimized configurations for common inference scenarios.

## Overview

The recipe system helps you:

- **Generate optimized recipe files** from high-level scenario constraints (model, GPU, ISL/OSL/concurrency)
- **Avoid manual tuning** of low-level parameters like EP_SIZE, MOE_BACKEND, DP_ATTENTION
- **Ensure validated configurations** through CI-tested recipes

**Note:** A recipe file is a comprehensive YAML containing `scenario`, `env`, and `config` sections. It serves as a complete deployment descriptor that can be used directly with `trtllm-bench` and `trtllm-serve`.

## Quick Start

### Generate recipe from scenario parameters:

```bash
trtllm-configure \
    --model nvidia/DeepSeek-R1-0528-FP4 \
    --gpu B200 \
    --num-gpus 8 \
    --target-isl 8192 \
    --target-osl 1024 \
    --target-concurrency 256 \
    --output recipe.yaml
```

## Profiles

The system includes three built-in profiles:

### 1. **dsr1-fp4** - DeepSeek-R1 FP4
- Complex EP_SIZE logic based on TP, ISL, OSL, CONC
- MOE_BACKEND: TRTLLM or CUTLASS (depends on concurrency)
- Optimized for high-throughput scenarios

### 2. **dsr1-fp8** - DeepSeek-R1 FP8
- EP_SIZE always equals TP
- MOE_BACKEND: DEEPGEMM
- Simpler configuration rules

### 3. **gptoss-fp4** - GPT-OSS FP4
- Simple concurrency-based rules
- Requires TRTLLM_ENABLE_PDL=1 environment variable
- Optimized for 120B parameter models

## Recipe Format

A recipe file contains:

```yaml
scenario:
  model: openai/gpt-oss-120b
  gpu: H100_SXM
  num_gpus: 8
  target_isl: 8000
  target_osl: 1000
  target_concurrency: 256
  profile: gptoss-fp4  # optional, auto-detected

env:
  TRTLLM_ENABLE_PDL: 1
  NCCL_GRAPH_REGISTER: 0

config:
  cuda_graph_config:
    enable_padding: true
    max_batch_size: 256
  enable_attention_dp: true
  kv_cache_config:
    dtype: fp8
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.85
  print_iter_log: true
  stream_interval: 20
  num_postprocess_workers: 4
  moe_config:
    backend: TRTLLM

# Optional overrides for power users
overrides:
  # kv_cache_config:
  #   free_gpu_memory_fraction: 0.9
```

## Example Recipes

See the `db/` directory for validated recipes:
- `gptoss-fp4-h100-throughput.yaml` - GPT-OSS 120B on H100 GPUs
- `dsr1-fp4-b200-throughput.yaml` - DeepSeek-R1 FP4 on B200 GPUs

## Adding Custom Profiles

For advanced users, custom profiles can be registered:

```python
from tensorrt_llm.recipes import ProfileBase, register_profile

class MyCustomProfile(ProfileBase):
    def compute_config(self, scenario):
        # Your logic here
        return {'config': {...}, 'env': {...}, 'cli_args': {...}}

    def get_defaults(self):
        return {...}

register_profile('my-profile', MyCustomProfile)
```

## Validation

The system validates:
- Required fields (model, ISL, OSL, concurrency)
- Numeric ranges (ISL > 0, concurrency > 0)
- TP divisibility (num_gpus % tp_size == 0)
- GPU compatibility
- Configuration parameters (memory fractions, batch sizes)

Use `--no-validate` to skip validation if needed.

## Integration with trtllm-serve and trtllm-bench

### Option 1: Generate Recipe with trtllm-configure, then use with trtllm-bench

Generate a recipe file from scenario parameters, then benchmark with it:

```bash
# Generate recipe from scenario
trtllm-configure \
    --model nvidia/DeepSeek-R1-0528-FP4 \
    --gpu B200 \
    --num-gpus 8 \
    --target-isl 8192 \
    --target-osl 1024 \
    --target-concurrency 256 \
    --output my-recipe.yaml

# Use with trtllm-bench (recommended)
trtllm-bench --recipe my-recipe.yaml
```

### Option 2: Use Existing Recipe YAML Directly (Comprehensive)

**Recipe YAMLs can now be used directly** with `trtllm-serve` and `trtllm-bench` via `--extra_llm_api_options`:

```bash
# Recipe YAML provides everything: config, env vars, and serves as deployment descriptor
trtllm-serve --extra_llm_api_options tensorrt_llm/recipes/db/gptoss-fp4-h100-throughput.yaml

# CLI flags override recipe values (priority: CLI > recipe > defaults)
trtllm-serve --tp_size 4 \
    --extra_llm_api_options tensorrt_llm/recipes/db/gptoss-fp4-h100-throughput.yaml
```

**Benefits of using recipe YAMLs directly:**
- ✅ Single file describes entire deployment (config + env vars + metadata)
- ✅ No need to manually set environment variables
- ✅ Self-documenting (scenario section describes the use case)
- ✅ CLI flags can still override any setting
- ✅ Backward compatible (simple config YAMLs still work)

**How it works:**
1. `trtllm-serve` and `trtllm-bench` detect recipe format (has `scenario` and `config` keys)
2. Automatically extracts `config:` section for LLM API parameters
3. Automatically sets environment variables from `env:` section (if not already set)
4. CLI flags take precedence over recipe values

### Priority Order

When using recipe YAMLs with serve/bench:

1. **CLI flags** (highest priority) - `--tp_size 4` overrides everything
2. **Recipe values** - `scenario:` and `config:` sections
3. **Built-in defaults** (lowest priority)

## Contributing

To contribute a new recipe:

1. Create a YAML file in `db/`
2. Test the configuration with your model
3. Submit a PR with CI test results
4. Document any specific requirements or constraints
