# AutoDeploy Model Registry

The AutoDeploy model registry provides a comprehensive, maintainable list of supported models for testing and coverage tracking.

## Format

**Version: 2.0** (Flat format with composable configurations)

### Structure

```yaml
version: '2.0'
description: AutoDeploy Model Registry - Flat format with composable configs
models:
- name: meta-llama/Llama-3.1-8B-Instruct
  yaml_extra: [dashboard_default.yaml, world_size_2.yaml]

- name: meta-llama/Llama-3.3-70B-Instruct
  yaml_extra: [dashboard_default.yaml, world_size_4.yaml, llama-3.3-70b.yaml]

- name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  yaml_extra: [dashboard_default.yaml, world_size_2.yaml, demollm_triton.yaml]
```

### Key Concepts

- **Flat list**: Models are in a single flat list (not grouped)
- **Composable configs**: Each model references YAML config files via `yaml_extra`
- **Deep merging**: Config files are merged in order (later files override earlier ones)
- **No inline args**: All configuration is in YAML files for reusability

## Configuration Files

Config files are stored in `configs/` subdirectory and define runtime parameters:

### Core Configs

| File | Purpose | Example Use |
|------|---------|-------------|
| `dashboard_default.yaml` | Baseline settings for all models | Always first in yaml_extra |
| `world_size_N.yaml` | GPU count (1, 2, 4, 8) | Defines tensor_parallel_size |

### Runtime Configs

| File | Purpose |
|------|---------|
| `multimodal.yaml` | Vision + text models |
| `demollm_triton.yaml` | DemoLLM runtime with Triton backend |
| `simple_shard_only.yaml` | Large models requiring simple sharding

### Model-Specific Configs

| File | Purpose |
|------|---------|
| `llama-3.3-70b.yaml` | Optimized settings for Llama 3.3 70B |
| `nano_v3.yaml` | Settings for Nemotron Nano V3 |
| `llama-4-scout.yaml` | Settings for Llama 4 Scout |
| `openelm.yaml` | Apple OpenELM (custom tokenizer) |
| `gemma3_1b.yaml` | Gemma 3 1B (sequence length) |
| `deepseek_v3_lite.yaml` | DeepSeek V3/R1 (reduced layers) |
| `llama4_maverick_lite.yaml` | Llama 4 Maverick (reduced layers) |

## Adding a New Model

### Simple Model (Standard Config)

```yaml
- name: organization/my-new-model-7b
  yaml_extra: [dashboard_default.yaml, world_size_2.yaml]
```

### Model with Special Requirements

```yaml
- name: organization/my-multimodal-model
  yaml_extra: [dashboard_default.yaml, world_size_4.yaml, multimodal.yaml]
```

### Model with Custom Config

1. Create `configs/my_model.yaml`:

```yaml
# Custom settings for my model
max_batch_size: 2048
kv_cache_free_gpu_memory_fraction: 0.95
cuda_graph_config:
  enable_padding: true
```

2. Reference it in `models.yaml`:

```yaml
- name: organization/my-custom-model
  yaml_extra: [dashboard_default.yaml, world_size_8.yaml, my_model.yaml]
```

## Config Merging

Configs are merged in order. Example:

```yaml
yaml_extra:
  - dashboard_default.yaml    # baseline: runtime=trtllm, benchmark_enabled=true
  - world_size_2.yaml         # adds: tensor_parallel_size=2
  - openelm.yaml              # overrides: tokenizer=llama-2, benchmark_enabled=false
```

**Result**: `runtime=trtllm, tensor_parallel_size=2, tokenizer=llama-2, benchmark_enabled=false`

## World Size Guidelines

| World Size | Model Size Range | Example Models |
|------------|------------------|----------------|
| 1 | \< 2B params | TinyLlama, Qwen 0.5B, Phi-4-mini |
| 2 | 2-15B params | Llama 3.1 8B, Qwen 7B, Mistral 7B |
| 4 | 20-80B params | Llama 3.3 70B, QwQ 32B, Gemma 27B |
| 8 | 80B+ params | DeepSeek V3, Llama 405B, Nemotron Ultra |

## Model Coverage

The registry contains models distributed across different GPU configurations (world sizes 1, 2, 4, and 8), including both text-only and multimodal models.

**To verify current model counts and coverage:**

```bash
cd /path/to/autodeploy-dashboard
python3 scripts/prepare_model_coverage_v2.py \
    --source local \
    --local-path /path/to/TensorRT-LLM \
    --output /tmp/model_coverage.yaml

# View summary
grep -E "^- name:|yaml_extra:" /path/to/TensorRT-LLM/examples/auto_deploy/model_registry/models.yaml | wc -l
```

When adding or removing models, use `prepare_model_coverage_v2.py` to validate the registry structure and coverage.

## Best Practices

1. **Always include `dashboard_default.yaml` first** - it provides baseline settings
1. **Always include a `world_size_N.yaml`** - defines GPU count
1. **Add special configs after world_size** - they override defaults
1. **Create reusable configs** - if 3+ models need same settings, make a config file
1. **Use model-specific configs sparingly** - only for unique requirements
1. **Test before committing** - verify with `prepare_model_coverage_v2.py`

## Testing Changes

```bash
# Generate workload from local changes
cd /path/to/autodeploy-dashboard
python3 scripts/prepare_model_coverage_v2.py \
    --source local \
    --local-path /path/to/TensorRT-LLM \
    --output /tmp/test_workload.yaml

# Verify output
cat /tmp/test_workload.yaml
```
