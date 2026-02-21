# TensorRT-LLM Autodeploy Model Registry

The model registry at `examples/auto_deploy/model_registry/` provides pre-configured settings for 100+ supported models.

## Registry Structure

**Location**: `examples/auto_deploy/model_registry/`

```
model_registry/
├── README.md           # Registry documentation
├── models.yaml         # Model list with config references
└── configs/            # Reusable configuration files
    ├── dashboard_default.yaml     # Baseline for all models
    ├── world_size_N.yaml          # GPU count (1, 2, 4, 8)
    ├── multimodal.yaml            # Vision + text models
    ├── nano_v3.yaml               # Nemotron Nano V3
    ├── llama3_3_70b.yaml          # Llama 3.3 70B
    └── ...
```

## Composable Configuration System

Models reference config files via `yaml_extra` arrays. Configs are merged in order (later overrides earlier):

```yaml
models:
- name: meta-llama/Llama-3.1-8B-Instruct
  yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml']

- name: meta-llama/Llama-3.3-70B-Instruct
  yaml_extra: ['dashboard_default.yaml', 'world_size_4.yaml', 'llama3_3_70b.yaml']

- name: nvidia/Nemotron-Nano-v3-8B
  yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml', 'nano_v3.yaml']
```

## Core Config Files

### dashboard_default.yaml (Always First)
```yaml
runtime: trtllm
attn_backend: flashinfer
compile_backend: torch-cudagraph
model_factory: AutoModelForCausalLM
skip_loading_weights: false
max_seq_len: 512
```

### world_size_N.yaml (Required)
```yaml
# world_size_2.yaml
world_size: 2
```

Available: `world_size_1.yaml`, `world_size_2.yaml`, `world_size_4.yaml`, `world_size_8.yaml`

## World Size Guidelines

Choose GPU count based on model size:

| World Size | Model Size Range | Example Models |
|------------|------------------|----------------|
| 1 | < 2B params | TinyLlama, Qwen 0.5B, Phi-4-mini |
| 2 | 2-15B params | Llama 3.1 8B, Qwen 7B, Mistral 7B |
| 4 | 20-80B params | Llama 3.3 70B, QwQ 32B, Gemma 27B |
| 8 | 80B+ params | DeepSeek V3, Llama 405B, Nemotron Ultra |

## Special Configuration Files

### multimodal.yaml
For vision-language models (LLaVA, Phi-3-vision, etc.)

### demollm_triton.yaml
For DemoLLM runtime with Triton backend

### attn_backend_triton.yaml
For models requiring Triton attention backend

### simple_shard_only.yaml
For large models requiring simple sharding strategy

## Using Registry Configs

### Method 1: Reference from Model Registry

For models in the registry, compose their configs:

```bash
# Find the model in models.yaml
grep "Llama-3.1-8B" examples/auto_deploy/model_registry/models.yaml

# Output shows:
# - name: meta-llama/Llama-3.1-8B-Instruct
#   yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml']

# Manually merge the configs or use them as templates
```

### Method 2: Build Custom Config from Registry Patterns

Use registry configs as building blocks:

```yaml
# my_custom_config.yaml
# Start with dashboard defaults
runtime: trtllm
attn_backend: flashinfer
compile_backend: torch-cudagraph
model_factory: AutoModelForCausalLM

# Add world size
world_size: 4

# Add custom overrides
max_batch_size: 256
max_seq_len: 4096
enable_chunked_prefill: true
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.9

# Add transforms
transforms:
  fuse_fp8_gemms:
    stage: post_load_fusion
    enabled: true
```

### Method 3: Extend Registry Config

Create model-specific config that builds on registry defaults:

```yaml
# configs/my_model_custom.yaml
# Assumes dashboard_default.yaml and world_size_N.yaml are merged first

max_batch_size: 512
max_seq_len: 8192
transforms:
  multi_stream_moe:
    stage: compile
    enabled: true
```

## Example Model Configurations

### Standard Llama Model (2 GPUs)
```yaml
- name: meta-llama/Llama-3.1-8B-Instruct
  yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml']
```

Merged result:
```yaml
runtime: trtllm
attn_backend: flashinfer
compile_backend: torch-cudagraph
model_factory: AutoModelForCausalLM
skip_loading_weights: false
max_seq_len: 512
world_size: 2
```

### Large Model with Custom Config (4 GPUs)
```yaml
- name: meta-llama/Llama-3.3-70B-Instruct
  yaml_extra: ['dashboard_default.yaml', 'world_size_4.yaml', 'llama3_3_70b.yaml']
```

### Mamba Hybrid Model
```yaml
- name: nvidia/Nemotron-Nano-v3-8B
  yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml', 'nano_v3.yaml']
```

The `nano_v3.yaml` config includes Mamba-specific transforms like:
- `fuse_mamba_a_log`
- `insert_cached_ssm_attention`
- `multi_stream_moe`
- Tensor parallelism sharding for hybrid layers

## Adding New Models to Registry

### Standard Model
```yaml
- name: organization/my-new-model-7b
  yaml_extra: ['dashboard_default.yaml', 'world_size_2.yaml']
```

### Model with Special Requirements
```yaml
- name: organization/my-multimodal-model
  yaml_extra: ['dashboard_default.yaml', 'world_size_4.yaml', 'multimodal.yaml']
```

### Model with Fully Custom Config

1. Create `configs/my_model.yaml`:
```yaml
max_batch_size: 2048
kv_cache_config:
  free_gpu_memory_fraction: 0.95
transforms:
  custom_transform:
    enabled: true
```

2. Reference in `models.yaml`:
```yaml
- name: organization/my-custom-model
  yaml_extra: ['dashboard_default.yaml', 'world_size_8.yaml', 'my_model.yaml']
```

## Best Practices

1. **Always include dashboard_default.yaml first** - Provides baseline settings
2. **Always include a world_size_N.yaml** - Defines GPU count
3. **Add special configs after world_size** - They override defaults as needed
4. **Create reusable configs** - If 3+ models need same settings, make a shared config
5. **Use model-specific configs sparingly** - Only for unique requirements

## Supported Models (Examples)

The registry includes 100+ models across various architectures:

**Text Models**:
- Llama 2/3/3.1/3.2/3.3/4
- Qwen 2.5/3
- Mistral 7B/NeMo
- Phi-4 variants
- Gemma 1.1/2/3
- DeepSeek R1/V2/V3
- Granite 3.x
- StarCoder2

**Multimodal Models**:
- Phi-3-vision
- LLaVA variants
- Qwen-VL

**Hybrid Models** (Mamba + Attention):
- Nemotron Nano v3
- Nemotron Super v3

See `models.yaml` for the complete list.

## Config Merging Example

Given:
```yaml
yaml_extra:
  - dashboard_default.yaml    # runtime=trtllm, max_seq_len=512
  - world_size_2.yaml         # world_size=2
  - custom.yaml               # max_seq_len=4096, custom_param=value
```

Result:
```yaml
runtime: trtllm               # from dashboard_default
attn_backend: flashinfer      # from dashboard_default
compile_backend: torch-cudagraph  # from dashboard_default
model_factory: AutoModelForCausalLM  # from dashboard_default
skip_loading_weights: false   # from dashboard_default
max_seq_len: 4096            # from custom (overrides dashboard_default)
world_size: 2                # from world_size_2
custom_param: value          # from custom (new)
```

## Related Files

- **Registry README**: `examples/auto_deploy/model_registry/README.md`
- **Model List**: `examples/auto_deploy/model_registry/models.yaml`
- **Config Files**: `examples/auto_deploy/model_registry/configs/*.yaml`
- **Example Configs**: See this skill's `config_examples.md` for complete working configs
