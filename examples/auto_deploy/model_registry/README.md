# AutoDeploy Model Registry

The AutoDeploy model registry provides a comprehensive, maintainable list of supported models for testing and coverage tracking.

## Architecture

The registry is split into two layers:

1. **AD defaults** (`ad_defaults`) — AD-internal knobs (transforms, compile backends, etc.) stored in `tensorrt_llm/_torch/auto_deploy/config/model_registry_internal/configs/`. These are shipped inside the package.
1. **User configs** (`user_configs`) — User-facing knobs (max_batch_size, kv_cache_config, etc.) stored in `examples/auto_deploy/model_registry/configs/`.

The central registry file is `tensorrt_llm/_torch/auto_deploy/config/model_registry_internal/models.yaml`.

## Format

**Version: 2.0** (Flat format with composable configurations)

### Structure

```yaml
version: '2.0'
models:
- name: meta-llama/Llama-3.1-8B-Instruct
  config_id: default_ws_2
  world_size: 2
  ad_defaults: ['ad_base.yaml']
  user_configs: ['dashboard_default.yaml']

- name: meta-llama/Llama-3.3-70B-Instruct
  config_id: llama3_3_70b
  world_size: 4
  ad_defaults: ['ad_base.yaml', 'llama3_3_70b_ad.yaml']
  user_configs: ['dashboard_default.yaml', 'llama3_3_70b.yaml']
```

### Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | HuggingFace model id |
| `ad_defaults` | Yes | List of AD-internal config YAML files |
| `config_id` | No | Variant selector when a model has multiple entries (defaults to `"default"`) |
| `world_size` | No | GPU count (defaults to 1) |
| `user_configs` | No | List of user-facing config YAML files |

### Key Concepts

- **Flat list**: Models are in a single flat list (not grouped)
- **Two config layers**: `ad_defaults` for AD-internal knobs, `user_configs` for user-facing knobs
- **Deep merging**: Config files are merged in order (later files override earlier ones)
- **world_size**: First-class field in the registry

## User Configuration Files

User-facing config files are stored in `configs/` and define runtime parameters:

### Core Configs

| File | Purpose | Example Use |
|------|---------|-------------|
| `dashboard_default.yaml` | Baseline settings for all models | Always first in user_configs |
| `world_size_N.yaml` | GPU count (1, 2, 4, 8) | Defines tensor_parallel_size |

### Model-Specific Configs

| File | Purpose |
|------|---------|
| `llama3_3_70b.yaml` | Optimized settings for Llama 3.3 70B |
| `nano_v3.yaml` | Settings for Nemotron Nano V3 |
| `llama4_scout.yaml` | Settings for Llama 4 Scout |
| `openelm.yaml` | Apple OpenELM (custom tokenizer) |
| `gemma3_1b.yaml` | Gemma 3 1B (sequence length) |

## Adding a New Model

### Simple Model (Standard Config)

```yaml
- name: organization/my-new-model-7b
  config_id: default_ws_2
  world_size: 2
  ad_defaults: ['ad_base.yaml']
  user_configs: ['dashboard_default.yaml']
```

### Model with Custom Config

1. If the model needs AD-internal overrides (transforms, backends), create `configs/my_model_ad.yaml` in `tensorrt_llm/_torch/auto_deploy/config/model_registry_internal/configs/`.

1. If the model needs user-facing overrides (batch sizes, KV cache), create `configs/my_model.yaml` in this directory.

1. Add the entry in `tensorrt_llm/_torch/auto_deploy/config/model_registry_internal/models.yaml`:

```yaml
- name: organization/my-custom-model
  config_id: my_model
  world_size: 8
  ad_defaults: ['ad_base.yaml', 'my_model_ad.yaml']
  user_configs: ['dashboard_default.yaml', 'my_model.yaml']
```

## Config Merging

Configs are deep-merged in order: `default.yaml` (AD base) → `ad_defaults[0]` → `ad_defaults[1]` → ... → `user_configs[0]` → `user_configs[1]` → ... → CLI/init args (highest priority).

## World Size Guidelines

| World Size | Model Size Range | Example Models |
|------------|------------------|----------------|
| 1 | \< 2B params | TinyLlama, Qwen 0.5B, Phi-4-mini |
| 2 | 2-15B params | Llama 3.1 8B, Qwen 7B, Mistral 7B |
| 4 | 20-80B params | Llama 3.3 70B, QwQ 32B, Gemma 27B |
| 8 | 80B+ params | DeepSeek V3, Llama 405B, Nemotron Ultra |

## Best Practices

1. **Always include `ad_base.yaml` first** in `ad_defaults` — it provides baseline AD settings
1. **Always include `dashboard_default.yaml` first** in `user_configs` — it provides baseline runtime settings
1. **Set `world_size` explicitly** — it's used by tests and `trtllm-serve` to determine GPU count
1. **Create reusable configs** — if 3+ models need same settings, make a config file
1. **Use model-specific configs sparingly** — only for unique requirements
