# Expert Configuration of LLM API

For advanced TensorRT LLM users, the full set of `tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs` is exposed. Use at your own risk. The argument list may diverge from the standard TensorRT LLM argument list.

- All configuration fields used by the AutoDeploy core pipeline, `InferenceOptimizer`, are exposed exclusively in `AutoDeployConfi`g in `tensorrt_llm._torch.auto_deploy.llm_args`.
  Please make sure to refer to those first.
- For advanced users, the full set of `LlmArgs` in `tensorrt_llm._torch.auto_deploy.llm_args` can be used to configure the AutoDeploy `LLM` API, including runtime options.
- Note that some fields in the full `LlmArgs`
  object are overlapping, duplicated, and/or _ignored_ in AutoDeploy, particularly arguments
  pertaining to configuring the model itself since AutoDeploy's model ingestion+optimize pipeline
  significantly differs from the default manual workflow in TensorRT-LLM.
- However, with the proper care the full `LlmArgs`
  objects can be used to configure advanced runtime options in TensorRT-LLM.
- Any valid field can be simply provided as keyword argument ("`**kwargs`") to the AutoDeploy `LLM` API.

# Expert Configuration of `build_and_run_ad.py`

For advanced users, `build_and_run_ad.py` provides advanced configuration capabilities using a flexible argument parser powered by PyDantic Settings and OmegaConf. You can use dot notation for CLI arguments, provide multiple YAML configuration files, and utilize sophisticated configuration precedence rules to create complex deployment configurations.

## CLI Arguments with Dot Notation

The script supports flexible CLI argument parsing using dot notation to modify nested configurations dynamically. You can target any field in both the `ExperimentConfig` in `examples/auto_deploy/build_and_run_ad.py` and nested `AutoDeployConfig` or `LlmArgs` objects in `tensorrt_llm._torch.auto_deploy.llm_args`:

```bash
# Configure model parameters
# NOTE: config values like num_hidden_layers are automatically resolved into the appropriate nested
# dict value ``{"args": {"model_kwargs": {"num_hidden_layers": 10}}}`` although not explicitly
# specified as CLI arg
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --args.model-kwargs.num-hidden-layers=10 \
  --args.model-kwargs.hidden-size=2048 \
  --args.tokenizer-kwargs.padding-side=left

# Configure runtime and backend options
python build_and_run_ad.py \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --args.world-size=2 \
  --args.compile-backend=torch-opt \
  --args.attn-backend=flashinfer

# Configure prompting and benchmarking
python build_and_run_ad.py \
  --model "microsoft/phi-4" \
  --prompt.batch-size=4 \
  --prompt.sp-kwargs.max-tokens=200 \
  --prompt.sp-kwargs.temperature=0.7 \
  --benchmark.enabled=true \
  --benchmark.bs=8 \
  --benchmark.isl=1024
```

## YAML Configuration Files

Both `ExperimentConfig` and `AutoDeployConfig`/`LlmArgs` inherit from `DynamicYamlMixInForSettings`, which enables you to provide multiple YAML configuration files that are automatically deep-merged at runtime.

Create a YAML configuration file (e.g., `my_config.yaml`):

```yaml
# my_config.yaml
args:
  model_kwargs:
    num_hidden_layers: 12
    hidden_size: 1024
  world_size: 4
  max_seq_len: 2048
  max_batch_size: 16
  transforms:
    detect_sharding:
      support_partial_config: true
    insert_cached_attention:
      backend: triton
    compile_model:
      backend: torch-compile

prompt:
  batch_size: 8
  sp_kwargs:
    max_tokens: 150
    temperature: 0.8
    top_k: 50
```

Create an additional override file (e.g., `production.yaml`):

```yaml
# production.yaml
args:
  world_size: 8
  max_batch_size: 32
  transforms:
    compile_model:
      backend: torch-opt
```

Then use these configurations:

```bash
# Using single YAML config
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --yaml-extra my_config.yaml

# Using multiple YAML configs (deep merged in order, later files have higher priority)
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --yaml-extra my_config.yaml production.yaml

# Targeting nested AutoDeployConfig with separate YAML
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --yaml-extra my_config.yaml \
  --args.yaml-extra autodeploy_overrides.yaml
```

## Configuration Precedence and Deep Merging

The configuration system follows a precedence order in which higher priority sources override lower priority ones:

1. **CLI Arguments** (highest priority) - Direct command line arguments
1. **YAML Configs** - Files specified via `--yaml-extra` and `--args.yaml-extra`
1. **Default Settings** (lowest priority) - Built-in defaults from the config classes

**Deep Merging**: Unlike simple overwriting, deep merging recursively combines nested dictionaries. For example:

```yaml
# Base config
args:
  model_kwargs:
    num_hidden_layers: 10
    hidden_size: 1024
  max_seq_len: 2048
```

```yaml
# Override config
args:
  model_kwargs:
    hidden_size: 2048  # This will override
    # num_hidden_layers: 10 remains unchanged
  world_size: 4  # This gets added
```

**Nested Config Behavior**: When using nested configurations, outer YAML configuration files become initialization settings for inner objects, giving them higher precedence:

```bash
# The outer yaml-extra affects the entire ExperimentConfig
# The inner args.yaml-extra affects only the AutoDeployConfig
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --yaml-extra experiment_config.yaml \
  --args.yaml-extra autodeploy_config.yaml \
  --args.world-size=8  # CLI override beats both YAML configs
```

## Built-in Default Configuration

Both `AutoDeployConfig` and `LlmArgs` classes automatically load a built-in `default.yaml` configuration file that provides defaults for the AutoDeploy inference optimizer pipeline. This file is specified in the `_get_config_dict()` function in `tensorrt_llm._torch.auto_deploy.llm_args` and defines default transform configurations for graph optimization stages.

The built-in defaults are automatically merged with your configurations at the lowest priority level, ensuring that your custom settings always override the defaults. You can inspect the current default configuration to understand the baseline transform pipeline:

```bash
# View the default configuration
cat tensorrt_llm/_torch/auto_deploy/config/default.yaml

# Override specific transform settings
python build_and_run_ad.py \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --args.transforms.export-to-gm.strict=true
```
