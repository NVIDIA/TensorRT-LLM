# 🔥🚀⚡ AutoDeploy Examples

This folder contains runnable examples for **AutoDeploy**. For general AutoDeploy documentation, motivation, support matrix, and feature overview, please see the [official docs](https://nvidia.github.io/TensorRT-LLM/torch/auto_deploy/auto-deploy.html).

______________________________________________________________________

## Quick Start

AutoDeploy is included with the TRT-LLM installation.

```bash
sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
```

You can refer to [TRT-LLM installation guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/linux.md) for more information.

Run a simple example with a Hugging Face model:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

______________________________________________________________________

## Example Run Script ([`build_and_run_ad.py`](./build_and_run_ad.py))

This script demonstrates end-to-end deployment of HuggingFace checkpoints using AutoDeploy’s graph-transformation pipeline.

You can configure your experiment with various options. Use the `-h/--help` flag to see available options:

```bash
python build_and_run_ad.py --help
```

Below is a non-exhaustive list of common configuration options:

| Configuration Key | Description |
|-------------------|-------------|
| `--model` | The HF model card or path to a HF checkpoint folder |
| `--args.model-factory` | Choose model factory implementation (`"AutoModelForCausalLM"`, ...) |
| `--args.skip-loading-weights` | Only load the architecture, not the weights |
| `--args.model-kwargs` | Extra kwargs that are being passed to the model initializer in the model factory |
| `--args.tokenizer-kwargs` | Extra kwargs that are being passed to the tokenizer initializer in the model factory |
| `--args.world-size` | The number of GPUs used for auto-sharding the model |
| `--args.runtime` | Specifies which type of Engine to use during runtime (`"demollm"` or `"trtllm"`) |
| `--args.compile-backend` | Specifies how to compile the graph at the end |
| `--args.attn-backend` | Specifies kernel implementation for attention |
| `--args.mla-backend` | Specifies implementation for multi-head latent attention |
| `--args.max-seq-len` | Maximum sequence length for inference/cache |
| `--args.max-batch-size` | Maximum dimension for statically allocated KV cache |
| `--args.attn-page-size` | Page size for attention |
| `--prompt.batch-size` | Number of queries to generate |
| `--benchmark.enabled` | Whether to run the built-in benchmark (true/false) |

For default values and additional configuration options, refer to the [`ExperimentConfig`](./build_and_run_ad.py) class in [build_and_run_ad.py](./build_and_run_ad.py) file.

The following is a more complete example of using the script:

```bash
cd examples/auto_deploy
python build_and_run_ad.py \
--model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--args.world-size 2 \
--args.runtime "demollm" \
--args.compile-backend "torch-compile" \
--args.attn-backend "flashinfer" \
--benchmark.enabled True
```

### Advanced Configuration

The script supports flexible configs:

- CLI dot notation for nested fields
- YAML configs with deep merge
- Precedence: CLI > YAML > defaults

The default level is `INFO`.

### Model Evaluation with LM Evaluation Harness

lm-evaluation-harness is supported. To run the evaluation, please use the following command:

```bash
# model is defined the same as above. Other config args can also be specified in the model_args (comma separated).
# You can specify any tasks supported with lm-evaluation-harness.
cd examples/auto_deploy
python lm_eval_ad.py \
--model autodeploy --model_args model=meta-llama/Meta-Llama-3.1-8B-Instruct,world_size=2 --tasks mmlu
```

### Mixed-precision Quantization using TensorRT Model Optimizer

TensorRT Model Optimizer [AutoQuantize](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) algorithm is a PTQ algorithm from ModelOpt which quantizes a model by searching for the best quantization format per-layer while meeting the performance constraint specified by the user. This way, `AutoQuantize` enables to trade-off model accuracy for performance.

Currently `AutoQuantize` supports only `effective_bits` as the performance constraint (for both weight-only quantization and weight & activation quantization). See
[AutoQuantize documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.model_quant.html#modelopt.torch.quantization.model_quant.auto_quantize) for more details.

#### 1. Quantize a model with ModelOpt

Refer to [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_autodeploy/README.md) for generating quantized model checkpoint.

#### 2. Deploy the quantized model with AutoDeploy

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model "<MODELOPT_CKPT_PATH>" --args.world-size 1
```

### Incorporating `auto_deploy` into your own workflow

AutoDeploy can be seamlessly integrated into your existing workflows using TRT-LLM's LLM high-level API. This section provides a blueprint for configuring and invoking AutoDeploy within your custom applications.

Here is an example of how you can build an LLM object with AutoDeploy integration:

```
from tensorrt_llm._torch.auto_deploy import LLM


# Construct the LLM high-level interface object with autodeploy as backend
llm = LLM(
    model=<HF_MODEL_CARD_OR_DIR>,
    world_size=<DESIRED_WORLD_SIZE>,
    compile_backend="torch-compile",
    model_kwargs={"num_hidden_layers": 2}, # test with smaller model configuration
    attn_backend="flashinfer", # choose between "triton" and "flashinfer"
    attn_page_size=64, # page size for attention (tokens_per_block, should be == max_seq_len for triton)
    skip_loading_weights=False,
    model_factory="AutoModelForCausalLM", # choose appropriate model factory
    free_mem_ratio=0.8, # fraction of available memory for cache
    max_seq_len=<MAX_SEQ_LEN>,
    max_batch_size=<MAX_BATCH_SIZE>,
)

```

Please consult the [AutoDeploy `LLM` API](../../tensorrt_llm/_torch/auto_deploy/llm.py) and the
[`AutoDeployConfig` class](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)
for more detail on how AutoDeploy is configured via the `**kwargs` of the `LLM` API.

### Expert Configuration of LLM API

For expert TensorRT LLM users, we also expose the full set of [`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)
*at your own risk* (the argument list diverges from TRT-LLM's argument list):

<details>
<summary>Click to expand for more details on using LlmArgs directly</summary>

- All config fields that are used by the AutoDeploy core pipeline (i.e. the `InferenceOptimizer`) are
  _exclusively_ exposed in the [`AutoDeployConfig` class](../../tensorrt_llm/_torch/auto_deploy/llm_args.py).
  Please make sure to refer to those first.
- For expert users we expose the full set of [`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)
  that can be used to configure the [AutoDeploy `LLM` API](../../tensorrt_llm/_torch/auto_deploy/llm.py) including runtime options.
- Note that some fields in the full [`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)
  object are overlapping, duplicated, and/or _ignored_ in AutoDeploy, particularly arguments
  pertaining to configuring the model itself since AutoDeploy's model ingestion+optimize pipeline
  significantly differs from the default manual workflow in TensorRT-LLM.
- However, with the proper care the full [`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)
  objects can be used to configure advanced runtime options in TensorRT-LLM.
- Note that any valid field can be simply provided as keyword argument ("`**kwargs`") to the
  [AutoDeploy `LLM` API](../../tensorrt_llm/_torch/auto_deploy/llm.py).

</details>

### Expert Configuration of `build_and_run_ad.py`

For expert users, `build_and_run_ad.py` provides advanced configuration capabilities through a flexible argument parser powered by PyDantic Settings and OmegaConf. You can use dot notation for CLI arguments, provide multiple YAML configuration files, and leverage sophisticated configuration precedence rules to create complex deployment configurations.

<details>
<summary>Click to expand for detailed configuration examples</summary>

#### CLI Arguments with Dot Notation

The script supports flexible CLI argument parsing using dot notation to modify nested configurations dynamically. You can target any field in both the [`ExperimentConfig`](./build_and_run_ad.py) and nested [`AutoDeployConfig`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)/[`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.) objects:

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

# Configure runtime and backend settings
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

#### YAML Configuration Files

Both [`ExperimentConfig`](./build_and_run_ad.py) and [`AutoDeployConfig`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py)/[`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py) inherit from [`DynamicYamlMixInForSettings`](../../tensorrt_llm/_torch/auto_deploy/utils/_config.py), enabling you to provide multiple YAML configuration files that are automatically deep-merged at runtime.

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

#### Configuration Precedence and Deep Merging

The configuration system follows a strict precedence order where higher priority sources override lower priority ones:

1. **CLI Arguments** (highest priority) - Direct command line arguments
1. **YAML Extra Configs** - Files specified via `--yaml-extra` and `--args.yaml-extra`
1. **YAML Default Config** - (**do not change**) Files specified via `--yaml-default` and `--args.yaml-default`
1. **Default Settings** (lowest priority) - Built-in defaults from the config classes

**Deep Merging**: Unlike simple overwriting, deep merging intelligently combines nested dictionaries recursively. For example:

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

**Nested Config Behavior**: When using nested configurations, outer YAML configs become init settings for inner objects, giving them higher precedence:

```bash
# The outer yaml-extra affects the entire ExperimentConfig
# The inner args.yaml-extra affects only the AutoDeployConfig
python build_and_run_ad.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --yaml-extra experiment_config.yaml \
  --args.yaml-extra autodeploy_config.yaml \
  --args.world-size=8  # CLI override beats both YAML configs
```

#### Built-in Default Configuration

Both [`AutoDeployConfig`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py) and [`LlmArgs`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py) classes automatically load a built-in [`default.yaml`](../../tensorrt_llm/_torch/auto_deploy/config/default.yaml) configuration file that provides sensible defaults for the AutoDeploy inference optimizer pipeline. This file is specified via the [`yaml_default`](../../tensorrt_llm/_torch/auto_deploy/llm_args.py) field and defines default transform configurations for graph optimization stages.

The built-in defaults are automatically merged with your configurations at the lowest priority level, ensuring that your custom settings always override the defaults. You can inspect the current default configuration to understand the baseline transform pipeline:

```bash
# View the default configuration
cat tensorrt_llm/_torch/auto_deploy/config/default.yaml

# Override specific transform settings
python build_and_run_ad.py \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --args.transforms.export-to-gm.strict=true
```

As indicated before, this can be overwritten via the `yaml_default` (`--yaml-default`) field but note that this will overwrite the entire Inference Optimizer pipeline.

</details>

## Roadmap

Check out our [Github Project Board](https://github.com/orgs/NVIDIA/projects/83) to learn more about
the current progress in AutoDeploy and where you can help.

## Disclaimer

This project is under active development and is currently in a prototype stage. The code is experimental, subject to change, and may include backward-incompatible updates. While we strive for correctness, there are no guarantees regarding functionality, stability, or reliability.
