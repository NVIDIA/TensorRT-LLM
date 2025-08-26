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

Please refer to [Expert Configuration of LLM API](https://nvidia.github.io/TensorRT-LLM/torch/auto_deploy/advanced/expert_configurations.html) for details.

## Disclaimer

This project is under active development and is currently in a prototype stage. The code is experimental, subject to change, and may include backward-incompatible updates. While we strive for correctness, there are no guarantees regarding functionality, stability, or reliability.
