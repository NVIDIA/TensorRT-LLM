<div align="center">

# 🔥🚀⚡ AutoDeploy

<h4> Seamless Model Deployment from PyTorch to TRT-LLM</h4>

<div align="left">

AutoDeploy is designed to simplify and accelerate the deployment of PyTorch models, including off-the-shelf models like those from Hugging Face, to TensorRT-LLM. It automates graph transformations to integrate inference optimizations such as tensor parallelism, KV-caching and quantization. AutoDeploy supports optimized in-framework deployment, minimizing the amount of manual modification needed.

______________________________________________________________________

## Motivation & Approach

Deploying large language models (LLMs) can be challenging, especially when balancing ease of use with high performance. Teams need simple, intuitive deployment solutions that reduce engineering effort, speed up the integration of new models, and support rapid experimentation without compromising performance.

AutoDeploy addresses these challenges with a streamlined, (semi-)automated pipeline that transforms in-framework PyTorch models, including Hugging Face models, into optimized inference-ready models for TRT-LLM. It simplifies deployment, optimizes models for efficient inference, and bridges the gap between simplicity and performance.

### **Key Features:**

- **Seamless Model Transition:** Automatically converts PyTorch/Hugging Face models to TRT-LLM without manual rewrites.
- **Unified Model Definition:** Maintain a single source of truth with your original PyTorch/Hugging Face model.
- **Optimized Inference:** Built-in transformations for sharding, quantization, KV-cache integration, MHA fusion, and CudaGraph optimization.
- **Immediate Deployment:** Day-0 support for models with continuous performance enhancements.
- **Quick Setup & Prototyping:** Lightweight pip package for easy installation with a demo environment for fast testing.

______________________________________________________________________

## Get Started

1. **Install AutoDeploy:**

AutoDeploy is accessible through TRT-LLM installation.

```bash
sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
```

You can refer to [TRT-LLM installation guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/linux.md) for more information.

2. **Run Llama Example:**

You are ready to run an in-framework LLama Demo now.

The general entrypoint to run the auto-deploy demo is the `build_and_run_ad.py` script, Checkpoints are loaded directly from Huggingface (HF) or a local HF-like directory:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

______________________________________________________________________

## Support Matrix

AutoDeploy streamlines the model deployment process through an automated workflow designed for efficiency and performance. The workflow begins with a PyTorch model, which is exported using `torch.export` to generate a standard Torch graph. This graph contains core PyTorch ATen operations alongside custom attention operations, determined by the attention backend specified in the configuration.

The exported graph then undergoes a series of automated transformations, including graph sharding, KV-cache insertion, and GEMM fusion, to optimize model performance. After these transformations, the graph is compiled using one of the supported compile backends (like `torch-opt`), followed by deploying it via the TRT-LLM runtime.

### Supported Models

**Bring Your Own Model**: AutoDeploy leverages `torch.export` and dynamic graph pattern matching, enabling seamless integration for a wide variety of models without relying on hard-coded architectures.

Additionally, we have officially verified support for the following models:

<details>
<summary>Click to expand supported models list</summary>

| Model Series | HF Model Card | Model Factory | Precision | World Size | Runtime | Compile Backend ||| Attention Backend |||
|--------------|----------------------|----------------|-----------|------------|---------|-----------------|--------------------|--------------------|--------------------|----------|----------|
|              |               |            |           |            |         | torch-simple    | torch-compile    | torch-opt          | triton | flashinfer | MultiHeadLatentAttention |
| LLaMA        | meta-llama/Llama-2-7b-chat-hf<br>meta-llama/Meta-Llama-3.1-8B-Instruct<br>meta-llama/Llama-3.1-70B-Instruct<br>codellama/CodeLlama-13b-Instruct-hf | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| LLaMA-4      | meta-llama/Llama-4-Scout-17B-16E-Instruct<br>meta-llama/Llama-4-Maverick-17B-128E-Instruct | AutoModelForImageTextToText | BF16 | 1,2,4,8 | demollm, trtllm | ✅ | ✅ | ❌ | ✅ | ✅ | n/a |
| Nvidia Minitron | nvidia/Llama-3_1-Nemotron-51B-Instruct<br>nvidia/Llama-3.1-Minitron-4B-Width-Base<br>nvidia/Llama-3.1-Minitron-4B-Depth-Base | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| Nvidia Model Optimizer | nvidia/Llama-3.1-8B-Instruct-FP8<br>nvidia/Llama-3.1-405B-Instruct-FP8 | AutoModelForCausalLM | FP8 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| DeepSeek     | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| Mistral      | mistralai/Mixtral-8x7B-Instruct-v0.1<br>mistralai/Mistral-7B-Instruct-v0.3 | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| BigCode      | bigcode/starcoder2-15b | AutoModelForCausalLM | FP32 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| Deepseek-V3      | deepseek-ai/DeepSeek-V3 | AutoModelForCausalLM | BF16 | 1,2,4 | demollm | ✅ | ❌ | ❌ | n/a | n/a | ✅ |
| Phi4      | microsoft/phi-4<br>microsoft/Phi-4-reasoning<br>microsoft/Phi-4-reasoning-plus | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅ | ✅ | ✅ | n/a |
| Phi3/2      | microsoft/Phi-3-mini-4k-instruct<br>microsoft/Phi-3-mini-128k-instruct<br>microsoft/Phi-3-medium-4k-instruct<br>microsoft/Phi-3-medium-128k-instruct<br>microsoft/Phi-3.5-mini-instruct | AutoModelForCausalLM | BF16 | 1,2,4 | demollm, trtllm | ✅ | ✅ | ✅(partly) | ✅ | ❌ | n/a |

</details>

### Runtime Integrations

AutoDeploy runs natively with the entire `TRT-LLM` stack via the `LLM` API. In addition, we provide a light-weight wrapper of the `LLM` API for onboarding and debugging new models:

| `"runtime"` | Description |
|-------------|-------------|
| `trtllm`    | A robust, production-grade runtime optimized for high-performance inference. |
| `demollm`   | A lightweight runtime wrapper designed for development and testing, featuring a naive scheduler and KV-cache manager for simplified debugging and testing. |

### Compile Backends

AutoDeploy supports multiple backends for compiling the exported Torch graph:

| `"compile_backend"` | Description |
|--------------------|-------------|
| `torch-simple`     | Exports the graph without additional optimizations. |
| `torch-compile`    | Applies `torch.compile` to the graph after all AutoDeploy transformations have been completed. |
| `torch-cudagraph`  | Performs CUDA graph capture (without torch.compile). |
| `torch-opt`        | Uses `torch.compile` along with CUDA Graph capture to enhance inference performance. |

### Attention backends

Optimize attention operations using different attention kernel implementations:

| `"attn_backend"` | Description |
|----------------------|-------------|
| `triton` | Custom fused multi-head attention (MHA) with KV Cache kernels for efficient attention processing. |
| `flashinfer`         | Uses off-the-shelf optimized attention kernels with KV Cache from the [`flashinfer`](https://github.com/flashinfer-ai/flashinfer.git) library. |

### Precision Support

AutoDeploy supports a range of precision formats to enhance model performance, including:

- BF16, FP32
- Quantization formats like FP8.

______________________________________________________________________

## Advanced Usage

### Example Run Script ([`build_and_run_ad.py`](./build_and_run_ad.py))

To build and run AutoDeploy example, use the [`build_and_run_ad.py`](./build_and_run_ad.py) script:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

You can arbitrarily configure your experiment. Use the `-h/--help` flag to see available options:

```bash
python build_and_run_ad.py --help
```

Below is a non-exhaustive list of common config options:

| Configuration Key | Description |
|-------------------|-------------|
| `--model` | The HF model card or path to a HF checkpoint folder |
| `--args.model-factory` | Choose model factory implementation (`"AutoModelForCausalLM"`, ...) |
| `--args.skip-loading-weights` | Only load the architecture, not the weights |
| `--args.model-kwargs` | Extra kwargs that are being passed to the model initializer in the model factory |
| `--args.tokenizer-kwargs` | Extra kwargs that are being passed to the tokenizer initializer in the model factory |
| `--args.world-size` | The number of GPUs for Tensor Parallel |
| `--args.runtime` | Specifies which type of Engine to use during runtime (`"demollm"` or `"trtllm"`) |
| `--args.compile-backend` | Specifies how to compile the graph at the end |
| `--args.attn-backend` | Specifies kernel implementation for attention |
| `--args.mla-backend` | Specifies implementation for multi-head latent attention |
| `--args.max-seq-len` | Maximum sequence length for inference/cache |
| `--args.max-batch-size` | Maximum dimension for statically allocated KV cache |
| `--args.attn-page-size` | Page size for attention |
| `--prompt.batch-size` | Number of queries to generate |
| `--benchmark.enabled` | Whether to run the built-in benchmark (true/false) |

For default values and additional configuration options, refer to the `ExperimentConfig` class in [build_and_run_ad.py](./build_and_run_ad.py) file.

Here is a more complete example of using the script:

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

#### Logging Level

Use the following env variable to specify the logging level of our built-in logger ordered by
decreasing verbosity;

```bash
AUTO_DEPLOY_LOG_LEVEL=DEBUG
AUTO_DEPLOY_LOG_LEVEL=INFO
AUTO_DEPLOY_LOG_LEVEL=WARNING
AUTO_DEPLOY_LOG_LEVEL=ERROR
AUTO_DEPLOY_LOG_LEVEL=INTERNAL_ERROR
```

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

<details>
<summary>Click to expand the example</summary>

```
from tensorrt_llm._torch.auto_deploy import LLM


# Construct the LLM high-level interface object with autodeploy as backend
llm = LLM(
    model=<HF_MODEL_CARD_OR_DIR>,
    world_size=<NUM_WORLD_RANK>,
    compile_backend="torch-compile",
    model_kwargs={"num_hidden_layers": 2}, # test with smaller model configuration
    attn_backend="flashinfer", # choose between "triton" and "flashinfer"
    attn_page_size=64, # page size for attention (tokens_per_block, should be == max_seq_len for triton)
    skip_loading_weights=False,
    model_factory="AutoModelForCausalLM", # choose appropriate model factory
    mla_backend="MultiHeadLatentAttention", # for models that support MLA
    free_mem_ratio=0.8, # fraction of available memory for cache
    simple_shard_only=False, # tensor parallelism sharding strategy
    max_seq_len=<MAX_SEQ_LEN>,
    max_batch_size=<MAX_BATCH_SIZE>,
)

```

</details>

For more examples on TRT-LLM LLM API, visit [`this page`](https://nvidia.github.io/TensorRT-LLM/examples/llm_api_examples.html).

______________________________________________________________________

## Roadmap

1. **Model Coverage:**

   - Expand support for additional LLM variants and features:
     - LoRA
     - Speculative Decoding
     - Model specialization for disaggregated serving

1. **Performance Optimization:**

   - Enhance inference speed and efficiency with:
     - MoE fusion and all-reduce fusion techniques
     - Reuse of TRT-LLM PyTorch operators for greater efficiency

______________________________________________________________________

## Disclaimer

This project is in active development and is currently in an early (beta) stage. The code is experimental, subject to change, and may include backward-incompatible updates. While we strive for correctness, we provide no guarantees regarding functionality, stability, or reliability. Use at your own risk.
