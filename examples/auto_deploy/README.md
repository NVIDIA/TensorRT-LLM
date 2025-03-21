<div align="center">

# 🔥🚀⚡ AutoDeploy

<h4> Seamless Model Deployment from PyTorch to TRT-LLM</h4>

<div align="left">

AutoDeploy is designed to simplify and accelerate the deployment of PyTorch models, including off-the-shelf models like those from Hugging Face, to TensorRT-LLM. It automates graph transformations to integrate inference optimizations such as tensor parallelism, KV-caching and quantization. AutoDeploy supports optimized in-framework deployment, minimizing the amount of manual modification needed.

______________________________________________________________________

## Latest News 🔥

- \[2025/02/14\] Initial experimental release of `auto_deploy` backend for TensorRT-LLM

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
sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools<77.0.1 && pip3 install tensorrt_llm
```

You can refer to [TRT-LLM installation guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation/linux.md) for more information.

2. **Run Llama Example:**

You are ready to run an in-framework LLama Demo now.

The general entrypoint to run the auto-deploy demo is the `build_and_run_ad.py` script, Checkpoints are loaded directly from Huggingface (HF) or a local HF-like directory:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --config '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'
```

______________________________________________________________________

## Support Matrix

AutoDeploy streamlines the model deployment process through an automated workflow designed for efficiency and performance. The workflow begins with a PyTorch model, which is exported using `torch.export` to generate a standard Torch graph. This graph contains core PyTorch ATen operations alongside custom attention operations, determined by the attention backend specified in the configuration.

The exported graph then undergoes a series of automated transformations, including graph sharding, KV-cache insertion, and GEMM fusion, to optimize model performance. After these transformations, the graph is compiled using one of the supported compile backends (like `torch-opt`), followed by deploying it via the TRT-LLM runtime.

### Supported Models

AutoDeploy officially supports a range of Hugging Face models that have been tested to work out of the box:

| Model Series | HF Model Card | Precision | Supported Config |
| ------ | ------ |------ | ------ |
| LLaMA | meta-llama/Llama-2-7b-chat-hf<br>meta-llama/Meta-Llama-3.1-8B-Instruct<br>meta-llama/Llama-3.1-70B-Instruct<br>codellama/CodeLlama-13b-Instruct-hf | BF16 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple, torch-opt<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |
| Nvidia Minitron | nvidia/Llama-3_1-Nemotron-51B-Instruct<br>nvidia/Llama-3.1-Minitron-4B-Width-Base<br>nvidia/Llama-3.1-Minitron-4B-Depth-Base | BF16 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple, torch-opt<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |
| Nvidia Model Optimizer | nvidia/Llama-3.1-8B-Instruct-FP8<br>nvidia/Llama-3.1-405B-Instruct-FP8 | FP8 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple, torch-opt<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |
| DeepSeek | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | BF16 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple, torch-opt<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |
| Mistral | mistralai/Mixtral-8x7B-Instruct-v0.1<br>mistralai/Mistral-7B-Instruct-v0.3 | BF16 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |
| BigCode | bigcode/starcoder2-15b | FP32 | world_size:1,2,4<br>runtime:demollm,trtllm<br>compile_backend:torch-simple, torch-opt<br>attn_backend:TritonWithFlattenedInputs,FlashInfer |

### Runtime Integrations

AutoDeploy runs natively with the entire `TRT-LLM` stack via the `LLM` API. In addition, we provide a light-weight wrapper of the `LLM` API for onboarding and debugging new models:

- **TRT-LLM runtime**: A robust, production-grade runtime optimized for high-performance inference. Enable it with `"runtime": "trtllm"` in the configuration.
- **Lightweight development environment**: A lightweight runtime wrapper designed for development and testing, featuring a naive scheduler and KV-cache manager for simplified debugging and testing. Activate it using `"runtime": "demollm"` in the configuration.

### Compile Backends

AutoDeploy supports multiple backends for compiling the exported Torch graph:

- **torch-simple:** Exports the graph without additional optimizations. Enable with `"compile_backend": "torch-simple"` in the configuration.
- **torch-opt:** Uses `torch.compile` along with CUDA Graph capture to enhance inference performance. Enable with `"compile_backend": "torch-opt"` in the configuration.

### Attention backends

Optimize attention operations using different attention kernel implementations:

- **Triton:** Custom fused multi-head attention (MHA) with KV Cache kernels for efficient attention processing. Enable with `"attn_backend": "TritonWithFlattenedInputs"`.
- **FlashInfer:** Uses off-the-shelf optimized attention kernels with KV Cache from the [`flashinfer`](https://github.com/flashinfer-ai/flashinfer.git) library. Enable with `"attn_backend": "FlashInfer"`.

### Precision Support

AutoDeploy supports a range of precision formats to enhance model performance, including:

- BF16, FP32
- Quantization formats like FP8.

______________________________________________________________________

## Advanced Usage

### Example Build Script ([`build_and_run_ad.py`](./build_and_run_ad.py))

#### Base Command

To build and run AutoDeploy example, use the following command with the [`build_and_run_ad.py`](./build_and_run_ad.py) script:

In the below example:

- `model` is the HF model card or path to a HF checkpoint folder.
- `world_size` is the number of GPUs for Tensor Parallel, default is `0` .
- `runtime` specify which type of Engine to use during runtime, default is `demollm` .
- `compile_backend` specify how to compile the graph from `torch.export`, default is `torch-opt` .
- `attn_backend` specify kernel implementation for attention, default is `TritonWithFlattenedInputs`.
- `benchmark` indicates whether to run the built-in benchmark for token generation, default is `False`.

```bash
cd examples/auto_deploy
python build_and_run_ad.py \
--config '{"model": {HF_modelcard_or_path_to_local_folder}, "world_size": {num_GPUs}, "runtime": {"demollm"|"trtllm"}, "compile_backend": {"torch-simple"|"torch-opt"}, "attn_backend": {"TritonWithFlattenedInputs"|"FlashInfer"}, "benchmark": {true|false} }'
```

#### Experiment Configuration

The experiment configuration `dataclass` is defined in
[simple_config.py](./simple_config.py). Check it out for detailed documentation on each
available configuration.

Arguments can be overwritten during runtime by specifying the `--config` argument on the command
line and providing a valid config dictionary in `json` format. For example, to run any experiment
with benchmarking enabled, use:

```bash
cd examples/auto_deploy
python build_and_run_ad.py --config '{"benchmark": true}'
```

The `model_kwargs` and `tokenizer_kwargs` dictionaries can be supplied on the command line via
`--model-kwargs '{}'` and `--tokenizer-kwargs '{}'`.

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
python build_and_run_ad.py --config '{"world_size": 1, "model": "{<MODELOPT_CKPT_PATH>}"}'
```

### Incorporating `auto_deploy` into your own workflow

AutoDeploy can be seamlessly integrated into your existing workflows using TRT-LLM's LLM high-level API. This section provides a blueprint for configuring and invoking AutoDeploy within your custom applications.

Here is an example of how you can build an LLM object with AutoDeploy integration:

```
from tensorrt_llm import LLM
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm._torch.auto_deploy.shim import AutoDeployConfig

# 1. Set up the build configuration
build_config = BuildConfig(
    max_seq_len=<MAX_SEQ_LEN>,
    max_batch_size=<MAX_BS>,
)
build_config.plugin_config.tokens_per_block = <PAGE_SIZE>
# if using "TritonWithFlattenedInputs" as backend, <PAGE_SIZE> should equal to <MAX_SEQ_LEN>
# Refer to examples/auto_deploy/simple_config.py (line 109) for details.

# 2. Set up AutoDeploy configuration
# AutoDeploy will use its own cache implementation
model_kwargs = {"use_cache":False}

ad_config = AutoDeployConfig(
    use_cuda_graph=True, # set True if using "torch-opt" as compile backend
    torch_compile_enabled=True, # set True if using "torch-opt" as compile backend
    model_kwargs=model_kwargs,
    attn_backend="TritonWithFlattenedInputs", # choose between "TritonWithFlattenedInputs" and "FlashInfer"
    skip_loading_weights=False,
)

# 3. Construct the LLM high-level interface object with autodeploy as backend
llm = LLM(
    model=<HF_MODEL_CARD_OR_DIR>,
    backend="autodeploy",
    build_config=build_config,
    pytorch_backend_config=ad_config,
    tensor_parallel_size=<NUM_WORLD_RANK>,
)

```

For more examples on TRT-LLM LLM API, visit [`this page`](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/).

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
