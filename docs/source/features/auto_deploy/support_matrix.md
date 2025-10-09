## Support Matrix

AutoDeploy streamlines model deployment with an automated workflow designed for efficiency and performance. The workflow begins with a PyTorch model, which is exported using `torch.export` to generate a standard Torch graph. This graph contains core PyTorch ATen operations alongside custom attention operations, determined by the attention backend specified in the configuration.

The exported graph then undergoes a series of automated transformations, including graph sharding, KV-cache insertion, and GEMM fusion, to optimize model performance. After these transformations, the graph is compiled using one of the supported compile backends (like `torch-opt`), followed by deploying it via the TRT-LLM runtime.

### Support Models

**Bring Your Own Model**: AutoDeploy leverages `torch.export` and dynamic graph pattern matching, enabling seamless integration for a wide variety of models without relying on hard-coded architectures.

AutoDeploy supports Hugging Face models compatible with `AutoModelForCausalLM` and `AutoModelForImageTextToText`.
In addition, the following models have been officially validated using the default configuration: `runtime=trtllm`, `compile_backend=torch-compile`, and `attn_backend=flashinfer`

<details>
<summary>Click to expand supported models list</summary>

- Qwen/QwQ-32B
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-235B-A22B
- Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-4B
- Qwen/Qwen3-8B
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- apple/OpenELM-1_1B-Instruct
- apple/OpenELM-270M-Instruct
- apple/OpenELM-3B-Instruct
- apple/OpenELM-450M-Instruct
- bigcode/starcoder2-15b-instruct-v0.1
- bigcode/starcoder2-7b
- deepseek-ai/DeepSeek-Prover-V1.5-SFT
- deepseek-ai/DeepSeek-Prover-V2-7B
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- google/codegemma-7b-it
- google/gemma-1.1-7b-it
- google/gemma-2-27b-it
- google/gemma-2-2b-it
- google/gemma-2-9b-it
- google/gemma-2b
- google/gemma-3-1b-it
- ibm-granite/granite-3.1-2b-instruct
- ibm-granite/granite-3.1-8b-instruct
- ibm-granite/granite-3.3-2b-instruct
- ibm-granite/granite-3.3-8b-instruct
- ibm-granite/granite-guardian-3.1-2b
- ibm-granite/granite-guardian-3.2-5b
- meta-llama/CodeLlama-34b-Instruct-hf
- meta-llama/CodeLlama-7b-Instruct-hf
- meta-llama/CodeLlama-7b-Python-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct
- meta-llama/Llama-3.3-70B-Instruct
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- microsoft/Phi-3-medium-128k-instruct
- microsoft/Phi-3-medium-4k-instruct
- microsoft/Phi-4-mini-instruct
- microsoft/Phi-4-mini-reasoning
- microsoft/Phi-4-reasoning
- microsoft/Phi-4-reasoning-plus
- microsoft/phi-4
- mistralai/Codestral-22B-v0.1
- mistralai/Mistral-7B-Instruct-v0.2
- mistralai/Mistral-7B-Instruct-v0.3
- mistralai/Mixtral-8x22B-Instruct-v0.1
- nvidia/Llama-3.1-405B-Instruct-FP8
- nvidia/Llama-3.1-70B-Instruct-FP8
- nvidia/Llama-3.1-8B-Instruct-FP8
- nvidia/Llama-3.1-Minitron-4B-Depth-Base
- nvidia/Llama-3.1-Minitron-4B-Width-Base
- nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
- nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- nvidia/Llama-3_1-Nemotron-51B-Instruct
- nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8
- nvidia/Llama-3_3-Nemotron-Super-49B-v1
- nvidia/Mistral-NeMo-Minitron-8B-Base
- perplexity-ai/r1-1776-distill-llama-70b

</details>

### Runtime Integrations

AutoDeploy runs natively with the complete `TRT-LLM` stack via the `LLM` API. In addition, we provide a light-weight wrapper of the `LLM` API for onboarding and debugging new models:

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

Optimize attention operations with different attention kernel implementations:

| `"attn_backend"` | Description |
|----------------------|-------------|
| `triton` | Custom fused multi-head attention (MHA) with KV Cache kernels for efficient attention processing. |
| `flashinfer`         | Uses optimized attention kernels with KV Cache from the [`flashinfer`](https://github.com/flashinfer-ai/flashinfer.git) library. |

### Precision Support

AutoDeploy supports models with various precision formats, including quantized checkpoints generated by [`TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

**Supported precision types include:**

- BF16 / FP16 / FP32
- FP8
- [NVFP4](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
