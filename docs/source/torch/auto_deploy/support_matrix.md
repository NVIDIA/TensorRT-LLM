## Support Matrix

AutoDeploy streamlines the model deployment process through an automated workflow designed for efficiency and performance. The workflow begins with a PyTorch model, which is exported using `torch.export` to generate a standard Torch graph. This graph contains core PyTorch ATen operations alongside custom attention operations, determined by the attention backend specified in the configuration.

The exported graph then undergoes a series of automated transformations, including graph sharding, KV-cache insertion, and GEMM fusion, to optimize model performance. After these transformations, the graph is compiled using one of the supported compile backends (like `torch-opt`), followed by deploying it via the TRT-LLM runtime.

### Supported Models

**Bring Your Own Model**: AutoDeploy leverages `torch.export` and dynamic graph pattern matching, enabling seamless integration for a wide variety of models without relying on hard-coded architectures.

We support Hugging Face models that are compatible with `AutoModelForCausalLM` and `AutoModelForImageTextToText`.
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

AutoDeploy supports models with various precision formats, including quantized checkpoints generated by [`TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

**Supported precision types include:**

- BF16 / FP16 / FP32
- FP8
- [NVFP4](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
