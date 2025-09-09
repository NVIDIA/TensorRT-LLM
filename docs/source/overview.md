(product-overview)=

# Overview

## About TensorRT LLM

[TensorRT LLM](https://developer.nvidia.com/tensorrt) is NVIDIA's comprehensive open-source library for accelerating and optimizing inference performance of the latest large language models (LLMs) on NVIDIA GPUs. 

## Key Capabilities

### 🔥 **Architected on Pytorch**

TensorRT LLM provides a high-level Python [LLM API](./quick-start-guide.md#run-offline-inference-with-llm-api) that supports a wide range of inference setups - from single-GPU to multi-GPU or multi-node deployments. It includes built-in support for various parallelism strategies and advanced features. The LLM API integrates seamlessly with the broader inference ecosystem, including NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) and the [Triton Inference Server](https://github.com/triton-inference-server/server).

TensorRT LLM is designed to be modular and easy to modify. Its PyTorch-native architecture allows developers to experiment with the runtime or extend functionality. Several popular models are also pre-defined and can be customized using [native PyTorch code](source:tensorrt_llm/_torch/models/modeling_deepseekv3.py), making it easy to adapt the system to specific needs.

### ⚡ **State-of-the-Art Performance**

TensorRT LLM delivers breakthrough performance on the latest NVIDIA GPUs:

- **DeepSeek R1**: [World-record inference performance on Blackwell GPUs](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- **Llama 4 Maverick**: [Breaks the 1,000 TPS/User Barrier on B200 GPUs](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

### 🎯 **Comprehensive Model Support**

TensorRT LLM supports the latest and most popular LLM architectures:

### FP4 Support
[NVIDIA B200 GPUs](https://www.nvidia.com/en-us/data-center/dgx-b200/) , when used with TensorRT LLM, enable seamless loading of model weights in the new [FP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/#what_is_nvfp4), allowing you to automatically leverage optimized FP4 kernels for efficient and accurate low-precision inference.

### FP8 Support

TensorRT LLM strives to support the most popular models on **Day 0**.

### 🚀 **Advanced Optimization & Production Features**
- **In-Flight Batching & Paged Attention**: {ref}`inflight-batching` eliminates wait times by dynamically managing request execution, processing context and generation phases together for maximum GPU utilization and reduced latency.
- **Multi-GPU Multi-Node Inference**: Seamless distributed inference with tensor, pipeline, and expert parallelism across multiple GPUs and nodes through the Model Definition API.
- **Advanced Quantization**: 
  - **FP4 Quantization**: Native support on NVIDIA B200 GPUs with optimized FP4 kernels
  - **FP8 Quantization**: Automatic conversion on NVIDIA H100 GPUs leveraging Hopper architecture
- **Speculative Decoding**: Multiple algorithms including EAGLE, MTP and NGram
- **KV Cache Management**: Paged KV cache with intelligent block reuse and memory optimization
- **Chunked Prefill**: Efficient handling of long sequences by splitting context into manageable chunks
- **LoRA Support**: Multi-adapter support with HuggingFace and NeMo formats, efficient fine-tuning and adaptation
- **Checkpoint Loading**: Flexible model loading from various formats (HuggingFace, NeMo, custom)
- **Guided Decoding**: Advanced sampling with stop words, bad words, and custom constraints
- **Disaggregated Serving (Beta)**: Separate context and generation phases across different GPUs for optimal resource utilization

### 🔧 **Latest GPU Architecture Support**

TensorRT LLM supports the full spectrum of NVIDIA GPU architectures:
- **NVIDIA Blackwell**: B200, GB200, RTX Pro 6000 SE with FP4 optimization
- **NVIDIA Hopper**: H100, H200,GH200 with FP8 acceleration
- **NVIDIA Ada Lovelace**: L40/L40S, RTX 40 series with FP8 acceleration
- **NVIDIA Ampere**: A100, RTX 30 series for production workloads

## What Can You Do With TensorRT LLM?

Whether you're building the next generation of AI applications, optimizing existing LLM deployments, or exploring the frontiers of large language model technology, TensorRT LLM provides the tools, performance, and flexibility you need to succeed in the era of generative AI.To get started, refer to the {ref}`quick-start-guide`.
