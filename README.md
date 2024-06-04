<div align="center">

TensorRT-LLM
===========================
<h4> A TensorRT Toolbox for Optimized Large Language Model Inference</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.4.1-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.0.1-green)](https://developer.nvidia.com/tensorrt)
[![version](https://img.shields.io/badge/release-0.11.0.dev-green)](./tensorrt_llm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Architecture](./docs/source/architecture/overview.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](./docs/source/performance/perf-overview.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](./examples/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](./docs/source/)

---
<div align="left">

## Latest News
* [*Weekly*] Check out **[@NVIDIAAIDev](https://twitter.com/nvidiaaidev?lang=en)** & **[NVIDIA AI](https://www.linkedin.com/showcase/nvidia-ai/)** LinkedIn for the latest updates!
* [2024/02/06] [🚀 Speed up inference with SOTA quantization techniques in TRT-LLM](./docs/source/blogs/quantization-in-TRT-LLM.md)
* [2024/01/30] [ New XQA-kernel provides 2.4x more Llama-70B throughput within the same latency budget](./docs/source/blogs/XQA-kernel.md)
* [2023/12/04] [Falcon-180B on a single H200 GPU with INT4 AWQ, and 6.7x faster Llama-70B over A100](./docs/source/blogs/Falcon180B-H200.md)
* [2023/11/27] [SageMaker LMI now supports TensorRT-LLM - improves throughput by 60%, compared to previous version](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
* [2023/11/13] [H200 achieves nearly 12,000 tok/sec on Llama2-13B](./docs/source/blogs/H200launch.md)
* [2023/10/22] [🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙](https://github.com/NVIDIA/trt-llm-rag-windows#readme)
* [2023/10/19] Getting Started Guide - [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available
](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
* [2023/10/17] [Large Language Models up to 4x Faster on RTX With TensorRT-LLM for Windows
](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)

## TensorRT-LLM Overview

TensorRT-LLM is an easy-to-use Python API to define Large
Language Models (LLMs) and build
[TensorRT](https://developer.nvidia.com/tensorrt) engines that contain
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
TensorRT-LLM contains components to create Python and C++ runtimes that
execute those TensorRT engines. It also includes a
[backend](https://github.com/triton-inference-server/tensorrtllm_backend)
for integration with the
[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server);
a production-quality system to serve LLMs.  Models built with TensorRT-LLM can
be executed on a wide range of configurations going from a single GPU to
multiple nodes with multiple GPUs (using
[Tensor Parallelism](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/parallelisms.html#tensor-parallelism)
and/or
[Pipeline Parallelism](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/parallelisms.html#pipeline-parallelism)).

The TensorRT-LLM Python API architecture looks similar to the
[PyTorch](https://pytorch.org) API. It provides a
[functional](./tensorrt_llm/functional.py) module containing functions like
`einsum`, `softmax`, `matmul` or `view`. The [layers](./tensorrt_llm/layers)
module bundles useful building blocks to assemble LLMs; like an `Attention`
block, a `MLP` or the entire `Transformer` layer. Model-specific components,
like `GPTAttention` or `BertAttention`, can be found in the
[models](./tensorrt_llm/models) module.

TensorRT-LLM comes with several popular models pre-defined. They can easily be
modified and extended to fit custom needs. Refer to the [Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) for a list of supported models.

To maximize performance and reduce memory footprint, TensorRT-LLM allows the
models to be executed using different quantization modes (refer to
[`support matrix`](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html#software)).  TensorRT-LLM supports
INT4 or INT8 weights (and FP16 activations; a.k.a.  INT4/INT8 weight-only) as
well as a complete implementation of the
[SmoothQuant](https://arxiv.org/abs/2211.10438) technique.

## Getting Started

To get started with TensorRT-LLM, visit our documentation:

- [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
- [Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [Installation Guide for Linux](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
- [Installation Guide for Windows](https://nvidia.github.io/TensorRT-LLM/installation/windows.html)
- [Supported Hardware, Models, and other Software](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)

## Community
- [Model zoo](https://huggingface.co/TheFloat16) (generated by TRT-LLM rel 0.9 a9356d4b7610330e89c1010f342a9ac644215c52)
