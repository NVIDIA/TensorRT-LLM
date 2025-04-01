(product-overview)=

# Overview

## About TensorRT-LLM

[TensorRT-LLM](https://developer.nvidia.com/tensorrt) accelerates and optimizes inference performance for the latest large language models (LLMs) on NVIDIA GPUs. This open-source library is available for free on the [TensorRT-LLM GitHub repo](https://github.com/NVIDIA/TensorRT-LLM) and as part of the [NVIDIA NeMo framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/).

LLMs have revolutionized the field of artificial intelligence and created entirely new ways of interacting with the digital world. But, as organizations and application developers around the world look to incorporate LLMs into their work, some of the challenges with running these models become apparent. Put simply, LLMs are large. That fact can make them expensive and slow to run without the right techniques.

TensorRT-LLM offers a comprehensive library for compiling and optimizing LLMs for inference. TensorRT-LLM incorporates all of the optimizations (that is, kernel fusion and quantization, runtime optimizations like C++ implementations, KV caching, continuous in-flight batching, and paged attention) and more, while providing an intuitive Model Definition API for defining and building new models.

Some of the major benefits that TensorRT-LLM provides are:

### Common LLM Support

TensorRT-LLM supports the latest LLMs. Refer to the {ref}`support-matrix-software` for the full list.

### In-Flight Batching and Paged Attention

{ref}`inflight-batching` takes advantage of the overall text generation process for an LLM can be broken down into multiple iterations of execution on the model. Rather than waiting for the whole batch to finish before moving on to the next set of requests, the TensorRT-LLM runtime immediately evicts finished sequences from the batch. It then begins executing new requests while other requests are still in flight. It's a {ref}`executor` that aims at reducing wait times in queues, eliminating the need for padding requests, and allowing for higher GPU utilization.

### Multi-GPU Multi-Node Inference

TensorRT-LLM consists of pre– and post-processing steps and multi-GPU multi-node communication primitives in a simple, open-source Model Definition API for groundbreaking LLM inference performance on GPUs. Refer to the {ref}`multi-gpu-multi-node` section for more information.

### FP8 Support

[NVIDIA H100 GPUs](https://www.nvidia.com/en-us/data-center/dgx-h100/) with TensorRT-LLM give you the ability to convert model weights into a new FP8 format easily and compile models to take advantage of optimized FP8 kernels automatically. This is made possible through [NVIDIA Hopper](https://blogs.nvidia.com/blog/h100-transformer-engine/) and done without having to change any model code.

### Latest GPU Support

TensorRT-LLM supports GPUs based on the NVIDIA Hopper, NVIDIA Ada Lovelace, and NVIDIA Ampere architectures.
Certain limitations might apply. Refer to the {ref}`support-matrix` for more information.

### Native Windows Support

Windows platform support is deprecated as of v0.18.0. All Windows-related code and functionality will be completely removed in future releases.

## What Can You Do With TensorRT-LLM?

Let TensorRT-LLM accelerate inference performance on the latest LLMs on NVIDIA GPUs. Use TensorRT-LLM as an optimization backbone for LLM inference in NVIDIA NeMo, an end-to-end framework to build, customize, and deploy generative AI applications into production. NeMo provides complete containers, including TensorRT-LLM and NVIDIA Triton, for generative AI deployments.

TensorRT-LLM improves ease of use and extensibility through an open-source modular Model Definition API for defining, optimizing, and executing new architectures and enhancements as LLMs evolve, and can be customized easily.

If you’re eager to dive into the world of LLMs, now is the time to get started with TensorRT-LLM. Explore its capabilities, experiment with different models and optimizations, and embark on your journey to unlock the incredible power of AI-driven language models. To get started, refer to the {ref}`quick-start-guide`.
