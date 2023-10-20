<div align="center">

TensorRT-LLM
===========================
<h4> A TensorRT Toolbox for Large Language Models </h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.2-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-9.1-green)](https://developer.nvidia.com/tensorrt)
[![version](https://img.shields.io/badge/release-0.5.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Architecture](./docs/source/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](./docs/source/performance.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](./examples/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](./docs/source/)

---
<div align="left">


## Table of Contents

- [TensorRT-LLM Overview](#tensorrt-llm-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Support Matrix](#support-matrix)
- [Performance](#performance)
- [Advanced Topics](#advanced-topics)
  - [Quantization](#quantization)
  - [In-flight Batching](#in-flight-batching)
  - [Attention](#attention)
  - [Graph Rewriting](#graph-rewriting)
  - [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)
- [Release Notes](#release-notes)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)

## TensorRT-LLM Overview

TensorRT-LLM provides users with an easy-to-use Python API to define Large
Language Models (LLMs) and build
[TensorRT](https://developer.nvidia.com/tensorrt) engines that contain
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
TensorRT-LLM also contains components to create Python and C++ runtimes that
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

The Python API of TensorRT-LLM is architectured to look similar to the
[PyTorch](https://pytorch.org) API. It provides users with a
[functional](./tensorrt_llm/functional.py) module containing functions like
`einsum`, `softmax`, `matmul` or `view`. The [layers](./tensorrt_llm/layers)
module bundles useful building blocks to assemble LLMs; like an `Attention`
block, a `MLP` or the entire `Transformer` layer. Model-specific components,
like `GPTAttention` or `BertAttention`, can be found in the
[models](./tensorrt_llm/models) module.

TensorRT-LLM comes with several popular models pre-defined. They can easily be
modified and extended to fit custom needs. See below for a list of supported
[models](#Models).

To maximize performance and reduce memory footprint, TensorRT-LLM allows the
models to be executed using different quantization modes (see
[`examples/gpt`](./examples/gpt) for concrete examples).  TensorRT-LLM supports
INT4 or INT8 weights (and FP16 activations; a.k.a.  INT4/INT8 weight-only) as
well as a complete implementation of the
[SmoothQuant](https://arxiv.org/abs/2211.10438) technique.

For a more detailed presentation of the software architecture and the key
concepts used in TensorRT-LLM, we recommend you to read the following
[document](./docs/source/architecture.md).

## Installation

*For Windows installation, see [`Windows/`](windows/).*

TensorRT-LLM must be built from source, instructions can be found
[here](./docs/source/installation.md). An image of a Docker container with
TensorRT-LLM and its Triton Inference Server Backend will be made available
soon.

The remaining commands in that document must be executed from the TensorRT-LLM
container.

## Quick Start

To create a TensorRT engine for an existing model, there are 3 steps:

1. Download pre-trained weights,
2. Build a fully-optimized engine of the model,
3. Deploy the engine.

The following sections show how to use TensorRT-LLM to run the
[BLOOM-560m](https://huggingface.co/bigscience/bloom-560m) model.

***0. In the BLOOM folder***

Inside the Docker container, you have to install the requirements:

```bash
pip install -r examples/bloom/requirements.txt
git lfs install
```

***1. Download the model weights from HuggingFace***

From the BLOOM example folder, you must download the weights of the model.

```bash
cd examples/bloom
rm -rf ./bloom/560M
mkdir -p ./bloom/560M && git clone https://huggingface.co/bigscience/bloom-560m ./bloom/560M

```
***2. Build the engine***

```python
# Single GPU on BLOOM 560M
python build.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/1-gpu/
```

See the BLOOM [example](examples/bloom) for more details and options regarding the `build.py` script.

***3. Run***

The `summarize.py` script can be used to perform the summarization of articles
from the CNN Daily dataset:

```python
python summarize.py --test_trt_llm \
                    --hf_model_location ./bloom/560M/ \
                    --data_type fp16 \
                    --engine_dir ./bloom/560M/trt_engines/fp16/1-gpu/
```

More details about the script and how to run the BLOOM model can be found in
the example [folder](examples/bloom). Many more [models](#models) than BLOOM
are implemented in TensorRT-LLM. They can be found in the
[examples](./examples/) directory.

## Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on
NVIDIA GPUs. The following sections provide a list of supported GPU
architectures as well as important features implemented in TensorRT-LLM.

### Devices

TensorRT-LLM is rigorously tested on the following GPUs:

* [H100](https://www.nvidia.com/en-us/data-center/h100/)
* [L40S](https://www.nvidia.com/en-us/data-center/l40s/)
* [A100](https://www.nvidia.com/en-us/data-center/a100/)/[A30](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
* [V100](https://www.nvidia.com/en-us/data-center/v100/) (experimental)

If a GPU is not listed above, it is important to note that TensorRT-LLM is
expected to work on GPUs based on the Volta, Turing, Ampere, Hopper and Ada
Lovelace architectures. Certain limitations may, however, apply.

### Precision

Various numerical precisions are supported in TensorRT-LLM. The support for
some of those numerical features require specific architectures:

|                              | FP32  | FP16  | BF16  | FP8  | INT8 | INT4 |
| :--------------------------- | :---- | :---- | :---- | :--- | :--- | :--- |
| Volta (SM70)                 | Y     | Y     | N     | N    | Y    | Y    |
| Turing (SM75)                | Y     | Y     | N     | N    | Y    | Y    |
| Ampere (SM80, SM86)          | Y     | Y     | Y     | N    | Y    | Y    |
| Ada-Lovelace (SM89)          | Y     | Y     | Y     | Y    | Y    | Y    |
| Hopper (SM90)                | Y     | Y     | Y     | Y    | Y    | Y    |

In this release of TensorRT-LLM, the support for FP8 and quantized data types
(INT8 or INT4) is not implemented for all the models. See the
[precision](./docs/source/precision.md) document and the
[examples](./examples/.) folder for additional details.

### Key Features

TensorRT-LLM contains examples that implement the following features.

* Multi-head Attention([MHA](https://arxiv.org/abs/1706.03762))
* Multi-query Attention ([MQA](https://arxiv.org/abs/1911.02150))
* Group-query Attention([GQA](https://arxiv.org/abs/2307.09288))
* In-flight Batching
* Paged KV Cache for the Attention
* Tensor Parallelism
* Pipeline Parallelism
* INT4/INT8 Weight-Only Quantization (W4A16 & W8A16)
* [SmoothQuant](https://arxiv.org/abs/2211.10438)
* [GPTQ](https://arxiv.org/abs/2210.17323)
* [AWQ](https://arxiv.org/abs/2306.00978)
* [FP8](https://arxiv.org/abs/2209.05433)
* Greedy-search
* Beam-search
* RoPE

In this release of TensorRT-LLM, some of the features are not enabled for all
the models listed in the [examples](examples/.) folder.

### Models

The list of supported models is:

* [Baichuan](examples/baichuan)
* [Bert](examples/bert)
* [Blip2](examples/blip2)
* [BLOOM](examples/bloom)
* [ChatGLM-6B](examples/chatglm6b)
* [ChatGLM2-6B](examples/chatglm2-6b/)
* [Falcon](examples/falcon)
* [GPT](examples/gpt)
* [GPT-J](examples/gptj)
* [GPT-Nemo](examples/gpt)
* [GPT-NeoX](examples/gptneox)
* [LLaMA](examples/llama)
* [LLaMA-v2](examples/llama)
* [MPT](examples/mpt)
* [OPT](examples/opt)
* [SantaCoder](examples/gpt)
* [StarCoder](examples/gpt)

## Performance

Please refer to the [performance](./docs/source/performance.md) page for
performance numbers. That page contains measured numbers for four variants of
popular models (GPT-J, LLAMA-7B, LLAMA-70B, Falcon-180B), measured on the H100,
L40S and A100 GPU(s).

## Advanced Topics

### Quantization

This [document](./docs/source/precision.md) describes the different
quantization methods implemented in TensorRT-LLM and contains a support matrix
for the different models.

### In-flight Batching

TensorRT-LLM supports in-flight batching of requests (also known as continuous
batching or iteration-level batching). It's a
[technique](./docs/source/batch_manager.md) that aims at reducing wait
times in queues, eliminating the need for padding requests and allowing for
higher GPU utilization.

### Attention

TensorRT-LLM implements several variants of the Attention mechanism that
appears in most the Large Language Models.  This
[document](./docs/source/gpt_attention.md) summarizes those implementations and
how they are optimized in TensorRT-LLM.

### Graph Rewriting

TensorRT-LLM uses a declarative approach to define neural networks and contains
techniques to optimize the underlying graph. For more details, please refer to
[doc](./docs/source/graph-rewriting.md)

### Benchmark

TensorRT-LLM provides [C++](./benchmarks/cpp/README.md) and
[Python](./benchmarks/python/README.md) tools to perform benchmarking. Note,
however, that it is recommended to use the C++ version.

## Troubleshooting


* It's recommended to add options `–shm-size=1g –ulimit memlock=-1` to the
  docker or nvidia-docker run command.  Otherwise you may see NCCL errors when
  running multiple GPU inferences. See
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#errors
  for details.

* When building models, memory-related issues such as
```
[09/23/2023-03:13:00] [TRT] [E] 9: GPTLMHeadModel/layers/0/attention/qkv/PLUGIN_V2_Gemm_0: could not find any supported formats consistent with input/output data types
[09/23/2023-03:13:00] [TRT] [E] 9: [pluginV2Builder.cpp::reportPluginError::24] Error Code 9: Internal Error (GPTLMHeadModel/layers/0/attention/qkv/PLUGIN_V2_Gemm_0: could not find any supported formats consistent with input/output data types)
```
may happen. One possible solution is to reduce the amount of memory needed by
reducing the maximum batch size, input and output lengths. Another option is to
enable plugins, for example: `--use_gpt_attention_plugin`.

## Release notes

  * TensorRT-LLM requires TensorRT 9.1.0.4 and 23.08 containers.

### Change Log

  * TensorRT-LLM v0.5.0 is the first public release.

### Known Issues

### Report Issues

You can use GitHub issues to report issues with TensorRT-LLM.
