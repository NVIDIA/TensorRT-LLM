<div align="center">

TensorRT-LLM
===========================
<h4> A TensorRT Toolbox for Optimized Large Language Model Inference</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.2-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-9.2-green)](https://developer.nvidia.com/tensorrt)
[![version](https://img.shields.io/badge/release-0.9.0.dev-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Architecture](./docs/source/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](./docs/source/performance.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](./examples/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](./docs/source/)

---
<div align="left">

## Latest News
* [2024/02/06] [🚀 Speed up inference with SOTA quantization techniques in TRT-LLM](./docs/source/blogs/quantization-in-TRT-LLM.md)
* [2024/01/30] [ New **XQA-kernel** provides **2.4x more Llama-70B throughput** within the same latency budget](./docs/source/blogs/XQA-kernel.md)
* [2023/12/04] [**Falcon-180B** on a **single H200** GPU with INT4 AWQ, and **6.7x faster Llama-70B** over A100](./docs/source/blogs/Falcon180B-H200.md)
* [2023/11/27] [SageMaker LMI now supports TensorRT-LLM - improves throughput by 60%, compared to previous version](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
* [2023/11/13] [H200 achieves nearly 12,000 tok/sec on Llama2-13B](./docs/source/blogs/H200launch.md)
* [2023/10/22] [🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙](https://github.com/NVIDIA/trt-llm-rag-windows#readme)
* [2023/10/19] Getting Started Guide - [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available
](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
* [2023/10/17] [Large Language Models up to 4x Faster on RTX With TensorRT-LLM for Windows
](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)


[2023/11/27 - Amazon Sagemaker](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
[2023/11/17 - Perplexity](https://blog.perplexity.ai/blog/turbocharging-llama-2-70b-with-nvidia-h100) ;
[2023/10/31 - Phind](https://www.phind.com/blog/phind-model-beats-gpt4-fast) ;
[2023/10/12 - Databricks (MosaicML)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) ;
[2023/10/04 - Perplexity](https://blog.perplexity.ai/blog/introducing-pplx-api) ;
[2023/09/27 - CloudFlare](https://www.cloudflare.com/press-releases/2023/cloudflare-powers-hyper-local-ai-inference-with-nvidia/);

## Table of Contents

- [TensorRT-LLM](#tensorrt-llm)
  - [Latest News](#latest-news)
  - [Table of Contents](#table-of-contents)
  - [TensorRT-LLM Overview](#tensorrt-llm-overview)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Support Matrix](#support-matrix)
    - [Devices](#devices)
    - [Precision](#precision)
    - [Key Features](#key-features)
    - [Models](#models)
  - [Performance](#performance)
  - [Advanced Topics](#advanced-topics)
    - [Quantization](#quantization)
    - [In-flight Batching](#in-flight-batching)
    - [Attention](#attention)
    - [Graph Rewriting](#graph-rewriting)
    - [Benchmark](#benchmark)
  - [Troubleshooting](#troubleshooting)
  - [Release notes](#release-notes)
    - [Change Log](#change-log)
      - [Versions 0.8.0](#versions-080)
      - [For history change log, please see CHANGELOG.md.](#for-history-change-log-please-see-changelogmd)
    - [Known Issues](#known-issues)
    - [Report Issues](#report-issues)

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

After installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit),
please run the following commands to install TensorRT-LLM for x86_64 users.

```bash
# Obtain and start the basic docker image environment.
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the latest preview version (corresponding to the main branch) of TensorRT-LLM.
# If you want to install the stable version (corresponding to the release branch), please
# remove the `--pre` option.
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check installation
python3 -c "import tensorrt_llm"
```

For developers who have the best performance requirements, debugging needs, or use the aarch64 architecture,
please refer to the instructions for [building from source code](docs/source/build_from_source.md).

For Windows installation, see [`Windows`](windows/README.md).

## Quick Start

Please be sure to complete the [installation steps](#installation) before proceeding with the following steps.

To create a TensorRT engine for an existing model, there are 3 steps:

1. Download pre-trained weights,
2. Build a fully-optimized engine of the model,
3. Deploy the engine, in other words, run the fully-optimized model.

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

```bash
# Single GPU on BLOOM 560M
python convert_checkpoint.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --output_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/
# May need to add trtllm-build to PATH, export PATH=/usr/local/bin:$PATH
trtllm-build --checkpoint_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/ \
                --gemm_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/1-gpu/
```

See the BLOOM [example](examples/bloom) for more details and options regarding the `trtllm-build` command.

***3. Run***

The `../summarize.py` script can be used to perform the summarization of articles
from the CNN Daily dataset:

```bash
python ../summarize.py --test_trt_llm \
                       --hf_model_dir ./bloom/560M/ \
                       --data_type fp16 \
                       --engine_dir ./bloom/560M/trt_engines/fp16/1-gpu/
```

More details about the script and how to run the BLOOM model can be found in
the example [folder](examples/bloom). Many more [models](#models) than BLOOM
are implemented in TensorRT-LLM. They can be found in the
[examples](./examples/) directory.

Beyond local execution, you can also use the NVIDIA Triton Inference Server to create a production-ready deployment of your LLM as described in this [blog](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/).

## Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on
NVIDIA GPUs. The following sections provide a list of supported GPU
architectures as well as important features implemented in TensorRT-LLM.

### Devices

TensorRT-LLM supports the following architectures:

* [NVIDIA Hopper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) (SM90), for example, H200, H100, H20
* [NVIDIA Ada Lovelace](https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/) (SM89), for example, L40S, L20, L4
* [NVIDIA Ampere](https://www.nvidia.com/en-us/data-center/ampere-architecture/) (SM80, SM86), for example, A100, A30, A10G
* [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/) (SM75), for example, T4
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) (SM70 - experimental), for example, V100


It is important to note that TensorRT-LLM is expected to work on all GPUs based on the Volta, Turing, Ampere, Hopper, and Ada Lovelace architectures. Certain limitations may apply.

### Precision

Various numerical precisions are supported in TensorRT-LLM. The support for
some of those numerical features require specific architectures:

|                     | FP32 | FP16 | BF16 | FP8  | INT8  | INT4  |
| :------------------ | :--- | :--- | :--- | :--- | :---- | :---- |
| Volta (SM70)        | Y    | Y    | N    | N    | Y (1) | Y (2) |
| Turing (SM75)       | Y    | Y    | N    | N    | Y (1) | Y (2) |
| Ampere (SM80, SM86) | Y    | Y    | Y    | N    | Y     | Y (3) |
| Ada-Lovelace (SM89) | Y    | Y    | Y    | Y    | Y     | Y     |
| Hopper (SM90)       | Y    | Y    | Y    | Y    | Y     | Y     |

(1) INT8 SmoothQuant is not supported on SM70 and SM75.<br>
(2) INT4 AWQ and GPTQ are not supported on SM < 80.<br>
(3) INT4 AWQ and GPTQ with FP8 activations require SM >= 89.

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
* [BART](examples/enc_dec)
* [BERT](examples/bert)
* [Blip2](examples/blip2)
* [BLOOM](examples/bloom)
* [ChatGLM](examples/chatglm)
* [FairSeq NMT](examples/enc_dec/nmt)
* [Falcon](examples/falcon)
* [Flan-T5](examples/enc_dec)
* [GPT](examples/gpt)
* [GPT-J](examples/gptj)
* [GPT-Nemo](examples/gpt)
* [GPT-NeoX](examples/gptneox)
* [InternLM](examples/internlm)
* [LLaMA](examples/llama)
* [LLaMA-v2](examples/llama)
* [mBART](examples/enc_dec)
* [Mistral](examples/llama#mistral-v01)
* [MPT](examples/mpt)
* [mT5](examples/enc_dec)
* [OPT](examples/opt)
* [Phi-1.5/Phi-2](examples/phi)
* [Qwen](examples/qwen)
* [Replit Code](examples/mpt)
* [RoBERTa](examples/bert)
* [SantaCoder](examples/gpt)
* [StarCoder1/StarCoder2](examples/gpt)
* [T5](examples/enc_dec)
* [Whisper](examples/whisper)

Note: [Encoder-Decoder](examples/enc_dec/) provides general encoder-decoder
functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, etc. We
unroll the exact model names in the list above to let users find specific
models easier.

The list of supported multi-modal models is:

* [BLIP2 w/ OPT-2.7B](examples/multimodal)
* [BLIP2 w/ T5-XL](examples/multimodal)
* [LLaVA-v1.5-7B](examples/multimodal)
* [Nougat family](examples/multimodal) Nougat-small, Nougat-base

Note: Multi-modal provides general multi-modal functionality that supports many multi-modal architectures such as BLIP family, LLaVA family, etc. We unroll the exact model names in the list above to let users find specific models easier.

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

* If you encounter accuracy issues in the generated text, you may want to increase
  the internal precision in the attention layer. For that, pass the `--context_fmha_fp32_acc enable` to
  `trtllm-build`.

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
enable plugins, for example: `--gpt_attention_plugin`.

* MPI + Slurm

TensorRT-LLM is a
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)-aware package
that uses [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/). If you are
running scripts in a [Slurm](https://slurm.schedmd.com/) environment, you might
encounter interferences:
```
--------------------------------------------------------------------------
PMI2_Init failed to initialize.  Return code: 14
--------------------------------------------------------------------------
--------------------------------------------------------------------------
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot
execute. There are several options for building PMI support under
SLURM, depending upon the SLURM version you are using:

  version 16.05 or later: you can use SLURM's PMIx support. This
  requires that you configure and build SLURM --with-pmix.

  Versions earlier than 16.05: you must use either SLURM's PMI-1 or
  PMI-2 support. SLURM builds PMI-1 by default, or you can manually
  install PMI-2. You must then build Open MPI using --with-pmi pointing
  to the SLURM PMI library location.

Please configure as appropriate and try again.
--------------------------------------------------------------------------
```
As a rule of thumb, if you are running TensorRT-LLM interactively on a Slurm
node, prefix your commands with `mpirun -n 1` to run TensorRT-LLM in a
dedicated MPI environment, not the one provided by your Slurm allocation.

For example: `mpirun -n 1 python3 examples/run.py ...`

## Release notes

  * TensorRT-LLM requires TensorRT 9.2 and 23.12 containers.

### Change Log

#### Versions 0.8.0

* Model Support
  - Phi-1.5/2.0
  - Mamba support (see examples/mamba/README.md)
    - The support is limited to beam width = 1 and single-node single-GPU
  - Nougat support (see examples/multimodal/README.md#nougat)
  - Qwen-VL support (see examples/qwenvl/README.md)
  - RoBERTa support, thanks to the contribution from @erenup
  - Skywork model support
  - Add example for multimodal models (BLIP with OPT or T5, LlaVA)
* Features
  - Chunked context support (see docs/source/gpt_attention.md#chunked-context)
  - LoRA support for C++ runtime (see docs/source/lora.md)
  - Medusa decoding support (see examples/medusa/README.md)
    - The support is limited to Python runtime for Ampere or newer GPUs with fp16 and bf16 accuracy, and the `temperature` parameter of sampling configuration should be 0
  - StreamingLLM support for LLaMA (see docs/source/gpt_attention.md#streamingllm)
  - Support for batch manager to return logits from context and/or generation phases
    - Include support in the Triton backend
  - Support AWQ and GPTQ for QWEN
  - Support ReduceScatter plugin
  - Support for combining `repetition_penalty` and `presence_penalty` #274
  - Support for `frequency_penalty` #275
  - OOTB functionality support:
    - Baichuan
    - InternLM
    - Qwen
    - BART
  - LLaMA
    - Support enabling INT4-AWQ along with FP8 KV Cache
    - Support BF16 for weight-only plugin
  - Baichuan
    - P-tuning support
    - INT4-AWQ and INT4-GPTQ support
  - Decoder iteration-level profiling improvements
  - Add `masked_select` and `cumsum` function for modeling
  - Smooth Quantization support for ChatGLM2-6B / ChatGLM3-6B / ChatGLM2-6B-32K
  - Add Weight-Only Support To Whisper #794, thanks to the contribution from @Eddie-Wang1120
  - Support FP16 fMHA on NVIDIA V100 GPU
* API
  - Add a set of High-level APIs for end-to-end generation tasks (see examples/high-level-api/README.md)
  - **[BREAKING CHANGES]** Migrate models to the new build workflow, including LLaMA, Mistral, Mixtral, InternLM, ChatGLM, Falcon, GPT-J, GPT-NeoX, Medusa, MPT, Baichuan and Phi (see docs/source/new_workflow.md)
  - **[BREAKING CHANGES]** Deprecate `LayerNorm` and `RMSNorm` plugins and removed corresponding build parameters
  - **[BREAKING CHANGES]** Remove optional parameter `maxNumSequences` for GPT manager
* Bug fixes
  - Fix the first token being abnormal issue when `--gather_all_token_logits` is enabled #639
  - Fix LLaMA with LoRA enabled build failure #673
  - Fix InternLM SmoothQuant build failure #705
  - Fix Bloom int8_kv_cache functionality  #741
  - Fix crash in `gptManagerBenchmark` #649
  - Fix Blip2 build error #695
  - Add pickle support for `InferenceRequest` #701
  - Fix Mixtral-8x7b build failure with custom_all_reduce #825
  - Fix INT8 GEMM shape #935
  - Minor bug fixes
* Performance
  - **[BREAKING CHANGES]** Increase default `freeGpuMemoryFraction` parameter from 0.85 to 0.9 for higher throughput
  - **[BREAKING CHANGES]** Disable `enable_trt_overlap` argument for GPT manager by default
  - Performance optimization of beam search kernel
  - Add bfloat16 and paged kv cache support for optimized generation MQA/GQA kernels
  - Custom AllReduce plugins performance optimization
  - Top-P sampling performance optimization
  - LoRA performance optimization
  - Custom allreduce performance optimization by introducing a ping-pong buffer to avoid an extra synchronization cost
  - Integrate XQA kernels for GPT-J (beamWidth=4)
* Documentation
  - Batch manager arguments documentation updates
  - Add documentation for best practices for tuning the performance of TensorRT-LLM (See docs/source/perf_best_practices.md)
  - Add documentation for Falcon AWQ support (See examples/falcon/README.md)
  - Update to the `docs/source/new_workflow.md` documentation
  - Update AWQ INT4 weight only quantization documentation for GPT-J
  - Add blog: Speed up inference with SOTA quantization techniques in TRT-LLM
  - Refine TensorRT-LLM backend README structure #133
  - Typo fix #739

#### For history change log, please see [CHANGELOG.md](./CHANGELOG.md).

### Known Issues

  * On windows, running context FMHA plugin with FP16 accumulation on LLaMA, Mistral and Phi models suffers from poor accuracy and the resulting inference output may be garbled. The suggestion to workaround these is to enable FP32 accumulation when building the models, i.e. passing the options `--context_fmha disable --context_fmha_fp32_acc enable` to `trtllm-build` command as a work-around, and this should be fixed in the next version

  * The hang reported in issue
    [#149](https://github.com/triton-inference-server/tensorrtllm_backend/issues/149)
    has not been reproduced by the TensorRT-LLM team. If it is caused by a bug
    in TensorRT-LLM, that bug may be present in that release

### Report Issues

You can use GitHub issues to report issues with TensorRT-LLM.
