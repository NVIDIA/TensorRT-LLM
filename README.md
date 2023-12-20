<div align="center">

TensorRT-LLM
===========================
<h4> A TensorRT Toolbox for Optimized Large Language Model Inference</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.2-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-9.2-green)](https://developer.nvidia.com/tensorrt)
[![version](https://img.shields.io/badge/release-0.7.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Architecture](./docs/source/architecture.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Results](./docs/source/performance.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](./examples/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](./docs/source/)

---
<div align="left">

## Latest News
* [2023/12/04] [**Falcon-180B** on a **single H200** GPU with INT4 AWQ, and **6.7x faster Llama-70B** over A100](./docs/source/blogs/Falcon180B-H200.md)

<img src="./docs/source/blogs/media/Falcon180B-H200_H200vA100.png" alt="H200 TPS" width="400" height="auto">

H200 with INT4 AWQ, runs Falcon-180B on a _single_ GPU.

H200 is now 2.4x faster on Llama-70B with recent improvements to TensorRT-LLM GQA; up to 6.7x faster than A100.

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

*For Windows installation, see [`Windows`](windows/README.md).*

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

```bash
# Single GPU on BLOOM 560M
python convert_checkpoint.py --model_dir ./bloom/560M/ \
                --dtype float16 \
                --output_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/
# May need to add trtllm-build to PATH, export PATH=/usr/local/bin:$PATH
trtllm-build --checkpoint_dir ./bloom/560M/trt_ckpt/fp16/1-gpu/ \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./bloom/560M/trt_engines/fp16/1-gpu/
```

See the BLOOM [example](examples/bloom) for more details and options regarding the `build.py` script.

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

## Support Matrix

TensorRT-LLM optimizes the performance of a range of well-known models on
NVIDIA GPUs. The following sections provide a list of supported GPU
architectures as well as important features implemented in TensorRT-LLM.

### Devices

TensorRT-LLM is rigorously tested on the following GPUs:

* [H100](https://www.nvidia.com/en-us/data-center/h100/)
* [L40S](https://www.nvidia.com/en-us/data-center/l40s/)
* [A100](https://www.nvidia.com/en-us/data-center/a100/)
* [A30](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)
* [V100](https://www.nvidia.com/en-us/data-center/v100/) (experimental)

If a GPU is not listed above, it is important to note that TensorRT-LLM is
expected to work on GPUs based on the Volta, Turing, Ampere, Hopper and Ada
Lovelace architectures. Certain limitations may, however, apply.

### Precision

Various numerical precisions are supported in TensorRT-LLM. The support for
some of those numerical features require specific architectures:

|                     | FP32 | FP16 | BF16 | FP8  | INT8 | INT4 |
| :------------------ | :--- | :--- | :--- | :--- | :--- | :--- |
| Volta (SM70)        | Y    | Y    | N    | N    | Y    | Y    |
| Turing (SM75)       | Y    | Y    | N    | N    | Y    | Y    |
| Ampere (SM80, SM86) | Y    | Y    | Y    | N    | Y    | Y    |
| Ada-Lovelace (SM89) | Y    | Y    | Y    | Y    | Y    | Y    |
| Hopper (SM90)       | Y    | Y    | Y    | Y    | Y    | Y    |

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
* [Bert](examples/bert)
* [Blip2](examples/blip2)
* [BLOOM](examples/bloom)
* [ChatGLM](examples/chatglm)
* [FairSeq NMT](examples/nmt)
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
* [Qwen](examples/qwen)
* [Replit Code](examples/mpt)
* [SantaCoder](examples/gpt)
* [StarCoder](examples/gpt)
* [T5](examples/enc_dec)
* [Whisper](examples/whisper)

Note: [Encoder-Decoder](examples/enc_dec/) provides general encoder-decoder
functionality that supports many encoder-decoder models such as T5 family, BART family, Whisper family, NMT family, etc. We
unroll the exact model names in the list above to let users find specific
models easier.

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

For example: `mpirun -n 1 python3 examples/gpt/build.py ...`

## Release notes

  * TensorRT-LLM requires TensorRT 9.2 and 23.10 containers.

### Change Log

#### Version 0.6.1

  * Models
      * ChatGLM3
      * InternLM (contributed by @wangruohui)
      * Mistral 7B (developed in collaboration with Mistral.AI)
      * MQA/GQA support to MPT (and GPT) models (contributed by @bheilbrun)
      * Qwen (contributed by @Tlntin and @zhaohb)
      * Replit Code V-1.5 3B (external contribution)
      * T5, mT5, Flan-T5 (Python runtime only)

  * Features
      * Add runtime statistics related to active requests and KV cache
        utilization from the batch manager (see
        the [batch manager](docs/source/batch_manager.md) documentation)
      * Add `sequence_length` tensor to support proper lengths in beam-search
        (when beam-width > 1 - see
        [tensorrt_llm/batch_manager/GptManager.h](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * BF16 support for encoder-decoder models (Python runtime - see
        [examples/enc_dec](examples/enc_dec/README.md))
      * Improvements to memory utilization (CPU and GPU - including memory
        leaks)
      * Improved error reporting and memory consumption
      * Improved support for stop and bad words
      * INT8 SmoothQuant and INT8 KV Cache support for the Baichuan models (see
        [examples/baichuan](examples/baichuan/README.md))
      * INT4 AWQ Tensor Parallelism support and INT8 KV cache + AWQ/weight-only
        support for the GPT-J model (see [examples/gptj](examples/gptj/README.md))
      * INT4 AWQ support for the Falcon models
        (see [examples/falcon](examples/falcon/README.md))
      * LoRA support (functional preview only - limited to the Python runtime,
        only QKV support and not optimized in terms of runtime performance) for
        the GPT model (see the
        [Run LoRA with the Nemo checkpoint](examples/gpt/README.md#Run-LoRA-with-the-Nemo-checkpoint)
        in the GPT example)
      * Multi-GPU support for encoder-decoder models (Python runtime - see
        [examples/enc_dec](examples/enc_dec/README.md))
      * New heuristic for launching the Multi-block Masked MHA kernel (similar
        to FlashDecoding - see
        [decoderMaskedMultiheadAttentionLaunch.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionLaunch.h))
      * Prompt-Tuning support for GPT and LLaMA models (see the
        [Prompt-tuning](examples/gpt/README.md#Prompt-tuning) Section in the GPT example)
      * Performance optimizations in various CUDA kernels
      * Possibility to exclude input tokens from the output (see `excludeInputInOutput` in
        [`GptManager`](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * Python binding for the C++ runtime (GptSession - see [`pybind`](cpp/tensorrt_llm/pybind))
      * Support for different micro batch sizes for context and generation
        phases with pipeline parallelism (see `GptSession::Config::ctxMicroBatchSize` and
        `GptSession::Config::genMicroBatchSize` in
        [tensorrt_llm/runtime/gptSession.h](cpp/include/tensorrt_llm/runtime/gptSession.h))
      * Support for "remove input padding" for encoder-decoder models (see
        [examples/enc_dec](examples/enc_dec/README.md))
      * Support for context and generation logits (see `mComputeContextLogits` and
        `mComputeGenerationLogits` in
        [tensorrt_llm/runtime/gptModelConfig.h](cpp/include/tensorrt_llm/runtime/gptModelConfig.h))
      * Support for `logProbs` and `cumLogProbs` (see `"output_log_probs"` and
        `"cum_log_probs"` in [`GptManager`](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * Update to CUTLASS 3.x

  * Bug fixes
      * Fix for ChatGLM2 #93 and #138
      * Fix tensor names error "RuntimeError: Tensor names
        (`host_max_kv_cache_length`) in engine are not the same as expected in
        the main branch" #369
      * Fix weights split issue in BLOOM when `world_size = 2` ("array split
        does not result in an equal division") #374
      * Fix SmoothQuant multi-GPU failure with tensor parallelism is 2 #267
      * Fix a crash in GenerationSession if stream keyword argument is not None
        #202
      * Fix a typo when calling PyNVML API [BUG] code bug #410
      * Fix bugs related to the improper management of the `end_id` for various
        models [C++ and Python]
      * Fix memory leaks [C++ code and Python models]
      * Fix the std::alloc error when running the gptManagerBenchmark -- issue
        gptManagerBenchmark std::bad_alloc error #66
      * Fix a bug in pipeline parallelism when beam-width > 1
      * Fix a bug with Llama GPTQ due to improper support of GQA
      * Fix issue #88
      * Fix an issue with the Huggingface Transformers version #16
      * Fix link jump in windows readme.md #30 - by @yuanlehome
      * Fix typo in batchScheduler.h #56 - by @eltociear
      * Fix typo #58 - by @RichardScottOZ
      * Fix Multi-block MMHA: Difference between `max_batch_size` in the engine
        builder and `max_num_sequences` in TrtGptModelOptionalParams? #65
      * Fix the log message to be more accurate on KV cache #224
      * Fix Windows release wheel installation: Failed to install the release
        wheel for Windows using pip #261
      * Fix missing torch dependencies: [BUG] The batch_manage.a choice error
        in --cpp-only when torch's cxx_abi version is different with gcc #151
      * Fix linking error during compiling google-test & benchmarks #277
      * Fix logits dtype for Baichuan and ChatGLM: segmentation fault caused by
        the lack of bfloat16 #335
      * Minor bug fixes

#### Version 0.5.0

  * TensorRT-LLM v0.5.0 is the first public release.

### Known Issues

  * The hang reported in issue
    [#149](https://github.com/triton-inference-server/tensorrtllm_backend/issues/149)
    has not been reproduced by the TensorRT-LLM team. If it is caused by a bug
    in TensorRT-LLM, that bug may be present in that release

### Report Issues

You can use GitHub issues to report issues with TensorRT-LLM.
