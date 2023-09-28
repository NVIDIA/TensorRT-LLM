# TensorRT-LLM: A TensorRT toolbox for Large Language Models

## Table of Contents

- [The TensorRT-LLM Overview](#the-tensorrt-llm-overview)
- [Installation](#installation)
- [Supported Models and Examples](#supported-models-and-examples)
- [Troubleshooting](#troubleshooting)
- [Release notes](#release-notes)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)

## The TensorRT-LLM Overview

TensorRT-LLM provides users with an easy-to-use Python API to define Large
Language Models (LLMs) and build
[TensorRT](https://developer.nvidia.com/tensorrt) engines that contain
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
TensorRT-LLM also contains components to create Python and C++ runtimes that
execute those TensorRT engines. It also includes a backend for integration with
the [NVIDIA Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server).  Models
built with TensorRT-LLM can be executed on a wide range of configurations going
from a single GPU to multiple nodes with multiple GPUs (using [Tensor
Parallelism](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/parallelisms.html#tensor-parallelism)).

The Python API of TensorRT-LLM is architectured to look similar to the
[PyTorch](https://pytorch.org) API. It provides users with a
[functional](./tensorrt_llm/functional.py) module containing functions like
`einsum`, `softmax`, `matmul` or `view`. The [layer](./tensorrt_llm/layer)
module bundles useful building blocks to assemble LLMs; like an `Attention`
block, a `MLP` or the entire `Transformer` layer. Model-specific components,
like `GPTAttention` or `BertAttention`, can be found in the
[model](./tensorrt_llm/model) module.

TensorRT-LLM provides users with predefined models that can easily be modified
and extended. The current version of TensorRT-LLM supports
[BERT](https://huggingface.co/docs/transformers/model_doc/bert),
[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt),
[NVIDIA GPT-2B](https://huggingface.co/nvidia/GPT-2B-001),
[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj),
[LLaMA](https://huggingface.co/docs/transformers/model_doc/llama),
[OPT](https://huggingface.co/docs/transformers/model_doc/opt),
[SantaCoder](https://huggingface.co/bigcode/santacoder)
and
[StarCoder](https://huggingface.co/bigcode/starcoder).
To maximize performance and reduce memory footprint, TensorRT-LLM allows the
models to be executed using different quantization modes (see
[`examples/gpt`](./examples/gpt) for concrete examples).  TensorRT-LLM supports
INT4 or INT8 weights (and FP16 activations; a.k.a.  INT4/INT8 weight-only) as
well as a complete implementation of the
[SmoothQuant](https://arxiv.org/abs/2211.10438) technique.

For a more detailed presentation of the software architecture and the key
concepts used in TensorRT-LLM, we recommend you to read the following
[document](./docs/architecture.md).

## Installation

TensorRT-LLM contains Python and C++ components, and must be compiled from
source to be used.  TensorRT-LLM is dependent on the latest versions of
TensorRT and [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) which are distributed separately, and should be copied
into this repository.

We recommend that you use a [Docker](https://www.docker.com) container to build
and run TensorRT-LLM. Instructions to install an environment to run Docker
containers for the NVIDIA platform can be found
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


Make sure you have fetched all the dependencies before compiling TensorRT-LLM:

```bash
git submodule update --init --recursive
```

### Docker Container

Use the following command to create a Docker image for development:

```bash
make -C docker build
```

This will create a docker image for development of TensorRT-LLM and tag it locally with `tensorrt_llm/devel:latest`.
To run the container, use the following command:

```bash
make -C docker run
```

If you prefer to work with your own user account in that container instead of `root`, include the option `LOCAL_USER=1`
in the command above like so:

```bash
make -C docker run LOCAL_USER=1
```

#### Systems without GNU `make`

On systems without GNU `make` or shell support, you can build the Docker image for development as follows:

```bash
docker build --pull  \
    --target devel \
    --file docker/Dockerfile.multi \
    --tag tensorrt_llm/devel:latest \
    .
```

Then run the container by issuing the following command:

```bash
docker run --rm -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
    --volume ${PWD}:/code/tensorrt_llm \
    --workdir /code/tensorrt_llm \
    tensorrt_llm/devel:latest
```

### Build From Source

```bash
# To build the TensorRT-LLM code.
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt

# Deploy TensorRT-LLM in your environment.
pip install ./build/tensorrt_llm*.whl
```

By default, `build_wheel.py` enables incremental builds. To clean the build
directory, add the `--clean` option:

```bash
python3 ./scripts/build_wheel.py --clean  --trt_root /usr/local/tensorrt
```

#### Fully automated release builds in Docker

The steps of creating a Docker image for development, building the wheel and installing it inside the container can be
executed in a single command:

```bash
make -C docker release_build
```

You can optionally append `CUDA_ARCHS="<list of architectures in CMake format>"` to specify which architectures should
be supported by the wheel. Once the image is built, run it in a Docker container with:

```bash
make -C docker release_run
```

Append `LOCAL_USER=1` to this command for switching to your local user account instead of `root` inside the container.
The examples of TensorRT-LLM are installed in directory `/app/tensorrt_llm/examples`.

### Building for Specific CUDA Architectures

Specific CUDA architectures may be passed as an argument to
[`build_wheel.py`](scripts/build_wheel.py). The script accepts a single
argument taking a semicolon separated list of CUDA architecture specifications
compatible with [CUDA_ARCHITECTURES in CMake].  For instance, to build for
compute capabilities 8.0 and 8.6, call `build_wheel.py` like so:

```bash
python3 ./scripts/build_wheel.py --cuda_architectures "80-real;86-real"
```

### Building and Linking Against the C++ Runtime of TensorRT-LLM

Running `build_wheel.py` will also compile the library containing the C++
runtime of TensorRT-LLM. If Python support and `torch` modules are not
required, the script provides the option `--cpp_only` which restricts the build
to the C++ runtime only:

```bash
python3 ./scripts/build_wheel.py --cuda_architectures "80-real;86-real" --cpp_only --clean
```

This is particularly useful to avoid linking problems which may be introduced
by particular versions of `torch` related to the [dual ABI support of
GCC](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). The
option `--clean` will remove the build directory before building. The default
build directory is `cpp/build`, which may be overridden using the option
`--build_dir`. Run `build_wheel.py --help` for an overview of all supported
options.

Clients may choose to link against the shared or the static version of the
library. These libraries can be found in the following locations:

```bash
cpp/build/tensorrt_llm/libtensorrt_llm.so
cpp/build/tensorrt_llm/libtensorrt_llm_static.a
```

In addition, one needs to link against the library containing the LLM plugins
for TensorRT available here:

```bash
cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
```

Add the following directories to your project include paths

```bash
cpp
cpp/include
```

Only header files contained in `cpp/include` are part of the supported API and
may be directly included. Other headers contained under `cpp` should not be
included directly since they might change in future versions.

For examples of how to use the C++ runtime, see the unit tests in
[gptSessionTest.cpp](cpp/tests/runtime/gptSessionTest.cpp) and the related
[CMakeLists.txt](cpp/tests/CMakeLists.txt) file.

## Supported Models and Examples

- [Bert](examples/bert)
- [BLOOM](examples/bloom)
- [ChatGLM-6B](examples/chatglm6b)
- [ChatGLM2-6B](examples/chatglm2-6b/)
- [Falcon](examples/falcon)
- [GPT](examples/gpt)
- [GPT-J](examples/gptj)
- [GPT-NeoX](examples/gptneox)
- [LLaMA](examples/llama)
- [OpenAI Triton](examples/openai_triton)
- [OPT](examples/opt)
- [SantaCoder](examples/gpt)
- [StarCoder](examples/gpt)

## Troubleshooting

- It's recommended to add options `–shm-size=1g –ulimit memlock=-1` to the
  docker or nvidia-docker run command.  Otherwise you may see NCCL errors when
  running multiple GPU inferences. See
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#errors
  for details.

- If you encounter
```text
NVIDIA H100 PCIe with CUDA capability sm_90 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

when building engines, you need to install the preview version of PyTorch that
corresponds to your CUDA version.  As an example, for CUDA 12.1, use:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

[CUDA_ARCHITECTURES in CMake]: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES

## Release notes

### Changelog

**August 2023**

  - TensorRT-LLM requires TensorRT 9.0.1.4 and 23.07 containers,
  - Support for Baichuan-13B, ChatGLM2, Falcon-40B,
  - Support for GPTQ for GPT-NeoX and LLaMA (experimental),
  - Support for AWQ for GPT-J (experimental),
  - Revised GPT Attention plugin,
    - The GPT Attention now supports in-flight batching,
    - The In-flight Batching Attention plugin will be removed in the next release (kept for debugging purposes in that release),
  - Support for Group-Query Attention (GQA)
    - LLama 70B can now be run with 4 GPUs,
  - ALiBi support in Multi-head Attention (context and generation),
  - Optimization of the MHA/MQA/GQA CUDA kernel (generation),
  - Enhancements and bug fixes for the beam-search implementation,
  - Support for "no_repeat_ngram_size" parameters,
  - Bug fixes for the "bad/stop words" features,
  - Embeddings can now be splitted along the hidden dimension,
  - Improvements to the in-flight batching feature and paged K/V cache manager (C++),
    - Included in the C++ Triton backend,
  - Multi-GPU support in the Triton backend,
  - Early-stopping support in the Triton backend,
  - First implementation of a graph rewriting feature (to be updated in the next release).

**July 2023**

  - TensorRT-LLM requires TensorRT 9.0,
  - Support for BLOOM, ChatGLM 6B, GPT-NeoX, LLaMA v2,
  - Support for BF16 and FP8 models,
  - Support for in-flight batching,
  - Support for a new C++ Triton Backend,
  - Refactoring of the KV cache to support paging,
    - The KV cache is now decomposed into blocks,
    - The layout of the K cache has changed to `[batch_size, num_heads, seq_length, dim_per_head]`,
  - Support for multi-GPU embeddings,
  - Support for embedding sharing (input embedding and LM head),
  - New example that shows how to integrate an OpenAI Triton kernel into TensorRT-LLM,
  - Improved documentation (Docstrings in `functional.py` and documentation in `docs`)

**June 2023**

  - Support Nemo-GPT Next, SantaCoder, StarCoder in FP16,
  - Support for a new C++ Runtime (with streaming support),
  - Support for beam-search,
  - Support for Multiquery Attention (MQA),
  - Support for RoPE,
  - Support for INT8 KV Cache,
  - Support INT4 weight-only (with GPT example), but the weight-only kernels will not be optimal on hopper

**May 2023**

  - **The initial release of TensorRT-LLM**
  - Support GPT, BERT, OPT, LLaMA in FP16,
  - Support single-node multi-GPU GPT, OPT, BERT, LLaMA FP16 using Tensor parallelism,
  - Support Triton Inference Server with a Python backend,
  - Support sampling features, including top-k, top-p, temperature, and sampling penalty,
  - Attention support
   - Optimized Flash-Attention-based Multihead Attention for Ampere, Ada and Hopper architectures,
   - Multi-Query Attention (MQA),
   - ALiBi in Multihead-Attention,
  - Support SmoothQuant INT8 (with GPT example),
  - Support INT8 weight-only (with GPT example), but the weight-only kernels will not be optimal on hopper

### Known issues
