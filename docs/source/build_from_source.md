# Build from Source

- [Overview](#overview)
- [Install From the Wheel Package](#install-from-the-wheel-package)
- [Fetch the Sources](#fetch-the-sources)
- [Build TensorRT-LLM in One Step](#build-tensorrt-llm-in-one-step)
- [Build Step-by-step](#build-step-by-step)
    - [Create the Container](#create-the-container)
      - [On Systems with GNU `make`](#on-systems-with-gnu-make)
      - [On Systems without GNU `make`](#on-systems-without-gnu-make)
    - [Build TensorRT-LLM](#build-tensorrt-llm)
    - [Link with the TensorRT-LLM C++ Runtime](#link-with-the-tensorrt-llm-c++-runtime)
    - [Supported C++ Header Files](#supported-c++-header-files)

## Overview

This document provides instructions for building TensorRT-LLM from source code on Linux.

We first recommend that you [`install TensorRT-LLM`](../../README.md#installation) directly.
Building from source code is necessary for users who require the best performance or debugging
capabilities, or if the [GNU C++11 ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) is required.

We recommend the use of [Docker](https://www.docker.com) to build and run TensorRT-LLM. Instructions
to install an environment to run Docker containers for the NVIDIA platform can be found
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Fetch the Sources

The first step to build TensorRT-LLM is to fetch the sources:

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

git lfs install
git lfs pull
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
```

Note: There are two options to create TensorRT-LLM Docker image and approximate disk space required to build the image is 63 GB

## Option 1: Build TensorRT-LLM in One Step

TensorRT-LLM contains a simple command to create a Docker image:

```bash
make -C docker release_build
```

It is possible to add the optional argument `CUDA_ARCHS="<list of architectures
in CMake format>"` to specify which architectures should be supported by
TensorRT-LLM. It restricts the supported GPU architectures but helps reduce
compilation time:

```bash
# Restrict the compilation to Ada and Hopper architectures.
make -C docker release_build CUDA_ARCHS="89-real;90-real"
```

Once the image is built, the Docker container can be executed using:

```bash
make -C docker release_run
```

The `make` command supports the `LOCAL_USER=1` argument to switch to the local
user account instead of `root` inside the container.  The examples of
TensorRT-LLM are installed in directory `/app/tensorrt_llm/examples`.

## Option 2: Build Step-by-step

For users looking for more flexibility, TensorRT-LLM has commands to create and
run a development container in which TensorRT-LLM can be built.

### Create the Container

#### On Systems with GNU `make`

The following command creates a Docker image for development:

```bash
make -C docker build
```

The image will be tagged locally with `tensorrt_llm/devel:latest`.  To run the
container, use the following command:

```bash
make -C docker run
```

For users who prefer to work with their own user account in that container
instead of `root`, the option `LOCAL_USER=1` must be added to the above command
above:

```bash
make -C docker run LOCAL_USER=1
```

#### On Systems Without GNU `make`

On systems without GNU `make` or shell support, the Docker image for
development can be built using:

```bash
docker build --pull  \
             --target devel \
             --file docker/Dockerfile.multi \
             --tag tensorrt_llm/devel:latest \
             .
```

The container can then be run using:

```bash
docker run --rm -it \
           --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
           --volume ${PWD}:/code/tensorrt_llm \
           --workdir /code/tensorrt_llm \
           tensorrt_llm/devel:latest
```

### Build TensorRT-LLM

Once in the container, TensorRT-LLM can be built from source using:

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

It is possible to restrict the compilation of TensorRT-LLM to specific CUDA
architectures. For that purpose, the `build_wheel.py` script accepts a
semicolon separated list of CUDA architecture as shown in the following
example:

```bash
# Build TensorRT-LLM for Ampere.
python3 ./scripts/build_wheel.py --cuda_architectures "80-real;86-real" --trt_root /usr/local/tensorrt
```

The list of supported architectures can be found in the
[`CMakeLists.txt`](source:cpp/CMakeLists.txt) file.

### Build the Python Bindings for the C++ Runtime

The C++ Runtime, in particular, [`GptSession`](source:cpp/include/tensorrt_llm/runtime/gptSession.h) can be exposed to
Python via [bindings](source:cpp/tensorrt_llm/pybind/bindings.cpp). This feature can be turned on through the default
build options:

```bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt
```

After installing the resulting wheel as described above, the C++ Runtime bindings will be available in
package `tensorrt_llm.bindings`. Running `help` on this package in a Python interpreter will provide on overview of the
relevant classes. The [associated unit tests](source:tests/bindings) should also be consulted for understanding the API.

This feature will not be enabled when [`building only the C++ runtime`](#link-with-the-tensorrt-llm-c++-runtime).

### Link with the TensorRT-LLM C++ Runtime

The `build_wheel.py` script will also compile the library containing the C++
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

### Supported C++ Header Files

When using TensorRT-LLM, you need to add the `cpp` and `cpp/include`
directories to the project's include paths.  Only header files contained in
`cpp/include` are part of the supported API and may be directly included. Other
headers contained under `cpp` should not be included directly since they might
change in future versions.

For examples of how to use the C++ runtime, see the unit tests in
[gptSessionTest.cpp](source:cpp/tests/runtime/gptSessionTest.cpp) and the related
[CMakeLists.txt](source:cpp/tests/CMakeLists.txt) file.
