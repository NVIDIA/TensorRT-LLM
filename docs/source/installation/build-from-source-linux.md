(build-from-source-linux)=

# Building from Source Code on Linux

This document provides instructions for building TensorRT-LLM from source code on Linux. Building from source code is necessary if you want the best performance or debugging capabilities, or if the [GNU C++11 ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) is required.

## Prerequisites

Use [Docker](https://www.docker.com) to build and run TensorRT-LLM. Instructions to install an environment to run Docker containers for the NVIDIA platform can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

## Building a TensorRT-LLM Docker Image

There are two options to create a TensorRT-LLM Docker image. The approximate disk space required to build the image is 63 GB.

### Option 1: Build TensorRT-LLM in One Step

TensorRT-LLM contains a simple command to create a Docker image. Note that if you plan to develop on TensorRT-LLM, we recommend using [Option 2: Build TensorRT-LLM Step-By-Step](#option-2-build-tensorrt-llm-step-by-step).

```bash
make -C docker release_build
```

You can add the `CUDA_ARCHS="<list of architectures in CMake format>"` optional argument to specify which architectures should be supported by TensorRT-LLM. It restricts the supported GPU architectures but helps reduce compilation time:

```bash
# Restrict the compilation to Ada and Hopper architectures.
make -C docker release_build CUDA_ARCHS="89-real;90-real"
```

After the image is built, the Docker container can be run.

```bash
make -C docker release_run
```

The `make` command supports the `LOCAL_USER=1` argument to switch to the local user account instead of `root` inside the container.  The examples of TensorRT-LLM are installed in the `/app/tensorrt_llm/examples` directory.

Since TensorRT-LLM has been built and installed, you can skip the remaining steps.

### Option 2: Build TensorRT-LLM Step-by-Step

If you are looking for more flexibility, TensorRT-LLM has commands to create and run a development container in which TensorRT-LLM can be built.

#### Create the Container

**On systems with GNU `make`**

1. Create a Docker image for development. The image will be tagged locally with `tensorrt_llm/devel:latest`.

    ```bash
    make -C docker build
    ```

2. Run the container.

    ```bash
    make -C docker run
    ```

    If you prefer to work with your own user account in that container, instead of `root`, add the `LOCAL_USER=1` option.

    ```bash
    make -C docker run LOCAL_USER=1
    ```

**On systems without GNU `make`**

1. Create a Docker image for development.

    ```bash
    docker build --pull  \
                --target devel \
                --file docker/Dockerfile.multi \
                --tag tensorrt_llm/devel:latest \
                .
    ```

2. Run the container.

    ```bash
    docker run --rm -it \
            --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
            --volume ${PWD}:/code/tensorrt_llm \
            --workdir /code/tensorrt_llm \
            tensorrt_llm/devel:latest
    ```
    Note: please make sure to set `--ipc=host` as a docker run argument to avoid `Bus error (core dumped)`.

Once inside the container, follow the next steps to build TensorRT-LLM from source.

## Build TensorRT-LLM

### Option 1: Full Build with C++ Compilation

The following command compiles the C++ code and packages the compiled libraries along with the Python files into a wheel. When developing C++ code, you need this full build command to apply your code changes.

```bash
# To build the TensorRT-LLM code.
python3 ./scripts/build_wheel.py
```

Once the wheel is built, install it by:

```bash
pip install ./build/tensorrt_llm*.whl
```

Alternatively, you can use editable installation, which is convenient if you also develop Python code.

```bash
pip install -e .
```

By default, `build_wheel.py` enables incremental builds. To clean the build
directory, add the `--clean` option:

```bash
python3 ./scripts/build_wheel.py --clean
```

It is possible to restrict the compilation of TensorRT-LLM to specific CUDA
architectures. For that purpose, the `build_wheel.py` script accepts a
semicolon separated list of CUDA architecture as shown in the following
example:

```bash
# Build TensorRT-LLM for Ampere.
python3 ./scripts/build_wheel.py --cuda_architectures "80-real;86-real"
```

To use the C++ benchmark scripts under [benchmark/cpp](/benchmarks/cpp/), for example `gptManagerBenchmark.cpp`, add the `--benchmarks` option:

```bash
python3 ./scripts/build_wheel.py --benchmarks
```

Refer to the {ref}`support-matrix-hardware` section for a list of architectures.

#### Building the Python Bindings for the C++ Runtime

The C++ Runtime, in particular, `GptSession` can be exposed to Python via bindings. This feature can be turned on through the default build options.

```bash
python3 ./scripts/build_wheel.py
```

After installing, the resulting wheel as described above, the C++ Runtime bindings will be available in
the `tensorrt_llm.bindings` package. Running `help` on this package in a Python interpreter will provide on overview of the
relevant classes. The associated unit tests should also be consulted for understanding the API.

This feature will not be enabled when [`building only the C++ runtime`](#link-with-the-tensorrt-llm-c++-runtime).

#### Linking with the TensorRT-LLM C++ Runtime

The `build_wheel.py` script will also compile the library containing the C++ runtime of TensorRT-LLM. If Python support and `torch` modules are not required, the script provides the option `--cpp_only` which restricts the build to the C++ runtime only.

```bash
python3 ./scripts/build_wheel.py --cuda_architectures "80-real;86-real" --cpp_only --clean
```

This is particularly useful to avoid linking problems which may be introduced by particular versions of `torch` related to the [dual ABI support of GCC](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). The option `--clean` will remove the build directory before building. The default build directory is `cpp/build`, which may be overridden using the option
`--build_dir`. Run `build_wheel.py --help` for an overview of all supported options.

The shared library can be found in the following location:

```bash
cpp/build/tensorrt_llm/libtensorrt_llm.so
```

In addition, link against the library containing the LLM plugins for TensorRT.

```bash
cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
```

#### Supported C++ Header Files

When using TensorRT-LLM, you need to add the `cpp` and `cpp/include` directories to the project's include paths.  Only header files contained in `cpp/include` are part of the supported API and may be directly included. Other headers contained under `cpp` should not be included directly since they might change in future versions.


### Option 2: Python-Only Build without C++ Compilation

If you only need to modify Python code, it is possible to package and install TensorRT-LLM without compilation.

```bash
# Package TensorRT-LLM wheel.
TRTLLM_USE_PRECOMPILED=1 pip wheel . --no-deps --wheel-dir ./build

# Install TensorRT-LLM wheel.
pip install ./build/tensorrt_llm*.whl
```

Alternatively, you can use editable installation for convenience during Python development.

```bash
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

Setting `TRTLLM_USE_PRECOMPILED=1` enables downloading a prebuilt wheel of the version specified in `tensorrt_llm/version.py`, extracting compiled libraries into your current directory, thus skipping C++ compilation.

You can specify a custom URL or local path for downloading using `TRTLLM_PRECOMPILED_LOCATION`. For example, to use version 0.16.0 from PyPI:

```bash
TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.16.0-cp312-cp312-linux_x86_64.whl pip install -e .
```

#### Known Limitations

Currently, our released TensorRT-LLM wheel packages are linked against public PyTorch hosted on PyPI, which disables C++11 ABI support. However, the Docker image built previously is based on an NGC container where PyTorch has C++11 ABI enabled; see [NGC PyTorch container page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). Therefore, we recommend performing a full build inside this container.

When using `TRTLLM_PRECOMPILED_LOCATION`, ensure that your wheel is compiled based on the same version of C++ code as your current directory; any discrepancies may lead to compatibility issues.
