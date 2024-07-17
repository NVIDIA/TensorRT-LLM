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

TensorRT-LLM contains a simple command to create a Docker image.

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

### Option 2: Build TensorRT-LLM Step-By-Step

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

#### Build TensorRT-LLM

Once in the container, build TensorRT-LLM from the source.

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

Refer to the Refer to the {ref}`support-matrix-hardware` section for a list of architectures.

## Building the Python Bindings for the C++ Runtime

The C++ Runtime, in particular, `GptSession` can be exposed to Python via bindings. This feature can be turned on through the default build options.

```bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt
```

After installing, the resulting wheel as described above, the C++ Runtime bindings will be available in
the `tensorrt_llm.bindings` package. Running `help` on this package in a Python interpreter will provide on overview of the
relevant classes. The associated unit tests should also be consulted for understanding the API.

This feature will not be enabled when [`building only the C++ runtime`](#link-with-the-tensorrt-llm-c++-runtime).

## Linking with the TensorRT-LLM C++ Runtime

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
