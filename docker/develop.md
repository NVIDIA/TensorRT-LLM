# Description

TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT LLM also contains components to
create Python and C++ runtimes that orchestrate the inference execution in a performant way.

# Overview

## TensorRT LLM Develop Container

The TensorRT LLM Develop container includes all necessary dependencies to build TensorRT LLM from source. It is
specifically designed to be used alongside the source code cloned from the official TensorRT LLM repository:

[GitHub Repository - NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

Full instructions for cloning the TensorRT LLM repository can be found in
the [TensorRT LLM Documentation](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html).

> **Note:**  
> This container does not contain a pre-built binary release of `TensorRT LLM` or tools like `trtllm-serve`.

### Running the TensorRT LLM Develop Container Using Docker

With the top-level directory of the TensorRT LLM repository cloned to your local machine, you can run the following
command to start the development container:

```bash
make -C docker ngc-devel_run LOCAL_USER=1 DOCKER_PULL=1 IMAGE_TAG=x.y.z
```

where `x.y.z` is the version of the TensorRT LLM container to use (cf. [release history on GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases) and [tags in NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel/tags)). This command pulls the specified container from the
NVIDIA NGC registry, sets up the local user's account within the container, and launches it with full GPU support. The
local source code of TensorRT LLM will be mounted inside the container at the path `/code/tensorrt_llm` for seamless
integration. Ensure that the image version matches the version of TensorRT LLM in your currently checked out local git branch. Not
specifying a `IMAGE_TAG` will attempt to resolve this automatically, but not every intermediate release might be
accompanied by a development container. In that case, use the latest version preceding the version of your development
branch.

If you prefer launching the container directly with `docker`, you can use the following command:

```bash
docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
           --gpus=all \
           --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
           --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
           --env "CONAN_HOME=/code/tensorrt_llm/cpp/.conan" \
           --workdir /code/tensorrt_llm \
           --tmpfs /tmp:exec \
           --volume .:/code/tensorrt_llm \
           nvcr.io/nvidia/tensorrt-llm/devel:x.y.z
```

Note that this will start the container with the user `root`, which may leave files with root ownership in your local
checkout.

### Building the TensorRT LLM Wheel within the Container

You can build the TensorRT LLM Python wheel inside the development container using the following command:

```bash
./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=native
```

#### Explanation of Build Flags:

- `--clean`: Clears intermediate build artifacts from prior builds to ensure a fresh compilation.
- `--use_ccache`: Enables `ccache` to optimize and accelerate subsequent builds by caching compilation results.
- `--cuda_architectures=native`: Configures the build for the native architecture of your GPU. Leave this away to build
  the wheel for all supported architectures. For additional details, refer to
  the [CUDA Architectures Documentation](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES).

For additional build options and their usage, refer to the help documentation by running:

```bash
./scripts/build_wheel.py --help
```

The wheel will be built in the `build` directory and can be installed using `pip install` like so:

```bash
pip install ./build/tensorrt_llm*.whl
```

For additional information on building the TensorRT LLM wheel, refer to
the [official documentation on building from source](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-full-build-with-c-compilation).

### Security CVEs

To review known CVEs on this image, refer to the Security Scanning tab on this page.

### License

By pulling and using the container, you accept the terms and conditions of
this [End User License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and [Product-Specific Terms](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
