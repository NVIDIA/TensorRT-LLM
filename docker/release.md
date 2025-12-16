# Description

TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT LLM also contains components to
create Python and C++ runtimes that orchestrate the inference execution in a performant way.

# Overview

## TensorRT LLM Release Container

The TensorRT LLM Release container provides a pre-built environment for running TensorRT-LLM.

Visit the [official GitHub repository](https://github.com/NVIDIA/TensorRT-LLM) for more details.

### Running TensorRT LLM Using Docker

A typical command to launch the container is:

```bash
docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
    		nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

where x.y.z is the version of the TensorRT LLM container to use (cf. [release history on GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases) and [tags in NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)). To sanity check, run the following command:

```bash
python3 -c "import tensorrt_llm"
```

This command will print the TensorRT LLM version if everything is working correctly. After verification, you can explore
and try the example scripts included in `/app/tensorrt_llm/examples`.

Alternatively, if you have already cloned the TensorRT LLM repository, you can use the following convenient command to
run the container:

```bash
make -C docker ngc-release_run LOCAL_USER=1 DOCKER_PULL=1 IMAGE_TAG=x.y.z
```

This command pulls the specified container from the NVIDIA NGC registry, sets up the local user's account within the
container, and launches it with full GPU support.

For comprehensive information about TensorRT-LLM, including documentation, source code, examples, and installation
guidelines, visit the following official resources:

- [TensorRT LLM GitHub Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT LLM Online Documentation](https://nvidia.github.io/TensorRT-LLM/latest/index.html)

### Security CVEs

To review known CVEs on this image, refer to the Security Scanning tab on this page.

### License

By pulling and using the container, you accept the terms and conditions of
this [End User License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and [Product-Specific Terms](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
