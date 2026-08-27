(installation-guide)=
(containers)=

# Installation Guide

There are multiple ways to install and run TensorRT LLM. The options below are ordered from simplest to most involved. Before installing, check the [Supported Hardware](../supported-hardware) page to ensure your GPU is compatible.

**This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.**

Nightly releases are development builds identified by a `.dev` segment in the version. They provide early access to
the latest changes, but their quality is not guaranteed and they may contain bugs or regressions.

## Option 1: Pre-built Release Container

Pre-built TensorRT LLM releases are available as [container images on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release). This is the simplest way to obtain TensorRT LLM.

Replace `x.y.z` with the desired version tag. The [available tags on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)
include both regular and `.dev` nightly releases.

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:x.y.z

docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

{{container_tag_admonition}}

View the source commit recorded in the image by running the following inside the container:

```bash
printenv TRT_LLM_GIT_COMMIT
```

Sanity check the installation by running the following inside the container:

```bash
python3 -c "import tensorrt_llm"
```

(linux)=
## Option 2: Install on Linux via `pip`

> **Note:** The TensorRT LLM wheel on PyPI is built with the [public PyTorch package](https://pypi.org/project/torch/). This version may be incompatible with the NVIDIA NGC PyTorch container, which uses a different PyTorch build.
> If you are using the NGC PyTorch container, install the wheel built specifically for that container. The pre-built NGC PyTorch container-specific wheel is located at `/app/tensorrt_llm` inside the TensorRT LLM NGC Release container.

Tested on Ubuntu 24.04.

### Install prerequisites

Before the pre-built Python wheel can be installed via `pip`, a few
prerequisites must be put into place:

Install CUDA Toolkit 13.2 following the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
and make sure `CUDA_HOME` environment variable is properly set.

The `cuda-compat-13-2` package may be required depending on your system's NVIDIA GPU
driver version. For additional information, refer to the [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

```bash
# By default, PyTorch CUDA 12.8 package is installed. Install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
pip3 install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cu130

sudo apt-get -y install libopenmpi-dev

# Optional step: Only required for disagg-serving
sudo apt-get -y install libzmq3-dev
```

```{tip}
Instead of manually installing the prerequisites as described
above, it is also possible to use the pre-built [TensorRT LLM Develop container
image hosted on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel)
(see [here](containers) for information on container tags).
```

### Install pre-built TensorRT LLM wheel

Once all prerequisites are in place, TensorRT LLM can be installed as follows:

Before installing the latest version, uninstall any previous CUTLASS DSL installation as described in the
[CUTLASS DSL installation guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html#installation):

```bash
pip3 uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
    nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13
```

```bash
pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm
```

#### Install a nightly release

Nightly wheels are published to a dedicated package index. Use `--extra-index-url` to add this index while keeping
PyPI available for dependency resolution.

To install the latest nightly release:

```bash
pip3 install --pre tensorrt_llm \
    --extra-index-url https://pypi.nvidia.com/trtllm_nightly/
```

To install a specific nightly release, choose a version from the [available nightly wheels](https://pypi.nvidia.com/trtllm_nightly/tensorrt-llm/)
and replace `<version>` in the following command:

```bash
pip3 install --pre "tensorrt_llm==<version>" \
    --extra-index-url https://pypi.nvidia.com/trtllm_nightly/
```

### Sanity check

View the source commit recorded in the wheel metadata:

```bash
pip3 show --verbose tensorrt_llm | grep "Source Commit"
```

Run a quick start example on supported environment:

```{literalinclude} ../../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

### Known limitations

There are some known limitations when you pip install the pre-built TensorRT LLM wheel package.

1. MPI in the Slurm environment

    If you encounter an error while running TensorRT LLM in a Slurm-managed cluster, you need to reconfigure the MPI installation to work with Slurm.
    The setup method depends on your Slurm configuration, please check with your admin. This is not TensorRT LLM specific, but rather a general MPI+Slurm issue.
    ```text
    The application appears to have been direct launched using "srun",
    but OMPI was not built with SLURM support. This usually happens
    when OMPI was not configured --with-slurm and we weren't able
    to discover a SLURM installation in the usual places.
    ```

2. Prevent `pip` from replacing existing PyTorch installation

   On certain systems, particularly Ubuntu 22.04, users installing TensorRT LLM would find that their existing, CUDA 13.0 compatible PyTorch installation (e.g., `torch==2.9.0+cu130`) was being uninstalled by `pip`. It was then replaced by a CUDA 12.8 version (`torch==2.9.0`), causing the TensorRT LLM installation to be unusable and leading to runtime errors.

   The solution is to create a `pip` constraints file, locking `torch` to the currently installed version. Here is an example of how this can be done manually:

   ```bash
   CURRENT_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
   echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
   pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm -c /tmp/torch-constraint.txt
   ```

## Option 3: Build from Source

For developers who wish to modify, customize, or contribute to TensorRT LLM, see [Build from Source](build-from-source).
