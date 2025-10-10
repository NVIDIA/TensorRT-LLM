(linux)=

# Installing on Linux via `pip`

1. Install TensorRT LLM (tested on Ubuntu 24.04).

   ### Install prerequisites

   Before the pre-built Python wheel can be installed via `pip`, a few
   prerequisites must be put into place:

   Install CUDA Toolkit following the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and
   make sure `CUDA_HOME` environment variable is properly set.

   ```bash
   # Optional step: Only required for NVIDIA Blackwell GPUs and SBSA platform
   pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   sudo apt-get -y install libopenmpi-dev
   
   # Optional step: Only required for disagg-serving
   sudo apt-get -y install libzmq3-dev hwloc libhwloc-dev
   export LIBFABRIC_REPO="https://github.com/ofiwg/libfabric.git"
   export LIBFABRIC_VERSION="v2.3.0"
   git clone --depth 1 -b ${LIBFABRIC_VERSION} ${LIBFABRIC_REPO}
   cd libfabric
   ./autogen.sh
   ./configure --prefix=/path/to/install/path \
        --disable-verbs \
        --disable-psm3 \
        --disable-opx \
        --disable-usnic \
        --disable-rstream \
        --enable-efa \
        --with-cuda=/path/to/cuda \
        --enable-cuda-dlopen
    make install -j$(nproc)
   ```

   PyTorch CUDA 12.8 package is required for supporting NVIDIA Blackwell GPUs and SBSA platform. On prior GPUs or Linux x86_64 platform, this extra installation is not required.

   ```{tip}
   Instead of manually installing the preqrequisites as described
   above, it is also possible to use the pre-built [TensorRT LLM Develop container
   image hosted on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel)
   (see [here](containers) for information on container tags).
   ```

   ### Install pre-built TensorRT LLM wheel

   Once all prerequisites are in place, TensorRT LLM can be installed as follows:

   ```bash
   pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
   ```
   **This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.**

2. Sanity check the installation by running the following in Python (tested on Python 3.12):

    ```{literalinclude} ../../../examples/llm-api/quickstart_example.py
        :language: python
        :linenos:
    ```

**Known limitations**

There are some known limitations when you pip install pre-built TensorRT LLM wheel package.

1. MPI in the Slurm environment

    If you encounter an error while running TensorRT LLM in a Slurm-managed cluster, you need to reconfigure the MPI installation to work with Slurm.
    The setup methods depends on your slurm configuration, pls check with your admin. This is not a TensorRT LLM specific, rather a general mpi+slurm issue.
    ```
    The application appears to have been direct launched using "srun",
    but OMPI was not built with SLURM support. This usually happens
    when OMPI was not configured --with-slurm and we weren't able
    to discover a SLURM installation in the usual places.
    ```
