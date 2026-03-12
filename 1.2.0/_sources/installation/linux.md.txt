(linux)=

# Installing on Linux via `pip`

1. Install TensorRT LLM (tested on Ubuntu 24.04).

   ### Install prerequisites

   Before the pre-built Python wheel can be installed via `pip`, a few
   prerequisites must be put into place:

   Install CUDA Toolkit 13.1 following the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
   and make sure `CUDA_HOME` environment variable is properly set.

   The `cuda-compat-13-1` package may be required depending on your system's NVIDIA GPU
   driver version. For additional information, refer to the [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

   ```bash
   # By default, PyTorch CUDA 12.8 package is installed. Install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
   pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

   sudo apt-get -y install libopenmpi-dev
   
   # Optional step: Only required for disagg-serving
   sudo apt-get -y install libzmq3-dev
   ```

   ```{tip}
   Instead of manually installing the preqrequisites as described
   above, it is also possible to use the pre-built [TensorRT LLM Develop container
   image hosted on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel)
   (see [here](containers) for information on container tags).
   ```

   ### Install pre-built TensorRT LLM wheel

   Once all prerequisites are in place, TensorRT LLM can be installed as follows:

   ```bash
   pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm
   ```

   > **Note:** The TensorRT LLM wheel on PyPI is built with PyTorch 2.9.1. This version may be incompatible with the NVIDIA NGC PyTorch 25.12 container, which uses a more recent PyTorch build from the main branch. If you are using this container or a similar environment, please install the pre-built wheel located at `/app/tensorrt_llm` inside the TensorRT LLM NGC Release container instead.

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

2. Prevent `pip` from replacing existing PyTorch installation

   On certain systems, particularly Ubuntu 22.04, users installing TensorRT LLM would find that their existing, CUDA 13.0 compatible PyTorch installation (e.g., `torch==2.9.0+cu130`) was being uninstalled by `pip`. It was then replaced by a CUDA 12.8 version (`torch==2.9.0`), causing the TensorRT LLM installation to be unusable and leading to runtime errors.

   The solution is to create a `pip` constraints file, locking `torch` to the currently installed version. Here is an example of how this can be done manually:

   ```bash
   CURRENT_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
   echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
   pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm -c /tmp/torch-constraint.txt
   ```
