(linux)=

# Installing on Linux via `pip`

1. Install TensorRT LLM (tested on Ubuntu 24.04).

   ### Install prerequisites

   Before the pre-built Python wheel can be installed via `pip`, a few
   prerequisites must be put into place:

   Install CUDA Toolkit following the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and
   make sure `CUDA_HOME` environment variable is properly set.

   ```bash
   # By default, PyTorch CUDA 12.8 package is installed. Install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
   pip3 install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130

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
