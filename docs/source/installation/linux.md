(linux)=

# Installing on Linux via `pip`

1. Install TensorRT-LLM (tested on Ubuntu 24.04).

   ### Install prerequisites

   Before the pre-built Python wheel can be installed via `pip`, a few
   prerequisites must be put into place:

   ```bash
   # Optional step: Only required for Blackwell and Grace Hopper
   pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   sudo apt-get -y install libopenmpi-dev
   ```

   PyTorch CUDA 12.8 package is required for supporting NVIDIA Blackwell and Grace Hopper GPUs. On prior GPUs, this extra installation is not required.

   ```{tip}
   Instead of manually installing the preqrequisites as described
   above, it is also possible to use the pre-built [TensorRT-LLM Develop container
   image hosted on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel)
   (see [here](containers) for information on container tags).
   ```

   ### Install pre-built TensorRT-LLM wheel

   Once all prerequisites are in place, TensorRT-LLM can be installed as follows:

   ```bash
   pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
   ```

2. Sanity check the installation by running the following in Python (tested on Python 3.12):

    ```{literalinclude} ../../../examples/llm-api/quickstart_example.py
        :language: python
        :linenos:
    ```

**Known limitations**

There are some known limitations when you pip install pre-built TensorRT-LLM wheel package.

1. MPI in the Slurm environment

    If you encounter an error while running TensorRT-LLM in a Slurm-managed cluster, you need to reconfigure the MPI installation to work with Slurm.
    The setup methods depends on your slurm configuration, pls check with your admin. This is not a TensorRT-LLM specific, rather a general mpi+slurm issue.
    ```
    The application appears to have been direct launched using "srun",
    but OMPI was not built with SLURM support. This usually happens
    when OMPI was not configured --with-slurm and we weren't able
    to discover a SLURM installation in the usual places.
    ```

2. CUDA Toolkit

    `pip install tensorrt-llm` won't install CUDA toolkit in your system, and the CUDA Toolkit is not required if want to just deploy a TensorRT-LLM engine.
    TensorRT-LLM uses the [ModelOpt](https://nvidia.github.io/TensorRT-Model-Optimizer/) to quantize a model, while the ModelOpt requires CUDA toolkit to jit compile certain kernels which is not included in the pytorch to do quantization effectively.
    Please install CUDA toolkit when you see the following message when running ModelOpt quantization.

    ```
    /usr/local/lib/python3.10/dist-packages/modelopt/torch/utils/cpp_extension.py:65:
    UserWarning: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
    Unable to load extension modelopt_cuda_ext and falling back to CPU version.
    ```
    The installation of CUDA toolkit can be found in [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/).
