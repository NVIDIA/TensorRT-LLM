(linux)=

# Installing on Linux

1. Install TensorRT-LLM (tested on Ubuntu 24.04).

    ```bash
    (Optional) pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
    ```

    PyTorch CUDA 12.8 package is required for supporting NVIDIA Blackwell GPUs. On prior GPUs, this extra installation is not required.

    If using the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) image, the prerequisite steps for installing NVIDIA Blackwell-enabled PyTorch package and `libopenmpi-dev` are not required.

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

3. Install inside the PyTorch NGC Container

   The PyTorch NGC Container may lock Python package versions via the `/etc/pip/constraint.txt` file. When installing the pre-built TensorRT-LLM wheel inside the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), you need to clear this file first.

   ```bash
   [ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
   ```

   PyTorch NGC Container typically includes a pre-installed `tensorrt` Python package. If there is a version mismatch between this pre-installed package and the version required by the TensorRT-LLM wheel, you will need to uninstall the existing `tensorrt` package before installing TensorRT-LLM.

   ```bash
   pip uninstall -y tensorrt
   ```
