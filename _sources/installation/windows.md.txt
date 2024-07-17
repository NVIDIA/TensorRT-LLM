(windows)=

# Installing on Windows

```{note}
The Windows release of TensorRT-LLM is currently in beta.
We recommend checking out the [v0.10.0 tag](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.10.0) for the most stable experience.
```

**Prerequisites**

1. Clone this repository using [Git for Windows](https://git-scm.com/download/win).

2. Install the dependencies one of two ways:

    1. Install all dependencies together.

       1. Run the provided PowerShell script `setup_env.ps1` located under the `/windows/` folder which installs Python and CUDA 12.4.1 automatically with default settings. Run PowerShell as Administrator to use the script.

       ```bash
       ./setup_env.ps1 [-skipCUDA] [-skipPython]
       ```

       2. Close and re-open any existing PowerShell or Git Bash windows so they pick up the new `Path` modified by the `setup_env.ps1` script above.

    2. Install the dependencies one at a time.

        1. Install [Python 3.10](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe).

            1. Select **Add python.exe to PATH** at the start of the installation. The installation may only add the `python` command, but not the `python3` command.
            2. Navigate to the installation path `%USERPROFILE%\AppData\Local\Programs\Python\Python310` (`AppData` is a hidden folder) and copy `python.exe` to `python3.exe`.

        2. Install [CUDA 12.4.1 Toolkit](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Windows&target_arch=x86_64). Use the Express Installation option. Installation may require a restart.


**Steps**

1. Install TensorRT-LLM.

  If you have an existing TensorRT installation (from older versions of `tensorrt_llm`), please execute

  ```bash
  pip uninstall -y tensorrt tensorrt_libs tensorrt_bindings
  pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
  ```

  before installing TensorRT-LLM with the following command.

  ```bash
  pip install tensorrt_llm==0.10.0 --extra-index-url https://pypi.nvidia.com
  ```

  Run the following command to verify that your TensorRT-LLM installation is working properly.

  ```bash
  python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
  ```

2. Build the model.
3. Deploy the model.
