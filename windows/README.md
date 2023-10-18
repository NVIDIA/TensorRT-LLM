# TensorRT-LLM for Windows

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Additional Setup for Building from Source](#additional-setup-for-building-from-source)
- [Building from Source](#building-from-source)
- [Installation](#installation)
- [Extra Steps for C++ Runtime Usage](#extra-steps-for-c-runtime-usage)
- [Next Steps](#next-steps)

## Overview

TensorRT-LLM is supported on bare-metal Windows for single-GPU inference. We provide a release wheel for Windows which can be downloaded from https://developer.nvidia.com/. Alternatively, you may build TensorRT-LLM for Windows from source. Building from source is an advanced option and is not necessary for building or running LLM engines. It is, however, required if you plan to use the C++ runtime directly or run C++ benchmarks.

## Quick Start

If you encounter difficulties with any prerequisites, check the [Detailed Setup](#detailed-setup) instructions below.

Prerequisites:
- [Python3 >= 3.9](https://www.python.org/downloads/windows/)
- [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
- [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [TensorRT 9.1.0.4](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)

```
pip install -r .\requirements-windows.txt
pip install tensorrt_llm-<version>-py3-none-any.whl
```

## Detailed Setup

### Python
Install [Python3 >= 3.9](https://www.python.org/downloads/windows/). When installing, add to the system `Path` and click "Disable path length limit." The installation may only add the `python` command, but not the `python3` command. Navigate to the installation path, `C:\Users\<username>\AppData\Local\Programs\Python\Python39` (note `AppData` is a hidden folder), and copy `python.exe` to `python3.exe`.

### CUDA
Install the [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64). You may use the Express Installation option. Installation may require a restart.

### Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467). You will be prompted to choose between an `exe`, which installs the MPI executable, and an `msi`, which installs the MPI SDK. Download and install both.

### TensorRT-LLM Repo
It may be useful to create a single folder for holding TensorRT-LLM and its dependencies, such as `C:\Users\<username>\inference\`. We will assume this directory structure in further steps.

Install [Git for Windows](https://git-scm.com/download/win).

Clone TensorRT-LLM using Git Bash (Powershell works as well):
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
```

### cuDNN and TensorRT

Download and unzip [TensorRT 9.1.0.4](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip). Move the folder to a location you can reference later, such as `C:\Users\<username>\inference\TensorRT`.

Download and unzip [cuDNN](https://developer.nvidia.com/cudnn). Move the folder to a location you can reference later, such as `C:\Users\<username>\inference\cuDNN`.

You'll need to add libraries and binaries for TensorRT and cuDNN to your system's `Path` environment variable. To do so, click the Windows button and search for "environment variables." Select "Edit the system environment variables." A "System Properties" window will open. Select the "Environment Variables" button at the bottom right, then in the new window under "System variables" click "Path" then the "Edit" button. Add "New" lines for the `bin` and `lib` dirs of both TensorRT and cuDNN. Your `Path` should include lines like this:

```
C:\Users\<username>\inference\TensorRT\bin
C:\Users\<username>\inference\TensorRT\lib
C:\Users\<username>\inference\cuDNN\bin
C:\Users\<username>\inference\cuDNN\lib
```

Click "OK" on all the open dialogue windows. Be sure to close and re-open any existing Powershell or Git Bash windows so they pick up the new `Path`.

Now, to install the TensorRT core libraries, run Powershell and use `pip` to install the Python wheel:
```
pip install C:\Users\<username>\inference\TensorRT\python\tensorrt-9.1.0.post12.dev4-cp39-none-win_amd64.whl
```

You may run the following command to verify that your TensorRT installation is working properly:
```
python -c "import tensorrt as trt; print(trt.__version__)"
```

If you are using the pre-built TensorRT-LLM release wheel (recommended unless you need to directly invoke the C++ runtime), skip to [Installation](#installation). If you are building your own wheel from source, proceed to [Additional Setup for Building from Source](#additional-setup-for-building-from-source).

## Additional Setup for Building from Source

*Advanced. Skip this section if you plan to use the pre-built TensorRT-LLM release wheel.*

Install [CMake](https://cmake.org/download/) and select the option to add it to the system path.

Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/). When prompted to select more Workloads, check "Desktop development with C++."

TensorRT-LLM on Windows currently depends on NVTX assets that do not come packaged with the CUDA12.2 installer. To install these assets, download the [CUDA11.8 Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). During installation, select "Advanced installation." Nsight NVTX is located in the CUDA drop down. Deselect *all* packages, and select Nsight NVTX.

## Building from Source

*Advanced. Skip this section if you plan to use the pre-built TensorRT-LLM release wheel.*

In Powershell, from the `TensorRT-LLM` root folder, run:
```
python .\scripts\build_wheel.py -a <architecture> --trt_root <path_to_trt_root> --build_type Release -D "ENABLE_MULTI_DEVICE=0"
```
`<architecture>` should correspond to the architecture or list of architectures you wish to support, e.g `"86-real;89-real"` to support GeForce 30-series and 40-series cards.

The `-D "ENABLE_MULTI_DEVICE=0"` is required on Windows. Multi-device inference is supported on Linux, but not on Windows.

The above command will generate `build\tensorrt_llm-<version>-py3-none-any.whl`. Other generated files include:

- `build\` - Contains the wheel and other built artifacts
- `cpp\build\` - Contains cpp-related build files
  - `cpp\build\tensorrt_llm\Release` - Contains shared and static libraries for TensorRT-LLM C++ runtime
- `tensorrt_llm\libs\` - Contains other C++ runtime dependencies that were built, namely `nvinfer_plugin_tensorrt_llm.dll` and `th_common.dll`

## Installation

In Powershell, from the root of this repo, run:
```
pip install -r .\requirements-windows.txt
pip install .\build\tensorrt_llm-<version>-py3-none-any.whl
```

You may run the following command to verify that your TensorRT-LLM installation is working properly:
```
python -c "import tensorrt_llm; print(tensorrt_llm._utils.trt_version())"
```

## Extra Steps for C++ Runtime Usage

*Advanced. Skip this section if you do not intend to use the TensorRT-LLM C++ runtime directly. Note that you have to have built from source to use the C++ runtime.*

Building from source creates libraries that can be used if you wish to directly link against the C++ runtime for TensorRT-LLM. These libraries are also required if you wish to run C++ unit tests and some benchmarks.

Building from source will produce the following library files:
- `tensorrt_llm` libraries located in `cpp\build\tensorrt_llm\Release`
  - `tensorrt_llm.dll` - Shared library
  - `tensorrt_llm.exp` - Export file
  - `tensorrt_llm.lib` - Stub for linking to `tensorrt_llm.dll`
  - `tensorrt_llm_static.lib` - Static library
- Dependency libraries (These get copied to `tensorrt_llm\libs\`)
  - `nvinfer_plugin_tensorrt_llm` libraries located in `cpp\build\tensorrt_llm\plugins\Release\`
    - `nvinfer_plugin_tensorrt_llm.dll`
    - `nvinfer_plugin_tensorrt_llm.exp`
    - `nvinfer_plugin_tensorrt_llm.lib`
  - `th_common` libraries located in `cpp\build\tensorrt_llm\thop\Release`
    - `th_common.dll`
    - `th_common.exp`
    - `th_common.lib`

The locations of the DLLs, in addition to some `torch` DLLs, must be added to the Windows `Path` in order to us the TensorRT-LLM C++ runtime. As in [Setup](#setup), append the locations of these libraries to your `Path`. When complete, your `Path` should include lines similar to these:

```
C:\Users\<username>\inference\TensorRT-LLM\cpp\build\tensorrt_llm\Release
C:\Users\<username>\AppData\Local\Programs\Python\Python39\Lib\site-packages\tensorrt_llm\libs
C:\Users\<username>\AppData\Local\Programs\Python\Python39\Lib\site-packages\torch\lib
```

For examples of how to use the C++ runtime, see the unit tests in
[gptSessionTest.cpp](cpp/tests/runtime/gptSessionTest.cpp) and the related
[CMakeLists.txt](cpp/tests/CMakeLists.txt) file.

## Next Steps

See [examples/llama](examples/llama) for a showcase of how to run a quick benchmark on LLaMa.

## Limitations

`openai-triton` examples are not supported on Windows.
