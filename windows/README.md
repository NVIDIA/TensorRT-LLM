# TensorRT-LLM for Windows

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Building from Source](#building-from-source)
- [Installation](#installation)
- [Extra Steps for C++ Runtime Usage](#extra-steps-for-c-runtime-usage)
- [Next Steps](#next-steps)
- [Limitations](#limitations)
- [Troubleshooting Common Errors](#troubleshooting-common-errors)

## Overview

**NOTE: The Windows release of TensorRT-LLM is currently in beta. We recommend using the `rel` branch for the most stable experience. The latest supported Windows release is 0.6.1. You are currently on `main`.**

TensorRT-LLM is supported on bare-metal Windows for single-GPU inference. The release supports GeForce 40-series GPUs.

The release wheel for Windows can be installed with `pip`. Alternatively, you may build TensorRT-LLM for Windows from source. Building from source is an advanced option and is not necessary for building or running LLM engines. It is, however, required if you plan to use the C++ runtime directly or run C++ benchmarks.

## Quick Start

You can clone this repository using [Git for Windows](https://git-scm.com/download/win).

We provide a Powershell script, `setup_env.ps1`, which installs Python, CUDA 12.2, and Microsoft MPI automatically with default settings. Be sure to run Powershell as Administrator to use the script. Usage:

```
./setup_env.ps1 [-skipCUDA] [-skipPython] [-skipMPI]
```

Close and reopen Powershell after running the script so that `Path` changes take effect. The script will install whichever components are not skipped. Any components may be installed manually instead of using the script. Further, cuDNN **must** be installed manually. For more details about manually installing prerequisites, check the [Detailed Setup](#detailed-setup) instructions below.

Prerequisites:
- [Python 3.10](https://www.python.org/downloads/windows/)
- [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Windows&target_arch=x86_64)
- [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
- [cuDNN](https://developer.nvidia.com/cudnn)

Once your prerequisites are installed, install TensorRT-LLM:

```
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```

You may now build and run models!

## Detailed Setup

### Python
Install [Python 3.10](https://www.python.org/downloads/windows/). Select "Add python.exe to PATH" at the start of the installation. The installation may only add the `python` command, but not the `python3` command. Navigate to the installation path, `%USERPROFILE%\AppData\Local\Programs\Python\Python310` (note `AppData` is a hidden folder), and copy `python.exe` to `python3.exe`.

### CUDA
Install the [CUDA 12.2 Toolkit](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Windows&target_arch=x86_64). You may use the Express Installation option. Installation may require a restart.

### Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467). You will be prompted to choose between an `exe`, which installs the MPI executable, and an `msi`, which installs the MPI SDK. Download and install both.

### TensorRT-LLM Repo
It may be useful to create a single folder for holding TensorRT-LLM and its dependencies, such as `%USERPROFILE%\inference\`. We will assume this folder structure in further steps.

Clone TensorRT-LLM:
```
git clone --branch rel https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
```

### cuDNN

Download and unzip [cuDNN](https://developer.nvidia.com/cudnn). Move the folder to a location you can reference later, such as `%USERPROFILE%\inference\cuDNN`.

You'll need to add libraries and binaries for cuDNN to your system's `Path` environment variable. To do so, click the Windows button and search for "environment variables." Select "Edit the system environment variables." A "System Properties" window will open. Select the "Environment Variables" button at the bottom right, then in the new window under "System variables" click "Path" then the "Edit" button. Add "New" lines for the `bin` and `lib` dirs of cuDNN. Your `Path` should include lines like this:

```
%USERPROFILE%\inference\cuDNN\bin
%USERPROFILE%\inference\cuDNN\lib
```

Click "OK" on all the open dialogue windows. Be sure to close and re-open any existing Powershell or Git Bash windows so they pick up the new `Path`.

If you are using the pre-built TensorRT-LLM release wheel (recommended unless you need to directly invoke the C++ runtime), skip to [Installation](#installation). If you are building your own wheel from source, proceed to [Building from Source](#building-from-source).

## Building from Source

*Advanced. Skip this section if you plan to use the pre-built TensorRT-LLM release wheel.*

Building from source requires extra prerequisites:
- [CMake](https://cmake.org/download/) (version 3.27.7 recommended)
- [Visual Studio 2022](https://visualstudio.microsoft.com/)
- [TensorRT 9.2.0.5 for TensorRT-LLM](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.windows10.x86_64.cuda-12.2.llm.beta.zip)
- Nsight NVTX

We provide a Docker container with these prerequisites already installed. Building with Docker will require you to install [Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/), build the container, build TensorRT-LLM, and copy files out of the Docker container for usage on your Windows host machine. Alternatively, you may install the prerequisites on a bare-metal machine and build there. See [Docker Build Instructions](#docker-build-instructions) or [Bare-Metal Build Instructions](#bare-metal-build-instructions) to proceed.

### Docker Build Instructions

#### Docker Desktop

Install [Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/). You may need to change the following configurations:
- Right click the Docker icon in the Windows system tray (bottom right of your taskbar) and select "Switch to Windows containers..."
- In Docker Desktop settings on the General tab, uncheck "Use the WSL 2 based image"
- On the Docker Engine tab, set you configuration file to
```
{
  "experimental": true
}
```

Note: After building, you'll need to copy files out of your container. `docker cp` is not supported on Windows for Hyper-V based images. Unless you are using WSL 2 based images, be sure to mount a folder, e.g. `trt-llm-build`, to your container when you run it for moving files between the container and host system.

#### Acquiring an Image

The Docker container will be hosted for public download in a future release. At this time, it must be built manually. See [windows/docker/README.md](/windows/docker/README.md) for image build instructions.

#### Running the Container

Run the container in interactive mode with your build folder mounted. Be sure to specify a memory limit with the `-m` flag - by default the limit is 2GB, which is not sufficient to build TensorRT-LLM.
```
docker run -it -m 12g -v .\trt-llm-build:C:\workspace\trt-llm-build tensorrt-llm-windows-build:latest
```

#### Build and Extract Files

Clone and setup the TensorRT-LLM repository within the container:
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
```

Build TensorRT-LLM
```
python .\scripts\build_wheel.py -a "89-real" --trt_root C:\workspace\TensorRT-9.2.0.5\
```

The above command will generate `build\tensorrt_llm-*.whl`. Copy or move this into your mounted folder so it can be accessed on your host machine. If you intend to use the C++ runtime, you'll also need to gather various DLLs from the build into your mounted folder. Complete information about these files can be found below in [Extra Steps for C++ Runtime Usage](#extra-steps-for-c-runtime-usage).

Once you've gathered your files into the mounted folder, you may exit the container and continue on to [Installation](#installation).

### Bare-Metal Build Instructions

We provide a second Powershell script, `setup_build_env.ps1`, which installs CMake, Microsoft Visual Studio Build Tools, and TensorRT automatically with default settings. Be sure to run Powershell as Administrator to use the script. Usage:

```
./setup_build_env.ps1 -TRTPath <TRT-containing-folder> [-skipCMake] [-skipVSBuildTools] [-skipTRT]
```

Close and reopen Powershell after running the script so that `Path` changes take effect. Note that you should supply to `-TRTPath` a directory that already exists to contain TensorRT - e.g. `-TRTPath ~/inference` may be valid, but `-TRTPath ~/inference/TensorRT` will not be valid if `TensorRT` does not exist. `-TRTPath` isn't required if `-skipTRT` is supplied.

The script will install whichever components are not skipped. Any components may be installed manually instead of using the script. Note that for Visual Studio, the script just installs the command-line Build Tools. You may prefer a full Visual Studio 2022 IDE installation, which is linked below.

Nsight NVTX **must** be installed manually. For more details about manually installing individual prerequisites, including NVTX, check the instructions below.

#### CMake

Install [CMake](https://cmake.org/download/) (version 3.27.7 recommended) and select the option to add it to the system path.

#### Visual Studio

Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/). When prompted to select more Workloads, check "Desktop development with C++."

#### TensorRT

Download and unzip [TensorRT 9.2.0.5 for TensorRT-LLM](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.windows10.x86_64.cuda-12.2.llm.beta.zip). Move the folder to a location you can reference later, such as `%USERPROFILE%\inference\TensorRT`.

You'll need to add libraries for TensorRT  to your system's `Path` environment variable. Follow the same instructions used for [cuDNN](#cuDNN). Your `Path` should include a line like this:

```
%USERPROFILE%\inference\TensorRT\lib
```

Be sure to close and re-open any existing Powershell or Git Bash windows so they pick up the new `Path`.

Now, to install the TensorRT core libraries, run Powershell and use `pip` to install the Python wheel:
```
pip install %USERPROFILE%\inference\TensorRT\python\tensorrt-*.whl
```

You may run the following command to verify that your TensorRT installation is working properly:
```
python -c "import tensorrt as trt; print(trt.__version__)"
```

#### Nsight NVTX

TensorRT-LLM on Windows currently depends on NVTX assets that do not come packaged with the CUDA12.2 installer. To install these assets, download the [CUDA11.8 Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). During installation, select "Advanced installation." Nsight NVTX is located in the CUDA drop down. Deselect all packages, and then select Nsight NVTX.

#### 64-bit Developer Powershell

In order to build, you'll need to launch a 64-bit Developer Powershell. From your usual Powershell terminal, run one of the following two commands.

If you installed Visual Studio Build Tools (e.g. using the `setup_build_env.ps1` script):

```
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
```

If you installed Visual Studio Community (e.g. via manual GUI setup):

```
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
```

#### Build

In Powershell, from the `TensorRT-LLM` root folder, run:
```
python .\scripts\build_wheel.py -a "89-real" --trt_root <path_to_trt_root>
```

The `-a` flag specifies the device architecture. `"89-real"` supports GeForce 40-series cards.

Note that the flag `-D "ENABLE_MULTI_DEVICE=0"`, while not specified here, is implied on Windows. Multi-device inference is supported on Linux, but not on Windows.

The above command will generate `build\tensorrt_llm-*.whl`.

## Installation

To download and install the wheel, in Powershell, run:
```
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```

Alternatively, if you built the wheel from source, locate your wheel (either in `build\` or in the folder you mounted to your Docker contailer) and run:
```
pip install tensorrt_llm-*.whl
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
  - `nvinfer_plugin_tensorrt_llm` libraries located in `cpp\build\tensorrt_llm\plugins\`
    - `nvinfer_plugin_tensorrt_llm.dll`
    - `nvinfer_plugin_tensorrt_llm.exp`
    - `nvinfer_plugin_tensorrt_llm.lib`
  - `th_common` libraries located in `cpp\build\tensorrt_llm\thop\`
    - `th_common.dll`
    - `th_common.exp`
    - `th_common.lib`

The locations of the DLLs, in addition to some `torch` DLLs, must be added to the Windows `Path` in order to us the TensorRT-LLM C++ runtime. As in [Detailed Setup](#detailed-setup), append the locations of these libraries to your `Path`. When complete, your `Path` should include lines similar to these:

```
%USERPROFILE%\inference\TensorRT-LLM\cpp\build\tensorrt_llm\Release
%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\tensorrt_llm\libs
%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\lib
```

Your `Path` additions may differ, particularly if you used the Docker method and copied all the relevant DLLs into a single folder.

For examples of how to use the C++ runtime, see the unit tests in
[gptSessionTest.cpp](../cpp/tests/runtime/gptSessionTest.cpp) and the related
[CMakeLists.txt](../cpp/tests/CMakeLists.txt) file.

## Next Steps

See [examples/llama](examples/llama) for a showcase of how to run a quick benchmark on LLaMa.

## Limitations

`openai-triton` examples are not supported on Windows.

## Troubleshooting Common Errors

Many build errors can be resolved by simply deleting the build tree. Try running the build script with `--clean` or running `rm -r cpp/build`.
