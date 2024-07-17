(build-from-source-windows)=

# Building from Source Code on Windows

```{note}
This section is for advanced users. Skip this section if you plan to use the pre-built TensorRT-LLM release wheel.
```

## Prerequisites

1. Install prerequisites listed in our [Installing on Windows](https://nvidia.github.io/TensorRT-LLM/installation/windows.html) document.
2. Install [CMake](https://cmake.org/download/), version 3.27.7 is recommended, and select the option to add it to the system path.
3. Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/).
4. Download and unzip [TensorRT 10.1.0.27](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/zip/TensorRT-10.1.0.27.Windows.win10.cuda-12.4.zip).

## Building a TensorRT-LLM Docker Image

### Docker Desktop

1. Install [Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/).
2. Set the following configurations:

  1. Right-click the Docker icon in the Windows system tray (bottom right of your taskbar) and select **Switch to Windows containers...**.
  2. In the Docker Desktop settings on the **General** tab, uncheck **Use the WSL 2 based image**.
  3. On the **Docker Engine** tab, set your configuration file to:

  ```
  {
    "experimental": true
  }
  ```

```{note}
After building, copy the files out of your container. `docker cp` is not supported on Windows for Hyper-V based images. Unless you are using WSL 2 based images, mount a folder, for example, `trt-llm-build`, to your container when you run it for moving files between the container and host system.
```

### Acquire an Image

The Docker container will be hosted for public download in a future release. At this time, it must be built manually. From the `TensorRT-LLM\windows\` folder, run the build command:

```bash
docker build -f .\docker\Dockerfile -t tensorrt-llm-windows-build:latest .
```

And your image is now ready for use.

### Run the Container

Run the container in interactive mode with your build folder mounted. Specify a memory limit with the `-m` flag. By default, the limit is 2 GB, which is not sufficient to build TensorRT-LLM.

```bash
docker run -it -m 12g -v .\trt-llm-build:C:\workspace\trt-llm-build tensorrt-llm-windows-build:latest
```

### Build and Extract Files

1. Clone and setup the TensorRT-LLM repository within the container.

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
```

2. Build TensorRT-LLM. This command generates `build\tensorrt_llm-*.whl`.

```bash
python .\scripts\build_wheel.py -a "89-real" --trt_root C:\workspace\TensorRT-10.1.0.27\
```

3. Copy or move `build\tensorrt_llm-*.whl` into your mounted folder so it can be accessed on your host machine. If you intend to use the C++ runtime, you'll also need to gather various DLLs from the build into your mounted folder. For more information, refer to [C++ Runtime Usage](#c-runtime-usage).



## Building TensorRT-LLM on Bare Metal

**Prerequisites**

1. Install all prerequisites (`git`, `python`, `CUDA`) listed in our [Installing on Windows](https://nvidia.github.io/TensorRT-LLM/installation/windows.html) document.
2. Install Nsight NVTX. TensorRT-LLM on Windows currently depends on NVTX assets that do not come packaged with the CUDA 12.4.1 installer. To install these assets, download the [CUDA 11.8 Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64).

    1. During installation, select **Advanced installation**.

    2. Nsight NVTX is located in the CUDA drop-down.

    3. Deselect all packages, and select **Nsight NVTX**.

3. Install the dependencies one of two ways:

    1. Run the `setup_build_env.ps1` script, which installs CMake, Microsoft Visual Studio Build Tools, and TensorRT automatically with default settings.

        1. Run PowerShell as Administrator to use the script.

        ```bash
        ./setup_build_env.ps1 -TRTPath <TRT-containing-folder> [-skipCMake] [-skipVSBuildTools] [-skipTRT]
        ```

        2. Close and reopen PowerShell after running the script so that `Path` changes take effect.

        3. Supply a directory that already exists to contain TensorRT to `-TRTPath`, for example, `-TRTPath ~/inference` may be valid, but `-TRTPath ~/inference/TensorRT` will not be valid if `TensorRT` does not exist. `-TRTPath` isn't required if `-skipTRT` is supplied.

    2. Install the dependencies one at a time.

        1. Install [CMake](https://cmake.org/download/), version 3.27.7 is recommended, and select the option to add it to the system path.
        2. Download and install [Visual Studio 2022](https://visualstudio.microsoft.com/). When prompted to select more Workloads, check **Desktop development with C++**.
        3. Download and unzip [TensorRT 10.1.0.27](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/zip/TensorRT-10.1.0.27.Windows.win10.cuda-12.4.zip). Move the folder to a location you can reference later, such as `%USERPROFILE%\inference\TensorRT`.

            1. Add the libraries for TensorRT  to your system's `Path` environment variable. Your `Path` should include a line like this:

            ```bash
            %USERPROFILE%\inference\TensorRT\lib
            ```

            2. Close and re-open any existing PowerShell or Git Bash windows so they pick up the new `Path`.

            3. Remove existing `tensorrt` wheels first by executing

            ```bash
            pip uninstall -y tensorrt tensorrt_libs tensorrt_bindings
            pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
            ```

            4. Install the TensorRT core libraries, run PowerShell, and use `pip` to install the Python wheel.

            ```bash
            pip install %USERPROFILE%\inference\TensorRT\python\tensorrt-*.whl
            ```

            5. Verify that your TensorRT installation is working properly.

            ```bash
            python -c "import tensorrt as trt; print(trt.__version__)"
            ```


**Steps**

1. Launch a 64-bit Developer PowerShell. From your usual PowerShell terminal, run one of the following two commands.

    1. If you installed Visual Studio Build Tools (that is, used the `setup_build_env.ps1` script):

    ```bash
    & 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
    ```

    2. If you installed Visual Studio Community (e.g. via manual GUI setup):

    ```bash
    & 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
    ```

2. In PowerShell, from the `TensorRT-LLM` root folder, run:

```bash
python .\scripts\build_wheel.py -a "89-real" --trt_root <path_to_trt_root>
```

The `-a` flag specifies the device architecture. `"89-real"` supports GeForce 40-series cards.

The flag `-D "ENABLE_MULTI_DEVICE=0"`, while not specified here, is implied on Windows. Multi-device inference is supported on Linux, but not on Windows.

This command generates `build\tensorrt_llm-*.whl`.

(c-runtime-usage)=
## Linking with the TensorRT-LLM C++ Runtime

```{note}
This section is for advanced users. Skip this section if you do not intend to use the TensorRT-LLM C++ runtime directly. You must build from source to use the C++ runtime.
```

Building from source creates libraries that can be used if you wish to directly link against the C++ runtime for TensorRT-LLM. These libraries are also required if you wish to run C++ unit tests and some benchmarks.

Building from source produces the following library files.
- `tensorrt_llm` libraries located in `cpp\build\tensorrt_llm`
  - `tensorrt_llm.dll` - Shared library
  - `tensorrt_llm.exp` - Export file
  - `tensorrt_llm.lib` - Stub for linking to `tensorrt_llm.dll`
- Dependency libraries (these get copied to `tensorrt_llm\libs\`)
  - `nvinfer_plugin_tensorrt_llm` libraries located in `cpp\build\tensorrt_llm\plugins\`
    - `nvinfer_plugin_tensorrt_llm.dll`
    - `nvinfer_plugin_tensorrt_llm.exp`
    - `nvinfer_plugin_tensorrt_llm.lib`
  - `th_common` libraries located in `cpp\build\tensorrt_llm\thop\`
    - `th_common.dll`
    - `th_common.exp`
    - `th_common.lib`

The locations of the DLLs, in addition to some `torch` DLLs, must be added to the Windows `Path` in order to use the TensorRT-LLM C++ runtime. Append the locations of these libraries to your `Path`. When complete, your `Path` should include lines similar to these:

```bash
%USERPROFILE%\inference\TensorRT-LLM\cpp\build\tensorrt_llm
%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\tensorrt_llm\libs
%USERPROFILE%\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\lib
```

Your `Path` additions may differ, particularly if you used the Docker method and copied all the relevant DLLs into a single folder.
