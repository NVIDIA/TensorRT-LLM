# Building the TensorRT-LLM Windows Docker Image

These instructions provide details on how to build the TensorRT-LLM Windows Docker image manually from source.

You should already have set up Docker Desktop based on the top-level [Windows README instructions](/windows/README.md#docker-desktop).

## Set up Build Context

cuDNN and NvToolsExt cannot be installed via the command line, so you'll need to manually install them and copy them to the build context in order to build this container.

### cuDNN

If you followed the top-level [Windows README](/windows/README.md), you'll already have a copy of cuDNN. If not, download and unzip [cuDNN](https://developer.nvidia.com/cudnn).

Copy the entire `cuDNN` folder into `TensorRT-LLM/windows/docker`.

### NvToolsExt

TensorRT-LLM on Windows currently depends on NVTX assets that do not come packaged with the CUDA12.2 installer. To install these assets, download the [CUDA11.8 Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). During installation, select "Advanced installation." Nsight NVTX is located in the CUDA drop down. Deselect all packages, and then select Nsight NVTX.

You will now have `C:\Program Files\NVIDIA Corporation\NvToolsExt`. Copy the entire `NvToolsExt` folder into `TensorRT-LLM/windows/docker`

### Build

Now that `TensorRT-LLM\windows\docker` contains `cuDNN\` and `NvToolsExt\`, run the build command:

```
docker build -t tensorrt-llm-windows-build:latest .
```

Your image is now ready for use. Return to [Running the Container](/windows/README.md#running-the-container) to proceed with your TensorRT-LLM build using Docker.
