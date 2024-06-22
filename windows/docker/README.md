# Building the TensorRT-LLM Windows Docker Image

These instructions provide details on how to build the TensorRT-LLM Windows Docker image manually from source.

You should already have set up Docker Desktop based on the [Windows source build instructions](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-windows.html#docker-desktop).

From the `TensorRT-LLM\windows\` folder, run the build command:

```bash
docker build -f .\docker\Dockerfile -t tensorrt-llm-windows-build:latest .
```

Your image is now ready for use. Return to [Run the Container](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-windows.html#run-the-container) to proceed with your TensorRT-LLM build using Docker.
