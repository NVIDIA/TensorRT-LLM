# Nitro TensorRT-LLM

<p align="center">
  <img alt="nitrologo" src="https://raw.githubusercontent.com/janhq/nitro/main/assets/Nitro%20README%20banner.png">
</p>

<p align="center">
  <a href="https://nitro.jan.ai/docs">Documentation</a> - <a href="https://nitro.jan.ai/api-reference">API Reference</a> 
  - <a href="https://github.com/janhq/nitro/releases/">Changelog</a> - <a href="https://github.com/janhq/nitro/issues">Bug reports</a> - <a href="https://discord.gg/AsJ8krTT3N">Discord</a>
</p>

> ⚠️ **Nitro is currently in Development**: Expect breaking changes and bugs!

## About 

Nitro TensorRT-LLM is an experimental implementation of [Nitro](https://nitro.jan.ai) that runs LLMs using [Nvidia's TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on Windows. 

- Pure C++ inference server on top of TensorRT-LLM's C++ Runtime
- OpenAI-compatible API with `/chat/completion` and `loadmodel` endpoints
- Packageable as a single runnable package (e.g. `nitro.exe`) to run seamlessly on bare metal in Windows
- Can be embedded in Windows Desktop apps

You can try this in [Jan](https://jan.ai) using the TensorRT-LLM Extension, with a Nvidia 3090 or 4090. 

> Read more about Nitro at https://nitro.jan.ai/

### Repo Structure

This repo is a fork of [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

The Nitro inference server is then included in `ROOT/cpp/tensorrt_llm/nitro`. Nitro inference server is built as the final step.

```
+-- cpp
|   +-- tensorrt_llm
|   |   +-- nitro
|   |   |   +-- nitro_deps
|   |   |   +-- main.cc
|   |   |   +-- ...
|   |   +-- CMakeLists.txt
```

### Package Structure

Nitro TensorRT-LLM can be compiled into a single Windows executable that runs seamlessly on bare metal.

The Nitro TensorRT-LLM executable is approximately ~730mb. Note: this excludes the TensorRT-LLM Engine for the Model. 

> NOTE: Nvidia Driver >=535 and CUDA Toolkit >=12.2 are pre-requisites, which are often pre-installed with Nvidia GPUs 

| Dependencies                    | Purpose                                                                                    | Size       |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ---------- |
| nitro.exe                       | Nitro                                                                                      | Negligible |
| tensorrt_llm.dll                | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | ~450mb     |
| nvinfer.dll                     | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | ~200mb     |
| nvinfer_plugin_tensorrt_llm.dll | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | Negligible |
| cudnn_ops_infer64_8.dll         | [cuDNN](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#cudnn)                    | ~80mb      |
| cudnn64_8.dll                   | [cuDNN](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#cudnn)                    | Negligible |
| msmpi.dll                       | [Microsoft MPI](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#microsoft-mpi)    | Negligible |
| **Total**                       |                                                                                            | **~730mb** |

## Quickstart

### Step 1: Pre-requisites

#### Nvidia Driver
Install an [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) supporting CUDA 12.2 or higher.

Use the following command to verify the installation:
```sh
nvidia-smi
```

#### CUDA Toolkit
Install a [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) compatible with your NVIDIA driver.

Use the following command to verify the installation:

```sh
nvcc --version
```

### Step 2: Get a TensorRT Engine

TensorRT Engines are precompiled binaries that contain the underlying LLM. These engines are OS and GPU-specific to the machines used to build them, and they are not yet cross-platform.

This means you need a specific TensorRT Engine based on: 
- Large Language Model
- Operating system
- GPU type

#### Option 1: Download a prebuilt TensorRT Engine

We've compiled some initial engines available for download: 
- OS: Windows 10, GPU: 3090s, Model: OpenHermes 7B
- OS: Windows 11, GPU: 4090s, Model: OpenHermes 7B

[TODO: add links]

#### Option 2: Build a TensorRT Engine from model

You can also build the TensorRT Engine directly on your machine, using your preferred model.

This process can take upwards of 1 hour. 

See [Compile from Source](#compile-from-source) instructions below. 

### Step 3: Run Nitro TensorRT-LLM Server

- [ ] TODO: Explain how user needs to run the ./nitro.exe

```bash title="Run Nitro server"
# Go to folder with `nitro.exe`
./nitro
```

### Step 4: Load Model Engine

```bash title="Load model"
curl -X POST   http://0.0.0.0:3928/inferences/tensorrtllm/loadmodel   
  -H 'Content-Type: application/json'
  -d '{
    "engine_path": <ENGINE_PATH_HERE>, 
    "ctx_len": 1000
  }'
```

#### Parameters

| Parameter     | Type    | Description                               |
| ------------- | ------- | ----------------------------------------- |
| `engine_path` | String  | The file path to the TensorRT-LLM engine. |
| `ctx_len`     | Integer | The context length for engine operations. |

### Step 5: Making an Inference Request

Note: `model` field is not used, as Nitro TensorRT-LLM only loads one model at a time. It is retained for OpenAI-compatibility but discarded

```bash title="Nitro TensorRT-LLM Inference"
curl --location 'http://0.0.0.0:3928/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Accept: text/event-stream' \
  --header 'Access-Control-Allow-Origin: *' \
  --data '{
    "messages": [
      {
        "content": "Hello there 👋",
        "role": "assistant"
      },
      {
        "content": "Write a long story about NVIDIA!!!!",
        "role": "user"
      }
    ],
    "stream": true,
    "model": <NON-NULL STRING>, 
    "max_tokens": 2048
  }'
```

## Compile from source

- [ ] TODO

## Download

- [ ] TODO

## Contact

- For support, please file a GitHub ticket.
- For questions, join our Discord [here](https://discord.gg/FTk2MvZwJH).
- For long-form inquiries, please email hello@jan.ai.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=janhq/tensorrt-llm-nitro&type=Date)](https://star-history.com/#janhq/tensorrt-llm-nitro&Date)
