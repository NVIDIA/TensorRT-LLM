# Nitro TensorRT-LLM for Windows

<p align="center">
  <img alt="nitrologo" src="https://raw.githubusercontent.com/janhq/nitro/main/assets/Nitro%20README%20banner.png">
</p>

<p align="center">
  <a href="https://nitro.jan.ai/docs">Documentation</a> - <a href="https://nitro.jan.ai/api-reference">API Reference</a> 
  - <a href="https://github.com/janhq/nitro/releases/">Changelog</a> - <a href="https://github.com/janhq/nitro/issues">Bug reports</a> - <a href="https://discord.gg/AsJ8krTT3N">Discord</a>
</p>

> ⚠️ **Nitro is currently in Development**: Expect breaking changes and bugs!

## About 

Nitro TensorRT-LLM is an experimental implementation of [Nitro](https://nitro.jan.ai) that runs LLMs using [Nvidia's TensorRT-LLM on Windows](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows). 

- Pure C++ inference server on top of TensorRT-LLM's C++ Runtime
- OpenAI-compatible API with `/chat/completion` and `loadmodel` endpoints
- Packageable as a single runnable package (e.g. `nitro.exe`) to run seamlessly on bare metal in Windows
- Can be embedded in Windows Desktop apps

You can try this in [Jan](https://jan.ai) using the TensorRT-LLM Extension. 

> Read more about Nitro at https://nitro.jan.ai/

### Package Contents

Nitro TensorRT-LLM can be compiled into a single Windows executable that runs seamlessly on bare metal.

The Nitro TensorRT-LLM package is approximately ~730mb. Note: this excludes the TensorRT-LLM Engine for the Model. 

| Dependencies                    | Purpose                                                                                    | Size       |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ---------- |
| nitro.exe                       | Nitro                                                                                      | Negligible |
| tensorrt_llm.dll                | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | ~450mb     |
| nvinfer.dll                     | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | ~200mb     |
| nvinfer_plugin_tensorrt_llm.dll | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#tensorrt-llm-repo) | Negligible |
| cudnn_ops_infer64_8.dll         | [cuDNN](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#cudnn)                    | ~80mb      |
| cudnn64_8.dll                   | [cuDNN](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#cudnn)                    | Negligible |
| msmpi.dll                       | [Microsoft MPI](https://github.com/NVIDIA/TensorRT-LLM/tree/main/windows#microsoft-mpi)    | Negligible |
| zlib.dll                        |                                                                                            | Negligible |
| **Total**                       |                                                                                            | **~730mb** |

## Quickstart

### Step 1: Prerequisites

> NOTE: Nvidia Driver >=535 and CUDA Toolkit >=12.2 are prerequisites, and are often pre-installed with Nvidia GPUs 

- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) for your specific GPU >=535 
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) >=12.2


```sh
# Verify Prerequisites
nvidia-smi  # Nvidia Driver
nvcc --version # CUDA toolkit
```

### Step 2: Get a Model's TensorRT Engine

Models in TensorRT-LLM are compiled to [TensorRT-LLM Engines](https://nvidia.github.io/TensorRT-LLM/architecture.html) for your GPU and Operating System.

#### Option 1: Prebuilt TensorRT Engine

| Model          | OS      | GPU Architecture | Download |
| -------------- | ------- | ---------------- | -------- |
| Llamacorn 1.1b | Windows | 3090s (Ampere)   | Download |
| Llamacorn 1.1b | Windows | 4090s (Ada)      | Download |
| OpenHermes 7b  | Windows | 3090s (Ampere)   | Download |
| OpenHermes 7b  | Windows | 4090s (Ada)      | Download |

#### Option 2: Build a TensorRT Engine from model

You can also build the TensorRT Engine directly on your machine, using your preferred model.

- This process can take upwards of 1 hour. 
- See [Building a TensorRT-LLM Engine](#building-a-tensorrt-llm-engine) instructions below. 

### Step 3: Run Nitro TensorRT-LLM for Windows

```bash title="Run Nitro server"
# Go to folder with `nitro.exe`
.\nitro.exe [thread_num] [host] [port] [uploads_folder_path]
.\nitro.exe 1 http://0.0.0.0 3928
```

### Step 4: Load Model's TensorRT-LLM Engine

```bash title="Load model"
# Powershell
Invoke-WebRequest -Uri "http://localhost:3928/inferences/tensorrtllm/loadmodel" `
    -Method Post `
    -ContentType "application/json" `
    -Body "{ `
        `"engine_path`": `"./openhermes-7b`", `
        `"ctx_len`": 512, `
        `"ngl`": 100 `
     }"

# WSL
curl --location 'http://localhost:3928/inferences/tensorrtllm/loadmodel' \
--header 'Content-Type: application/json' \
--data '{
    "engine_path": "./llamacorn-1.1b",
    "ctx_len": 512,
    "ngl": 100
  }'
```

#### Parameters

| Parameter     | Type    | Description                               |
| ------------- | ------- | ----------------------------------------- |
| `engine_path` | String  | The file path to the TensorRT-LLM engine. |
| `ctx_len`     | Integer | The context length for engine operations. |
| `ngl`         | Integer | The number of GPU layers to use.          |

### Step 5: Making an Inference Request

Nitro TensorRT-LLM offers a drop-in replacement for OpenAI's' `/chat/completions`, including streaming responses. 

> Note: `model` field is a placeholder for OpenAI compatibility. It is not used as Nitro TensorRT-LLM currently only loads 1 model at a time

```bash title="Nitro TensorRT-LLM Inference"
# Powershell
$url = "http://localhost:3928/v1/chat/completions"
$headers = @{
    "Content-Type" = "application/json"
    "Accept" = "text/event-stream"
    "Access-Control-Allow-Origin" = "*"
}

$body = @{
    "messages" = @(
        @{
            "content" = "Hello there 👋"
            "role" = "assistant"
        },
        @{
            "content" = "Write a long story about NVIDIA!!!!"
            "role" = "user"
        }
    )
    "stream" = $true
    "model" = "operhermes-mistral"
    "max_tokens" = 2048
} | ConvertTo-Json

Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body -UseBasicParsing -TimeoutSec 0

# WSL
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
## Contributing

### Repo Structure

This repo is a fork of [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), with the intent of keeping pace with TensorRT-LLM developments.

The actual Nitro code is in a subfolder, which is then used in the Build process. 

```
+-- cpp
|   +-- tensorrt_llm
|   |   +-- nitro
|   |   |   +-- nitro_deps
|   |   |   +-- main.cc
|   |   |   +-- ...
|   |   +-- CMakeLists.txt
```
## Building a TensorRT-LLM Engine

- [ ] TODO

## Contact

- For support, please file a GitHub ticket.
- For questions, join our Discord [here](https://discord.gg/FTk2MvZwJH).
- For long-form inquiries, please email hello@jan.ai.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=janhq/tensorrt-llm-nitro&type=Date)](https://star-history.com/#janhq/tensorrt-llm-nitro&Date)
