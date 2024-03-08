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
- Packageable as a single runnable package (e.g. `nitro.exe`) that can be run seamlessly
- Can be embedded in Windows Desktop apps

You can try this in [Jan](https://jan.ai) using the TensorRT-LLM Extension, with a Nvidia 3090 or 4090. 


> Read more about Nitro at https://nitro.jan.ai/

### Repo Structure

- [ ] Explain that this repo is a fork of TensorRT-LLM repo
- [ ] Explain we are adding the Nitro inference server into it

```
+-- tensorrt_llm
|   +-- nitro
|   |   +-- controllers
|   |   +-- models
|   |   +-- nitro_deps 
|   |   +-- utils
```

### Package Structure

- [ ] Explain that Nitro TensorRT-LLM can be compiled (?) to a single executable
- [ ] Can be packaged with `.dll` dependencies to remove need to for manual install steps

#### Windows

```
+-- nitro.exe
+-- tensorrt-llm.dll
```

## Quickstart

### Step 1: Installation

- [ ] Windows Installation instructions
- [ ] Linux Installation instructions

### Step 2: Downloading an Engine

- [ ] Explain Engines in TensorRT-LLM == Models in GGUF
- [ ] Explain how Engines are OS and GPU-specific
- [ ] Link to precompiled Engines (start with 3090, 4090)

### Step 3: Run Nitro server

- [ ] Explain how user needs to run the ./nitro.exe
- [ ] TODO: Implement thread, host, port, folder path

```bash title="Run Nitro server"
# Go to folder with `nitro.exe`
./nitro
```

### Step 4: Load model

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

### Step 5: Making an Inference

- [ ] Where is the model name defined? (in Engine)
- [ ] Re-emphasize 'Engine' == 'Model' in TensorRT-LLM

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
    "model": "operhermes-mistral",
    "max_tokens": 2048
  }'



```

## Compile from source

## Download

### Contact

- For support, please file a GitHub ticket.
- For questions, join our Discord [here](https://discord.gg/FTk2MvZwJH).
- For long-form inquiries, please email hello@jan.ai.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=janhq/tensorrt-llm-nitro&type=Date)](https://star-history.com/#janhq/tensorrt-llm-nitro&Date)
