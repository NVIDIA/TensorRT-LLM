<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Mistral Large 4 on TensorRT LLM - Blackwell (GB300)

## Introduction

This guide describes how to serve Mistral Large 4 with TensorRT LLM on NVIDIA GB300 GPUs. It covers obtaining the checkpoint and its format, the supported deployments, building the software, launching `trtllm-serve` on one or two nodes, the model-specific API behavior, example requests, accuracy verification and the known limitations. The configurations below are validated on GB300, most of them by the accuracy tests in `tests/integration/defs/accuracy/modeling_v2/test_mistral_large4.py`; configurations outside these tables are not validated.

Mistral Large 4 is a Mixture-of-Experts model with about 1T parameters, 49B of them active per token (52B including the embedding and output layers). It takes text and image input, supports a context of up to 1,048,576 tokens, and uses latent attention with a sparse top-k indexer. Like Mistral Large 3, it is published on Hugging Face in the Mistral format: `params.json` for the configuration, `tekken.json` for the tokenizer and `consolidated-*.safetensors` for the weights. TensorRT LLM loads its text model as the architecture `MistralLarge4ForCausalLM`; the checkpoint itself names no architecture. The repository also ships Hugging Face tokenizer, chat-template, processor and generation-config files, but no Transformers `config.json`. TensorRT LLM reads the Mistral-format files only; see [Checkpoint Format](#checkpoint-format).

TensorRT LLM serves Mistral Large 4 through **modeling_v2** targets (`tensorrt_llm/_torch/_experimental/modeling_v2/models/mistral_large4/`). A modeling_v2 target is one implementation for one combination of checkpoint, GPU architecture and parallel layout. When the model is loaded, TensorRT LLM picks the target from the checkpoint's quantization, the GPU's compute capability, the parallel mapping and the speculative-decoding configuration. There is no other implementation of this architecture, so a deployment that no target claims fails to load. modeling_v2 is a prototype feature.

## Prerequisites

* GPU: NVIDIA GB300 (`sm_103`). Other GPUs are not supported: no target claims another compute capability.
  * The 4-GPU deployments (TEP4, DEP4 and TP4) run on one node with 4 GPUs.
  * The 8-GPU deployments (TEP8, DEP8, TP8) run on two nodes with 4 GPUs each. Both nodes must be in the same multi-node NVLink domain.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin and PMIx (`srun --mpi=pmix`), or an equivalent MPI launcher, to start one rank per GPU.
* A shared filesystem visible to all ranks for the repository, the checkpoint and the configuration file. The NVFP4 checkpoint, including its speculative-decoding draft, is about 610 GB.
* Container runtime with NVIDIA GPU support, and the TensorRT LLM development container built from `docker/Dockerfile.multi` (base `nvcr.io/nvidia/pytorch:26.09-py3`). Use the SBSA (aarch64) variant on GB300.
* Python packages from `requirements.txt`, including `mistral-common[opencv]` 1.13.0 or later, `xgrammar` and `llguidance`. Install mistral-common from PyPI: the checkpoint's tokenizer declares reasoning efforts that earlier releases do not know. Image input needs OpenCV, which the `opencv` extra installs.

## Models

* NVFP4 model: [mistralai/Mistral-Large-4-1T-A52B-NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4). This is the checkpoint TensorRT LLM serves.
* FP8 model: [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B). Not supported by TensorRT LLM; it is refused at load time.

| Checkpoint | Weights | Features | Deployments |
| :-- | :-- | :-- | :-- |
| NVFP4 | ModelOpt `MIXED_PRECISION`: NVFP4 routed experts (group size 16), FP8 128x128 block-scaled linear layers, BF16 vision encoder; the DSpark draft layers are part of the checkpoint | Text, images, DSpark speculative decoding, contexts up to 1,048,576 tokens | TEP4, DEP4, TP4, TEP8, DEP8, TP8 |

### Download the Checkpoint

Download the NVFP4 repository once to the shared filesystem. It is about 610 GB.

```bash
hf auth login   # if the repository asks for a Hugging Face access token
hf download mistralai/Mistral-Large-4-1T-A52B-NVFP4 --local-dir /path/to/mistral-large-4-nvfp4
```

Pass the downloaded directory to TensorRT LLM as it is; no conversion is needed. The examples below use `/path/to/mistral-large-4-nvfp4`; replace it with your local path. The configurations in this guide are validated with a local directory, not with a Hugging Face model ID.

### Checkpoint Format

The repository contains two sets of files. TensorRT LLM reads only the Mistral-format set:

| Files | Contents | Read by TensorRT LLM |
| :-- | :-- | :-- |
| `params.json`, `consolidated.safetensors.index.json`, `consolidated-*.safetensors` | Mistral format: model configuration and weights, including the DSpark draft | Yes |
| `tekken.json` | Mistral format: tokenizer, chat format and reasoning efforts, read through mistral-common | Yes |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Hugging Face tokenizer and chat template | No |
| `processor_config.json` | Hugging Face image processor (`PixtralProcessor`) | No |
| `generation_config.json` | Hugging Face sampling defaults (temperature 1.0, top_p 0.95) | No |

Every Mistral Large 4 configuration must set `checkpoint_format: mistral_large_4`. With it, the model configuration, the weights, the tokenizer, chat rendering and image preprocessing all come from the Mistral-format files and mistral-common, and the Hugging Face files are ignored. Every component of the server that tokenizes, including the post-processing workers, the multimodal input loader and the disaggregated router, uses `tekken.json`. Use `mistral_large_4`, not the `mistral` format that Mistral Large 3 uses.

Without `checkpoint_format`, TensorRT LLM expects the Hugging Face format, which needs a `config.json`, and fails with an error that names `mistral_large_4`. Mistral Large 4 is not served from the Hugging Face format.

## Supported Configurations

| Deployment | GPUs | Checkpoint | Parallel settings | Validated `max_seq_len` |
| :-- | --: | :-- | :-- | :-- |
| TEP4 | 4 | NVFP4 | `tensor_parallel_size: 4`, `moe_expert_parallel_size: 4`, `enable_attention_dp: false` | 32768, 1048576 |
| DEP4 | 4 | NVFP4 | `tensor_parallel_size: 4`, `moe_expert_parallel_size: 4`, `enable_attention_dp: true` | 32768, 1048576 |
| TP4 | 4 | NVFP4 | `tensor_parallel_size: 4`, `moe_tensor_parallel_size: 4`, `moe_expert_parallel_size: 1`, `enable_attention_dp: false` | 32768, 1048576 |
| TEP8 | 8 (2 x 4) | NVFP4 | `tensor_parallel_size: 8`, `moe_expert_parallel_size: 8`, `enable_attention_dp: false` | 32768, 1048576 |
| DEP8 | 8 (2 x 4) | NVFP4 | `tensor_parallel_size: 8`, `moe_expert_parallel_size: 8`, `enable_attention_dp: true` | 32768, 1048576 (prompts up to 128K tokens) |
| TP8 | 8 (2 x 4) | NVFP4 | `tensor_parallel_size: 8`, `moe_tensor_parallel_size: 8`, `moe_expert_parallel_size: 1`, `enable_attention_dp: false` | 32768, 1048576 |

The 8-GPU deployments use the same targets as their 4-GPU counterparts. The 1M-token configurations of TEP8 and TP8 are validated with prompts of up to 1M tokens; DEP8 is validated with prompts of up to 128K tokens, and DEP4 covers 1M-token prompts under attention data parallelism.

Features of the NVFP4 deployments:

| Feature | Status |
| :-- | :-- |
| DSpark speculative decoding (3 draft tokens) | Supported; the draft is part of the checkpoint. It is the only speculative-decoding method. |
| Image input | Supported. Images are encoded on every rank. |
| KV cache block reuse (prefix caching) | Supported. |
| Chunked prefill | Supported; required when prompts can be longer than `max_num_tokens`. |
| CUDA graphs with padding | Supported. |
| Structured outputs (`response_format`) | Supported with the `xgrammar` and `llguidance` backends. See [`guided_decoding_backend`](#guided_decoding_backend). |
| Tool calling and reasoning output | Supported through the `mistral_large_4` tool and reasoning parsers. |
| Disaggregated serving | Validated for one TEP4 context server and one TEP4 generation server at `max_seq_len` 32768, without speculative decoding. See [Disaggregated Serving](#disaggregated-serving). |
| Pipeline parallelism, speculative-decoding methods other than DSpark | Not supported. |
| LoRA, beam search | Not validated. |

## Deployment Steps

### Build TensorRT LLM from Source

Mistral Large 4 support requires TensorRT LLM built from source. Inside the TensorRT LLM development container, from the repository root:

```bash
python3 scripts/build_wheel.py --use_ccache -a "103-real" --skip_building_wheel --install --yes
```

`-a "103-real"` builds for GB300. `--install` installs TensorRT LLM in editable mode into the virtual environment that `build_wheel.py` creates at the repository root, named after the container's Python version: `.venv-3.12` for the current containers. The launch commands below put its `bin` directory first on `PATH`, so build with the repository at the same path the servers use. With `--build_root`, the virtual environment is created under the build root instead. See [Building from Source](../installation/build-from-source.md) for the other options.

### Configuration File

Create a YAML configuration file on the shared filesystem. The configuration below is the tested TEP4 configuration with DSpark speculative decoding at a 32K-token context:

```shell
EXTRA_LLM_API_FILE=/path/to/mistral-large-4-config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
checkpoint_format: mistral_large_4
modeling_v2: require
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
max_batch_size: 16
max_num_tokens: 32768
max_seq_len: 32768
enable_chunked_prefill: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 16
kv_cache_config:
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.85
speculative_config:
  decoding_type: DSpark
  max_draft_len: 3
  speculative_model: /path/to/mistral-large-4-nvfp4
guided_decoding_backend: xgrammar
EOF
```

Change these keys for the other deployments:

| Deployment | Changes to the configuration above |
| :-- | :-- |
| DEP4 | `enable_attention_dp: true`. `max_batch_size` and `cuda_graph_config.max_batch_size` then apply to each rank. |
| TP4 | `moe_expert_parallel_size: 1` and `moe_tensor_parallel_size: 4` |
| TEP8, DEP8, TP8 | The TEP4, DEP4 or TP4 settings with `tensor_parallel_size: 8`, and `moe_expert_parallel_size: 8` (TEP8, DEP8) or `moe_tensor_parallel_size: 8` (TP8) |
| 1M-token context | `max_seq_len: 1048576` and `enable_chunked_prefill: true` |
| Without speculative decoding | Remove `speculative_config`. |

Always set `max_seq_len`. Without it the engine sizes the KV cache for the checkpoint's 1,048,576-token limit.

### Launch the TensorRT LLM Server

Mistral Large 4 is launched through the `trtllm-llmapi-launch` wrapper, one rank per GPU. Rank 0 serves HTTP and the other ranks become workers. The example below starts the 4-GPU server with Slurm:

```bash
MODEL=/path/to/mistral-large-4-nvfp4
REPO=/path/to/TensorRT-LLM   # in-place build from the previous step

srun -N 1 --ntasks 4 --ntasks-per-node 4 \
    --mpi=pmix --gpus-per-node=4 \
    --container-image=/path/to/tensorrt-llm-container.sqsh \
    --container-mounts=${REPO}:${REPO},${MODEL}:${MODEL},${EXTRA_LLM_API_FILE}:${EXTRA_LLM_API_FILE}:ro \
    bash -c "
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
        exec trtllm-llmapi-launch trtllm-serve ${MODEL} \
            --config ${EXTRA_LLM_API_FILE} \
            --host 0.0.0.0 --port 8000 \
            --tool_parser mistral_large_4 \
            --reasoning_parser mistral_large_4
    "
```

Outside Slurm, start the same command under MPI on one node:

```bash
export PATH="${REPO}/.venv-3.12/bin:$PATH"
mpirun -n 4 trtllm-llmapi-launch trtllm-serve ${MODEL} \
    --config ${EXTRA_LLM_API_FILE} \
    --host 0.0.0.0 --port 8000 \
    --tool_parser mistral_large_4 \
    --reasoning_parser mistral_large_4
```

For the 8-GPU deployments, use an 8-GPU configuration file and start 8 ranks on two nodes, for example `srun -N 2 --ntasks 8 --ntasks-per-node 4 --mpi=pmix --gpus-per-node=4 ...` with the same command. Both nodes must be in one multi-node NVLink domain (on Slurm clusters with topology segments, request both nodes in one segment, e.g. `--segment=2`).

> [!NOTE]
> Pass `--tool_parser mistral_large_4` and `--reasoning_parser mistral_large_4` by name. `auto` cannot detect them, because it reads a Hugging Face `config.json`, which the checkpoint does not have. A command-line option that is also set in the YAML file overrides the YAML value.

Startup is dominated by weight loading and depends mostly on the filesystem. On a networked filesystem, the validated servers became ready in 6 to 13 minutes, and the TP8 deployment in about 23 minutes. The server is ready when the following log is shown:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Quick Start Without a Server (LLM API)

The script below runs the TEP4 configuration through the LLM API. Save it as `quickstart.py`:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, DSparkDecodingConfig,
                                 KvCacheConfig)


def main():
    model = "/path/to/mistral-large-4-nvfp4"
    with LLM(
            model,
            checkpoint_format="mistral_large_4",
            modeling_v2="require",
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=False,
            max_batch_size=16,
            max_num_tokens=32768,
            max_seq_len=32768,
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=16,
                                              enable_padding=True),
            kv_cache_config=KvCacheConfig(enable_block_reuse=True,
                                          free_gpu_memory_fraction=0.85),
            speculative_config=DSparkDecodingConfig(max_draft_len=3,
                                                    speculative_model=model),
    ) as llm:
        outputs = llm.generate(["The capital of France is"],
                               SamplingParams(max_tokens=32, temperature=0))
        print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

Run it with one rank per GPU under `trtllm-llmapi-launch`, as for the server. With Slurm:

```bash
srun -N 1 --ntasks 4 --ntasks-per-node 4 \
    --mpi=pmix --gpus-per-node=4 \
    --container-image=/path/to/tensorrt-llm-container.sqsh \
    --container-mounts=${REPO}:${REPO},${MODEL}:${MODEL} \
    bash -c "
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
        exec trtllm-llmapi-launch python3 ${REPO}/quickstart.py
    "
```

Outside Slurm, `python3 quickstart.py` on a node with 4 GPUs also works: the LLM API then starts its worker ranks itself through MPI. Inside a Slurm job step, MPI cannot start them, so use `trtllm-llmapi-launch`. For the 8-GPU deployments, start 8 ranks on two nodes, as for the server.

### Key YAML Options

These options are set in the YAML file passed to `trtllm-serve --config`. See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `checkpoint_format`

* **Description:** Must be `mistral_large_4`, which reads the checkpoint in the Mistral format and ignores its Hugging Face files (see [Checkpoint Format](#checkpoint-format)). The default, the Hugging Face format, fails with an error that names `mistral_large_4`.

#### `modeling_v2`

* **Description:** Set it to `require`. With the default, `off`, loading fails with `Unknown architecture for AutoModelForCausalLM: MistralLarge4ForCausalLM`. With `require`, a deployment that no target claims fails at load time with a message listing what was checked (compute capability, parallel mapping, speculative decoding and quantization) and which deployments the targets serve. `auto` serves the supported deployments as well, but for a deployment that no target claims it falls back to the default path and fails with the `Unknown architecture` error, which does not say why; use `require`.

#### `tensor_parallel_size`, `moe_expert_parallel_size`, `moe_tensor_parallel_size`, `enable_attention_dp`

* **Description:** Select the deployment, as listed in [Supported Configurations](#supported-configurations). `tensor_parallel_size` must be 4 or 8, and the world size must equal it: pipeline parallelism is not supported.

#### `max_seq_len`, `max_num_tokens`, `enable_chunked_prefill`

* **Description:** The validated values are `max_num_tokens: 32768` with either `max_seq_len: 32768` and chunked prefill off, or `max_seq_len: 1048576` and chunked prefill on.

#### `max_batch_size` and `cuda_graph_config`

* **Description:** The accuracy tests use `max_batch_size: 16` (per rank with attention DP) with CUDA graphs captured up to the same batch size and `enable_padding: true`. Larger batch sizes run, but are not covered by the accuracy tests.

#### `kv_cache_config`

* **Options:**
  * `dtype`: Leave unset. Mistral Large 4 keeps its KV cache in the `fp8_ds_mla` format: an FP8 latent cache with its scales and a BF16 rotary part, plus FP8 indexer keys. `fp8` and `fp8_ds_mla` are also accepted; other values are rejected.
  * `free_gpu_memory_fraction`: `0.85` is the validated value.
  * `enable_block_reuse`: Prefix caching across requests. Validated on and off.
  * `tokens_per_block`: Leave at the default (32).
* The KV cache manager V2 (`use_kv_cache_manager_v2`, on by default) is required.

#### `speculative_config`

* **Description:** DSpark speculative decoding with the draft carried in the NVFP4 checkpoint. `decoding_type` must be `DSpark`, `max_draft_len` must be `3` (the depth the draft was trained with), and `speculative_model`, if set, must be the checkpoint directory itself. Do not set `max_concurrency` or a draft-length schedule.

#### `guided_decoding_backend`

* **Description:** The guided-decoding backend: `xgrammar` or `llguidance`. It constrains structured outputs (`response_format`), and with `llguidance` also `tool_choice: "required"` and named tool calls. Without speculative decoding, Mistral Large 4 defaults to `xgrammar`. **With speculative decoding, guided decoding stays off unless this key is set**, and requests with `response_format` are then not constrained. With `xgrammar`, a JSON response starts at the first generated token, so no reasoning is returned with it; with `llguidance`, the JSON may follow a reasoning section.

### Mistral Large 4 API Behavior

* **Prompt rendering:** Chat messages are rendered by mistral-common from the checkpoint's `tekken.json`, not by a Jinja chat template; the checkpoint's `chat_template.jinja` is not used. A server-level `--chat_template` is ignored with a warning, and a request-level `chat_template` returns HTTP 400. Of `chat_template_kwargs`, only `reasoning_effort` and `continue_final_message` are read; other keys are ignored with a warning. Use the top-level `response_format` for structured outputs.
* **Reasoning:** `reasoning_effort` accepts `low`, `medium` and `high`. Unset, the checkpoint's default, `high`, applies, so the model reasons before it answers. `none` and other values return HTTP 400. The effort can be given at the top level of the request or in `chat_template_kwargs`; when both are given they must agree. With `--reasoning_parser mistral_large_4`, the reasoning text is returned in `reasoning_content` and the answer in `content`. `thinking_token_budget` is not supported.
* **Tool calling:** With `--tool_parser mistral_large_4`, tool calls are returned in `tool_calls`. `tool_choice: "auto"` works with every configuration. `tool_choice: "required"` and a named function need `guided_decoding_backend: llguidance`; the call and its arguments are then constrained to the declared tools. Otherwise these requests return HTTP 400, as they do when combined with `response_format`. Under `auto`, tools with `strict: true` are accepted, but their arguments are not constrained. For `tool_choice: "none"`, see [Known Limitations](#known-limitations).
* **Images:** The deployments accept images as `image_url` content parts, as data URLs (PNG or JPEG; the tested form) or as `http(s)` URLs that the server downloads. URLs that resolve to private addresses are refused unless `TRTLLM_MEDIA_ALLOW_PRIVATE_URLS=1` is set on the server; keep it unset in production. The server sets no limit on the number of images per prompt (prompts with ten 1540x1540 images have been tested), so enforce one in front of the server if you need it. All image tokens count against `max_seq_len`. Video, audio and precomputed image embeddings are not supported.
* **Sampling:** The model card recommends `temperature` 1.0 and `top_p` 0.95 at every reasoning effort, the values in the checkpoint's `generation_config.json`. TensorRT LLM does not apply that file, also not with `generation_config: auto`, so set both in each request. A chat request that sets neither samples with the server defaults, temperature 1.0 and top_p 1.0; an LLM API `SamplingParams` that sets no `temperature`, `top_p` or `top_k` decodes greedily. The examples below use `temperature: 0` for reproducible output.
* **Output length:** On `/v1/chat/completions`, a `max_tokens` or `max_completion_tokens` larger than `max_seq_len` returns HTTP 400. Prompts longer than `max_num_tokens` return HTTP 400 when chunked prefill is off.
* **Log probabilities:** Requests for `logprobs` or `top_logprobs` return HTTP 400 when speculative decoding is enabled.

## Testing API Endpoint

The server runs on the rank-0 node and listens on port `8000`. Send requests to that node's hostname; `localhost` only works on the rank-0 node itself.

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

### Basic Chat

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/mistral-large-4-nvfp4",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 256,
    "temperature": 0
}' -w "\n"
```

The response contains the answer in `choices[0].message.content` and, because the default reasoning effort is `high`, the model's reasoning in `choices[0].message.reasoning_content`. Add `"stream": true` to receive the answer as server-sent events.

### Reasoning

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/mistral-large-4-nvfp4",
    "messages": [{"role": "user", "content": "What is 17 * 23? Give the final number."}],
    "reasoning_effort": "low",
    "max_tokens": 2048,
    "temperature": 0
}' -w "\n"
```

`reasoning_effort` takes `low`, `medium` or `high` (the default). The reasoning is returned in `choices[0].message.reasoning_content` and the answer in `choices[0].message.content`. Leave room in `max_tokens` for the reasoning.

### Tool Calling

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/mistral-large-4-nvfp4",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Paris"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0
}' -w "\n"
```

The call is returned in `choices[0].message.tool_calls`, with `finish_reason` set to `tool_calls`. To continue the conversation, append the assistant message with its `tool_calls` and a `tool` message with the matching `tool_call_id` and the result.

### Structured Output

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/mistral-large-4-nvfp4",
    "messages": [{"role": "user", "content": "Describe a fictional software engineer."}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "languages": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "age", "languages"],
                "additionalProperties": false
            }
        }
    },
    "max_tokens": 256,
    "temperature": 0
}' -w "\n"
```

`choices[0].message.content` holds a JSON object that matches the schema. `{"type": "json_object"}` requests any JSON object. With the `xgrammar` backend of the configuration above, the JSON starts at the first generated token and no reasoning is returned. With `llguidance`, the JSON may follow a reasoning section, which is returned in `reasoning_content` and counts against `max_tokens`; raise `max_tokens` or lower `reasoning_effort` so that the JSON is not cut off.

### Image Input

```shell
IMAGE_B64=$(base64 -w0 /path/to/image.png)

curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/mistral-large-4-nvfp4",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"${IMAGE_B64}"'"}},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }],
    "max_tokens": 256,
    "temperature": 0
}' -w "\n"
```

Several images can be passed in one message; they are seen in the order given.

## Disaggregated Serving

Disaggregated serving is validated with one context server and one generation server, each a TEP4 deployment on its own GB300 node, with the NVFP4 checkpoint without speculative decoding, at `max_seq_len` 32768. The KV cache is transferred with NIXL. Both servers use this configuration:

```yaml
checkpoint_format: mistral_large_4
modeling_v2: require
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
max_batch_size: 16
max_num_tokens: 32768
max_seq_len: 32768
enable_chunked_prefill: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 16
kv_cache_config:
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.85
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
```

Start each server as in [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server), one per node, with `--port 8001`, and then the disaggregated router with the two server addresses:

```yaml
# disagg_config.yaml
hostname: 0.0.0.0
port: 8000
context_servers:
  num_instances: 1
  urls:
    - "<context-node>:8001"
generation_servers:
  num_instances: 1
  urls:
    - "<generation-node>:8001"
```

```bash
trtllm-serve disaggregated -c disagg_config.yaml --server_start_timeout 1800
```

See [Disaggregated Serving](../features/disagg-serving.md) for the general setup. Other layouts, the 1M-token configuration, speculative decoding and image input with disaggregated serving are not validated.

## Running Evaluations to Verify Accuracy (Optional)

The accuracy tests in `tests/integration/defs/accuracy/modeling_v2/test_mistral_large4.py` build the configurations above through the LLM API and compare the scores with the references in `tests/integration/defs/accuracy/references/` (`gsm8k.yaml`, `gpqa_diamond.yaml`, `mmmu.yaml`, `ruler.yaml` and `acceptance_length.yaml`). They read the checkpoint from `${LLM_MODELS_ROOT}/Mistral-Large-4-1T-A52B-NVFP4`, the [downloaded](#download-the-checkpoint) NVFP4 repository, and the datasets from `LLM_MODELS_ROOT`. On a node with 4 GB300 GPUs:

```bash
cd tests/integration/defs
export LLM_MODELS_ROOT=/path/to/models

# GSM8K on the NVFP4 TEP4 target, without speculative decoding.
pytest -v "accuracy/modeling_v2/test_mistral_large4.py::TestModelingV2MistralLarge4Nvfp4Sm103Tep4::test_gsm8k"

# The quick profile on TEP4 with DSpark: GSM8K, RULER, image and acceptance-length checks.
pytest -v "accuracy/modeling_v2/test_mistral_large4.py::TestModelingV2MistralLarge4Nvfp4Accuracy::test_accuracy[quick-tep4]"
```

`test_accuracy` is parameterized by profile and deployment, as `test_accuracy[<profile>-<deployment>]`:

| Profile | Deployments | Contents |
| :-- | :-- | :-- |
| `quick` | `tep4`, `dep4`, `tp4`, `tep8`, `dep8`, `tp8` and their `_1m` variants | GSM8K, a reduced RULER set, a reduced MMMU set, image, structured-output and cache checks, DSpark acceptance length |
| `full` | the same | The quick checks with the full RULER and MMMU sets, more image cases, GPQA Diamond, and GSM8K with speculative decoding off |
| `long_lite`, `long_full` | the `_1m` deployments | RULER at 64K, 128K and 1M tokens, cache and retrieval checks over prompts of up to 1M tokens, then the image checks and GSM8K |

GPQA Diamond and MMMU run with `reasoning_effort: low`.

The 8-GPU cases run on two nodes with 4 GPUs each, under `trtllm-llmapi-launch`. The test lists `tests/integration/test_lists/test-db/l0_gb300_multi_gpus.yml` and `l0_gb300_multi_nodes_node2_gpu8.yml` show the cases that CI runs.

## Known Limitations

* **GB300 only.** No other GPU, including GB200 and B200, is supported.
* **NVFP4 checkpoint only.** Other exports of Mistral Large 4, including the FP8 checkpoint, are refused at load time.
* **Mistral format only.** The checkpoint is read through `checkpoint_format: mistral_large_4`; the Hugging Face format is not supported. The Hugging Face files in the repository are not read, so the sampling defaults in its `generation_config.json` do not apply.
* **Responses API.** Use `/v1/chat/completions`. On this revision `/v1/responses` drops image inputs, and with speculative decoding every `/v1/responses` request returns HTTP 400 because the request asks for log probabilities.
* **Streamed requests that fail validation.** A streamed request that the engine rejects after the response has started (for example, with chunked prefill on, a prompt longer than `max_seq_len`; or, with speculative decoding, a request for log probabilities) receives HTTP 200 and then a closed connection instead of an error status. The same request without streaming returns HTTP 400.
* **`tool_choice: "none"`.** The tools are still shown to the model. If it calls one anyway, the call is dropped and the response has empty content. To get a text answer, send the request without `tools`.
* **`tool_choice: "required"` and named tool calls** need `guided_decoding_backend: llguidance`. Under `auto`, tools with `strict: true` are not grammar-constrained.
* **`reasoning_effort`.** `low`, `medium` and `high` only; reasoning cannot be turned off.
* **OpenAI `store` field.** The server keeps no completions: chat requests with `store: true` return HTTP 400. `store: false` is accepted.
* **Log probabilities** are not available with speculative decoding.
* **Speculative decoding.** Only DSpark with the draft in the NVFP4 checkpoint, with 3 draft tokens. Guided decoding must be enabled explicitly with it (see `guided_decoding_backend` in [Key YAML Options](#key-yaml-options)).
* **Image count.** The server does not limit the number of images per prompt.
* **Long contexts.** A 1M-token request holds about 46 GiB of KV cache on each rank, so the 4-GPU deployments hold only one or two such requests at a time. TEP and TP keep a full copy of the KV cache on each rank; for many concurrent requests, use DEP.
* **Disaggregated serving.** Validated only for TEP4 to TEP4 at a 32K-token context without speculative decoding.
* **Determinism.** Greedy decoding is not bitwise reproducible across runs and batch compositions; scores vary by a few tenths of a point between runs.

## Troubleshooting Tips

| Symptom | Cause and fix |
| :-- | :-- |
| `... is a Mistral-Large-4 checkpoint in the Mistral-native format; load it with checkpoint_format='mistral_large_4'` | Set `checkpoint_format: mistral_large_4`. |
| `Mistral-Large-4 is served from the ModelOpt MIXED_PRECISION export, whose params.json states quantization_config; found quantization=...` | The checkpoint is not the NVFP4 checkpoint. Only the NVFP4 checkpoint is supported. |
| `The tekken.json of ... declares reasoning efforts the installed mistral-common does not know` | Install mistral-common 1.13.0 or later from PyPI. |
| `Unknown architecture for AutoModelForCausalLM: MistralLarge4ForCausalLM` | Set `modeling_v2: require`. |
| `modeling_v2='require', but no target fits 'MistralLarge4ForCausalLM': ...` | The deployment is not supported. The message lists what was checked; compare the GPU, the parallel settings, the speculative-decoding configuration and the checkpoint with [Supported Configurations](#supported-configurations). |
| `Cannot auto-detect tool parser ...` or `Cannot auto-detect reasoning parser ...` | Pass `--tool_parser mistral_large_4` and `--reasoning_parser mistral_large_4` by name. |
| `the Mistral-Large-4 latent cache is fp8_ds_mla ...` | Leave `kv_cache_config.dtype` unset, or set it to `fp8` or `fp8_ds_mla`. |
| `DSA for Mistral-Large-4 requires KV cache manager V2` | Leave `use_kv_cache_manager_v2` unset, or set it to `true`. |
| `Mistral DSpark max_draft_len must equal the trained block depth 3` or `Native Mistral DSpark must use the target checkpoint` | Set `max_draft_len: 3`, and set `speculative_model` to the checkpoint directory or leave it unset. |
| Warning `max_seq_len ... is greater than max_num_tokens ... that can be allocated in kv cache manager, setting max_seq_len to ...` | Set `max_seq_len` explicitly; unset, it defaults to 1,048,576. |
| Image requests fail with an error about OpenCV (`cv2`) | Install `mistral-common[opencv]`, or `opencv-python-headless`. |
| Sampled outputs differ from another serving stack for the same request | The other stack may apply the checkpoint's `generation_config.json` (top_p 0.95); TensorRT LLM does not. Set `temperature` and `top_p` in the request. |
| CUDA out-of-memory errors at startup | Keep `free_gpu_memory_fraction` at 0.85 or lower, and check `max_batch_size` and `max_seq_len`. |
| The 8-GPU server does not start, or ranks cannot reach each other | Check that both nodes are in one multi-node NVLink domain and that the checkout, the checkpoint and the configuration file resolve to the same paths on both nodes. |
