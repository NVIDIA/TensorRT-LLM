<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Inkling-2-Small on TensorRT LLM - Blackwell

## Introduction

This deployment guide provides step-by-step instructions for running Thinking Machines' Inkling-2-Small model with TensorRT LLM on NVIDIA GB300 GPUs. It covers the validated deployment on one 4-GPU GB300 tray: preparing the configuration, launching the server, and checking reasoning and tool-call output.

Inkling-2-Small is a sparse Mixture-of-Experts (MoE) model with about 594B total and 17B active parameters. Its text decoder has 48 layers with hidden size 4,608. Forty sliding-window layers attend the last 1,024 tokens, and eight global layers (5, 11, ..., 47) attend the whole prefix. Attention has 48 query and 8 KV heads of dimension 128 and no rotary embeddings: position enters through a learned relative bias over distances below 1,024, and each token and head also has a sink logit and an output gate. Short causal convolutions run on the keys, the values and each sublayer's output, and every six layers the residual stream closes a block that later layers mix back in. Each MoE layer routes every token to 8 of 384 experts. The released checkpoint stores the experts of layers 0-39 in NVFP4, with per-token NVFP4 activations, and those of layers 40-47 in MXFP8; the rest of the model is BF16. The checkpoint also carries a vision tower and a DSpark draft model, which this deployment does not use.

TensorRT LLM runs the model (`model_type: inkling2`, architecture `Inkling2ForConditionalGeneration`) as a Modeling V2 target, `inkling_2_small_nvfp4__sm_103__tep4`: attention tensor-parallel and the experts expert-parallel across four GB300 GPUs, with a BF16 KV cache.

## Prerequisites

* GPU: 4x NVIDIA GB300 GPUs (`SM103`) on one node. This guide was validated on a GB300 NVL tray.
* OS: Linux
* Drivers: CUDA Driver 580 or later
* Container runtime with NVIDIA GPU support
* About 400 GB of disk or shared-filesystem space for the checkpoint

## Models

* [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Apache-2.0, 393.5 GB). The checkpoint ships pre-quantized; no quantization step is needed.

The examples below serve the model by its Hub id. Replace it with a local checkpoint directory if you have downloaded one.

## Feature Support Notes

* **The validated deployment is attention TP4 with MoE EP4 on one GB300 tray** (`tensor_parallel_size: 4`, `moe_expert_parallel_size: 4`), text input only, with a BF16 KV cache, on the default KV cache manager (v2). The forward runs eagerly.
* **TensorRT LLM builds the model only for that deployment.** The settings below are required: `cuda_graph_config: null`, `disable_overlap_scheduler: true`, `enable_chunked_prefill: false`, `kv_cache_config.enable_block_reuse: false`, and an explicit `max_seq_len` of at most 262,144. Any other deployment fails at startup; see [Troubleshooting Tips](#troubleshooting-tips) for how to read the reason.
* **Not supported yet:** CUDA graphs, the overlap scheduler, chunked prefill, KV-cache block reuse, speculative decoding (DSpark), disaggregated serving, image input, GPUs other than GB300, and the MXFP8 variant of the checkpoint.
* **Without chunked prefill, a prompt must fit in `max_num_tokens`.** The configuration below sets `max_num_tokens` equal to `max_seq_len`.
* **Reasoning and tool calling are supported** with the `inkling` reasoning and tool parsers.
* **Performance:** this deployment is the functional baseline. Eager decode is host-bound; CUDA graphs and the overlap scheduler come next.

## Deployment Steps

### Run Docker Container

Run the docker container using the TensorRT LLM NVIDIA NGC image. The image is multi-arch (x86_64 and aarch64), so it runs on GB300 trays directly:

```shell
docker run --rm -it \
--ipc=host \
--gpus all \
-p 8000:8000 \
-v ~/.cache:/root/.cache:rw \
--name tensorrt_llm \
nvcr.io/nvidia/tensorrt-llm/release:x.y.z \
/bin/bash
```

Note:

* The command mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. If the `~/.cache` directory doesn't exist please create it using `$ mkdir ~/.cache`.
* You can mount additional directories and paths using the `-v <host_path>:<container_path>` flag if needed, such as mounting the downloaded weight paths.
* The command also maps port `8000` from the container to your host so you can access the LLM API endpoint from your host.
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use a release that includes Inkling 2 support; the `rc` releases have the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT LLM, the steps refer to [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html)

> **All commands below should be run inside the Docker container.**

### Configuration

Create a YAML configuration file with the validated settings:

```shell
EXTRA_LLM_API_FILE=/path/to/inkling2.yaml

cat << EOF > ${EXTRA_LLM_API_FILE}
tensor_parallel_size: 4
moe_expert_parallel_size: 4
max_seq_len: 32768
max_num_tokens: 32768
cuda_graph_config: null
disable_overlap_scheduler: true
enable_chunked_prefill: false
kv_cache_config:
  enable_block_reuse: false
EOF
```

### Launch the TensorRT LLM Server

Inkling-2-Small at TP4/EP4 fits on a single 4-GPU node, so `trtllm-serve` can be launched directly:

```bash
trtllm-serve thinkingmachines/Inkling-2-Small \
    --host 0.0.0.0 --port 8000 \
    --config ${EXTRA_LLM_API_FILE} \
    --reasoning_parser inkling \
    --tool_parser inkling
```

The `--reasoning_parser` and `--tool_parser` flags return the model's reasoning in `reasoning_content` and its tool calls in `tool_calls` on `/v1/chat/completions`. `auto` also selects `inkling` for both when the model argument is a local checkpoint directory, because the resolution reads the checkpoint's `config.json`; when serving by Hub id, pass `inkling` explicitly.

Each GPU reads only its own experts' weights from the checkpoint; in our runs the server was ready about 3.5 minutes after launch. During startup the log names the Modeling V2 target being built:

```text
[TRT-LLM] [W] [_torch][RANK 0] modeling_v2: building tensorrt_llm._torch.modeling_v2.models.inkling2.inkling_2_small_nvfp4__sm_103__tep4.modeling.ModelingV2Inkling2SmallNvfp4Sm103Tep4 on all 4 rank(s)
```

The server is ready when the log shows:

```text
INFO:     Application startup complete.
```

### Key YAML Options

These options are set within the YAML file passed to `trtllm-serve` via the `--config` argument; see the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `tensor_parallel_size` / `moe_expert_parallel_size`

Both must be 4. Each GPU runs 12 query and 2 KV heads of every attention layer and 96 of the 384 experts of every MoE layer.

#### `max_seq_len` / `max_num_tokens`

`max_seq_len` must be set explicitly, at most 262,144; it bounds the prompt plus the generated tokens, including reasoning. Chunked prefill is not available, so `max_num_tokens` must cover the longest prompt you serve. A longer prompt is rejected with `The prompt length (...) should not exceed max_num_tokens (...)`.

#### `cuda_graph_config` / `disable_overlap_scheduler` / `enable_chunked_prefill`

Required as shown: the forward runs eagerly, without CUDA graphs, the overlap scheduler or chunked prefill.

#### `kv_cache_config`

`enable_block_reuse: false` is required: besides the KV cache, each request carries the windows of its short convolutions, which block reuse does not restore. Lower `free_gpu_memory_fraction` (default 0.9) if other processes share the GPUs.

### LLM API

The same deployment through the Python LLM API:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def main():
    llm = LLM(
        "thinkingmachines/Inkling-2-Small",
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        max_seq_len=32768,
        max_num_tokens=32768,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
    )
    for output in llm.generate(["The capital of France is"], SamplingParams(max_tokens=16)):
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
```

## Testing API Endpoint

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries.

### Basic Test

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "thinkingmachines/Inkling-2-Small",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "max_tokens": 3072
}' -w "\n"
```

The answer is in `choices[0].message.content` and the model's reasoning in `choices[0].message.reasoning_content`. A response from our validation run, abridged:

```json
{
  "object": "chat.completion",
  "model": "thinkingmachines/Inkling-2-Small",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "reasoning_content": "The user asks a simple factual question and wants one sentence. Need answer exactly one sentence. Capital of France is Paris. ...",
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 27, "completion_tokens": 66, "total_tokens": 93}
}
```

With `"stream": true`, the reasoning arrives first, in `delta.reasoning_content`, followed by the answer in `delta.content`.

### Reasoning Effort

The chat template inserts a thinking-effort level into every conversation. It takes `reasoning_effort` as a number from 0 to 0.99 or as a name: `none` (0), `minimal` (0.1), `low` (0.2), `medium` (0.7), `high` (0.9), `xhigh` or `max` (0.99). The default is 0.9. Reasoning tokens count toward `max_tokens`, so at a high effort a small `max_tokens` can be spent on reasoning and leave `content` empty, with `finish_reason` `length`.

Pass the level through `chat_template_kwargs`:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "thinkingmachines/Inkling-2-Small",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "max_tokens": 1024,
    "chat_template_kwargs": {"reasoning_effort": "none"}
}' -w "\n"
```

At `none`, the model answers without reasoning, and `reasoning_content` is empty. The OpenAI API's request-level `reasoning_effort` field is accepted but does not reach this model's chat template; use `chat_template_kwargs`.

### Tool Calling Test

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "thinkingmachines/Inkling-2-Small",
    "messages": [{"role": "user", "content": "What'\''s the weather in Paris right now?"}],
    "max_tokens": 3072,
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, for example Paris"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }]
}' -w "\n"
```

The tool call is returned in `choices[0].message.tool_calls`, with `finish_reason` `tool_calls`. From our validation run, abridged:

```json
"message": {
  "role": "assistant",
  "content": "",
  "reasoning_content": "The user asks current weather in Paris. ...",
  "tool_calls": [
    {
      "id": "chatcmpl-tool-9892aae685384e13b6063a707c4f7e7d",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"unit\": \"celsius\"}"}
    }
  ]
},
"finish_reason": "tool_calls"
```

Streaming returns the call in `delta.tool_calls`. To continue, append the assistant message with its `tool_calls` as returned, and the result as a `tool` message with the call's `tool_call_id`, then send the conversation again with the same `tools`:

```json
{"role": "tool", "tool_call_id": "chatcmpl-tool-9892aae685384e13b6063a707c4f7e7d",
 "content": "{\"city\": \"Paris\", \"temperature\": 18, \"unit\": \"celsius\", \"condition\": \"sunny\"}"}
```

The model then answers from the result; in our run: `Paris is currently sunny and 18°C.`

## Accuracy

Measured on one GB300 tray with this target (TP4/EP4, eager), against the model's SGLang implementation (`lmsysorg/sglang:dev-inkling-2-small`, TP4/EP4) with the same client:

| | SGLang | TensorRT LLM |
|---|---|---|
| GSM8K, InferenceX protocol (chat template, strict-match) | 97.50 ± 0.43 | 97.73 ± 0.41 |
| GSM8K, 8-shot raw completions (strict-match) | 93.71 ± 0.67 | 92.95 ± 0.71 |
| Long-context retrieval, 5 prompts of 2K to 17.8K tokens | 5/5 | 5/5 |

The InferenceX protocol sends each problem through the chat template at the default reasoning effort, with a budget of 12,288 tokens.

## Troubleshooting Tips

* **Startup fails with `Unknown architecture for AutoModelForCausalLM: Inkling2ForConditionalGeneration`:** the deployment differs from the validated one, so the Modeling V2 target declined it, and TensorRT LLM has no other implementation of this model. An earlier line of the server log names the unmet conditions; without `cuda_graph_config: null`, for example:

  ```text
  modeling_v2: no target fits 'Inkling2ForConditionalGeneration' (the inkling2 target is Inkling-2-Small on GB300 with attention TP4 and MoE EP4, text only and eager; this deployment lacks: no CUDA graphs (the forward runs eagerly) (...)); using the built-in implementation
  ```

  With `modeling_v2: require` in the YAML file, the startup error itself states the unmet conditions: `ValueError: modeling_v2='require', but no target fits 'Inkling2ForConditionalGeneration': ...`.
* **A request fails with `The prompt length (...) should not exceed max_num_tokens (...)`:** raise `max_num_tokens`; chunked prefill is not available.
* **`content` is empty and `finish_reason` is `length`:** the reasoning used up `max_tokens`. Raise `max_tokens` or lower the reasoning effort.
* **Reasoning effort has no effect:** pass it as `chat_template_kwargs.reasoning_effort`. The request-level `reasoning_effort` field is accepted but does not reach this model's chat template.
* **A request with an image fails with `Unknown modality: image` (HTTP 400):** this deployment takes text input only.
* **Tool calls or reasoning content not parsed:** pass `--tool_parser inkling` and `--reasoning_parser inkling` on the `trtllm-serve` command line; `auto` resolves to `inkling` only for a local checkpoint directory.
* **CUDA out-of-memory errors:** reduce `kv_cache_config.free_gpu_memory_fraction`, `max_batch_size`, or `max_seq_len`.
