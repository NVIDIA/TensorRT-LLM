<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Beam on TensorRT LLM

This guide serves the complete 52-layer Beam model with NVFP4 weights on **four
GB300 GPUs on one node**, using TensorRT LLM's experimental StairCase/ModelingV2
implementation. It provides streaming and non-streaming text chat completions
through the native OpenAI-compatible server.

This is a functional deployment baseline. The expert GEMMs use native NVFP4
kernels, but expert dispatch currently uses a host-driven loop; this recipe is
not a throughput-optimized configuration.

## Prerequisites

- Four NVIDIA GB300 GPUs (SM103), visible to the same Linux environment, with
  working GPU peer communication. The full-model target requires TP4; single-GPU
  execution and other parallel layouts are not supported by this recipe.
- A TensorRT LLM installation containing the Beam ModelingV2 target and matching
  native libraries. Follow the [source build instructions](../installation/build-from-source.md)
  for the checkout containing this guide. An older release image may not contain
  this experimental target.
- An authorized, compatible Beam NVFP4 checkpoint directory, accessible to every
  rank, containing `config.json`, the tokenizer files, `chat_template.jinja`,
  `model.safetensors.index.json`, and all indexed weight shards. Preserve the
  checkpoint's quantization metadata and scales. The launcher does not download
  or convert weights.

## Deployment configuration

| Setting | Configuration |
|---|---|
| Model | All 52 layers, with checkpoint-defined sliding-window and full attention |
| Parallelism | Attention TP4, MoE TP4/EP1, PP1; attention data parallelism disabled |
| Precision | Packed NVFP4 experts with native W4A4 GEMMs; BF16 excluded modules and activations; FP32 router |
| Attention backend | TRTLLM |
| KV cache | BF16, `KVCacheManagerV2`, full-block prefix reuse enabled, partial-block reuse disabled |
| Sampling | Native `TorchSampler`; greedy decoding is the evaluated policy |
| Execution | Eager; CUDA graphs, overlap scheduling, and chunked prefill disabled |
| Server | Native OpenAI-compatible server, one frontend, text-only |

The Beam launcher configures the checkpoint loader, tokenizer, reasoning parser,
ModelingV2 routing, and native server together. Use this launcher for the recipe;
plain `trtllm-serve` with default arguments does not establish the same setup.
There is no separate serving YAML required for this launch.

## Launch the server

Run one launcher process with access to all four GPUs; it starts the TP4 workers.
Replace `/path/to/checkpoint` with your checkpoint directory.

```bash
python -m tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.serving \
  --checkpoint /path/to/checkpoint \
  --served-model-name beam-nvfp4 \
  --host 127.0.0.1 --port 8000 \
  --max-seq-len 69632 \
  --max-num-tokens 8192 \
  --max-batch-size 1
```

The server uses the checkpoint's `chat_template.jinja` by default. Use
`--chat-template /path/to/chat_template.jinja` only when supplying a compatible
template explicitly. The example binds to loopback; choose a reachable bind
address when deploying behind your service's access controls.

### Capacity and batching

- `--max-seq-len` bounds prompt plus generated tokens. The configured capacity
  is not a claim of evaluated accuracy at that context length.
- `--max-num-tokens` bounds tokens processed in an engine step. With chunked
  prefill disabled, the prompt must fit this budget. This target accepts at most
  8,192 new token rows per model call.
- `--max-batch-size 1` is the conservative starting configuration. Increasing it
  enables native scheduler batching and requires a corresponding KV memory budget.
- `--kv-cache-max-tokens` sets the shared cache capacity; by default it is twice
  `--max-seq-len`. Capacity must account for all concurrently active requests.
  The GSM8K evaluation used a batch limit of 128 and a cache token budget of
  1,048,576, which are workload-specific settings, not general capacity guarantees.
- `--iter-stats` enables native iteration statistics exposed through `/metrics`.

## Send a chat request

After the server is ready:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beam-nvfp4",
    "messages": [{"role": "user", "content": "What is 12 plus 7?"}],
    "temperature": 0,
    "max_tokens": 4096,
    "stream": false,
    "chat_template_kwargs": {"reasoning_effort": 128}
  }'
```

The response's `choices[0].message.content` contains the final answer;
`choices[0].message.reasoning_content` contains analysis-channel text. The parser
removes the channel delimiters. Set `"stream": true` and add curl's `-N` option
for streaming; read `content` and `reasoning_content` from each choice's `delta`.

`max_tokens` includes reasoning tokens. If the response reaches its token limit
while reasoning, it may have no final answer; increase the completion allowance
within the configured sequence capacity. The example illustrates the API and
does not guarantee completion within 4,096 tokens.

## Accuracy and limitations

The initial full-model evaluation scored **1,263/1,319 (95.75%) on GSM8K**, using
10-shot prompting, greedy decoding, reasoning effort 128, and a 65,536-token
completion allowance. All questions completed without execution errors or
truncated completions. The recorded comparison is a **95.53% BF16 mean** over
three runs, not a same-precision NVFP4 reference. Differences in precision,
runtime, and batching prevent interpreting the small score increase as a benefit
of quantization.

Greedy outputs can differ across batch sizes and between fresh and reused-cache
execution. The evaluation does not establish exact token parity for these modes.
Long-context accuracy, performance targets, and a checkpoint-backed CI regression
remain separate qualification work.

FP8 KV cache, speculative decoding, CUDA graphs, overlap scheduling, chunked
prefill, tool calling, other GPU architectures/layouts, and disaggregated serving
are not qualified by this recipe. The model's implementation and generated-input
regression coverage are described in the
[Beam source README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/_experimental/modeling_v2/models/beam/README.md).
