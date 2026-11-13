<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Muse Spark Open

This example serves Muse Spark Open (816B-A42B) with TensorRT-LLM on a single
4-GPU NVIDIA Blackwell node. The validated aggregated serving configuration
(TP4/EP4) is the curated recipe `examples/configs/curated/musespark-open.yaml`;
this directory adds an overlay that enables DFlash2 speculative decoding;
suffix-automaton (SA) drafting is covered in the speculative-decoding
deployment guide linked below.

For the full walkthrough — model facts, key YAML options, multimodal token
budgets, benchmarking, and troubleshooting — see the
[Muse Spark Open deployment guide](../../docs/source/deployment-guide/deployment-guide-for-musespark-open-on-trtllm.md).

## Hardware support

NVIDIA Blackwell GPUs (`SM100` family). The configuration in this example was
validated on GB300 NVL (`SM103`) with 4 GPUs (TP4/EP4).

## Prerequisites

- A TensorRT-LLM release container that includes Muse Spark Open support, or
  an installation built from source. The deployment guide's
  [Run Docker Container](../../docs/source/deployment-guide/deployment-guide-for-musespark-open-on-trtllm.md#run-docker-container)
  section shows how to run it.
- A complete Hugging Face-format Muse Spark Open checkpoint and tokenizer, e.g.
  `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`
  (`model_type: muse_spark`). The checkpoint ships the routed-expert weights
  pre-quantized to NVFP4; no additional quantization step is needed.

## Serve the model

From the repository root:

```bash
MODEL=/path/to/musespark-open-checkpoint

trtllm-serve ${MODEL} \
    --host 0.0.0.0 --port 8000 \
    --config examples/configs/curated/musespark-open.yaml \
    --tool_parser musespark \
    --reasoning_parser musespark
```

The server is ready when the log shows `Application startup complete`. Then:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/musespark-open-checkpoint",
    "messages": [{"role": "user", "content": "Where is New York?"}],
    "max_tokens": 1024,
    "temperature": 0,
    "reasoning_effort": "low"
}' -w "\n"
```

The checkpoint's chat template defaults to the highest reasoning effort
(`xhigh`), and reasoning tokens count toward `max_tokens`, so a small budget can
be spent entirely on reasoning and leave `content` empty. The request above sets
`reasoning_effort` to `low`; raise the effort (`medium`, `high`, `xhigh`) and
`max_tokens` together for harder prompts. With `--reasoning_parser musespark`,
the reasoning is returned separately in `reasoning_content`.

Guided decoding (JSON mode / JSON schema via the `xgrammar` backend) is
enabled by `guided_decoding_backend: xgrammar` in the recipe and used
through the standard OpenAI `response_format` field.

## Speculative decoding (opt-in)

`spec_dflash.yaml` holds the keys that enable DFlash2 speculative decoding
(a small drafter cross-attending to target hidden states, drafting 7 tokens
per step) with the published drafter checkpoint
[`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2).
Merge the keys into a copy of the recipe and launch with that merged file:

```bash
cat examples/configs/curated/musespark-open.yaml examples/musespark_open/spec_dflash.yaml \
    > musespark-open-dflash.yaml
trtllm-serve ${MODEL} \
    --host 0.0.0.0 --port 8000 \
    --config musespark-open-dflash.yaml \
    --tool_parser musespark \
    --reasoning_parser musespark
```

Acceptance is strict: an accepted draft token is exactly the token the target
model emits at that position. At temperature 0, completions still often differ
from a run with speculation off: draft verification runs the target through a
different attention path, and small numeric differences flip near-tied tokens,
which are common in this model's reasoning text. KV-cache prefix reuse and
batching change temperature-0 completions the same way; at temperature > 0
this disappears into ordinary sampling variation.

Model-free suffix-automaton (SA) drafting is also available; see the
[speculative-decoding guide](../../docs/source/deployment-guide/deployment-guide-for-musespark-open-speculative-decoding-on-trtllm.md)
for its configuration and measured trade-offs.

## Configuration notes

- **FP8 KV cache is required on `SM103` with chunked prefill or block
  reuse.** For head dimension 64 on `SM103`, the fused paged-context FMHA
  kernel exists only for the FP8 KV-cache dtype, and any request entering
  attention with a cached prefix (chunked prefill, block reuse, draft
  verification) uses that path. The recipe sets
  `kv_cache_config.dtype: fp8` accordingly; a 16-bit KV cache in that
  combination is refused.
- **Disaggregated serving:** some disaggregated configurations currently
  require `TLLM_FMHA_LIBS=-fallback` in the server environment (removes the
  fallback attention library from the default selection order).
- **Parsers:** pass `--tool_parser musespark` and
  `--reasoning_parser musespark`. `auto` resolves to `musespark` too, but only
  when the model argument is a local checkpoint directory, because it reads the
  checkpoint's `config.json`.
