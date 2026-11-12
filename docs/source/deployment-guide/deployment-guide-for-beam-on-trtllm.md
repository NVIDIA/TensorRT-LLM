<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Beam on TensorRT LLM

This guide serves the complete 52-layer Beam model with NVFP4 weights through the standard
`trtllm-serve` command, using TensorRT LLM's experimental ModelingV2 implementation. The same
configuration is available to the Python `LLM` API. Two recipes share one configuration:

- **Aggregated**: one node with four GB300 GPUs serves prefill and decode.
- **Disaggregated**: a context (prefill) worker on one four-GB300 node and a generation (decode)
  worker on a second four-GB300 node, behind the native `trtllm-serve disaggregated` router.

Both recipes add the DFlash2 drafter for speculative decoding and store the target's and the
drafter's KV caches as FP8 with an explicit per-layer scale table. Both provide streaming and
non-streaming text chat completions through the native OpenAI-compatible server.

This is a functional deployment baseline, not a throughput-optimized configuration: expert
dispatch currently uses a host-driven loop, and execution is eager.

## Prerequisites

- **GPUs**: four NVIDIA GB300 GPUs (SM103) per node, visible to the same Linux environment, with
  working GPU peer communication. The aggregated recipe uses one node; the disaggregated recipe
  uses two nodes that can reach each other for NIXL KV-cache transfer. Each worker runs attention
  TP4; single-GPU execution and other parallel layouts are not supported.
- **TensorRT LLM**: an installation containing the Beam ModelingV2 target and matching native
  libraries. Follow the [source build instructions](../installation/build-from-source.md) for
  the checkout containing this guide. An older release image may not contain this experimental
  target.
- **Three inputs**, accessible to every rank of every worker. The commands below use these
  placeholders:

  | Placeholder | Input |
  |---|---|
  | `<TARGET_CHECKPOINT>` | An authorized, compatible Beam NVFP4 checkpoint directory containing `config.json`, the tokenizer files (`tokenizer.json`), `chat_template.jinja`, `model.safetensors.index.json` and all indexed weight shards, with the checkpoint's quantization metadata and scales preserved. |
  | `<DRAFTER_CHECKPOINT>` | The matching DFlash2 drafter checkpoint directory (`config.json` and weights) for this target. |
  | `<KV_SCALES>` | The FP8 KV-cache scale table: a JSON file with one scale per logical layer for the target and for the drafter (format below). The checkpoints hold no KV-cache scales and none are assumed. |

  The server does not download or convert weights.

The scale table maps `"target"` (52 layers) and `"draft"` (5 layers) to their per-layer scales.
Each scale is a finite positive value whose FP32 value and reciprocal are normal; it is shared by
K and V (a value `x` is stored as `e4m3(x / s)` and read as `code * s`):

```json
{
  "target": {"layers": 52, "scales": {"0": 1.0, "1": 1.0, "...": 1.0, "51": 1.0}},
  "draft": {"layers": 5, "scales": {"0": 2.0, "1": 2.0, "2": 2.0, "3": 8.0, "4": 4.0}}
}
```

The values above illustrate the format only; use the calibrated table for your checkpoints.

## Deployment configuration

| Setting | Configuration |
|---|---|
| Model | All 52 layers, with checkpoint-defined sliding-window and full attention |
| Parallelism | Attention TP4, MoE TP4/EP1, PP1 per worker; attention data parallelism disabled |
| Precision | Packed NVFP4 experts with native W4A4 GEMMs; BF16 excluded modules and activations; FP32 router |
| Speculative decoding | DFlash2 drafter (`decoding_type: DFlash`), seven proposals per step (eight verification rows), TRTLLM draft attention, strict greedy verification |
| KV cache | One `KVCacheManagerV2` holding the target's 52 and the drafter's 5 KV layers, FP8 E4M3 with the explicit scale table; full-block prefix reuse enabled, partial-block reuse disabled |
| Attention backend | TRTLLM |
| Execution | Eager: CUDA graphs, overlap scheduling and chunked prefill disabled |
| Disaggregated KV transfer | Python cache transceiver over NIXL, 60,000 ms transfer timeout (both workers) |
| Server | Native OpenAI-compatible server, one frontend, text-only |

The recipes under `examples/configs/curated/` hold these settings:

| Recipe | Role |
|---|---|
| [`beam-nvfp4-dflash2-fp8kv-aggregated.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/beam-nvfp4-dflash2-fp8kv-aggregated.yaml) | Aggregated server |
| [`beam-nvfp4-dflash2-fp8kv-disagg-ctx.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/beam-nvfp4-dflash2-fp8kv-disagg-ctx.yaml) | Disaggregated context worker |
| [`beam-nvfp4-dflash2-fp8kv-disagg-gen.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/configs/curated/beam-nvfp4-dflash2-fp8kv-disagg-gen.yaml) | Disaggregated generation worker |

Set `max_num_tokens` to an integer from 1 through 8192; `null` is not supported.
Paged-context attention (cache reuse, chunked prefill, or supported speculative decoding)
requires `kv_cache_config.tokens_per_block: 32`.

Each recipe selects the Beam target and its supporting pieces by name:

- `modeling_v2: require` builds the ModelingV2 Beam target and fails clearly if the configuration
  does not fit it.
- `checkpoint_format: beam_nvfp4` selects the target's checkpoint loader.
- `custom_tokenizer: tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer`
  names the checkpoint's fast tokenizer class, so the server reads `tokenizer.json` directly
  instead of parsing `config.json` as a generic configuration.
- `reasoning_parser: beam_harmony_text` separates the model's analysis channel from its final
  answer. Set it in the recipe; the `--reasoning_parser` command-line flag lists only the
  built-in parsers.
- `speculative_config` adds the DFlash2 drafter. Its KV layers are placed in the target's KV
  cache after the target's layers, so `kv_cache_config.max_tokens` sizes one cache for both.

Settings that equal today's defaults are written out on purpose: they are part of the qualified
configuration. An explicit command-line flag still overrides the same recipe value, so do not
add flags such as `--enable_chunked_prefill` or `--max_batch_size` unless you intend to leave the
qualified configuration.

The FP8 KV cache is an explicit opt-in. Without `kv_cache_config.dtype: fp8` the target keeps its
default BF16 KV cache, which takes no scale table.

### Environment settings

Export these in the shell that starts each `trtllm-serve` worker. The MPI session forwards
`TRTLLM*` variables to every rank of the worker.

| Variable | Value | Purpose |
|---|---|---|
| `TRTLLM_SKIP_KV_CACHE_ESTIMATION` | `1` | Size the KV caches from the recipe's explicit `kv_cache_config.max_tokens` instead of an estimation pass. |
| `TRTLLM_MODELING_V2_KV_CACHE_SCALES` | `<KV_SCALES>` | The FP8 KV-cache scale table (experimental, process-scoped: one engine configuration per server process). Every checkpoint loader a rank builds reads and validates it once. It is required with `kv_cache_config.dtype: fp8` and rejected with a BF16 KV cache; a missing or malformed file, a missing `target` or `draft` section, or ranks holding different tables stop the worker at load time with an error. |
| `TLLM_LOG_LEVEL` | `info` (optional) | Shows the rank logs used in [Verify the deployment](#verify-the-deployment). |

### Capacity and batching

All three recipes use the same limits:

| Limit | Recipe value | Basis |
|---|---|---|
| Concurrent requests per engine step (`max_batch_size`) | 64 | In each serving mode, 64 concurrent GSM8K requests were answered (64/64, finish reason `stop`). |
| Prompt plus generated tokens per request (`max_seq_len`) | 69,632 | The 65,536-token completion allowance plus the rendered prompt. |
| New token rows per engine step (`max_num_tokens`) | 8,192 | The target's maximum per model call. |
| KV-cache capacity in tokens (`kv_cache_config.max_tokens`) | 4,456,448 | 64 × 69,632: every concurrent request can reach `max_seq_len`. |

- `max_seq_len` bounds the prompt plus all generated tokens, including reasoning tokens. A
  configured capacity is not a claim of evaluated accuracy at that length.
- With chunked prefill disabled, a prompt must fit `max_num_tokens` in one step.
- The KV-cache capacity is allocated when a worker starts. With the recipe values each GPU held
  about 103 GB in the 64-request runs. A lower `max_batch_size` is safe with the same capacity;
  a higher one needs a larger capacity and was not qualified.
- The drafter is trained with a larger block than the eight verification rows of this recipe, so
  the server logs a warning recommending a larger `max_draft_len`. Seven proposals is the
  qualified setting; the target accepts no other draft length.

## Aggregated serving

### Launch the server

Run one `trtllm-serve` process with access to all four GPUs of the node; it starts the TP4
workers. Copy the aggregated recipe and replace `<DRAFTER_CHECKPOINT>` in it with the drafter
directory.

```bash
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export TRTLLM_MODELING_V2_KV_CACHE_SCALES=<KV_SCALES>

trtllm-serve <TARGET_CHECKPOINT> \
  --config beam-nvfp4-dflash2-fp8kv-aggregated.yaml \
  --chat_template <TARGET_CHECKPOINT>/chat_template.jinja \
  --served_model_name beam-nvfp4 \
  --host 127.0.0.1 --port 8000
```

The example binds to loopback; choose a reachable bind address when deploying behind your
service's access controls. Loading takes several minutes. The server is ready when
`curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health` returns `200`.

### Send chat requests

Non-streaming:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beam-nvfp4",
    "messages": [{"role": "user", "content": "What is 12 plus 7?"}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"reasoning_effort": 128}
  }'
```

The response's `choices[0].message.content` holds the final answer and
`choices[0].message.reasoning_content` holds the analysis-channel text; the parser removes the
channel delimiters.

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beam-nvfp4",
    "messages": [{"role": "user", "content": "What is 12 plus 7?"}],
    "temperature": 0,
    "max_tokens": 256,
    "stream": true,
    "chat_template_kwargs": {"reasoning_effort": 128}
  }'
```

Each server-sent event carries `choices[0].delta.content` or `choices[0].delta.reasoning_content`;
the stream ends with `data: [DONE]`.

`max_tokens` includes reasoning tokens. A response that reaches its limit while reasoning may
have no final answer (`finish_reason` is `length`); raise `max_tokens` within the configured
sequence capacity.

### Stop the server

Stop the server with Ctrl+C in its terminal, or send SIGTERM to the `trtllm-serve` process. The
TP4 workers exit with it. Confirm that `nvidia-smi` lists no remaining processes on the node's
GPUs before starting another deployment there.

### Python LLM API

The aggregated recipe's settings, passed to `LLM()` in one process on the same node. The
checkpoint's tokenizer applies its chat template, and the registered `beam_harmony_text` parser
splits the reply:

```python
import os

os.environ["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] = "1"
os.environ["TRTLLM_MODELING_V2_KV_CACHE_SCALES"] = "<KV_SCALES>"

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

llm = LLM(
    model="<TARGET_CHECKPOINT>",
    modeling_v2="require",
    checkpoint_format="beam_nvfp4",
    custom_tokenizer="tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer",
    reasoning_parser="beam_harmony_text",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
    attn_backend="TRTLLM",
    disable_overlap_scheduler=True,
    enable_chunked_prefill=False,
    cuda_graph_config=None,
    max_batch_size=64,
    max_seq_len=69632,
    max_num_tokens=8192,
    kv_cache_config=KvCacheConfig(
        enable_block_reuse=True,
        enable_partial_reuse=False,
        use_kv_cache_manager_v2=True,
        max_tokens=4456448,
        dtype="fp8",
    ),
    speculative_config=DFlashDecodingConfig(
        max_draft_len=7, speculative_model="<DRAFTER_CHECKPOINT>", attention_backend="TRTLLM"
    ),
)
messages = [{"role": "user", "content": "What is 12 plus 7?"}]
prompt = llm.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, reasoning_effort=128
)
output = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=256))
reply = ReasoningParserFactory.create_reasoning_parser("beam_harmony_text").parse(
    output.outputs[0].text
)
print(reply.content)  # the final answer
print(reply.reasoning_content)  # the analysis-channel text
llm.shutdown()
```

Set both environment variables before constructing `LLM`, as above, or export them in the shell
that starts Python: the worker processes it starts read them.

## Disaggregated serving

The context worker runs on one node and the generation worker on a second node. Both use the same
checkpoints, scale table and settings; before any KV transfer they check each other's model, cache
format and scale identity, and refuse to transfer between mismatched workers. The context worker
moves each request's KV cache over NIXL to the generation worker. The drafter's prompt context
travels in the same cache, in the drafter's layers, together with the drafter's context position.
These recipes rely on that shared cache, which is the default: do not set
`TRTLLM_DISAGG_SHARED_DRAFT_KV=0`, with which the workers refuse to start.

### Launch the workers

On the context node, with the disaggregated context recipe (replace `<DRAFTER_CHECKPOINT>` in it):

```bash
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export TRTLLM_MODELING_V2_KV_CACHE_SCALES=<KV_SCALES>

trtllm-serve <TARGET_CHECKPOINT> \
  --config beam-nvfp4-dflash2-fp8kv-disagg-ctx.yaml \
  --chat_template <TARGET_CHECKPOINT>/chat_template.jinja \
  --served_model_name beam-nvfp4 \
  --server_role CONTEXT \
  --host 0.0.0.0 --port 8001
```

On the generation node, with the disaggregated generation recipe:

```bash
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export TRTLLM_MODELING_V2_KV_CACHE_SCALES=<KV_SCALES>

trtllm-serve <TARGET_CHECKPOINT> \
  --config beam-nvfp4-dflash2-fp8kv-disagg-gen.yaml \
  --chat_template <TARGET_CHECKPOINT>/chat_template.jinja \
  --served_model_name beam-nvfp4 \
  --server_role GENERATION \
  --host 0.0.0.0 --port 8002
```

Wait until both workers answer `/health` with `200`.

NIXL moves the KV cache with UCX, which must be able to register GPU memory. If your environment
restricts UCX to TCP (`UCX_TLS=tcp`), add the CUDA transports before starting each worker, for
example `export UCX_TLS=tcp,cuda_copy,cuda_ipc`. With TCP alone, every worker rank fails during
startup with `VRAM memory is detected as host by UCX`.

### Launch the router

Write the router configuration, replacing `<CONTEXT_HOST>` and `<GENERATION_HOST>` with the two
nodes' addresses:

```yaml
# disagg-router.yaml
hostname: 0.0.0.0
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "<CONTEXT_HOST>:8001"
generation_servers:
  num_instances: 1
  urls:
    - "<GENERATION_HOST>:8002"
```

Start the router on either node, or on a separate host that reaches both:

```bash
trtllm-serve disaggregated -c disagg-router.yaml --request_timeout 86400
```

`--request_timeout` is the router's limit, in seconds, for one request (default 180). When it
expires the router abandons the generation worker's attempt and retries it; such a retry failed
in testing with `Error in kv cache transfer for generation requests`. Long reasoning answers from
this deployment take longer than the default, so set the limit above the longest request you
serve.

### Send chat requests

Send the same requests as in the aggregated case to the router; the router hands each request's
prefill to the context worker and its decode to the generation worker:

```bash
curl http://<ROUTER_HOST>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beam-nvfp4",
    "messages": [{"role": "user", "content": "What is 12 plus 7?"}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"reasoning_effort": 128}
  }'

curl -N http://<ROUTER_HOST>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beam-nvfp4",
    "messages": [{"role": "user", "content": "What is 12 plus 7?"}],
    "temperature": 0,
    "max_tokens": 256,
    "stream": true,
    "chat_template_kwargs": {"reasoning_effort": 128}
  }'
```

The response fields are the same as in the aggregated case. The router does not serve
`/v1/models`; clients must name the model (`beam-nvfp4`).

### Stop the deployment

Stop the router first, then each worker, with Ctrl+C or SIGTERM to its `trtllm-serve` process.
Confirm with `nvidia-smi` on both nodes that no worker processes remain.

## Verify the deployment

With `TLLM_LOG_LEVEL=info`, each rank of a worker logs the facts that identify this
configuration:

- `[modeling_v2] ModelingV2BeamFullModelNvfp4Sm103Tp4: 52 layers, rank <r> of 4, ... speculative decoding DFlash, KV cache FP8`
- `checkpoint_format beam_nvfp4: BeamNvfp4CheckpointLoader, KV-cache scale table <KV_SCALES> sha256 <digest> (TRTLLM_MODELING_V2_KV_CACHE_SCALES)`
- `[modeling_v2] FP8 KV cache: {...}` with the scale table's digest and the per-layer target and draft scales
- `DFlash: target and draft KV layers share one KV cache manager (57 layers, disagg=<False|True>)`

The server process logs `custom_tokenizer BeamFastTokenizer: ...`. Check that every rank reports
the same scale-table digest, and that the two disaggregated workers report the same digest.

## Serve with NVIDIA Dynamo

The same recipes run behind the standard [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
frontend and TensorRT LLM worker (`python3 -m dynamo.frontend` and `python3 -m dynamo.trtllm`),
aggregated on one node or disaggregated with a prefill worker on one node and a decode worker on
a second node. These steps were tested with the `ai-dynamo` and `ai-dynamo-runtime`
`1.7.0.dev20261101` packages, unmodified; that release pins `tensorrt-llm==1.4.0rc2`, the version
of the TensorRT LLM build containing this guide.

### Install Dynamo next to TensorRT LLM

Install the two Dynamo packages into a virtual environment that sees the TensorRT LLM
installation, without Dynamo's `[trtllm]` extra (which would install a released TensorRT LLM
wheel over this build):

```bash
python3 -m venv --system-site-packages <DYNAMO_VENV>
<DYNAMO_VENV>/bin/pip install --dry-run ai-dynamo==1.7.0.dev20261101 ai-dynamo-runtime==1.7.0.dev20261101
<DYNAMO_VENV>/bin/pip install ai-dynamo==1.7.0.dev20261101 ai-dynamo-runtime==1.7.0.dev20261101
<DYNAMO_VENV>/bin/python -c "import tensorrt_llm, dynamo.trtllm; print(tensorrt_llm.__file__)"
```

The dry run must not plan to replace `torch`, `transformers`, `tensorrt`, `triton` or any `nvidia-*`
package of the environment, and the last command must print the path of this TensorRT LLM
installation. Run the frontend and workers with `<DYNAMO_VENV>/bin/python3`.

### Who processes the text

With Dynamo, the frontend renders the chat template, tokenizes the prompt, detokenizes the output
and separates reasoning from the final answer; the TensorRT LLM worker receives and returns token
IDs only. Configure the frontend's text processing through the worker's registration flags:

- `--custom-jinja-template <TARGET_CHECKPOINT>/chat_template.jinja`: the checkpoint's chat template.
- `--dyn-reasoning-parser beam --dyn-tool-call-parser beam`: Dynamo's built-in Beam parser. Both
  flags are needed: Dynamo selects its Beam parser only when both name `beam`.

Use the curated recipes as the worker's `--extra-engine-args`, with `<DRAFTER_CHECKPOINT>` replaced
and the `reasoning_parser` line removed (reasoning parsing belongs to the frontend here). The commands
below call the three prepared files `beam-dynamo-aggregated.yaml`, `beam-dynamo-disagg-ctx.yaml` and
`beam-dynamo-disagg-gen.yaml`. Export the
same environment variables as for `trtllm-serve` before starting each worker
(`TRTLLM_SKIP_KV_CACHE_ESTIMATION=1`, `TRTLLM_MODELING_V2_KV_CACHE_SCALES=<KV_SCALES>`, and the UCX
transports for disaggregation).

The tested steps use file-based discovery on a directory every process can read and write, and
the TCP request plane:

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=<SHARED_DISCOVERY_DIR> DYN_REQUEST_PLANE=tcp
```

### Aggregated

On one node:

```bash
python3 -m dynamo.frontend --http-host 0.0.0.0 --http-port 8000 \
  --discovery-backend file --request-plane tcp --reasoning-field-name reasoning_content &

python3 -m dynamo.trtllm --model-path <TARGET_CHECKPOINT> --served-model-name beam-nvfp4 \
  --extra-engine-args beam-dynamo-aggregated.yaml --disaggregation-mode agg \
  --dyn-reasoning-parser beam --dyn-tool-call-parser beam \
  --custom-jinja-template <TARGET_CHECKPOINT>/chat_template.jinja
```

### Disaggregated

Start the prefill worker on the first node and the decode worker on the second node, both with
the same `DYN_FILE_KV` directory, and start the frontend on the first node once both workers are
ready. File discovery watches its directory for changes made on its own node only, so a frontend
started earlier does not see the remote decode worker; deployments whose workers restart or join
later need etcd discovery instead. Each worker must advertise an address the frontend can reach
(`DYN_TCP_RPC_HOST`), and the frontend an address the workers can reach for response streams
(`DYN_TCP_RESPONSE_STREAM_HOST`):

```bash
# first node: prefill worker
DYN_TCP_RPC_HOST=<FIRST_NODE_ADDRESS> python3 -m dynamo.trtllm --model-path <TARGET_CHECKPOINT> \
  --served-model-name beam-nvfp4 --extra-engine-args beam-dynamo-disagg-ctx.yaml \
  --disaggregation-mode prefill --dyn-reasoning-parser beam --dyn-tool-call-parser beam \
  --custom-jinja-template <TARGET_CHECKPOINT>/chat_template.jinja &

# second node: decode worker
DYN_TCP_RPC_HOST=<SECOND_NODE_ADDRESS> python3 -m dynamo.trtllm --model-path <TARGET_CHECKPOINT> \
  --served-model-name beam-nvfp4 --extra-engine-args beam-dynamo-disagg-gen.yaml \
  --disaggregation-mode decode --dyn-reasoning-parser beam --dyn-tool-call-parser beam \
  --custom-jinja-template <TARGET_CHECKPOINT>/chat_template.jinja

# first node, after both workers are ready: frontend
DYN_TCP_RESPONSE_STREAM_HOST=<FIRST_NODE_ADDRESS> python3 -m dynamo.frontend \
  --http-host 0.0.0.0 --http-port 8000 --discovery-backend file --request-plane tcp \
  --reasoning-field-name reasoning_content
```

A worker is ready when its status server answers `/health` with `200`. To enable the status server,
set `DYN_SYSTEM_PORT` to a free port for each worker. Ports 8080 and 8081 were already in use on the
tested nodes.

Send the same chat requests as in the aggregated case to the frontend at port 8000. The Dynamo
frontend lists its models at `/v1/models`, and the GSM8K
evaluator below works against it in endpoint mode (`--endpoint http://<FIRST_NODE_ADDRESS>:8000
--served-model-name beam-nvfp4`).

### Differences from the native server

- A reply that starts without a channel header is returned with empty content by Dynamo's Beam
  parser, while the native `beam_harmony_text` parser returns that text as the final content.
  Replies with an analysis or final channel are parsed the same way.
- Dynamo's Beam parser attaches its turn grammar to every request as guided-decoding data, and
  `--dyn-structural-tag false` does not remove it in the tested release. These recipes configure
  no guided-decoding backend, so the TensorRT LLM worker does not apply it.
- A request whose `max_tokens` exceeds the 69,632-token context is accepted and served; the native
  server refuses it with HTTP 400.
- Only the eager recipes (CUDA graphs and overlap scheduling disabled) were tested with Dynamo.

## Accuracy and limitations

The accuracy of this combined configuration (NVFP4 target, DFlash2, FP8 target and draft KV
caches, aggregated and disaggregated) is measured separately and is not established by this
guide. An earlier full GSM8K evaluation of the ordinary configuration (BF16 KV cache, no
speculative decoding) does not qualify this configuration.

- FP8 KV-cache quantization changes attention numerics relative to the BF16 KV cache.
- Speculative decoding commits only tokens that match the target's own greedy choices; the
  drafter changes how many tokens a step commits. Greedy outputs can still differ across batch
  sizes, between fresh and reused-cache execution, between speculative and ordinary decoding,
  and between aggregated and disaggregated serving, because the target's own numerics differ
  across these execution shapes. Exact token parity across these modes is not established.
- Tool calling, CUDA graphs, overlap scheduling, chunked prefill, attention data parallelism,
  other GPU architectures and other parallel layouts are not qualified by these recipes.

The model's implementation and generated-input regression coverage are described in the
[Beam source README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/_experimental/modeling_v2/models/beam/README.md).

## Evaluate GSM8K accuracy

`examples/llm-api/eval_gsm8k.py` asks the first N questions (1 to 1319) of the pinned GSM8K test
split through `/v1/chat/completions` and scores the final answers. Stage the dataset at revision
`cc7b047b6e5bb11b4f1af84efc572db110a51b3c` as described in
[the evaluator README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/gsm8k_eval/README.md).

Against a running deployment (endpoint mode needs only `requests` and `pyarrow` on the client and
never starts or stops the server). The disaggregated router does not list its models, so name the
model:

```bash
python examples/llm-api/eval_gsm8k.py \
  --endpoint http://<ROUTER_HOST>:8000 --served-model-name beam-nvfp4 \
  --dataset-dir <DATASET> --num-questions 100 --output-dir <RESULTS>
```

With a server that the evaluator launches, owns and stops (managed mode; the aggregated recipe
with `<DRAFTER_CHECKPOINT>` replaced, as for the aggregated launch):

```bash
python examples/llm-api/eval_gsm8k.py \
  --model <TARGET_CHECKPOINT> --config beam-nvfp4-dflash2-fp8kv-aggregated.yaml \
  --chat-template <TARGET_CHECKPOINT>/chat_template.jinja \
  --server-env TRTLLM_SKIP_KV_CACHE_ESTIMATION=1 \
  --server-env TRTLLM_MODELING_V2_KV_CACHE_SCALES=<KV_SCALES> \
  --monitor-gpus 0,1,2,3 \
  --dataset-dir <DATASET> --num-questions 100 --output-dir <RESULTS>
```

Each run writes `results.jsonl` (one record per answered question; rerunning with the same output
directory resumes) and `summary.json` (score, failures, timing and GPU memory).

## First-100 GSM8K regression

The QA/LR functional list selects the first 100 GSM8K questions on one four-GB300 node,
with a 45-minute timeout and a minimum score of 95/100. Execution errors, timeouts,
truncations and parsing failures also fail the test. Two independent runs of the
aggregated recipe on separate nodes scored 97/100 and 98/100 in approximately 25 and
30 minutes, including model startup. This is a separate public
evaluation protocol; its score is not interchangeable with the full evaluation baseline.
Neither this job nor full GSM8K runs in L0. Full GSM8K is excluded from all CI lists.

The CI environment must stage the pinned dataset at `datasets/openai/gsm8k` below
`LLM_MODELS_ROOT` and set `TRTLLM_TEST_TARGET_RELATIVE_PATH`,
`TRTLLM_TEST_DRAFTER_RELATIVE_PATH` and `TRTLLM_TEST_KV_SCALES_RELATIVE_PATH` to the staged
target checkpoint, drafter checkpoint and FP8 KV scale table, respectively. These paths
are relative to `LLM_MODELS_ROOT`; private checkpoint locations are not stored in the repo.
Selecting the test enables regression mode; missing inputs fail rather than silently skip.
The CI service controls the LR/nightly cadence and provisions those inputs.
