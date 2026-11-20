# Embeddings (Encoder-Only Models)

`trtllm-serve` can serve **encoder-only models** (BERT-style classifiers, reward
models, text-embedding models) through an OpenAI-compatible **`POST /v1/embeddings`**
endpoint with **native dynamic batching** — coalescing many independent concurrent
requests into a single forward pass for high throughput, the way the NVIDIA Triton
Inference Server dynamic batcher does.

This replaces the need to run a separate Triton Inference Server in front of an
encoder model: point your existing OpenAI-style embeddings client at `trtllm-serve`
and it works unchanged.

## Quick start

Launch an embeddings server with the `embeddings` subcommand:

```bash
trtllm-serve embeddings <hf_model_or_path> \
    --max_batch_size 32 \
    --max_queue_delay 0.005 \
    --max_queue_size 2048 \
    --host 0.0.0.0 --port 8000
```

Send a request with any OpenAI-compatible client or `curl`:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "input": ["hello world", "foo bar"]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="tensorrt_llm")
resp = client.embeddings.create(model="<model>", input=["hello world", "foo bar"])
for item in resp.data:
    print(item.index, len(item.embedding))
```

The response is the standard OpenAI embeddings shape:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [ ... ]},
    {"object": "embedding", "index": 1, "embedding": [ ... ]}
  ],
  "model": "<model>",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

## Request fields

The endpoint accepts the standard OpenAI `/v1/embeddings` fields:

| Field | Type | Notes |
|---|---|---|
| `model` | str | Model name. |
| `input` | str \| list[str] \| list[int] \| list[list[int]] | Text(s) or pre-tokenized token-id list(s). |
| `encoding_format` | `"float"` (default) \| `"base64"` | `base64` packs little-endian float32 values. |
| `dimensions` | int (optional) | Matryoshka output size. Only supported by Matryoshka-trained text-embedding models; rejected with `400` otherwise. None of the served models are Matryoshka-trained (BERT classifiers / reward models emit label/score tensors; Qwen3-Embedding emits a fixed-width pooled vector), so this is currently always rejected. |
| `user` | str (optional) | Ignored; accepted for compatibility. |
| `add_special_tokens` | bool (default `true`) | TRT-LLM extension. Encoder models such as BERT generally need their special tokens (e.g. `[CLS]`/`[SEP]`) added during tokenization. |

There are **no required TRT-LLM-specific request fields** — existing OpenAI-compatible
embeddings clients work by pointing at the `trtllm-serve` URL.

## Dynamic batching

A lightweight in-server batcher coalesces concurrent requests in front of the
encoder forward pass. It exposes three knobs that mirror the Triton dynamic batcher:

| `trtllm-serve embeddings` flag | Behavior | Triton equivalent |
|---|---|---|
| `--max_batch_size` | Upper bound on the number of requests fused into one forward pass. A batch reaching this size is dispatched immediately. | maximum / `preferred_batch_size` |
| `--max_queue_delay` (seconds) | Hold window: how long an incoming request waits for others to join its batch before dispatch. | `max_queue_delay_microseconds` |
| `--max_queue_size` | Maximum number of in-flight queued requests. Further requests are rejected with HTTP 429 (backpressure). | `default_queue_policy.max_queue_size` |

A batch is dispatched as soon as **any** of these fires: it reaches `--max_batch_size`,
adding the next request would exceed the engine's `--max_num_tokens` budget, or the
`--max_queue_delay` hold window elapses.

### Migrating from the Triton Inference Server dynamic batcher

If you currently serve an encoder model with the Triton `inflight_batcher_llm` backend
and a `config.pbtxt` `dynamic_batching { ... }` block, map the settings directly:

| Triton `config.pbtxt` | `trtllm-serve embeddings` |
|---|---|
| `dynamic_batching.preferred_batch_size` / model max batch | `--max_batch_size` |
| `dynamic_batching.max_queue_delay_microseconds` | `--max_queue_delay` (in **seconds**, e.g. `100 µs` → `0.0001`) |
| `dynamic_batching.default_queue_policy.max_queue_size` | `--max_queue_size` |

Adopt the same values you tuned in Triton as a starting point, then adjust for your
latency/throughput budget.

## Error handling

| Condition | HTTP status |
|---|---|
| Input longer than `--max_seq_len` | 400 |
| Request queue full (`--max_queue_size` reached) | 429 |
| Invalid request body | 400 |

Embedding responses are unary (non-streaming).

## Output semantics and scope

The endpoint is **model-output-agnostic**: it returns whatever per-request vector the
model emits, serialized into the OpenAI embeddings schema.

- **Classifier / reward models** (e.g. a BERT sequence classifier): the returned
  vector is the model's class-logit / score vector (`[num_labels]`).
- **Text-embedding models** — the **Qwen3-Embedding family** (`Qwen3-Embedding-0.6B`,
  `-4B`, `-8B`) is supported. These ship as a `Qwen3ForCausalLM` decoder plus a
  sentence-transformers pooling pipeline; the embeddings server detects this and serves
  the **L2-normalized last-token hidden state** (a `[hidden_size]` sentence-embedding
  vector — 1024 / 2560 / 4096 respectively), with no extra flags. A configurable pooling
  method (CLS / mean) for other sentence-transformers backbones remains a follow-up.

Notes:

- The embeddings path uses the synchronous `llm.encode()` fast path
  (`EncoderExecutor`): a single forward pass per batch, **no KV cache, sampler, or
  decode loop**.
- One encoder model per server instance. Generation and embedding modes are not mixed
  in one server.
- **Single-GPU per server.** The encode path runs in-process and does not use the
  multi-GPU worker proxy, so the `embeddings` command does not expose tensor/pipeline
  parallelism flags (if a `--config` file sets them, startup fails with a clear error).
  To scale out, see [Scaling out across GPUs](#scaling-out-across-gpus) below.
- A single in-server worker drives the GPU (no `num_workers` knob): the GPU serializes
  forwards and the underlying executor is not safe for concurrent calls. Increase
  throughput with `--max_batch_size` / `--max_queue_delay`, not more workers.

## Scaling out across GPUs

Embedding / encoder-only models are usually small and fit comfortably on a single GPU.
The recommended way to use more GPUs is therefore **data parallelism**: run one
single-GPU `trtllm-serve embeddings` instance per GPU and put a load balancer in front
of them. There is no cross-GPU communication, so throughput scales close to linearly
with the number of replicas.

```bash
# One replica per GPU (8x B200 example), each on its own port.
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i trtllm-serve embeddings <model> --port $((8000 + i)) &
done
# Then point any HTTP load balancer (nginx, k8s Service, etc.) at ports 8000-8007.
```

**Tensor / pipeline parallelism** (sharding a single model across GPUs) is only needed
for an embedding model too large to fit on one GPU — uncommon for encoder-only models.
It is **not yet supported** by the `embeddings` command and is planned as a follow-up.

## Relationship to `llm.encode()`

The server reuses the existing Python `llm.encode()` API
(`LLM(..., encode_only=True)`) under the hood; the only addition is the async
coalescing layer plus the HTTP surface. The synchronous `llm.encode()` API continues
to work unchanged for direct Python callers.
