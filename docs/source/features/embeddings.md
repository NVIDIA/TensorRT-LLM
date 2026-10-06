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

## Encoder CUDA graphs

Encoder CUDA graphs capture the encoder forward pass once per input shape at startup
and replay it for each batch, removing the per-kernel launch overhead of eager
execution. The gain is largest for small encoders and small batches, where launching
kernels, not running them, dominates the latency.

Enable them with an `EncodeCudaGraphConfig` on an `encode_only=True` LLM:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EncodeCudaGraphConfig

llm = LLM(
    model="<model>",
    encode_only=True,
    cuda_graph_config=EncodeCudaGraphConfig(
        batch_sizes=[1, 2, 4, 8],
        num_tokens=[128, 256, 512, 1024],
        seq_lens=[64, 128],
        enable_padding=True,
    ),
)
outputs = llm.encode(["hello world", "foo bar"])
```

For `trtllm-serve embeddings`, put the same settings in the `--config` YAML:

```yaml
cuda_graph_config:
  batch_sizes: [1, 2, 4, 8]
  num_tokens: [128, 256, 512, 1024]
  seq_lens: [64, 128]
  enable_padding: true
```

| Field | Meaning |
|---|---|
| `batch_sizes` | Number of requests per batch to capture graphs for. |
| `num_tokens` | Total token counts per batch (all requests' tokens, packed) to capture graphs for. |
| `seq_lens` | Longest-request lengths to capture graphs for. |
| `enable_padding` | Round each batch up to the nearest captured shape. Recommended; without it only exact matches use a graph. |
| `max_num_token` / `max_seq_len` | Alternative to listing `num_tokens` / `seq_lens`: the buckets are generated up to this value. |

A graph is captured at startup for every feasible combination of these buckets. At
runtime a batch uses the graph for its (batch size, total tokens, longest request)
shape, after rounding up if `enable_padding` is set. A batch with no matching graph,
such as one larger than the largest bucket, runs eagerly: the result is the same, just
not accelerated. More buckets cover more traffic at the cost of startup time and graph
memory.

Encoder CUDA graphs require the `TRTLLM` attention backend; with other backends every
batch runs eagerly. Multi-item scoring batches also run eagerly. For encoder-decoder
models, see `encoder_cuda_graph_config` in
[Use encoder-decoder models with the PyTorch backend](../models/encoder-decoder.md).

### Extra model inputs

`llm.encode()` passes extra keyword arguments through to the model's `forward()`, for
example `token_type_ids` for BERT. With encoder CUDA graphs, declare each tensor
argument in `extra_model_inputs` so the graph is captured with it:

```python
import torch
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EncodeCudaGraphConfig, EncodeExtraInputSpec

llm = LLM(
    model="<bert_model>",
    encode_only=True,
    cuda_graph_config=EncodeCudaGraphConfig(
        batch_sizes=[1, 2, 4, 8],
        num_tokens=[128, 256, 512, 1024],
        seq_lens=[64, 128],
        enable_padding=True,
        extra_model_inputs=[
            EncodeExtraInputSpec(name="token_type_ids", shape=("num_tokens",), dtype="int32"),
        ],
    ),
)

# Sentence pairs: the tokenizer marks the second sentence of each pair as segment 1.
pairs = [("What is TensorRT-LLM?", "An inference library."),
         ("Is it fast?", "Yes.")]
tokenizer = AutoTokenizer.from_pretrained("<bert_model>")
encoded = tokenizer([q for q, _ in pairs], [a for _, a in pairs])

prompts = [{"prompt_token_ids": ids} for ids in encoded["input_ids"]]
# Packed: every prompt's values concatenated in prompt order, no padding.
token_type_ids = torch.tensor(
    [t for row in encoded["token_type_ids"] for t in row], dtype=torch.int32)

outputs = llm.encode(prompts, token_type_ids=token_type_ids)
```

Each `EncodeExtraInputSpec` has a `name` (the `forward()` argument), a `dtype`, and a
`shape` with exactly one symbolic dimension; the other dimensions are fixed integers:

| Symbolic dimension | Size at call time | Example |
|---|---|---|
| `"num_tokens"` | Total tokens in the batch, packed in prompt order | `token_type_ids`: `shape=("num_tokens",)` |
| `"batch_size"` | Number of prompts, one row each | per-request features: `shape=("batch_size", 40)` |

Rules:

- Every tensor argument passed to `llm.encode()` must be declared; an undeclared tensor
  raises `ValueError`. A non-tensor argument is allowed but runs that call eagerly.
- Every declared input must be passed on every call, with the declared dtype and shape.
  To omit one, pass zeros.
- These checks apply only to calls that can use a graph. Multi-item scoring batches
  and non-`TRTLLM` attention backends always run eagerly, so their arguments pass
  through unchecked.
- Pass tensors where they already live, on the host or on the device.
- `input_ids`, `position_ids`, `seq_lens`, `multi_item_part_lens`, `attn_metadata` and
  `return_context_logits` are managed by the runtime and cannot be declared.
- Extra model inputs are available through the Python `llm.encode()` API only.
  `trtllm-serve embeddings` does not support them yet and rejects a `--config` that
  declares `extra_model_inputs` at startup. They are also not supported for
  encoder-decoder models or encoders that take fixed-shape feature tensors.
