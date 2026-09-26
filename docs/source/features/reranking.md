# Reranking

`trtllm-serve rerank` serves text reranking models through HTTP endpoints with
native dynamic batching. For each request, the server pairs the query with every
candidate document, evaluates the pairs in the encode-only path, and returns
the documents ordered by relevance.

The encode-only path performs one forward pass per batch, with no KV cache,
sampler, or decode loop. Concurrent query-document pairs are coalesced into
larger GPU batches.

## Quick start

Launch a server with a [supported reranker checkpoint](#supported-model-architectures):

```bash
trtllm-serve rerank <hf_model_or_path> \
    --max_batch_size 32 \
    --max_queue_delay 0.005 \
    --max_queue_size 2048 \
    --host 0.0.0.0 --port 8000
```

For example, the Qwen3-Reranker family is supported:

```bash
trtllm-serve rerank Qwen/Qwen3-Reranker-0.6B \
    --max_batch_size 32 \
    --max_queue_delay 0.005 \
    --host 0.0.0.0 --port 8000
```

The server exposes the following routes:

| Route | Compatibility |
|---|---|
| `POST /rerank` | Jina/vLLM-style request and response |
| `POST /v1/rerank` | Alias of `/rerank` |
| `POST /v2/rerank` | Cohere v2-compatible request and response |

Send a request with `curl`:

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "What is the capital of China?",
    "documents": [
      "The capital of China is Beijing.",
      "Bananas grow in tropical climates."
    ],
    "top_n": 1,
    "return_documents": true
  }'
```

Results are ordered by descending relevance score. `index` always refers to
the document's position in the original request.

```json
{
  "id": "rerank-...",
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.98,
      "document": {"text": "The capital of China is Beijing."}
    }
  ],
  "usage": {"prompt_tokens": 91, "total_tokens": 91}
}
```

## Request fields

The two API styles share the query, documents, and result-ordering semantics,
but differ in a few compatibility fields:

| Field | `/rerank` and `/v1/rerank` | `/v2/rerank` |
|---|---|---|
| `model` | Optional string | Required string |
| `query` | Required non-empty string | Required non-empty string |
| `documents` | Non-empty list of strings or `{"text": "..."}` objects | Non-empty list of strings |
| `top_n` | Optional non-negative integer; `0` returns all results | Optional positive integer |
| `return_documents` | Optional boolean; defaults to `false` | Not part of the v2 response |
| `max_tokens_per_doc` | Optional positive integer | Positive integer; defaults to `4096` |
| `priority` | Not supported | Only `0` is currently supported |
| `instruction` | Optional TensorRT-LLM extension | Optional TensorRT-LLM extension |

`max_tokens_per_doc` limits document content, not the complete model input.
The server preserves the query and any model-specific prompt suffix while
truncating a document.

## Dynamic batching

The in-server batcher coalesces query-document pairs from concurrent HTTP
requests before running the encode-only forward pass.

| `trtllm-serve rerank` flag | Behavior |
|---|---|
| `--max_batch_size` | Maximum number of query-document pairs in one forward pass. A full batch is dispatched immediately. |
| `--max_num_tokens` | Maximum total number of input tokens in one batch. |
| `--max_queue_delay` | Maximum time, in seconds, that a pair waits for other work to join its batch. |
| `--max_queue_size` | Maximum number of queued pairs. Additional requests are rejected with HTTP 429. |

A batch is dispatched when it reaches `--max_batch_size`, adding another pair
would exceed `--max_num_tokens`, or the queue-delay window expires. Tune batch
size and delay together for the desired latency-throughput tradeoff.

## Scoring contract

The reranking service expects the model implementation to emit one scalar,
pre-sigmoid relevance logit for each query-document pair. The server applies a
numerically stable sigmoid, sorts the resulting probabilities, and selects
`top_n` when requested.

The HTTP and batching layers are shared infrastructure. Supporting another
reranker family also requires model-specific input formatting and a TensorRT-LLM
architecture that satisfies this scalar-output contract.

## Supported model architectures

The HTTP surface is designed for reranking generally. The model-specific
preprocessing and scoring implementation currently supports the
**Qwen3-Reranker family**:

- `Qwen3-Reranker-0.6B`
- `Qwen3-Reranker-4B`
- `Qwen3-Reranker-8B`

Qwen3-Reranker checkpoints are published as causal language models that score
the tokens `yes` and `no`. TensorRT-LLM selects `Qwen3ForTextReranking` for
these checkpoints and derives a one-row scoring head at load time:

```text
W_score = W_yes - W_no
relevance_logit = hidden_state_last @ W_score
relevance_score = sigmoid(relevance_logit)
```

This is mathematically equivalent to the checkpoint's two-token softmax while
avoiding a projection over the full vocabulary. The Qwen3 input formatter uses
the model's official query-document prompt. The optional `instruction` field
replaces its default web-search instruction.

## Error handling

| Condition | HTTP status |
|---|---|
| Invalid request body or unsupported `priority` | 400 |
| Input longer than the model or batch token limit | 400 |
| Request queue full | 429 |
| Model does not return one scalar per pair | 500 |

Reranking responses are unary and non-streaming.

## Deployment scope

- One reranker model is served by each server instance. Generation, embeddings,
  and reranking modes are not mixed in one instance.
- The reranking command is single-GPU. Tensor, pipeline, and context parallel
  values greater than one in `--config` are rejected at startup.
- A single worker drives the GPU because the underlying encode executor is not
  safe for concurrent calls. Increase throughput with dynamic batching rather
  than additional in-process workers.
- To scale across GPUs, run one server replica per GPU and distribute requests
  across the replicas with a load balancer.

The server uses the same synchronous `llm.encode()` path as other encode-only
models, with an asynchronous batching layer and reranking HTTP APIs added on
top.
