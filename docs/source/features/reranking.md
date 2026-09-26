# Reranking (Cross-Encoder Models)

`trtllm-serve` can serve **cross-encoder reranking models** through OpenAI/Cohere-style
**`POST /rerank`**, **`POST /v1/rerank`**, and **`POST /v2/rerank`** endpoints with
**native dynamic batching** — coalescing many independent concurrent (query, document)
pairs into a single forward pass for high throughput, the same way
[embeddings serving](embeddings.md) does.

Currently supported: the **Qwen3-Reranker family** (`Qwen3-Reranker-0.6B`, `-4B`,
`-8B`). These ship as a `Qwen3ForCausalLM` decoder scored as a cross-encoder: the
model judges whether a document is relevant to a query and answers `"yes"` or `"no"`;
the reranking score is `yes_logit - no_logit` at the last token, read directly out of
the model's own `lm_head` (no decode loop, sampling, or detokenization needed).

## Quick start

Launch a rerank server with the `rerank` subcommand:

```bash
trtllm-serve rerank <hf_model_or_path> \
    --max_batch_size 32 \
    --max_queue_delay 0.005 \
    --max_queue_size 2048 \
    --host 0.0.0.0 --port 8000
```

Send a request with `curl` (any of `/rerank`, `/v1/rerank`, `/v2/rerank` works):

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital and most populous city of France.",
      "Berlin is the capital of Germany."
    ]
  }'
```

The response is Cohere's `/v2/rerank` shape:

```json
{
  "id": "rerank-...",
  "model": "<model>",
  "results": [
    {"index": 0, "relevance_score": 8.4},
    {"index": 1, "relevance_score": -6.1}
  ],
  "usage": {"total_tokens": 96}
}
```

`results` is sorted by descending `relevance_score`.

## Request fields

| Field | Type | Notes |
|---|---|---|
| `model` | str | Model name. |
| `query` | str | The search query. |
| `documents` | list[str \| {"text": str}] | Candidate documents, as plain strings or Cohere-style `{"text": ...}` objects. |
| `top_n` | int (optional) | Return only the top-`n` scoring results. Defaults to all documents. |
| `return_documents` | bool (default `false`) | Include each document's text in the response results. |
| `instruction` | str (optional) | TRT-LLM extension. Overrides the Qwen3-Reranker task instruction embedded in the scoring prompt (default: a generic web-search-retrieval instruction). |

## Dynamic batching

Rerank serving reuses the same in-server dynamic batcher as [embeddings
serving](embeddings.md#dynamic-batching), with identical `--max_batch_size` /
`--max_queue_delay` / `--max_queue_size` flags and Triton `dynamic_batching`
migration mapping. Each `(query, document)` pair is tokenized into one prompt and
coalesced into the batch independently — a single request with many `documents`
still benefits from batching alongside other concurrent requests.

## Prompt construction and truncation

Each `(query, document)` pair is scored using the official Qwen3-Reranker prompt: a
fixed system instruction, an `<Instruct>/<Query>/<Document>` user turn, and an
assistant preamble ending right where the model emits its `yes`/`no` answer. Only the
**document** is truncated to fit the model's `max_seq_len`; the query and instruction
are always kept intact, so a truncated request still asks the same question, just
against less of the candidate document. If the instruction and query alone exceed
`max_seq_len`, the request fails with `400` rather than silently dropping part of the
query.

## Error handling

| Condition | HTTP status |
|---|---|
| `documents` is empty | 400 |
| Instruction + query alone exceed `max_seq_len` | 400 |
| Request queue full (`--max_queue_size` reached) | 429 |
| Invalid request body | 400 |

Rerank responses are unary (non-streaming).

Notes:

- The rerank path uses the synchronous `llm.encode()` fast path (`EncoderExecutor`): a
  single forward pass per batch, **no KV cache, sampler, or decode loop**.
- **Single-GPU per server.** Like the `embeddings` command, the encode path runs
  in-process and does not use the multi-GPU worker proxy, so `rerank` does not expose
  tensor/pipeline parallelism flags. To scale out, follow the same [data-parallel
  replica pattern](embeddings.md#scaling-out-across-gpus) as embeddings serving — run
  one single-GPU `trtllm-serve rerank` instance per GPU behind a load balancer.

## Relationship to `llm.encode()`

The server reuses the existing Python `llm.encode()` API (`LLM(..., encode_only=True)`)
under the hood, with the model's `architectures` and the tokenizer-resolved
`reranking_token_true_id`/`reranking_token_false_id` overridden via `model_kwargs`; the
only addition is the async coalescing layer, prompt construction, and the HTTP surface.
