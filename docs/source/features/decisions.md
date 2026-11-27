# Decision APIs

The generation service and the PPLX-Decider v1.1 embedding service answer typed questions about shared text. Each instance serves one loaded model and accepts uncompressed JSON. Both endpoints return the loaded model's ID.

## Supported text profile

| Endpoint | Shared input | Question types | Answers |
| --- | --- | --- | --- |
| `/v1/systemone` | Text, JSON object or array | `noul`, `choice`, `score` (1–10 levels) | Keyed by question ID |
| `/v1/decisions` | Text or user messages with text content | `predicate`, `choice`, `score` (2–10 levels) | In request order, 1–64 questions |

## TypeSafe System One

The TypeSafe Python SDK sends `model="jev-latest"` by default. On this endpoint, TRT-LLM accepts that name and `jev-preview` so existing clients can keep their model setting. For example, if this instance serves Qwen, both names use Qwen. You can also pass the served model ID directly.

A `noul` returns the probability of yes. A `choice` returns the winning key and candidate probabilities. A `score` returns the expected zero-based level index, the probabilities and a legend. Choice and score include confidence computed from the candidate distribution. [System One API reference](https://docs.typesafe.ai/api/system-one) · [TypeSafe confidence definitions](https://docs.typesafe.ai/confidence)

```bash
curl -fsS http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": {"headline": "Acme signed an agreement to acquire Beta."},
    "questions": {
      "event": {
        "type": "choice",
        "instructions": "Classify the headline.",
        "criteria": {"acquisition": "One company buys another", "earnings": "Financial results"}
      }
    }
  }'
```

Read the winning label from `answers.event.choice` and its distribution from `answers.event.probabilities`. TypeSafe Python SDK 0.7.5 and later accept `compression=False` to send plain JSON. The 0.5.7 and 0.6.0 clients send plain JSON by default. [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)

## OpenAI Decisions

`POST /v1/decisions` accepts text instructions for each question. User messages can contain strings or `input_text` parts. Answers retain repeated or omitted names. Success and error responses carry a server-generated `x-request-id`. A `predicate` returns the probability of true. A `choice` preserves each string or boolean candidate value. A `score` returns per-level probabilities and the expected level index. [OpenAI API reference for creating a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)

Get the ID of the loaded model. The existing OpenAI-style list endpoint returns one entry.

```bash
curl -fsS http://localhost:8000/v1/models
```

Replace `MODEL_ID` with `data[0].id`. The required `model` field identifies the model served by this instance. An unknown ID returns HTTP 404.

```bash
curl -fsS http://localhost:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_ID",
    "input": "Acme signed an agreement to acquire Beta.",
    "questions": [{"type": "predicate", "instructions": "Has a deal been signed?"}]
  }'
```

The probability is in `answers[0].probability`. OpenAI Python SDK 3.26.0 and later expose the same endpoint through `client.decisions.create`. [OpenAI Python SDK 3.26.0](https://github.com/openai/openai-python/releases/tag/v3.26.0)

## Model execution and limits

Qwen3.5-35B-A3B runs with `trtllm-serve serve MODEL`. The service reads the next-token logits for the specified answer labels. PPLX-Decider v1.1-27B runs on one GPU with `trtllm-serve embeddings MODEL`. Its trained prompt, ordered answer codes, last-position classification head and checkpoint temperature determine each distribution. Both services enforce the configured model context length.

System One accepts 1–255 choices. On Qwen, each answer label must encode as a distinct single token. OpenAI Decisions accepts 2–26 choices on Qwen and 2–255 on PPLX. The text profile excludes media, `box`, `input_json`, tool histories, per-question refusal and explicit cache controls.

Usage sums the prompt and output tokens consumed by each question. PPLX reads consume zero output tokens. OpenAI `cached_tokens` sums generation executor hits, and `cache_write_tokens` is zero because the executor does not report prefix writes. Generation choice and score confidence multiply the winning probability by the vocabulary mass on the candidates. PPLX scores the candidates directly.
