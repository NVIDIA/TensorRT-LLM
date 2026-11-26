# Decision APIs

The generation service and the PPLX-Decider v1.1 embedding service answer typed questions about shared text. Each endpoint accepts an uncompressed JSON request. Use the served model ID from the OpenAI-style `GET /v1/models` response. System One also accepts the TypeSafe SDK aliases `jev-latest` and `jev-preview`. Responses identify the loaded model.

## TypeSafe System One

`POST /v1/systemone` accepts a string, JSON object or array as `state`. Give each question a key. That key identifies its answer. A `noul` returns the probability of yes. A `choice` returns the winning key and the probabilities of its 1–255 candidates. A `score` returns the expected zero-based level index, the probabilities and a legend for its 1–10 ordered levels. Choice and score include confidence computed from the candidate distribution using the [TypeSafe definitions](https://docs.typesafe.ai/confidence).

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

Read the winning label from `answers.event.choice` and its distribution from `answers.event.probabilities`. For the official Python client, set `compression=False` so it sends plain JSON. [TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)

## OpenAI Decisions

`POST /v1/decisions` accepts a string or an ordered array of user messages containing strings or `input_text` parts. Supply 1–64 questions with text instructions. Answers retain question order, including repeated or omitted names. Success and error responses carry a server-generated `x-request-id`. A `predicate` returns the probability of true. A `choice` preserves each string or boolean candidate value. A `score` returns per-level probabilities and the expected index on a 2–10 level scale.

Get the served model ID.

```bash
curl -fsS http://localhost:8000/v1/models
```

Replace `MODEL_ID` with the `id` from the first entry in `data`.

```bash
curl -fsS http://localhost:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_ID",
    "input": "Acme signed an agreement to acquire Beta.",
    "questions": [{"type": "predicate", "instructions": "Has a deal been signed?"}]
  }'
```

The probability is in `answers[0].probability`. The official Python client exposes the same endpoint through `client.decisions.create`. [OpenAI Python SDK](https://github.com/openai/openai-python)

## Model execution and limits

Qwen3.5-35B-A3B runs with `trtllm-serve serve MODEL`. The service reads the next-token logits for the specified answer labels. OpenAI choices allow 2–26 candidates on this path. Wider System One choices require distinct single-token labels at the answer position. PPLX-Decider v1.1-27B runs with `trtllm-serve embeddings MODEL`. It supports 2–255 OpenAI choice candidates. Its trained prompt, ordered answer codes, last-position classification head and checkpoint temperature determine each distribution. Both services enforce the configured model context length.

Usage sums the prompt and output tokens consumed by each question. PPLX reads consume zero output tokens. OpenAI `cached_tokens` sums generation executor hits, and `cache_write_tokens` is zero because the executor does not report prefix writes. Generation choice and score confidence multiply the winning probability by the vocabulary mass on the candidates. PPLX scores the candidates directly.

Media, tool histories, per-question refusal, explicit cache controls, compressed uploads and framework-specific decision dialects are outside this text profile. See the [OpenAI Decisions reference](https://developers.openai.com/api/reference/resources/decisions/methods/create) and the [TypeSafe System One reference](https://docs.typesafe.ai/api/system-one) for the provider contracts.
