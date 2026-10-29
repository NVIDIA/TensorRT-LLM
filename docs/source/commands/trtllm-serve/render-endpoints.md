# Render endpoints

The render endpoints turn an OpenAI-style request into the prompt token ids a worker would use, without running
the model. A router or gateway can use the ids to choose a worker, and a client can send them to a worker to skip
its tokenization and chat templating.

- `POST /v1/chat/completions/render`
- `POST /v1/completions/render`

Both take the same body as the inference route of the same name. They are a prototype and are not registered with
`trtllm-serve` by default. In this page, a *worker* is a `trtllm-serve` server that runs the model.

## Start the endpoints

On a worker, set `TRTLLM_ENABLE_RENDER_ENDPOINTS=1` before starting `trtllm-serve`.

On a CPU-only host, `trtllm-render` loads only the tokenizer, configuration and processor files of a checkpoint:

```bash
trtllm-render --model <checkpoint> --port 8100
```

In a container with no GPU driver, `trtllm-render` loads the CUDA toolkit's `libcuda.so` stub from `$CUDA_HOME` or
`/usr/local/cuda*` so that `tensorrt_llm` can be imported. `python -m tensorrt_llm.serve.render` does not do this.

Give it the same model, tokenizer, chat template and parser options as the workers it fronts. Both processes publish
a rendering fingerprint on `GET /server_info`. Equal `render_fingerprint.digest` values mean the ids are valid for
that worker, so compare them once when a worker is added.

## The response

Chat returns one rendered request. This is the response for a chat request with a system and a user message:

```json
{
  "schema_version": 1,
  "kind": "chat",
  "request_id": "chatcmpl-9651e194058642d9b538b695bb179c68",
  "token_ids": [151644, 8948, 198, 2610, 525, 50537, "..."],
  "model": "Qwen3-0.6B",
  "reasoning_ended": null,
  "reasoning_parser_kwargs": {"chat_template_kwargs": {}},
  "cache_salt": null,
  "request": {
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Name three primary colors."}
    ],
    "max_tokens": 24,
    "temperature": 0
  }
}
```

Completions return a list with one rendered request per prompt. In each, `request` is the original body with `prompt`
set to that item's own prompt:

```json
{
  "schema_version": 1,
  "kind": "completion",
  "request_id": "cmpl-2eb4d58b52984d25916f70687f565577-0",
  "token_ids": [785, 6722, 315, 9625, 374],
  "model": "Qwen3-0.6B",
  "reasoning_ended": null,
  "reasoning_parser_kwargs": null,
  "cache_salt": null,
  "request": {"model": "Qwen3-0.6B", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
}
```

| Field | Meaning |
|---|---|
| `kind` | `"chat"` or `"completion"`: which route `request` belongs to. |
| `request_id` | Identifies this rendered request. |
| `token_ids` | The rendered prompt. |
| `model`, `cache_salt` | Copied from the request. |
| `reasoning_ended` | Whether reasoning had already ended before the first generated token, read off the prompt for the reasoning parser the renderer is started with. `null` when there is no such parser or the prompt does not say. |
| `reasoning_parser_kwargs` | The `chat_template_kwargs` the prompt was rendered with. Set for chat when the renderer has a reasoning parser, otherwise `null`. |
| `request` | The original request body. For a completions request with several prompts, each item carries its own prompt. |

## Send the result to the inference endpoints

The original request is always valid to send to a worker. The ids only let the worker skip rendering.

### Completions

Add the ids as the `prompt` of each item's `request`:

```bash
curl -s $RENDERER/v1/completions/render -H 'Content-Type: application/json' -d @request.json \
  | jq -c '.[] | .request + {prompt: .token_ids}' \
  | while read -r body; do
      curl -s $WORKER/v1/completions -H 'Content-Type: application/json' -d "$body"
    done
```

### Chat

Add the ids as `prompt_token_ids`:

```bash
curl -s $RENDERER/v1/chat/completions/render -H 'Content-Type: application/json' -d @request.json \
  | jq '.request + {prompt_token_ids: .token_ids}' \
  | curl -s $WORKER/v1/chat/completions -H 'Content-Type: application/json' -d @-
```

```python
import requests

rendered = requests.post(f"{RENDERER}/v1/chat/completions/render", json=body).json()
send = {**rendered["request"], "prompt_token_ids": rendered["token_ids"]}
reply = requests.post(f"{WORKER}/v1/chat/completions", json=send).json()
```

A request with a named `tool_choice` (the model must call one specific function) is the exception. The worker
appends a forced tool-call prefix when it renders, and rejects the request together with `prompt_token_ids`. Send
`rendered["request"]` as it is for those.

## Limits

Multimodal requests (images, video, audio) are not yet supported. The endpoints return `400` for them.

## When the KV-aware router forwards its ids

The `kv_cache_aware` router renders each chat request with the same code to choose a worker. It also sends the ids it
rendered to the worker, so the worker does not render again, only when every active worker reports the same rendering
fingerprint as the router in `/server_info`. Otherwise the worker renders the prompt itself and the router only uses
its ids to route. The router logs a warning that names each worker that blocks forwarding and why, and logs again
only when that changes. The causes:

- A worker reports no fingerprint (an older version, or its `/server_info` could not be fetched).
- A worker renders differently: another chat template (`--chat_template`), tokenizer, model type, or a model whose
  processor builds the prompt (vision models).
- A tokenizer that cannot be fully identified.
- The router has no model path to resolve the model type from, so it renders as a generic text model.

Some requests are never forwarded, whatever the workers report: a named `tool_choice`; a model whose usage count
excludes tokens of the rendered prompt (Kimi-K3); and a prompt that carries a reasoning mode (`<think>` prefilled) when a
worker runs a reasoning parser that reads the mode from the prompt text (`poolside_v1`, `laguna`, `minimax_m3`).
