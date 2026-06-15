(post-processor-hook)=

# Post-Processing Hook

`trtllm-serve` supports a user-supplied **post-processing hook**: a native, per-request seam that runs
on each generated output *after* detokenization and *before* the per-endpoint response formatter. It
lets a deployment rewrite, redact, suppress, or terminate model output — including stateful logic that
spans the chunks of a streamed response — without modifying TensorRT LLM source.

The hook is a plain Python callable class supplied by import path, mirroring `--custom_tokenizer`. It
is owned by the `LLM` instance (and built once in each post-processing worker process when those are
enabled) and invoked once per output, per streaming chunk (plus a final call), so it can hold its own
per-request state. Independent `LLM` instances in one process each own a separate hook instance.

```{note}
This feature is a prototype and its interface may change in a future release.
```

For the interface definitions referenced below, see
[`tensorrt_llm/executor/postprocessor_hook.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/executor/postprocessor_hook.py).

## Enabling the hook

Pass the dotted import path of your hook class to `--post_processor`:

```bash
trtllm-serve <model> --post_processor my_pkg.guardrail.MyPostProcessor
```

Equivalently, set it in a YAML config passed via `--extra_llm_api_options`:

```yaml
post_processor: my_pkg.guardrail.MyPostProcessor
```

The class must be:

- **Importable** — installed (`pip install`) or on `PYTHONPATH` when the server (and its
  post-processing worker processes, if any) start.
- **Picklable and no-argument-constructible** — the hook is reconstructed by reference inside each
  process; `__init__` takes no required arguments and is the one-time setup point.

The hook works the same with or without the out-of-process post-processing worker pool
(`--num_postprocess_workers`).

## The hook interface

A hook implements a single method, `__call__(self, chunk) -> verdict`:

```python
from tensorrt_llm.executor.postprocessor_hook import (
    PostProcChunk,
    PostProcVerdict,
    emit,
    suppress,
    terminate,
)


class MyPostProcessor:
    def __call__(self, chunk: PostProcChunk) -> PostProcVerdict:
        return emit(chunk.text_diff)  # pass through unchanged
```

### `PostProcChunk`

The payload handed to the hook for one output chunk:

| Field | Description |
|-------|-------------|
| `request_id` | Stable identifier for the request; the same value is passed for every chunk of a response, so the hook can key its per-request state on it. |
| `output_index` | Index of the output/beam within the request. |
| `text_diff` | Newly detokenized text produced by this chunk (streaming). For non-streaming requests this equals `text`. |
| `text` | Full accumulated detokenized text so far for this output. |
| `token_ids_diff` | Newly generated token ids for this chunk. |
| `is_final` | `True` on the terminating call for this output. |
| `aborted` | `True` if the request has been marked aborted in this process. Output-side observation only. |
| `streaming` | `True` for streaming requests. |

### Verdicts

Return one of the following helpers from `__call__`:

| Helper | Effect |
|--------|--------|
| `emit(text)` | Emit `text` for this chunk. Use it to pass output through unchanged (`emit(chunk.text_diff)`) or to rewrite/redact it. |
| `suppress()` | Withhold this chunk entirely (no client-visible output for it). |
| `terminate(reason)` | Stop the stream for this request. `reason` is surfaced as the response `stop_reason`, and the engine request is cancelled. |

## Usage examples

### Rewrite output

A stateless hook that upper-cases every chunk:

```python
from tensorrt_llm.executor.postprocessor_hook import PostProcChunk, PostProcVerdict, emit


class UpperCaseHook:
    def __call__(self, chunk: PostProcChunk) -> PostProcVerdict:
        return emit(chunk.text_diff.upper())
```

### Stateful guardrail that terminates on a banned phrase

This hook accumulates text per request (keyed by `request_id`), stops the stream as soon as a banned
phrase appears, and releases its state when the request finishes:

```python
from tensorrt_llm.executor.postprocessor_hook import (
    PostProcChunk, PostProcVerdict, emit, terminate,
)


class BannedPhraseGuard:
    _BANNED = ("forbidden phrase",)

    def __init__(self):
        # Per-request accumulators owned entirely by the hook.
        self._buffers: dict[int, str] = {}

    def __call__(self, chunk: PostProcChunk) -> PostProcVerdict:
        buffer = self._buffers.get(chunk.request_id, "") + chunk.text_diff.lower()
        self._buffers[chunk.request_id] = buffer

        if any(phrase in buffer for phrase in self._BANNED):
            self._buffers.pop(chunk.request_id, None)
            return terminate("banned_phrase")

        if chunk.is_final:
            self._buffers.pop(chunk.request_id, None)
        return emit(chunk.text_diff)
```

### Suppress output

A hook that withholds all client-visible text:

```python
from tensorrt_llm.executor.postprocessor_hook import PostProcChunk, PostProcVerdict, suppress


class SuppressHook:
    def __call__(self, chunk: PostProcChunk) -> PostProcVerdict:
        return suppress()
```

## Per-request state

The hook instance is owned by the `LLM` (built once in each post-processing worker process when the
pool is enabled) and shared across all requests it handles, so any per-request state must be keyed by
`chunk.request_id` and released when `chunk.is_final` is seen (or after a `terminate`). State is not
shared across processes or across separate `LLM` instances; when the post-processing worker pool is
enabled, all chunks of a single request are still routed to the same worker, so per-request state
remains consistent for that request.

Engine-level batching is transparent to the hook: even when many requests are batched and run together
in the engine, the hook is still invoked **once per request** (per output, per chunk), with
`chunk.request_id` identifying which request the chunk belongs to. There is no batched-call form — the
hook never receives more than one request's data in a single call, so keying state on `request_id` is
sufficient to keep concurrent requests isolated.

## Supported endpoints and limitations

- **Endpoints**: `chat/completions` and `completions`, both streaming and non-streaming. The hook also
  applies to the `responses` endpoint for non-harmony models.
- **harmony / gpt-oss models**: not supported. Because the harmony output path is reconstructed from
  raw token ids, it would bypass the text-based hook, so the server fails fast at startup when
  `--post_processor` is combined with a harmony model.
- **Text vs. token ids**: rewriting or suppressing text does not rewrite the underlying `token_ids` or
  `logprobs`. Clients that read both should expect them to diverge.
- **Reasoning / tool parsing**: the hook runs before the reasoning and tool-call parsers. A hook that
  rewrites or suppresses text may desync those parsers; prefer `terminate`, or apply such hooks to
  plain-text requests.

## Tests

The hook's unit and end-to-end tests double as runnable usage examples:

```bash
# Unit tests for the hook contract (rewrite / suppress / terminate, per-request state, loader)
pytest tests/unittest/executor/test_postprocessor_hook.py -v
```

- End-to-end serving tests across endpoints, streaming modes, and worker-pool settings:
  `tests/unittest/llmapi/apps/_test_openai_post_processor.py`.
- The deterministic sample hooks used by those tests:
  `tests/unittest/llmapi/apps/_postproc_hook_samples.py`.
