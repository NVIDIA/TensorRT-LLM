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
[`tensorrt_llm/executor/postprocessor_hook.py`](../../../tensorrt_llm/executor/postprocessor_hook.py).

## Enabling the hook

Pass the dotted import path of your hook class to `--post_processor_hook`:

```bash
trtllm-serve <model> --post_processor_hook my_pkg.guardrail.MyPostProcessorHook
```

Equivalently, set it in a YAML config passed via `--extra_llm_api_options`:

```yaml
post_processor_hook: my_pkg.guardrail.MyPostProcessorHook
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
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
    suppress,
    terminate,
)


class MyPostProcessorHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return emit(chunk.text_diff)  # pass through unchanged
```

### `PostProcessorHookChunk`

The payload handed to the hook for one output chunk — `request_id`, `output_index`, `text_diff`,
`text`, `token_ids_diff`, `is_final`, `aborted`, `streaming`, and `disagg_request_id`. See the
`PostProcessorHookChunk` dataclass in [`postprocessor_hook.py`](../../../tensorrt_llm/executor/postprocessor_hook.py)
for the authoritative field-by-field descriptions.

### Verdicts

Return one of the following helpers from `__call__`:

| Helper | Effect |
|--------|--------|
| `emit(text)` | Emit `text` for this chunk. Use it to pass output through unchanged (`emit(chunk.text_diff)`) or to rewrite/redact it. Affects the **text** channel only — it does not synthesize matching `token_ids` (see *Text vs. token ids* below). |
| `suppress()` | Withhold this chunk entirely across **all** client-visible channels — text, `token_ids`, and `logprobs` (so `detokenize=false` token output is withheld too). |
| `terminate(reason)` | Stop the stream for this request, withholding the terminating chunk on all channels. `reason` is surfaced as the response `stop_reason`, and the engine request is cancelled. |

Verdicts are **per chunk**: `suppress()` withholds the current chunk, and `terminate()` stops generation while keeping the chunks already emitted before it. This is consistent across streaming and non-streaming — a non-streaming response contains exactly the content the hook emitted before the first `suppress`/`terminate`, i.e. the same content a streaming client would have received. A hook that must withhold the *entire* output (all-or-nothing) should `suppress()` from the first chunk (it sees every chunk) rather than emitting and then terminating.

## Usage examples

### Rewrite output

A stateless hook that upper-cases every chunk:

```python
from tensorrt_llm.executor.postprocessor_hook import PostProcessorHookChunk, PostProcessorHookVerdict, emit


class UpperCaseHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return emit(chunk.text_diff.upper())
```

### Stateful guardrail that terminates on a banned phrase

This hook accumulates text per request (keyed by `request_id`), stops the stream as soon as a banned
phrase appears, and releases its state when the request finishes:

```python
from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk, PostProcessorHookVerdict, emit, terminate,
)


class BannedPhraseGuard:
    _BANNED = ("forbidden phrase",)

    def __init__(self):
        # Per-request accumulators owned entirely by the hook.
        self._buffers: dict[int, str] = {}

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
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
from tensorrt_llm.executor.postprocessor_hook import PostProcessorHookChunk, PostProcessorHookVerdict, suppress


class SuppressHook:
    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return suppress()
```

## Built-in hook: raw output dump

`tensorrt_llm.executor.raw_output_hook.RawOutputDump` ships with TensorRT LLM. It records what the
model generated for each request — after detokenization, and before the reasoning and tool-call
parsers and the response formatter — without changing the served output. Comparing a record with the
response a client received tells an error in the generated output from one introduced by a parser or
by the response formatting.

It records only when both the environment variable and the flag are set:

```bash
export TRTLLM_RAW_OUTPUT_DIR=/path/to/raw_output
trtllm-serve <model> \
    --post_processor_hook tensorrt_llm.executor.raw_output_hook.RawOutputDump
```

With `TRTLLM_RAW_OUTPUT_DIR` unset, the hook logs a warning when it is built, then passes every chunk
through and writes nothing. Each process that runs the hook (the server, or each post-processing
worker) appends JSON lines for finished outputs and partial capture segments — not per chunk — to
`$TRTLLM_RAW_OUTPUT_DIR/<UTC hour>/raw-<pid>.jsonl`, where `<UTC hour>` looks like `2026-11-04T05`:

| Field | Meaning |
|---|---|
| `request_id` | The engine request id (`chunk.request_id`), as a string. A streaming chat response, or a streaming completions response for a single prompt, carries it in its id, `chatcmpl-<id>` / `cmpl-<id>`. |
| `disagg_request_id` | The id the disaggregated server assigned the request (`chunk.disagg_request_id`), as a string; `null` when serving is not disaggregated. |
| `output_index` | The output's index within the request (`n` > 1, beam search). |
| `text` | The output's full detokenized text. |
| `frames` | The text each chunk added, in order; chunks that added none are left out. |
| `frames_consistent` | Whether `frames` add up to `text`; see below. |
| `token_ids`, `num_output_tokens` | The captured history of generated token ids and its count, not necessarily the final output's total. |
| `streaming`, `aborted` | From the output's last observed chunk. |
| `started_at`, `finished_at` | UTC ISO-8601 times of the current retained segment's first chunk and of the record. |
| `partial`, `end_reason` | Whether capture ended early or its history is incomplete or uncertain, and why; may be `true` and set even after a normal final chunk. These are not the model's stop reason. |
| `event` | Always `"raw_output"`. |

The ids are written as strings: a disaggregated id can be larger than 2^53, past which a JSON reader
that parses numbers as doubles loses precision. `frames` is a list rather than one string because
incremental parsers fail at chunk boundaries: a tool-call marker split across two chunks looks the same
as one emitted whole once the chunks are joined. `token_ids` also shows tokens that detokenization
leaves out of `text`, such as skipped special tokens.

`frames` and `token_ids` preserve the history handed to the hook, rather than reconstructing a
final output snapshot. Even with one sequence (`n` and `best_of` of 1), a stop-word correction can
remove text or token ids already seen in earlier chunks; those earlier frames and ids stay in the
record. When a request generates several sequences (`n` or `best_of` above 1, which includes beam
search), an output index need not retain one sequence's own history: `token_ids` can repeat ids, and
when sampling with `best_of` above `n` the index is a rank that can change between chunks, so one
record can join chunks of different sequences. `text` is always the text at the output's last
observed chunk, and `frames_consistent` is `false` when `frames` do not add up to it. Neither a true
`frames_consistent` nor a false `partial` guarantees that the captured history equals the final
output snapshot.

When `on_request_end` reports an error (see [Per-request state](#per-request-state)), retained
segments that produced text or token ids are written with `"partial": true` and `end_reason` set
to that reason. Already evicted segments are not rewritten, and an empty retained segment adds
no line for the callback. An output that gets neither a final chunk nor
`on_request_end` (see the same section) stays in flight. The hook keeps at most `max_pending` outputs
(4096) in flight: past that, it writes the least recently updated one with `"partial": true` and
`end_reason` `"evicted"`. If that output later resumes, the hook records a new segment: `text` is
still the full current text, but `frames`, `token_ids`, and `num_output_tokens` cover only the
retained segment. It is also marked `"partial": true` and `"end_reason": "evicted"`, even if it
receives a normal final chunk. An actual error reported by `on_request_end` takes precedence over
the capture-history reason.

The hook remembers at most `max_pending` evicted keys, releasing a request's markers on
`on_request_end` and consuming a key's marker when it resumes. If the marker budget overflows,
recording continues, but subsequent new segments without a known marker are conservatively marked
`"partial": true` and `"end_reason": "history_unavailable"`. This includes genuinely new outputs:
the hook cannot distinguish them from an output whose eviction identity did not fit. This
uncertainty stays latched until the hook instance restarts, even if all remembered markers are
later released. Already retained segments keep their history status. `partial` being false means
no known capture-history gap, not a guarantee of final-snapshot equality. To keep more or fewer
pending outputs and markers, set `max_pending` in a subclass and pass the subclass's import path
to `--post_processor_hook`.

A hook that raises fails the request (see *Hook errors fail closed* below). A debug dump must not, so
this hook never raises: a record that cannot be written — for example, because the directory is not
writable — is dropped, the drop is logged as a warning, and the response is served unchanged.

## Per-request state

The hook instance is owned by the `LLM` (built once in each post-processing worker process when the
pool is enabled) and shared across all requests it handles, so any per-request state must be keyed by
`chunk.request_id` and released when `chunk.is_final` is seen (or after a `terminate`). State is not
shared across processes or across separate `LLM` instances; when the post-processing worker pool is
enabled, all chunks of a single request are still routed to the same worker, so per-request state
remains consistent for that request.

When the post-processing worker pool is enabled, a request that ends in an error — the executor
reports one, or post-processing its response raises — gets no final chunk. A hook that keeps
per-request state can define an optional `on_request_end(self, request_id, reason)` method: the worker
calls it when it drops such a request, with `reason` set to `"error"` or `"postprocessing_error"`, so
the hook can release (or record) what it holds for that request. A hook without the method is left
alone, and an exception raised by it is logged rather than propagated.

Without the worker pool, the hook runs as the server reads each response of a request. A request that
ends in an executor error gets no final chunk there either (the error is raised before the hook
runs), and `on_request_end` is called for it with `reason` set to `"error"`. A stream whose client
disconnects before it finishes gets neither: the server stops reading its responses, so the hook sees
no further chunk for it.

Engine-level batching is transparent to the hook: even when requests are batched together in the
engine, the hook is invoked **once per request** (per output, per chunk) — there is no batched-call
form — so keying state on `chunk.request_id` is sufficient to keep concurrent requests isolated.

## Supported endpoints and limitations

- **Endpoints**: `chat/completions` and `completions`, both streaming and non-streaming. The hook is
  expected to apply to the `responses` endpoint for non-harmony models as well (shared detokenization
  path), though that endpoint is not covered by the current end-to-end tests.
- **Not client-bypassable**: the hook is a server-side guardrail, so it runs on every response even
  when a `completions` request sets `detokenize=false`. The server detokenizes for the hook regardless;
  the `detokenize` flag still controls only the returned channel (text vs. token ids). A
  `suppress`/`terminate` verdict withholds **all** client-visible channels — text, `token_ids`, and
  `logprobs` — on both the streaming and non-streaming paths, so a client cannot recover withheld
  content through any channel.
- **Requires a tokenizer**: the hook needs detokenized text to inspect, so `--post_processor_hook` combined
  with `skip_tokenizer_init` is rejected at startup rather than silently disabled.
- **harmony / gpt-oss models**: not supported. Because the harmony output path is reconstructed from
  raw token ids, it would bypass the text-based hook, so the server fails fast at startup when
  `--post_processor_hook` is combined with a harmony model.
- **Disaggregated serving**: the context and generation servers are separate processes, each running
  the hook on its own phase under a different `request_id`; per-request state is not shared across the
  two. Both phases see the same `chunk.disagg_request_id`, which correlates what each of them saw. A
  `terminate` on one phase does not propagate to the other.
- **Text vs. token ids**: `emit` rewrites the **text** channel only — it does not rewrite the underlying
  `token_ids`/`logprobs`, so a client reading both should expect them to diverge. (`suppress`/`terminate`
  withhold all channels, so they do not diverge.)
- **Reasoning / tool parsing**: the hook runs before the reasoning and tool-call parsers. A hook that
  rewrites or suppresses text may desync those parsers; prefer `terminate`, or apply such hooks to
  plain-text requests.
- **Hook errors fail closed**: if the hook raises, the request fails with an error rather than serving
  the un-vetted chunk (the server and other requests stay alive). A returned verdict with an unknown
  action is rejected the same way.
- **`n` > 1 / beam search**: `emit` and `suppress` act per output sequence, but `terminate` cancels the
  **whole** request (all sequences), because the engine request is the unit of cancellation — a
  `terminate` on one candidate ends the others too. Hooks needing per-sequence state should key on
  `(request_id, output_index)`.

## Tests

The hook's unit and end-to-end tests double as runnable usage examples:

```bash
# Unit tests for the hook contract (rewrite / suppress / terminate, per-request state, loader)
pytest tests/unittest/executor/test_postprocessor_hook.py -v
# Unit tests for the built-in raw output dump
pytest tests/unittest/executor/test_raw_output_hook.py -v
```

- End-to-end serving tests across endpoints, streaming modes, and worker-pool settings:
  `tests/unittest/llmapi/apps/_test_openai_post_processor.py`.
- The deterministic sample hooks used by those tests:
  `tests/unittest/llmapi/apps/_postproc_hook_samples.py`.
