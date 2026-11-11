(request-trace)=

# Request Trace

`trtllm-serve` can record every request it receives and the exact reply it sent, as JSON
lines on disk. The trace is meant for offline analysis of real traffic: what clients sent,
what the server answered, how each exchange ended, and the join key between the two.

```{warning}
The trace writes **full prompts and full completions** to disk: every request body,
every reply body (streamed frames included) and every request header except credentials.
Only credential **headers** are redacted: `cookie`, `set-cookie`, and any header whose
name contains `auth`, `api-key`, `api_key`, `apikey`, `token`, `secret`, `password`, `jwt`,
`assertion`, `credential`, `signature` or `cert` (so `authorization`,
`x-forwarded-authorization`, `x-goog-iap-jwt-assertion`, `x-amz-credential`,
`x-hub-signature-256` and forwarded client certificates among others). Credentials
**inside a request body** -- an API key in a tool definition, a bearer token in an MCP
server's `headers`, a password in a prompt -- are written as received, as is everything
else, personal data a client puts in a prompt included. The trace is **off by default**.
Turning it on is a decision about the data the deployment handles: the writer creates its
files with mode 0600 and its directories with mode 0700, but who may read the directory,
how long the files are kept, and whether the deployment's users have agreed to it are the
operator's to settle.
```

## Enabling the trace

Set `TRTLLM_REQUEST_TRACE_DIR` to a directory before starting the server. There is no
command-line option. Unset or empty, the trace is not installed at all: no middleware, no
files, no change to the server's behavior.

```bash
TRTLLM_REQUEST_TRACE_DIR=/data/trace trtllm-serve <model> --port 8000
```

Both the aggregated server and the disaggregated server record their own clients'
requests. On a worker of a disaggregated deployment the internal hops the disaggregated
server fans a request out to (`disaggregated_params.request_type` of `context_only` or
`generation_only`) leave no trace, so one client request is recorded once, by the server
the client talked to.

Only requests that carry a body are recorded: a `POST`, `PUT` or `PATCH` whose body is not
empty. `GET` routes (health, metrics, the model list) and body-less `POST`s (the
disaggregated router's `/kv_cache_events` polls, `/start_profile`, `/steady_clock_offset`)
leave no trace. In a multi-frontend deployment with the trace on in every frontend
process, a request an attached frontend forwards to the launcher (`/v1/messages/batches`)
is recorded by both processes; the two request lines share a `body_sha256`.

## Layout

```text
$TRTLLM_REQUEST_TRACE_DIR/
  2026-11-10T12/                              one directory per UTC hour
    requests-<host>-<pid>-<start>.jsonl       one line per request, as soon as its body is complete
    responses-<host>-<pid>-<start>.jsonl      one line per request, when its reply has ended
  _writer/<generation>.json                   the writer's accounting (see below)
```

A line goes to the hour of its own timestamp (`recorded_at` or `finished_at`), so a
request that started in one hour and ended in the next has its two lines in two
directories. The two lines of one request share a `trace_id`.

## The request line

Written when the request body has been received, before the handler runs, so a request
the process does not survive is still on disk.

| Field | Meaning |
|-------|---------|
| `event` | `"request"` |
| `trace_id` | Joins the request line to its response line. |
| `session` | The request's conversation id (body `conversation_params.conversation_id`, else the conversation-id headers the server supports), reduced to `[A-Za-z0-9._-]` and at most 128 characters; `_no_session` when there is none. |
| `recorded_at` | UTC timestamp of the line. |
| `server_arrival_time` | The arrival stamp the server takes for its performance metrics, on the same steady clock, so a line can be matched to a perf-metrics record. Not a wall clock. |
| `method`, `route` | HTTP method and URL path. |
| `headers` | Request headers as ordered `[name, value]` pairs, repeats kept, credential values replaced by `[redacted]`. |
| `body` | The decoded body: JSON, or msgpack when the request carried `x-trtllm-msgpack: 1`. A body that does not decode is recorded as text with a `body_parse_error` field. |
| `body_sha256` | SHA-256 of the body bytes as received, so a retry can be told from a new request. |
| `body_complete` | Present and `false` only when the handler answered before the body was fully received. |
| `sampling` | The generation knobs the client sent (`max_tokens`, `max_output_tokens`, `temperature`, `top_p`, `stream`, `reasoning_effort`, `tool_choice`, ...), keyed as sent and only when sent. |

## The response line

Written when the reply has ended, however it ended.

| Field | Meaning |
|-------|---------|
| `event` | `"response"` |
| `trace_id`, `session` | As on the request line. |
| `finished_at` | UTC timestamp of the line. |
| `status` | How the exchange ended; see the table below. |
| `http_status` | The HTTP status sent, or `null` when no reply started. |
| `disconnected` | Present and `true` when the server saw the client disconnect before the reply was complete, whatever the `status`. A non-streaming handler that notices the disconnect still hands its reply to the server, which accepts it silently; such a reply is `completed` and `disconnected`. (A disconnect message after the reply is complete is how an ASGI server answers any later `receive`, and is not recorded.) |
| `headers` | Response headers as sent, redacted like request headers. Absent when no reply started. |
| `response` | `{"kind": "sse_text", "body": "<the frames>"}` for a stream, `{"kind": "json", "body": <object>}` for a JSON reply, `{"kind": "text", "body": "..."}` for anything else. The body is exactly what the server sent, after every other built-in middleware ran (the performance-metrics headers and SSE event included); whether the client read all of it is not knowable on the server (see `disconnected`). Absent when no reply started. |
| `error` | `"<ExceptionType>: <message>"` when the handler raised or a send failed. |
| `finish_reasons`, `usage` | Lifted from a completion reply (a JSON `chat.completion` or `text_completion`, or the `chat.completion.chunk` frames of a chat stream): the finish reason of each choice, and the `usage` object when the reply carries one (a chat stream does so only when the client asked for it with `stream_options.include_usage`). Completions streams and Messages replies carry no summary. |
| `response_status`, `incomplete_reason`, `error_code`, `response_id`, `usage`, `max_output_tokens`, `output_item_types`, `output_call_ids` | Lifted from the terminal object of a Responses API reply, so a reader can tell a finished generation from a truncated or failed one without parsing the body. The terminal event is looked for among the stream's last eight frames (it is normally the last one; the performance-metrics event or a relay's error frame may follow it). |

| `status` | Meaning |
|----------|---------|
| `completed` | The reply was handed to the server in full with a 1xx-3xx status. |
| `rejected_<code>` | The reply was handed to the server in full with a 4xx status, for example `rejected_400` for a request validation failure. The request line holds the rejected body. |
| `error` | A 5xx reply, an exception out of the handler (the response line then has `error`, and the framework's own 500 is not observed), or a reply that stopped without its final chunk and without a disconnect to explain it. |
| `client_disconnected` | The client went away before the reply was complete: the server saw the disconnect and stopped streaming, the handler raised because the client was gone (Starlette's `ClientDisconnect` while reading the body), or a send failed. The frames sent so far are recorded. |
| `not_started` | The handler ended without sending anything and without an error, as a streaming response does when the client disconnects before the reply begins. |
| `cancelled` | The request's task was cancelled (server shutdown) without a client disconnect. |

### What only the handler knows

The middleware sees requests and replies; four facts live only in the handler that served
the request. The middleware seeds `request.state.request_trace` with a `RequestTraceFacts`
object for every request it records, the handlers fill it in as they learn things, and the
set fields are added to the response line:

| Field | Meaning |
|-------|---------|
| `engine_requests` | One entry per engine request the handler submitted for the reply: `{"request_id": <engine request id>, "prompt_tokens": {"count", "sha256", "encoding": "int32le"}}`. The digest is over the token ids the engine was given, as a little-endian int32 buffer (`hashlib.sha256(numpy.asarray(ids, dtype="<i4").tobytes())`), so a reader can check that its own rendering of the recorded body is the prompt the model saw. Only the digest is stored, never the ids. Recorded by the aggregated server for chat completions, completions (one entry per prompt), Responses and Messages. |
| `served_model` | The model name the aggregated server served the request under. |
| `disaggregated` | What the disaggregated server knows: `disagg_request_id`, `ctx_request_id`, `ctx_server`, `gen_server` (which workers served the request) and `prompt_tokens`, the digest of the prompt the context worker tokenized (taken from its reply, which the server relays and never hands to the client; a prefill run again on another worker replaces it). |
| `termination` | Why a stream's generation failed, as the producer saw it: `{"cause": "engine_error" \| "upstream_error" \| "internal_error", "detail": "<exception type, and its message for engine errors>"}`. Recorded whether the failure cut the stream (`status` is then `error`) or was answered inside it (a completions stream ends with an SSE `error` event and `[DONE]`, so its `status` is `completed`). Absent for a generation that ran to its end and for a stream the client cut. |

A handler that never touches the object -- a route this server does not instrument, or
a deployment that overrides a handler -- leaves the response line exactly as above. On
the untraced path the handlers do one attribute lookup and nothing else; with the trace
on, a handler does a few attribute writes and keeps a reference to the prompt token ids
it already holds, which the writer's thread digests when it writes the line, so the
event loop's cost does not grow with the prompt.

The trace never changes a reply: every step it takes runs under a guard, and a failure
inside the trace (a body the decoder cannot follow is not one -- it is recorded as text
with `body_parse_error`) costs the line it was building, is counted, and is logged once
and then every thousandth time. A request whose request line was lost still gets its
response line, with the same `trace_id`.

## Deployment-specific fields

A deployment that stamps its own identity on requests (a gateway's request id, a
tenant, an experiment label) can add fields to each request line through an import-path
plugin:

```bash
TRTLLM_REQUEST_TRACE_IDENTITY=my_pkg.trace_identity:identity
```

The callable receives `(headers, body, route)` -- the request's case-insensitive header
mapping (unredacted), the decoded body and the URL path -- and returns a mapping of
JSON-serializable fields, or `None`. Fields named like the request line's own keys
(`session`, `trace_id`, `body`, ...) are refused. A plugin that raises, or returns a value
JSON cannot hold, costs that request its plugin fields, never the line; a plugin that
cannot be imported stops the server at startup. Nothing on the response line comes from
the plugin.

## The writer and its accounting

Lines are queued on the event loop and appended by a worker thread, in batches, with each
append flushed and `fsync()`ed before its records count as written. A full queue drops
records rather than slowing the server; drops are counted and logged.

Each writer publishes its accounting as `_writer/<generation>.json`: how many records
were submitted, persisted, dropped (by reason: `queue_full`, `unserializable`,
`write_error`, `shutdown`), left `unknown` (their bytes may have landed, but the write or
`fsync` holding them failed) or still pending, and for each file it appended to the byte
range its durable appends occupy. The document is rewritten at start, on every write
failure, every ten seconds while the counts change, and at shutdown. A reader that needs
to know whether a copy of the trace is complete compares the files to these claims
rather than trusting what happens to be on disk: a line in the claimed range of a shard
was written by the writer that claims it, and `submitted == persisted + dropped + unknown
+ pending` holds in every published document.

A file whose last append stopped mid-line (a crash, a full disk) is repaired by the next
append, which starts on a fresh line; the fragment stays as one line a JSONL reader
skips. Two server lifetimes that reuse a pid append to different files, because the
writer's start time is part of the file name.

## Cost

With the trace on, the event loop decodes each request body once more than it otherwise
would (the record needs the parsed body for the session key and the internal-hop check;
about 2 ms per MB of JSON), keeps a reference to each reply chunk until the reply has
ended, and queues two records per request. Hashing the body, joining and decoding the
reply, lifting its summaries, serializing both lines and appending them happen on the
writer's worker thread; a long stream costs the loop one list append per chunk. The
reply the client receives is not changed in any way: the trace never adds
`stream_options`, headers or events.
