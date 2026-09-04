# Claude Code on TensorRT-LLM

`trtllm-serve` speaks the Anthropic Messages API, so Claude Code and anything
built on the Claude Agent SDK can run against a local model with no translation
proxy in between.

```bash
./launch.sh /path/to/your/model
```

The script starts the server, starts the gateway in front of it, waits for both
to report healthy, and prints the exact environment variables to run Claude Code
with. Ctrl-C tears everything down.

## What the endpoint provides

| Endpoint | |
|---|---|
| `POST /v1/messages` | streaming and non-streaming |
| `POST /v1/messages/count_tokens` | context sizing |
| `POST /v1/messages/batches` and the five routes under it | asynchronous batches |

Both the aggregated and the disaggregated server register all of them, so a
client cannot tell the two deployments apart.

## Connecting Claude Code

`launch.sh` prints this for you, filled in:

```bash
ANTHROPIC_BASE_URL=http://<host>:8333 \
ANTHROPIC_AUTH_TOKEN=<your-username> \
ANTHROPIC_MODEL=<model-id> \
  claude
```

Three things to know about those variables:

**`ANTHROPIC_MODEL` is not optional.** Claude Code never enumerates models — the
model is a field in each request body — so it uses whatever you give it. The
server does not validate the name, but sending the real one keeps its logs and
usage readable.

**`ANTHROPIC_AUTH_TOKEN` must match a line in the gateway's users file.** The
username *is* the key. That makes it an allowlist and an attribution trail, not
a secret: fine among colleagues on an internal network, not something to expose
beyond it. Without the gateway (`--no-gateway`) the server does not authenticate
at all and any value works.

**Set `ANTHROPIC_BASE_URL` to the gateway, not the server**, unless you have a
reason not to — see below.

To keep a Claude Code session against a local model separate from your real
Claude Code login, give it its own `HOME`:

```bash
env -i HOME=/tmp/claude-local PATH="$PATH" TERM="$TERM" \
    ANTHROPIC_BASE_URL=... ANTHROPIC_AUTH_TOKEN=... ANTHROPIC_MODEL=... \
    "$(command -v claude)"
```

`~/.claude/.credentials.json` is then unreachable, so it can be neither read nor
overwritten.

## Why the gateway

A Claude Code session bakes in the URL it was started with. Restart the server
on a different port or host and the session is over — it cannot be told to look
somewhere else.

The gateway holds one stable address and forwards to whichever backend is
currently healthy. Servers announce themselves by writing a JSON file into the
fleet directory and refreshing its `heartbeat`; the newest healthy one wins
routing, and a heartbeat that stops is how the gateway learns a server is gone.
Nothing else couples the two, so an unhealthy or missing registry costs routing
but never correctness.

That matters most where servers are replaced on a schedule — a batch scheduler
reclaiming an allocation at its wall clock, for instance. `launch.sh` uses
`--no-relay`, which turns off the half of the gateway that submits successor
jobs, since a local run has no scheduler to submit to.

Skip it with `--no-gateway` if you are running one short-lived server and do not
mind restarting the client with it.

## Aggregated vs disaggregated

```bash
./launch.sh --disagg /path/to/your/model
```

Disaggregated mode splits prefill and decode across two workers, each holding a
full copy of the weights — so it needs at least two GPUs, and gains nothing on
one. Use `CTX_GPU` and `GEN_GPU` to choose which.

For the client the two are identical. Internally, `count_tokens` on the
disaggregated server is answered by forwarding to a context worker, since that
server holds no tokenizer of its own and the context worker is what actually
tokenizes prompts.

## Tools

Tool calling needs a tool parser on the **worker**:

```bash
./launch.sh /path/to/your/model -- --tool_parser qwen3
```

Without one the worker never populates `message.tool_calls`, the adapter has
nothing to convert, and the model's tool call comes back as ordinary text with
`stop_reason: end_turn`. Claude Code then reads a finished answer and never runs
the tool — no error anywhere. Pick the parser that matches your model; `auto`
resolves it from the checkpoint for most.

`tool_choice` support is partial: `auto` and `none` work. `any` and a named
`tool` are rejected at request time, because the chat pipeline emits a forced
call without running the tool parser and the arguments arrive empty — an
up-front 400 beats an opaque 500 after generation.

## Extended thinking

`thinking` is passed through to the chat template:

```json
{"thinking": {"type": "enabled", "budget_tokens": 4096}}
```

`enabled` requires `budget_tokens` ≥ 1024 and strictly less than `max_tokens`.
`adaptive` leaves the budget to the server and `disabled` turns thinking off;
neither accepts `budget_tokens`. Whether the budget is honoured depends on the
model's chat template — a model without a thinking mode ignores it.

## Token counting

`count_tokens` counts the fully rendered prompt: system blocks, tool
definitions, and the thinking prefix included. Those are exactly the parts a
client's own estimate misses, and a client sizing its context against a wrong
number either overflows the window or compacts when it did not need to.

Comparing it against a response's `usage` needs care:

```
count_tokens == usage.input_tokens + usage.cache_read_input_tokens
```

`usage.input_tokens` excludes tokens served from the prefix cache, so on a cache
hit the two differ by exactly the cached prefix. That is the documented
Anthropic semantic, not a discrepancy.

## Batches

Batched requests share the engine with interactive traffic, so they are capped
by a semaphore rather than dispatched all at once — otherwise a large batch
starves the users who are waiting on a response.

Batches live in the process. A server restart loses them and their ids then
404. Anthropic's contract is 24h durability, so this is a deliberate deviation:
callers who need results across a restart must re-submit.

## Troubleshooting

**`401` on every request** — `ANTHROPIC_AUTH_TOKEN` is not in the gateway's
users file. `launch.sh` writes your username there; if you set `GATEWAY_USER`,
the token must match it.

**`503` with `overloaded_error`** — the gateway is up but has no healthy
backend. Check `<run-dir>/server.log`; during a handover this is expected and
retrying works.

**`404` on `/v1/models`** — the disaggregated server does not serve that route.
It is an OpenAI-surface endpoint and no Anthropic client needs it, but a wrapper
script that auto-discovers the model id will trip over it. Pass the model id
explicitly.

**Tool calls arrive as text** — no tool parser; see [Tools](#tools).

**Gateway dies when your shell exits** — `nohup` only ignores `SIGHUP`; the
process stays in your shell's process group and goes down with it. Use `setsid`
to give it its own session.

Logs for everything `launch.sh` starts are in the run directory it prints.
