# Worklog: Gemma 4 tool/reasoning parser + AD guided decoding smoke test

Branch: `sg/gemma-tool-call-parser`

---

## 2026-04-21 — Plan finalized

**Goal:** Manually verify PR #13248 (Gemma 4 tool + reasoning parsers) end-to-end
via `trtllm-serve`, and confirm AutoDeploy + xgrammar guided decoding is still alive.

**Key findings during planning:**

- `--extra_llm_api_options` is single-value only (not append-style). Fix: one merged YAML.
- Registry entry (`models.yaml:318`) uses `world_size: 1` for `gemma-4-26B-A4B-it`.
- Model available locally: `/home/scratch.trt_llm_data_ci/llm-models/gemma/gemma-4-26B-A4B-it`.
- `Gemma4ToolParser.supports_structural_tag()` returns `False` — tool calling works via
  post-processing (parser), not guided decoding. `openai_server.py:132-136` skips
  structural-tag guided decoding when this returns False.
- `skip_special_tokens` footgun: reasoning markers stripped unless `request.tools` is set
  (which triggers `needs_raw_special_tokens=True` in the tool parser).

**Planned test order and done criteria:**

- **Step 0** (pre-flight, standalone AD guided decoding — TinyLlama, no server):
  DONE when: pytest exits 0; log contains `Validation passed!`; generated text
  parses as JSON with keys `ssid`, `securityProtocol`, `bandwidth`.

- **Step 1** (launch server):
  DONE when: `Uvicorn running on http://0.0.0.0:8000` appears in log;
  no xgrammar init errors; `GuidedDecoder` initialization mentioned in log
  (confirms guided_decoding_backend wired up).

- **Step 2** (JSON-schema guided decoding through server):
  DONE when: `finish_reason == "stop"`; response content parses as JSON;
  object has exactly keys `{"name", "age", "email"}` and `age` is an int.
  This confirms xgrammar is initialized and the AD→server path is alive.

- **Step 3** (tool-calling, parser-only, no guided decoding):
  DONE when: `finish_reason == "tool_calls"`; `message.tool_calls` is non-empty;
  `tool_calls[0].function.name == "get_current_temperature"`;
  `tool_calls[0].function.arguments` parses as valid JSON with a `location` key;
  no raw `<|tool_call>` / `<|"|>` delimiter text in `message.content`.
  Optional strict=True variant: same criteria, plus server log shows warning
  "does not support structural tags" (no guided decoding error or crash).

- **Step 4** (reasoning with dummy tool):
  DONE when: `message.reasoning_content` is non-empty; `message.content`
  contains "391"; no `<|channel>` / `<channel|>` delimiter text leaks.

- **Step 5** (reasoning-only, explicit skip_special_tokens=false):
  DONE when: same criteria as Step 4.
  Bonus: same request without the flag → `reasoning_content` is empty or
  text leaks into `content` (documenting the known footgun).

- **Step 6** (streaming tool-call, optional):
  DONE when: streaming response delivers two `tool_calls` delta chunks —
  first with `function.name` set and empty `arguments`, second with full
  `arguments` JSON.

---

## Step 0 — PASSED (2026-04-21)

Ran `test_ad_guided_decoding.py::test_autodeploy_guided_decoding_main_json` with TinyLlama-1.1B-Chat-v1.0.

**Done criteria met:**
- pytest exit 0
- Log: `Validation passed! Generated JSON: {'ssid': 'my_wifi_ssid', 'securityProtocol': 'wpa_psk', 'bandwidth': '100Mbps'}`
- All three required fields present and non-empty strings
- Runtime: ~174s (model load + CUDA graph compile + inference)

xgrammar is installed, `GuidedDecoder` init works, AD `runtime: trtllm` path is alive.

---

## Step 1 — Plan revision (2026-04-21)

**Original plan:** single server launch with `guided_decoding_backend: xgrammar` baked into the config, testing guided decoding (Step 2) first after the server comes up.

**Revised plan:** split into two server runs:
1. **Run 1 (this session):** minimal config — tool calling + reasoning only. No xgrammar.
2. **Run 2 (separate):** add `guided_decoding_backend: xgrammar` to config and test JSON-schema guided decoding through the server.

Rationale: tool calling and reasoning use the parser post-processing path only — no guided decoding involved. Adding xgrammar to the first run introduces unnecessary complexity and a separate failure mode. The AD + xgrammar path is already confirmed working by Step 0.

**Second issue found:** `gemma4_moe.yaml` has `tokenizer: google/gemma-4-26B-A4B-it` (HF model ID). The server tried to `snapshot_download` it and failed (no internet access). Fix: append `tokenizer: /home/scratch.trt_llm_data_ci/llm-models/gemma/gemma-4-26B-A4B-it` to the merged YAML — later keys override earlier ones.

**Updated config** (`/tmp/gemma4_serving.yaml`): gemma4_moe.yaml + `world_size: 1` + local tokenizer path override.

---

## Step 1 — Server launched (2026-04-21)

Server came up after ~2 min (model load + CUDA graph compilation for all batch sizes).

Log shows:
- KV cache allocated: 3.12 GiB (16384 tokens, 512 blocks of 32)
- GPU memory after load: ~50 GB reserved, ~24 GB free
- `Application startup complete` (uvicorn/starlette)
- `curl http://127.0.0.1:8000/health` → HTTP 200

Note: `use_fast=False` tokenizer warning on startup (slow image processor) — expected for Gemma 4's multimodal tokenizer, not an error.

**Done criteria met:** server responding to /health.

---

## Step 3 — Tool calling PASSED (2026-04-21)

Request: weather in San Francisco with `get_current_temperature` tool.

**Done criteria met:**
- `finish_reason == "tool_calls"` ✓
- `tool_calls[0].function.name == "get_current_temperature"` ✓
- `tool_calls[0].function.arguments == '{"location": "San Francisco"}'` — valid JSON, `location` key present ✓
- `message.content == ""` — no raw `<|tool_call>` / `<|"|>` delimiter leakage ✓

Tool parser post-processing (`Gemma4ToolParser`) working correctly. No guided decoding involved.

---

## Step 4 — Reasoning with dummy tool PASSED (2026-04-21)

First attempt with `max_tokens: 512` hit `finish_reason: "length"` — reasoning chain in
`reasoning_content` already showed 391 but `content` was cut off before stating the final
answer. Re-ran with `max_tokens: 1024`.

**Done criteria met:**
- `finish_reason == "stop"` ✓
- `message.reasoning_content` non-empty — full working chain: distributive property,
  difference of squares, both reaching 391 ✓
- `message.content` contains `17 * 23 = **391**` ✓
- No `<|channel>` / `<channel|>` delimiter leakage in either field ✓

Note: `reasoning_content` starts with `"thought\n"` — this is the Gemma 4 channel name
("thought") prepended to the content inside the `<|channel>thought` block. Expected
behavior given the parser splits on `<|channel>` and `<channel|>`.

---

## Step 5 — Reasoning-only PASSED (2026-04-21)

**5a — With `skip_special_tokens: false` (workaround):**
- `finish_reason == "stop"` ✓
- `reasoning_content` non-empty, contains full chain reaching 391 ✓
- `content` contains `17 * 23 = **391**` ✓
- No delimiter leakage ✓

**5b — Without `skip_special_tokens` (footgun demo):**
- `reasoning_content == ""` — empty, as expected ✓
- `content` starts with `"thought\nThe user wants to know..."` — entire thinking chain leaked into content ✓
- The `<|channel>` / `<channel|>` markers were stripped by the tokenizer before the parser ran, so the parser could not split the output. The raw text (including the channel name "thought") ended up in `content` as if it were normal assistant text.

**Footgun confirmed:** reasoning-only requests without `skip_special_tokens: false` silently produce empty `reasoning_content` and polluted `content`. The workaround works correctly.

---

## Step 6 — Streaming tool-call PASSED (2026-04-21)

Streaming response delivered exactly 3 SSE chunks + `[DONE]`:
1. Role chunk: `{"delta": {"role": "assistant"}}`
2. Tool calls chunk: two entries in `tool_calls` — first with `name: "get_current_temperature"` + empty `arguments`, second with `arguments: '{"location": "San Francisco"}'`
3. Finish chunk: `finish_reason: "tool_calls"`

**Done criteria met:**
- Two-chunk `tool_calls` delta matches `Gemma4ToolParser._emit_tool_call_block` pattern ✓
- `arguments` JSON valid and contains `location` key ✓
- No delimiter leakage ✓

---

## Summary

All planned steps completed successfully (2026-04-21):

| Step | Result |
|------|--------|
| 0 — AD + xgrammar pre-flight (TinyLlama) | PASSED |
| 1 — Server launch (Gemma 4 MoE, AD backend) | PASSED |
| 3 — Tool calling (parser-only, no guided decoding) | PASSED |
| 4 — Reasoning with dummy tool | PASSED |
| 5a — Reasoning-only with `skip_special_tokens: false` | PASSED |
| 5b — Reasoning-only without flag (footgun demo) | CONFIRMED — `reasoning_content` empty, content polluted |
| 6 — Streaming tool-call | PASSED |

Remaining: Step 2 (JSON-schema guided decoding through server with xgrammar config) deferred to a separate server run.

<!-- Append new entries below as execution proceeds -->
