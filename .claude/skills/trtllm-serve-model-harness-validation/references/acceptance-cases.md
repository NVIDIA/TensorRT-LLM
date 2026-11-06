<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Acceptance cases: controls that qualify the validator

Validator-owned. A check that never fires proves nothing. A validator version, identified by
its tree SHA-256, issues verdicts only after its control matrix passes (check V-01).

## Running the matrix

From the repository root:

```bash
python .claude/skills/trtllm-serve-model-harness-validation/scripts/validate/controls.py --out-dir <dir>  # -> <dir>/controls.json
MH_RUN_CONTROLS=1 pytest .claude/skills/trtllm-serve-model-harness-validation/tests
```

`controls.py` builds its own synthetic corpora and control harness profiles, so no trace is
read. The receipt passes only when every case of the full scope (Responses, Anthropic, Chat and
cross) passes, and it lists each case with its result. Run it with the validator's environment
(`scripts/validate/requirements.txt`): the receipt binds the SDK versions as well as the
validator tree.

## 0. How a control is judged

- **Every fault mode declares its expected set.** That is the check it must trip, plus its
  declared consequences. For example, `delta_differs` trips `delta_done_mismatch` and also
  `done_item_mismatch`.
- **A fault passes when it trips its expected set and nothing outside it.** `ok` and the other
  normal controls must trip nothing.
- **Normal controls also run on real traffic.** Use the last known-good window of recorded
  traffic for each protocol.
- **Wording-only tests do not count.** A test that only searches for text or echoes the
  expected status does not qualify a check.
- **After any edit to a check, re-run the whole matrix.** A check can fail every reply of one
  model while the synthetic controls stay green: only a normal control on real traffic shows it.
- **A control is implemented to its documented stimulus and decision.** A control that tests
  another case than its row describes qualifies nothing.
- **A second reader earns its place with faults only it reads.** When C-12 was added, five
  modes (`no_event_lines`, `created_without_output`, `wrong_output_index`, `tool_index_gap`,
  `repeated_call_id`) tripped C-12 and nothing else. Each then became an adapter rule, so their
  expected sets now hold both. Any fault the SDK also reads trips C-12 as well, like
  `delta_before_start`.

## 1. Responses: `scripts/validate/fake_responses_server.py`

`controls.py` runs these through `verdict.py`. The reference checkers still run them with
`scripts/validate/run_controls.sh <corpus.jsonl> <out> [python]`, given recorded or
synthetic Responses requests with tool declarations. The requests are replayed as recorded, so
the `two_calls` row needs at least one with `parallel_tool_calls: false`; without one,
`ptc_violation` is NOT_APPLICABLE and the row reports BAD.

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing | none |
| `overlength` | nothing: a correct `response.failed` with `context_length_exceeded` | none |
| `drop_part_done` | a `content_part.done` is missing | `unpaired_content_part` |
| `snapshot_text_differs` | the snapshot text differs from the stream | `item_content_differs` |
| `delta_differs` | the joined deltas differ from the done event | `delta_done_mismatch`, `done_item_mismatch` |
| `no_terminal` | there is no terminal event | `no_terminal_event` |
| `custom_as_function` | a custom tool is sent as `function_call` | `type_mismatch` |
| `two_calls` | two calls under `parallel_tool_calls=false` | `ptc_violation` |
| `prefixed_name` | a namespace-prefixed name | `prefixed_name_of_declared` |
| `unknown_event` | an event type no rule covers (`response.reasoning_summary_text.delta`) | C-11, WIRE-05 |
| `created_without_output` | `response.created` carries a response without `output` | RSP-01, C-12 (SDK-01) |
| `wrong_output_index` | the message's part and text events carry the reasoning item's `output_index` | RSP-15, C-12 (SDK-01) |
| `usage_missing` | `response.completed` has no `usage` (`ok` reports correct usage) | C-14 (USG-01) |
| `usage_total_mismatch` | `total_tokens` is one more than input plus output | C-14 (USG-02) |
| `error_string` | HTTP 400 whose `error` is a bare string, as main once answered an unknown request field | RSP-16 |

## 2. Anthropic Messages: `scripts/validate/fake_agent_server.py`

The fault mode is chosen per request with the `x-fake-mode` header.

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing; includes `ping` events and one `tool_use` | none |
| `ok_thinking` | nothing; includes a thinking block with `signature_delta` | none |
| `overlength_profile` | nothing: the over-length error in the shape the harness profile requires | none |
| `no_message_delta` | the body ends after `content_block_stop` | ANT-08, ANT-10 |
| `no_message_stop` | there is no `message_stop` | ANT-10 |
| `delta_before_start` | a delta arrives before its `content_block_start` | ANT-02, C-12 (SDK-01) |
| `interleaved_blocks` | block 1 starts before block 0 stops | ANT-03 |
| `delta_type_mismatch` | a `text_delta` inside a `tool_use` block | ANT-04 |
| `bad_partial_json` | the joined `partial_json` is not a JSON object | ANT-06 |
| `duplicate_tool_use_id` | two `tool_use` blocks share an id | ANT-06 |
| `stop_reason_mismatch` | a `tool_use` block with `stop_reason: end_turn` | ANT-09 |
| `two_tools_disable_parallel` | two `tool_use` blocks under `disable_parallel_tool_use` | C-07 |
| `undeclared_tool` | the call names a tool the request did not declare | M-01 (`undeclared`); conformance must stay PASS |
| `invalid_input` | the call's input misses a required property | M-01 (`payload_invalid`) |
| `html_body` | an HTML page instead of SSE or JSON | WIRE-01 |
| `json_for_stream` | a JSON message for a `stream: true` request | WIRE-03 |
| `unknown_block` | a `server_tool_use` block, which no rule covers | C-11, WIRE-05 |
| `no_event_lines` | every frame is a bare `data:` line | ANT-15, C-12 (SDK-01: the SDK sees no `message_start`) |
| `usage_zero_output` | text and a call delivered, `output_tokens: 0` | C-14 (USG-04) |
| `count_tokens_off` | `count_tokens` answers 3 more than the turn billed; every Messages corpus entry is preceded by its count | C-14 (USG-05) |
| `error_untyped` | an error event without `error.type` | ANT-11 |
| `content_after_stop` | an event after `message_stop` | ANT-10 |
| `silent_gap` | no bytes, ping included, for longer than the profile's idle watchdog | R-02 |
| `overlength_400_wrong_wording` | over-length as a 400 whose wording the harness does not recognize | C-10, when the profile requires recognizable wording |

## 3. Chat Completions: `scripts/validate/fake_agent_server.py`

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing; one tool call with fragmented arguments | none |
| `ok_reasoning` | nothing; `reasoning_content` deltas, then content | none |
| `overlength_profile` | nothing: the over-length error in the harness profile's shape | none |
| `no_done` | there is no `[DONE]` | CHT-09 |
| `two_finish_reasons` | a second non-null `finish_reason` | CHT-06 |
| `fragment_without_id` | a tool-call index whose first fragment has no `id` or `name` | CHT-04 |
| `arguments_not_object` | the joined arguments are not a JSON object | CHT-05 |
| `finish_reason_mismatch` | tool calls with `finish_reason: stop` | CHT-07 |
| `duplicate_tool_call_id` | two indices share an id | CHT-04 |
| `two_calls_ptc_false` | two indices under `parallel_tool_calls: false` | C-07 |
| `reasoning_in_content` | think markup in `delta.content` | C-09 (flag), M-03 |
| `model_changes` | `model` differs between chunks | CHT-02 |
| `invalid_input` | the call's arguments miss a required property | M-01 (`payload_invalid`) |
| `html_body` | an HTML page instead of SSE or JSON | WIRE-01 |
| `json_for_stream` | a JSON completion for a `stream: true` request | WIRE-03 |
| `unknown_delta_field` | a `delta.refusal` field, which no rule covers | C-11, WIRE-05 |
| `bad_extension_field` | `avg_decoded_tokens_per_iter` as a string (`ok` sends it as a number, as trtllm-serve does) | CHT-13 |
| `bad_stop_reason` | `stop_reason` as an object (`ok` sends the stop token id, an integer, on the finishing chunk, as trtllm-serve does) | CHT-16 |
| `tool_index_gap` | the first call is announced at index 1 | CHT-04, C-12 (SDK-01) |
| `repeated_call_id` | later fragments repeat the call's `id` and `name` | CHT-04, C-12 (SDK-02: openai-python concatenates them) |
| `usage_cached_exceeds` | `prompt_tokens_details.cached_tokens` above `prompt_tokens` | C-14 (USG-03) |
| `usage_requested_missing` | `include_usage` asked for, no usage chunk | C-14 (USG-01) |
| `two_choices` | nothing: `n: 2`, both choices streamed and finished | none |
| `choice_missing` | `n: 2`, only choice 0 streamed | CHT-14 |
| `choice_without_finish` | `n: 2`, choice 1 never finishes | CHT-06 |
| `error_string` | HTTP 400 whose `error` is a bare string | CHT-15 |
| `silent_gap` | no bytes for longer than the idle watchdog | R-02 |

**Measurement modes** (both servers, not wire faults and not in the control matrix). They measure
what a harness does when the server misbehaves: its timeouts, retries and error mapping
(`references/contract.md` section 4). `--fault-requests N` applies a mode to N requests after the
first `--fault-skip` ones and answers the others `ok`, so the harness is seen recovering; a fault
after a few tool turns gives a compaction something to summarize. `--arguments JSON` is merged
over every call's arguments (a failing command, a large file to read).

| Mode | Server answer | Measured on |
|---|---|---|
| `slow_first_event` | nothing, headers included, for `--gap-seconds` | first-byte timeout |
| `silent_gap`, `silent_gap_pings`, `silent_gap_comments` | silence after the first event; with a `ping` (Messages) or an SSE comment every 5 s | idle timeout; whether keepalives hold it off, and for how long |
| `http_400`, `http_429`, `http_500` | an HTTP error in the protocol's envelope (429 with `retry-after: 1`) | which errors are retried, how often, with what body |
| `overlength_top_level` | over-length in trtllm-serve's own top-level error body (`{"object": "error", "message", ...}`); also `overlength_http_400` (Responses, OpenAI's envelope) | whether the harness reads the wording in that body |
| `cut_mid_stream` | the connection dropped after the first text delta | a cut stream (a recorder in between relays it as a clean end) |
| `no_finish` (Chat) | a clean end after the content, without `finish_reason` or `[DONE]` | a stream that ends without its terminal |
| `length_with_call` | a call, then `finish_reason: length`, `stop_reason: max_tokens` or `response.incomplete` | what the harness does with calls in a reply cut at its token limit |
| `undeclared_tool`, `invalid_input` (Chat and Responses added) | a call the harness cannot run | the harness's outcome strings |

The Responses server answers a request that declares no tool without a call, as a harness's
summary request needs.

## 4. Cross-protocol cases

`controls.py` implements X-01 to X-06, X-08 to X-10 and X-13 to X-128. X-07 needs H-05, and
X-11 and X-12 need the loop's invalidation log and phase ownership; they are not implemented
yet.

Build these as fixture traces with an adapter-produced next turn. No server is needed.

| Case | Stimulus | Expected decision |
|---|---|---|
| X-01 echo mismatch | The harness's next request carries a different payload under our call id | C-03 FAIL |
| X-02 echo lossless | The next request re-serializes our JSON arguments with different spacing and key order | C-03 PASS |
| X-03 pending at cutoff | A delivered call has no result before the window's cutoff | I-08 counts it as pending; not a failure |
| X-04 outcome strings | Tool results carry each `outcomes` string in the harness profile | H-01 classifies each correctly; an unknown string becomes `result` with a count of unmatched strings |
| X-05 unknown outcome profile | The harness profile has no `outcomes` | H-01 is BLOCKED, not PASS |
| X-06 vacuous turn | The reply is only whitespace and EOS, with no call | M-06 counts it. A reply with an empty text but a valid call is not vacuous |
| X-07 compaction placeholder | The harness's compaction request is answered with an unexecuted tool tag in place of a summary | H-05 flags it |
| X-08 client disconnect | The reply has no terminal event and the transport status is a client disconnect | C-04 is not failed; H-03 attributes it |
| X-09 degenerate stop | A `[PAD]` run is stopped and the reply fails with no output: Kimi-K3, a Messages stream that opens and ends with an error event; its raw record, written when the request ended (`partial`, `aborted`, `end_reason`), holds 25 `[PAD]` ids | R-04 PASS, `stopped` 1; M-02 counts the stop; `runtime` and `conformance` PASS, C-13 `stopped_raw` 1 |
| X-09b stopped after some text | The same stop after the stream delivered a text block, which the client drops with the failed reply | R-04 PASS, `stopped` 1; M-02 counts the stop |
| X-10 rate bound | 0 events in 20 sessions against a 1% threshold | BLOCKED (insufficient n), not PASS |
| X-11 invalidated assurance | A control proves an old check skipped a case; the hashes are unchanged | The dependent PASS verdicts are recorded as invalidated |
| X-12 stale owner | Two loop controllers act on the same ready stage | Only one acts; the stale owner's write is refused |
| X-13 raw equal | The reply delivers exactly what the GLM-format raw record says | C-13 PASS over 1 joined reply |
| X-14 call lost | The raw record has a call; the reply delivers none | C-13 FAIL (RAW-01) |
| X-15 arguments rewritten | The delivered call's argument differs from the written one | C-13 FAIL (RAW-02) |
| X-16 content rewritten | The delivered text lost a character the model wrote | C-13 FAIL (RAW-03) |
| X-17 model broke its format | The raw call is never closed | M-07 flagged; C-13 does not compare the reply: `joined` 1, denominator 0 |
| X-18 first call only | Two written calls under `disable_parallel_tool_use`; the first is delivered | C-13 PASS (policy) |
| X-19 no raw output | The same trace without `--raw` | C-13 NOT_APPLICABLE |
| X-20 ambiguous engine id | Two raw records share one `disagg_request_id` | Nothing joins; C-13 BLOCKED |
| X-21 text echo equal | Messages: signed thinking and text come back verbatim at the appended position | C-03 PASS, one text echo equal |
| X-22 text echo differs | The echoed text lost a character | C-03 FAIL (`field: text`) |
| X-23 signature dropped | The echoed thinking lost its signature | C-03 FAIL (`field: reasoning`) |
| X-24 echo policy unknown | The harness profile's `echo.text` is TODO | C-03 BLOCKED |
| X-25 Chat text echo | The echoed assistant `content` differs | C-03 FAIL |
| X-26 server prompt count | The trace's prompt count differs from the billed input | C-14 FAIL (USG-07) |
| X-27 declared offset | Kimi-K3 over Messages: the trace's prompt count is the billed input plus 3 | C-14 PASS |
| X-28 engine output count | The raw record counts 7 output tokens; the reply billed 1 | C-14 FAIL (USG-06) |
| X-29 next prompt shrank | The turn after an echoed call bills 1 input token where the turn billed 50 | C-14 FAIL (USG-08) |
| X-30 history cleared on request | The same, with `context_management` on the next request | C-14 not tripped |
| X-31 protocol differential, equal | One turn over Chat and over Messages: same request, billed input and reply | `protocol_diff.py` exit 0, `equal` |
| X-32 protocol differential, render | The same, but Messages billed 9 input tokens and Chat 7 | exit 1, `server_render` |
| X-33 protocol differential, harness | The two requests carry different system prompts | exit 0, `harness`: nothing is attributed to the server |
| X-34 protocol differential, divergence | Same request and billed input, other text from the first character | exit 2, `divergent_generation`: two generations, not a protocol path |
| X-35 protocol differential, conversion | The same text delivered as reasoning on one path and as content on the other | exit 1, `server_output` |
| X-36 protocol differential, reasoning divergence | Two turns. Turn 1 generates other reasoning per path; turn 2 replays each path's own and is billed differently | exit 2, turn 2 `after_divergence`: the bill follows the replayed reasoning |
| X-37 protocol differential, harness replay | Two turns. Turn 1 agrees; the harness replays other reasoning over one path on turn 2, billed differently | exit 0, turn 2 `harness` |
| X-38 protocol differential, render after replay | Two turns. Everything replayed agrees and turn 2 is billed differently | exit 1, turn 2 `server_render` |
| X-39 declared rewrite, schema default | Messages: the echoed Edit adds `replace_all: false`, its declared default; the profile declares `schema_defaults` | C-03 PASS, one `echo_rewritten` |
| X-40 undeclared rewrite | The same echo under a profile that declares no rewrite | C-03 FAIL |
| X-41 added value is not the default | The echoed Edit adds `replace_all: true` | C-03 FAIL |
| X-42 declared rewrite, cd prefix | We delivered `cd /w && ls`; the echo is `ls`; the profile declares `cd_cwd_prefix` | C-03 PASS, one `echo_rewritten` |
| X-43 cd added by the harness | We delivered `ls`; the echo is `cd /w && ls`. The cut is one-sided | C-03 FAIL |
| X-44 cd into another directory | We delivered `cd /w && ls`; the echo is `cd /v && ls` | C-03 FAIL |
| X-45 unknown rewrite | The profile declares a rewrite the validator does not know | the pipeline refuses the profile (no silent pass) |
| X-46 raw join by message id | An aggregated server's raw record has no orchestrator id; the Anthropic reply is named `msg_5_<hex>` | C-13 PASS over 1 joined reply |
| X-47 answered twice | The same request is answered twice (a retry after the first answer was lost in transit); the next turn echoes the second | C-03 PASS, one superseded answer, one text echo equal |
| X-48 asked again after an empty turn | The harness drops an empty answer and asks again at the same position, with a text block appended to its last user message (Claude Code); the next turn echoes the second answer | C-03 PASS: one superseded answer, one of them re-asked, one text echo equal |
| X-49 changed, not asked again | As X-48, but the second request changes the last user message instead of appending to it | C-03 FAIL: the first answer is still judged, and its echo differs |
| X-50 the recorder's own replies | The recorder answers 502 while the server is down, and loses another stream mid-reply (`upstream_error`) | C-01 and C-04 not failed; R-06 FAIL, count 2 |
| X-51 a lost request | A request has no response record at all | inventory BLOCKED, one `I-01:request_without_response` |
| X-52 the recorder stopped | A request's record says `recorder_shutdown` | pending at the capture's end: not judged, inventory not blocked |
| X-53 a declared capability rejection | The server rejects `thinking.display` with the wording the harness profile declares; the harness retries without the field | C-16 FLAGGED with the field, conformance PASS, no H-02 |
| X-54 an undeclared rejection | The same 400 with a wording the profile does not declare | C-16 FAIL |
| X-55 an operator probe | A request carries `x-mh-probe` | counted as a probe, not a reply of the cell |
| X-56 a rejected `count_tokens` | `count_tokens` answered 400 with an undeclared wording | C-16 FAIL |
| X-57 cut before the first byte | Against a 30 s first-byte cutoff, one client gave up after 35 s and one after 5 s, both before the reply | R-01 FAIL for the first only |
| X-58 cuts attributed | The same two cuts | H-03 FLAGGED, count 2 |
| X-59 an answer in the next hour | A window's last request is answered in the next hour's shard | without `--responses` inventory BLOCKED; with it PASS, and a response there for another request is ignored |
| X-60 a blind degenerate check | A clean reply, judged with a model profile that names no degenerate tokens, under a lenient threshold | M-02 BLOCKED, not PASS |
| X-60b tokens named by id only, no raw ids | The same reply, judged with GLM's profile (ids only) and no `--raw` | M-02 BLOCKED (by id only), not PASS |
| X-61 a degenerate check that can see | The same reply with a profile that names them (K3) | M-02 PASS |
| X-62 parallel sub-agents at one position | Two sub-agents of a session ask with different first messages and the same last one; the second's echo arrives first | C-03 PASS, both texts echo equal (found by call ids) |
| X-63 a side request before the echo | A small request without tools comes between a turn and the request that sends its call back, which grew | No USG-08; one pair |
| X-64 compaction | The request that sends the call back replaced the history with a summary and bills less | No USG-08; `next_input_shrank_by_compaction` 1 |
| X-115 a compaction request without tools | The request that sends the call back keeps the whole history but declares no tools (Codex's compaction summary request) and bills less, by less than its tools' declarations take | No USG-08; `next_input_shrank_by_tools_change` 1 |
| X-115b the same with the turn's tools | Same tools, whole history, smaller bill | USG-08 |
| X-115c one tool fewer, a bill far smaller | Responses: the turn declares `exec_command` and `view_image` and bills 1,000; the request that sends the call back keeps the whole history, declares `exec_command` only and bills 20 | USG-08 (`next_input_shrank_beyond_removed_tools`) |
| X-115d one tool fewer, the bill smaller by its declaration | The same, billing less by exactly the UTF-8 bytes of `view_image`'s declaration | No USG-08; `next_input_shrank_by_tools_change` 1 |
| X-115e one token more | The same, billing one token less again | USG-08 |
| X-115f other tools, not fewer | The request declares `exec_command` and `write_stdin` instead, and bills one token less | USG-08 (`next_input_shrank`) |
| X-115g no tool left | The turn declares two functions and a namespace of two; the request declares none and bills less by every declaration's bytes (the namespace's fields with each of its tools) plus 512 | No USG-08; `next_input_shrank_by_tools_change` 1 |
| X-115h one token more | The same, billing one token less again | USG-08 |
| X-116 GLM markup inside a value closed once | GLM raw `<arg_key>status</arg_key><arg_value>A</arg_key><arg_value>B</arg_value>`; the delivered value is `A</arg_key><arg_value>B` | M-07 (`markup_in_arg_value`); C-13 does not compare it: `not_judged` 1, denominator 0 |
| X-116b the same slip closed twice | `<arg_value>A</arg_value><arg_value>B</arg_key><arg_value></arg_value>` | M-07 (`text_between_args`), as before; not compared |
| X-117 a compaction summary on Responses | A turn with a call; a request with the whole history, one user message appended and no tools, answered with reasoning and a summary; then a request that starts over from the summary | `text_not_echoed_compacted` 1, `reasoning_not_echoed_compacted` 1, nothing missing |
| X-117b the summary carries a call | The same, the summary reply also carrying a call that no later request sends back or answers | Labelled compacted as in X-117; the call is `echo_missing` 1 and `I-08:unanswered` 1 |
| X-118 every reply in the C-13 record | Responses: one reply joins its raw record; the client cut the second before its end, and the hook wrote no raw record for it; the third is an HTTP 503 | C-13 PASS over 1, `replies` 3 = `joined` 1 + `unjoined` 0 + `client_cut` 1 + `http_error` 1; the reason names the cut |
| X-119 Codex's compaction, measured profile | Under the hosted Codex profile (`profiles/harness/codex-hosted.yaml`): a turn (reasoning with its carrier, a call; two tools); the compaction request (whole history, `tools: []`, `parallel_tool_calls: false`), billed 100 less, answered with reasoning, a summary and a call; the next request starts over | conformance PASS: C-07 PASS over 1 single-call reply, `next_input_shrank_by_tools_change` 1, the carrier echoed equal, the summary `text_not_echoed_compacted` 1, the dropped call `echo_missing` 1 |
| X-120 pi's `edit`, one edit object | Under `pi-chat`: the reply calls `edit` with `edits` as one edit object; the next request sends it back as `[edit]`, as pi keeps it | C-03 PASS, `echo_rewritten` 1 |
| X-120b `edits` as a JSON string | `edits` delivered as a JSON string of an array, sent back as the array | C-03 PASS, `echo_rewritten` 1 |
| X-120c a legacy pair | `oldText` and `newText` at the top level, sent back as delivered (pi's fold makes a new object, which its history does not keep) | C-03 PASS, `echo_rewritten` 0 |
| X-120d another edit sent back | One edit object delivered, `[edit]` sent back with another `newText` | C-03 FAIL |
| X-120e the rewrite not declared | X-120's traffic under a pi profile without `echo.tool_input` and `tools.tolerated_payload` | C-03 FAIL; M-01 `payload_invalid` 1 |
| X-121 a call pi prepares into a valid one | X-120's call: invalid as written (`edits` is not an array), valid once prepared | M-01 `payload_invalid` 0, `tolerated` 1 |
| X-121b a call still invalid once prepared | `edits` an object without `newText`, not one edit | M-01 `payload_invalid` 1, `tolerated` 0 |
| X-121c a legacy pair | `oldText` and `newText` at the top level, no `edits`: valid once folded into `edits` | M-01 `payload_invalid` 0, `tolerated` 1 |
| X-122 Kimi-K3 over Responses, the whole prompt billed | The trace's prompt count equals the billed input (the Responses route bills the whole prompt) | C-14 PASS; `usage_vs_trace_prompt` 1 |
| X-122b the same, 3 less billed | The prompt count is the billed input plus 3, as the Chat route bills | C-14 FAIL (USG-07) |
| X-122c Kimi-K3 over Chat | The prompt count is the billed input plus 3 | C-14 PASS, as on Messages (X-27) |
| X-123 a duplicate key, pi's re-serialization | Under `pi-chat`: a call's arguments repeat a key; the next request sends back JSON of the last value, as pi does | C-03 PASS, `echo_rewritten` 1; M-01 `payload_invalid` 1 (`arguments_not_strict_json`), not tolerated |
| X-123b the first value sent back | The same arguments, sent back with the repeated key's first value | C-03 FAIL |
| X-123c a raw newline in a string | Arguments with a raw newline inside a string, sent back with it escaped (pi's repair) | C-03 PASS, `echo_rewritten` 1 |
| X-123d arguments pi cannot read | Arguments that are not JSON, sent back as `{}` | C-03 PASS, `echo_rewritten` 1 |
| X-123e the rewrite not declared | X-123 under a pi profile without `pi_json_reserialize` | C-03 FAIL |
| X-124 a long output, then bash's abort | Under `pi-chat`: a result of 12,000 characters of output, then pi's `Command aborted` | H-01 `aborted` 1 (a rated class) |
| X-124b a long output, then its exit status | The same, ending with `Command exited with code 1` | H-01 `tool_error` 1 |
| X-124c an exit status inside a long output | `Command exited with code 1` in the middle of 24,000 characters that end otherwise | `result`: the line is not where pi writes it |
| X-124d a harness's word inside a long result | Claude Code profile: `InputValidationError` in the middle of a long tool result | `result`, not `arguments_rejected`: Claude Code writes it at the start of its own rejection |
| X-125 a stop string before the finish | A Chat chunk carries `stop_reason` before the chunk that finishes the choice | C-01 FAIL (CHT-16) |
| X-125b the stop token id where it belongs | `stop_reason` null on earlier chunks and 163586 on the finishing one | nothing tripped |
| X-125c a list | A non-streamed body whose choice carries `stop_reason: [2]` | C-01 FAIL (CHT-16) |
| X-126 the same examples in either order | Two shards of five empty turns each, read by two workers, merged in either order | The same five M-06 examples, those that come first by file and line |
| X-127 the same value, other bytes | Chat, a profile that declares no re-serialization: the arguments come back compact | C-03 PASS, `echo_reserialized` 1, `echo_equal` 0 |
| X-127b the same under pi's profile | `pi-chat` declares `pi_json_reserialize` | C-03 PASS, `echo_rewritten_by` `{pi_json_reserialize: 1}`, `echo_equal` 0 |
| X-127c byte for byte | The arguments come back as delivered | `echo_equal` 1 |
| X-127d a Messages echo | The `input` object comes back with the delivered value (the delivered `partial_json` had other spacing) | `echo_equal` 1: a Messages echo is compared by value |
| X-128 NaN in Chat arguments | `{"path": NaN}` delivered | C-01 FAIL (CHT-05); M-01 `payload_invalid` (`arguments_not_strict_json`) |
| X-128b Infinity in a Messages input | `partial_json` `{"file_path": Infinity}` | C-01 FAIL (ANT-06) |
| X-128c a number beyond a double | `{"path": "x", "n": 1e400}`: a JSON number | No CHT-05 |
| X-65 clearing that removes nothing | The echo bills less and carries `clear_thinking` with `keep: all` | USG-08 |
| X-66 clearing that may remove history | The echo bills less and carries `clear_tool_uses` | No USG-08 |
| X-67 a request without thinking (V4.1) | Raw output "OK" with no `</think>` for a Messages request without `thinking` | C-13 PASS on 1 reply, no M-07 |
| X-68 the same output with thinking on | The same raw output for a request with `thinking: adaptive` | M-07 (unclosed think) |
| X-69 checks that tested nothing | One turn: no request limits calls or sets tool_choice, nothing over length | C-07, C-08, H-04 NOT_APPLICABLE |
| X-70 the same checks with something to test | A named tool_choice with at most one call, answered by one call | C-07 PASS, C-08 PASS, H-04 NOT_APPLICABLE |
| X-71 rates per session | 160 clean turns from 2 sessions, 2% thresholds, `unit_for_rates: session` | M-03, M-06 BLOCKED (n = 2) |
| X-72 rates per turn | The same with `unit_for_rates: turn` | M-03, M-06 PASS (0 of 160), `sessions` 2 in the record |
| X-73 an unknown unit | The same with `unit_for_rates: turns` | BLOCKED, not guessed |
| X-74 under known lower bounds | First event 0.5 s, longest silence 5 s, against `{at_least: 10}` and `{at_least: 30}` | R-01 PASS, R-02 PASS |
| X-75 a silence past a lower bound | A 34.5 s silence against `idle_s: {at_least: 30}` | R-02 BLOCKED (the cut is unknown) |
| X-76 the same silence, exact cut | The same against `idle_s: 30` | R-02 FAIL |
| X-77 a first event past a lower bound | A first event at 12 s against `first_event_s: {at_least: 10}` | R-01 BLOCKED |
| X-78 an abort is an abort | Claude Code's rejection tool_result (`is_error: true`) under a profile that names `aborted` | H-01:aborted counts 1, not PASS |
| X-79 a rated outcome the profile cannot see | The same traffic under a profile without `aborted` | H-01:aborted BLOCKED (declares no `aborted`), claim BLOCKED |
| X-80 an outcome declared absent | `aborted: {none: reason}` | H-01:aborted NOT_APPLICABLE with that reason |
| X-81 nothing to compare | Anthropic streams (no final snapshot) | C-02 NOT_APPLICABLE |
| X-82 a signature echoed with another value | Thinking delivered with one signature, echoed with another | conformance FAIL (C-03) |
| X-82b the same signature | Echoed with the delivered value | conformance PASS |
| X-83 a run the server kept out of the text | Raw output with special tokens kept and 25 pad tokens; delivered text without them | M-02 counts 1 of 1 |
| X-84 a raw text that cannot show the run | Raw output without special tokens | M-02 denominator 0, `raw_unobservable` 1, BLOCKED |
| X-85 a Responses reasoning carrier echoed as delivered | `encrypted_content` sent back by item id, same value | C-03 PASS, `reasoning_echo_equal` 1 |
| X-86 a carrier echoed with another value | Same item id, another `encrypted_content` | C-03 FAIL (field `reasoning`) |
| X-87 a carrier dropped | Same item id, no `encrypted_content` | C-03 FAIL (field `reasoning`) |
| X-88 reasoning echo not declared | `echo.reasoning: TODO` | C-03 BLOCKED, `reasoning_echo_unchecked` 1 |
| X-89 a reasoning item left out | The next request has the call but not the reasoning item | C-03 PASS, `reasoning_echo_missing` 1 |
| X-90 a Responses reply joined by its header | `engine_request_id` recorded from `x-trtllm-engine-request-id`, a raw record with that `request_id` | C-13 PASS over 1 joined reply |
| X-90b the same reply without it | No engine id anywhere in the reply | C-13 BLOCKED (nothing joined) |
| X-90c a disaggregated Responses reply named `resp_<disagg_request_id>_<hex>` | One raw record with that `disagg_request_id`, another whose engine `request_id` is the same number | C-13 PASS over 1 reply, joined to the record with the `disagg_request_id` |
| X-90d the usual `resp_<hex>` name | No id in the name, no header | C-13 BLOCKED (nothing joined) |
| X-90e a disagg id in the name, an engine id in the header | The reply is `resp_9001_<hex>` and its header names engine id 5; engine id 5 names one record, another request's, and disagg id 9001 names this reply's | C-13 PASS over 1, joined to the record with the `disagg_request_id`: a disagg id comes before any engine id |
| X-90f the name's number is never an engine id | `resp_9001_<hex>`, no header; no record carries disagg id 9001, and one record's engine id is 9001 | Nothing joins: C-13 BLOCKED, `unjoined` 1, no RAW-* |
| X-91 an argument the harness tolerates | `exec_command` with an undeclared argument; profile `tools.tolerated_payload: [additionalProperties]` | M-01:payload_invalid 0, `tolerated` 1 |
| X-91b the same without the tolerance | Same call, profile without it | M-01:payload_invalid 1 |
| X-91c tolerated only when every violation is | Undeclared argument and a wrong type | M-01:payload_invalid 1 |
| X-92 a snapshot status that differs | Streamed message `completed`, snapshot `incomplete` | RSP-11 (C-02) |
| X-93 a snapshot carrier that differs | Reasoning `encrypted_content` differs between stream and snapshot | RSP-11 (C-02) |
| X-94 a gap in sequence numbers | One number skipped | RSP-04 |
| X-94b numbers not from 0 | The stream starts at 1 | RSP-04 |
| X-95 a keepalive that says something new | A repeated `response.in_progress` with another status | RSP-17 |
| X-95b an equal keepalive | Identical apart from `sequence_number` | nothing tripped |
| X-96 a claim BLOCKED by a check names it | M-01 BLOCKED for insufficient n | `claim_blockers` lists `M-01:payload_invalid: ...` |
| X-97 a text no later request echoes | One reply, no next turn | C-03 `text_no_next_turn` 1 |
| X-98 a run seen on token ids | Raw record with `token_ids` holding 25 pad ids; decoded text without special tokens | M-02 counts 1 of 1, not `raw_unobservable` |
| X-98b token ids without a run | Raw record with `token_ids`, no run | M-02 counts 0 of 1 |
| X-99 a call opened in the reasoning, the owed close after it | GLM raw `Plan.<tool_call>Read...</tool_call></think>`; delivered thinking `Plan.` and the call, as serving repairs it; GLM's profile declares `owed_think_close` | C-13 PASS over 1 compared reply, `repair:owed_think_close` 1, no M-07 |
| X-99b the same, the repair not declared | The same reply under a GLM profile without `owed_think_close` | M-07 (`call_in_reasoning`); C-13 does not compare it: denominator 0, `not_judged` 1 |
| X-99c a call in the reasoning and no close | Raw `Plan.<tool_call>Read...</tool_call>`, GLM's profile | M-07 (`unclosed_think`), as before; not compared, no repair noted |
| X-100 nothing compared | X-17: the only joined reply has a model format anomaly | C-13 BLOCKED (none was compared), not PASS over 0 of 1 |
| X-100b some compared | Two joined replies: one equal to its raw record, one with a model format anomaly | C-13 PASS over 1, `joined` 2, `not_judged` 1, and the reason says 1 of 2 was not compared |
| X-101 thinking echoes are reasoning echoes | X-21: a signed thinking block and a text come back verbatim | `reasoning_echo_equal` 1, `text_echo_equal` 1 |
| X-102 a side request's reply | A structured-output request (`output_config.format`, no tools) comes between a turn and the request that sends it back; its reply is never appended | `text_not_echoed_side_request` 1, `text_echo_missing` 0, C-03 PASS |
| X-103 a compaction summary | The reply to a compaction request (thinking and text, no call); the next request starts from a summary, with a shorter history | `text_not_echoed_compacted` 1, `reasoning_not_echoed_compacted` 1, nothing missing |
| X-103b a sub-agent's last reply | Followed by its parent's next request, which is shorter than the sub-agent's history but continues the parent's conversation | `text_echo_missing` 1, not compacted |
| X-104 an outcome flag the profile does not declare | Claude Code traffic under a measured profile without `unavailable_in_mode` | H-01:unavailable_in_mode NOT_APPLICABLE with that reason; the declared H-01:unsupported PASS |
| X-104b the same, the profile not measured | The same profile with `status: partial` | H-01:unavailable_in_mode BLOCKED |
| X-105 an engine id two workers share, told apart by time | The reply's window runs from 00:00:10Z (request) to 00:00:20Z (response); its message id carries engine id 5, which names a record on each of two generation workers, both with the reply's output count (1). One ran inside the window, starting 0.75 s before the request's second (inside the 2 s skew); the other, another request's text, ran minutes later | C-13 PASS over 1 compared reply, `resolved_ambiguous_id` 1; C-14 `usage_vs_engine_output` 0 (a resolved join is not compared again) |
| X-105b the only record in the window, another count | The record inside the window, another request's text, wrote 9 tokens where the reply reports 1; the other ran outside the window | Nothing joins: C-13 BLOCKED, `unjoined_ambiguous_id` 1, no RAW-* |
| X-105c two records fit | Both records ran inside the window and wrote 1 token | Nothing joins (fail closed): C-13 BLOCKED, `unjoined_ambiguous_id` 1, no RAW-* |
| X-105d an engine id naming one record | The only record under engine id 5 ran minutes after the reply | It joins without a window check, as before: C-13 PASS over 1, `resolved_ambiguous_id` 0, `usage_vs_engine_output` 1 |
| X-105e a request for two choices | Chat `n: 2`: one record per choice under engine id 5 (1 token each, 2 reported), and another worker's record inside the window that wrote 2 tokens | Nothing joins: C-13 BLOCKED, `unjoined_ambiguous_id` 1, no RAW-* |
| X-106 a reply named after the orchestrator's id | The reply is `msg_9001_<hex>`; the only record has `disagg_request_id` 9001 and engine id 3 | C-13 PASS over 1 compared reply |
| X-106b the engine id comes first | The reply's id 5 is the engine id of one record and the `disagg_request_id` of another (another request's text) | The engine id joins, as before: C-13 PASS over 1 |
| X-107 a run delivered in the text | Kimi-K3: 25 `[PAD]` in a delivered text block, `end_turn` | R-04 FAIL; M-02 counts it |
| X-107b a run delivered as an answer cut at its budget | The run is only in the raw ids; the reply delivers what decoding left and ends with `max_tokens` | R-04 FAIL; M-02 counts it |
| X-107c a run delivered as a finished answer | The same run in a generation that then ended; the reply ends with `end_turn` | R-04 FAIL; M-02 counts it |
| X-108 an aborted generation, a prefix delivered | The raw record (`aborted: true`) wrote `Reading.`; the reply completed with `Read` | C-13 PASS over 1 compared reply (RAW-05 holds), `stopped_raw` 1 |
| X-108b the same reply, a finished generation | The same raw record without `aborted` | C-13 FAIL (RAW-03), as before |
| X-108c more than a stopped generation wrote | A partial record (the hook's `partial`, `aborted`, `end_reason`) wrote `Reading.`; the reply completed with `Reading. More.` | C-13 FAIL (RAW-05), not RAW-03 |
| X-109 records stamped, no revision declared | Replies carry `build_id` b1; `--build-id` not given | inventory BLOCKED, the reason names `--build-id`; `revision`: declared null, not checked, b1 seen once |
| X-109b an empty revision | The same with `--build-id ''` | inventory BLOCKED (`--build-id` empty); `revision` declared "" |
| X-109c the declared revision | The same with `--build-id b1` | inventory PASS; `revision` checked |
| X-109d another revision | The same with `--build-id b2` | inventory BLOCKED (not matching), one reply with another build id |
| X-109e nothing stamped, nothing declared | Replies without `build_id`, no `--build-id` | inventory PASS, as before; `runtime` BLOCKED (R-05) |
| X-110 the evidence a verdict read | A snapshot (`trace/`, `raw/`, `MANIFEST.sha256`); the same copied elsewhere; the copy with one byte added | `evidence_shards` lists each shard with its role and SHA-256; `evidence_sha256` is the documented digest, the same for the copy and another for the changed copy; the manifest is recorded with its SHA-256 |
| X-111 over-length read by its text | A Chat profile with `errors.overlength.shape: any_error` (patterns and exclusions, as pi reads errors); a 400 in OpenAI's envelope says "maximum context length is 32768 tokens" | C-10 PASS over 1 over-length reply, no C-16 |
| X-111b the same words, trtllm-serve's top-level body | `{"object": "error", "message": ..., "code": 400}` | C-10 PASS |
| X-111c words the harness does not read | "The prompt exceeds the context length of this model.": over-length to the validator, matching none of the harness's patterns | C-10 FAIL |
| X-111d an excluded wording | "Rate limit reached: too many tokens per minute.": a pattern matches, and so does an exclusion | C-10 FAIL |
| X-111e over-length in the stream | A Responses `response.failed` with `context_length_exceeded` under `any_error` | C-10 PASS |
| X-111f a pattern that does not compile | `matches: ["(unclosed"]` | exit 3, no verdict |
| X-112 a resend after a capacity answer | A 429, then the identical request | H-02 PASS; counted as `H-02:retry_after_failure` |
| X-112b a resend after a client error | A 400 that is not over-length, then the identical request | H-02 FAIL |
| X-113 pi's results over Chat | Under the measured `pi-chat` profile: one reply with five calls, and the next request with pi's measured results for them (Chat has no error flag) | H-01 counts 1 each of `aborted`, `unsupported`, `arguments_rejected` and `tool_error` |
| X-113b pi's results under every pi profile | Each measured string, the source-only ones (`Operation aborted`, bash's `Command aborted`, `Invalid arguments for tool`, Node's `ENOENT`, the edit tool's `Could not find the exact text in ...`), classified by `pi-chat`, `pi-responses` and `pi-messages` (with `is_error`) | Every string lands in its class; a plain result, and one that only looks like an error (`Example: ...`), is `result` |
| X-114 pi reads OpenAI's wording | `pi-chat`, a 400 "maximum context length is 32768 tokens" | C-10 PASS |
| X-114b trtllm-serve's own wording | `pi-chat`, a 400 "The prompt exceeds the context length (32768)." in trtllm-serve's top-level body | C-10 FAIL: pi would end the run, not compact |
| X-114c pi over Messages | `pi-messages`, a 400 "prompt is too long: 40000 tokens > 32768 maximum" | C-10 PASS |
| X-114d pi over Responses | `pi-responses`, an in-stream `response.failed` with `context_length_exceeded` | C-10 PASS |

## 5. Calibration with seeded bugs (Step 3)

**Trace-level seeded faults.** `scripts/validate/mutate_traces.py` copies recorded Responses
replies whose single call is valid, and gives each copy one fault in the shape of a known serving
bug, under a 0700 directory removed at exit:
- a custom tool delivered as `function_call`, and a namespace-prefixed name (C-06);
- a second call under `parallel_tool_calls=false` (C-07);
- a terminal snapshot whose text differs from the stream (C-02);
- a lost terminal event (C-04).

Every copy must be caught by its own check, and the unmutated sample must trip nothing. See
`validate.md` section 8.

**Code-level seeded bugs (open).** Revert the fix, rebuild, and serve.

Revert one serving fix at a time on a throwaway branch, rebuild the evidence, and confirm
detection. A miss is a validator defect.

| Fix reverted | Expected detection |
|---|---|
| Bare and qualified tool names | C-06 `type_mismatch` or `prefixed_name_of_declared`; H-01 `aborted` in a Codex smoke run |
| `service_tier` accepted | Replies with HTTP 400 for the Codex corpus; H-02 resends |
| In-stream `context_length_exceeded` | C-10 FAIL; H-04 FAIL |
| Keepalive and early-open | R-01 / R-02 against the harness profile's cutoffs, with a slow fake engine |
| `single_call_stop` | C-07 FAIL with a two-call model sample |
| Snapshot equals stream | C-02 FAIL |
| Degenerate stop | R-04 FAIL with a `[PAD]` run sample |

Not every one of these fixes is in main yet: calibrate on a build that carries them (`<ref>`).
The open PR #19883 carries some of them onto main.
