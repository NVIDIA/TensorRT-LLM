<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation: claims, checks, verdicts

Validator-owned. Changing this file is a policy change (see `SKILL.md` section 2).

## 0. Where the rules come from

- **A data-validation gate catalogue** (gates G01 to G20): a claim and verdict algebra, and a
  loop and evidence contract. Only the gates that fired on serving evidence carry over here;
  section 4 lists the ones that do not apply to a cell, such as evaluation and training.
- **Serving verdict rules:** three claim levels, native-text-first attribution, the deployment
  check, window discipline.
- **What is new here:**
  - wire rules for three protocols (`protocols.md`);
  - a second reading of every reply by the official SDKs, compared with ours (C-12);
  - one turn record that every check reads (`turn-ir.md`);
  - per-harness outcome classes;
  - rate rules with intervals;
  - two tiers and exit codes, so an agent loop can stop on them.
- **How the validator itself is qualified** (section 8): its control matrix, the mutation gate
  on its properties, calibration on recorded traces of real deployments, and full tiers that
  re-derive the numbers independently.

## 1. Claims

A verdict is issued per cell, per claim and per evaluated population (a frozen window, a replay
run or a smoke run). Each claim states its population, its unit and its denominators.

| Claim | Question | Verdict style |
|---|---|---|
| `inventory` | What was evaluated, what is missing, and which code served it | Reconciliation. Missing data is stated, never "complete" |
| `conformance` | Did trtllm-serve speak this harness's protocol correctly, for each request's own declarations? | Hard: any confirmed violation is FAIL |
| `runtime` | Is the running process the code under test, and does it behave correctly under overload, over-length and silence? | Hard, using controls with normal and adverse cases |
| `model-behavior` | With serving correct, how often does the model misbehave in this harness? | Rate against a threshold (section 3) |
| `harness-outcome` | Does the harness accept our replies: no aborts, rejections or resends caused by us? | Rate against a threshold, plus hard zero for resends caused by our error mapping |
| `task-outcome` (L4, optional) | Did tasks succeed? | Needs a qualified oracle (gates G09/G10); reference agreement alone is not task correctness |
| `comparison` (optional) | Is A better than B: models, harnesses, or us against a vendor API? | Comparable populations, with selection and dependence handled (gate G18) |

**Healthy serving is not a good model, and a good model is not task success.** These are
separate claims, and none of them substitutes for another.

## 2. Verdict algebra

- **Per claim:**
  - FAIL wins over everything;
  - otherwise, missing required evidence is BLOCKED;
  - PASS needs every applicable check to have run over a stated denominator;
  - NOT_APPLICABLE needs a reason based on an absent feature, e.g. no request set
    `parallel_tool_calls=false`.
- **Three claim levels, never merged:**
  - *flagged*: a screen fired;
  - *content change proven*: two views of one generation differ, or a replay reproduces the
    difference;
  - *failure proven*: a consumer-side causal chain exists, e.g. the harness answered `aborted`
    to the delivered call.

  Report each level with its own denominator.
- **Attribution starts from the native text.** If the anomaly is already in the raw generation,
  the model wrote it, and serving can only be guilty of making two views diverge or of rewriting
  bytes. Unknown causes stay unknown, and correlation is not attribution.
- **Corrections are additive.** Published verdicts are never edited. New evidence that disproves
  an assurance invalidates every verdict that depended on it (gate G20). Record the invalidation,
  then re-check or BLOCK the affected claims.

## 3. Rates and thresholds

- **Thresholds come from the run's matrix file and are set by the human before the run.** This
  skill sets no universal quality percentage. An unknown threshold makes the claim BLOCKED, with
  the measurement reported.
- **The interval.** For `k` events in `n` units, compute the one-sided 95% Wilson interval:
  - PASS when its upper bound is at most the threshold;
  - FAIL when its lower bound exceeds the threshold;
  - otherwise BLOCKED (insufficient n).
- **Zero observed is not a zero rate.** With 0 events in n independent units, the 95% upper bound
  is about 3/n:
  - claiming below 0.1% needs about 3,000 units;
  - a defect of one in hundreds of thousands of turns needs S5 scale.
- **Units.** The matrix's `thresholds.unit_for_rates` sets the unit for every rate check:
  - `session`: sessions with at least one event, out of all sessions;
  - `turn`: each check's own unit, a reply (one model output) or, for call-level checks
    (M-01, H-01), a call. It fits the classes this skill judges: a tool-parser,
    reasoning-parser or protocol defect shows in the output where it happens, so every output
    is an occasion for it;
  - any other value is BLOCKED, never guessed.
- **What a turn-level bound leaves open.** The interval treats its units as independent. Turns
  of one session share their context, tools and conditions, so a defect that only some sessions'
  conditions trigger clusters in those sessions, and the turn-level bound is then optimistic.
  Every rate record therefore also carries `sessions` and `sessions_with_events`, and the
  variety of conditions still comes from the number and spread of sessions.
- **Semantic review.** For claims judged by sampling, such as "compaction summaries are
  substantive" (gate G12), record the rubric, strata, sample IDs and seed. A model-judge score is
  not ground truth.

## 4. Gates mapped to checks

IDs: `I` inventory, `C` conformance, `R` runtime, `M` model behavior, `H` harness outcome,
`V` validator closure. The "Now" column says for which protocols `scripts/validate/verdict.py`
implements the check today ("all" = Responses, Anthropic Messages and Chat Completions).

| ID | Gate | Check | Verdict rule | Now |
|---|---|---|---|---|
| I-01 | G01 | Joins: request ↔ response ↔ raw ↔ next turn. Ids above 2^53 are compared as decimal strings | Unjoined records are counted. A required join that is missing gives BLOCKED_PARTIAL: `verdict.py` blocks inventory and names the partial join (a request without any response record). For a window, pass the next hour's response shards with `--responses`: they answer the window's requests only, and `--requests` boundary requests are never counted as lost. A `recorder_shutdown` record is pending at the capture's end, not missing. Operator probes (header `x-mh-probe`, or the older `x-mh-selftest`) are counted and left out of the cell. So are requests of another protocol than the cell's (a gateway's health probes, another client), counted per protocol: they are not the harness's traffic, nor its resends. Ids that name several raw records are counted (`I-01:raw_ambiguous_id:r` engine ids, `:d` disagg ids); a reply under one joins as C-13 says, or not at all (`unjoined_ambiguous_id`) | all |
| I-02 | G01 | Counts reconciled per unit (requests, streams, calls, turns, sessions) against the writer sidecars | Mismatch is BLOCKED until explained | planned |
| I-03 | G01 | Freeze manifest present and verified (full tier) | Missing gives BLOCKED | planned |
| I-04 | G01 | Credential scan: digests and counts only, including `NAME=value` secrets in environment dumps (`secret_assignment`: names with TOKEN, SECRET, PASSWORD or API_KEY, also after a JSON-escaped newline) | NEW digests are reported to the human and the contexts held | all (scan_credentials.py) |
| I-05 | G02 | `build_id` on every record, and equal to the declared revision (`--build-id`) | Missing or mixed gives BLOCKED for the claims that name a revision. Replies that carry a `build_id` while no revision is declared (`--build-id` not given, empty or blank) give inventory BLOCKED, the reason naming `--build-id`: nothing was checked against them. `verdict.json` records `--build-id` as given and the build ids the replies carry (`revision`) | all |
| I-06 | G02 | Cell identity resolvable: model profile and template hash, harness profile and version, protocol | Unresolved gives BLOCKED | partial (profiles bound) |
| I-07 | G02 | Capture boundary declared: recorder, request trace, replay or mirror | Missing gives BLOCKED | planned |
| I-08 | G05 | Every delivered call is answered next turn, or pending at the cutoff | Unanswered calls are counted; pending is not a failure | all |
| I-09 | G06 | Sessions, threads and sub-agents grouped by the harness profile's identity keys | Ungroupable gives BLOCKED for session-level rates | all |
| I-10 | G17 | Strata coverage: tool kinds, effort levels, context buckets, sub-agents, images, errors, compaction | An empty stratum BLOCKS any claim that needs it | planned |
| C-01 | G03 | Wire rules for the protocol (`protocols.md`) | Any violation is FAIL | all |
| C-02 | G03 | Stream equals the final snapshot item by item: ids, text, names, arguments, input | FAIL. N/A where the protocol has no streamed snapshot (Anthropic, Chat) | Responses (N/A elsewhere) |
| C-03 | G03 | Echo: what the harness sends back next turn equals what we delivered (call id, name, payload, text, reasoning). Calls are found by call id, Responses texts by item id. Messages and Chat texts of a turn with calls are found in the first later request whose history holds those call ids (parallel sub-agents of one session can share a position and a previous message, and a compacted history moves the turn); texts of a turn without calls at the position the reply was appended to (the message before it must be the one we answered) | FAIL when we delivered something other than what the harness recorded. Texts and reasoning are compared where the harness profile's `echo` says `exact`, a reasoning carrier by its value where the protocol gives one (Anthropic's thinking signature, a Responses reasoning item's `encrypted_content`, found by item id); a `TODO` there makes C-03 BLOCKED for echoed texts and reasoning items. A reasoning item the next request leaves out counts as `reasoning_echo_missing`, as a text does as `text_echo_missing`; a Messages or Chat reply counts once per kind, its text blocks under `text_*` and its thinking or reasoning blocks under `reasoning_*`. Two non-echoes are by design and have their own labels: `*_not_echoed_side_request`, the reply to a side request (Messages structured output, `output_config.format`, without tools, e.g. Claude Code's titles and classifiers, never appended to the conversation), and `*_not_echoed_compacted`, a reply none of whose calls a later request sends back, after which the conversation went on from a summary (the next request of the thread, side requests aside, has a shorter history, as USG-08 reads a compaction, and continues no earlier request: the reply to a compaction request is that summary, and a call in it is dropped with it). The labels hold on every protocol; a Responses request names no position, so there the shorter history decides (e.g. Codex gives a sub-agent a thread of its own). A text or reasoning item delivered in a reply that no later request follows counts as `text_no_next_turn` / `reasoning_no_next_turn`, as a call does as `no_next_turn`, so the record adds up to what was delivered. Call arguments: on Responses and Chat, where the arguments travel as a string, an echo of the same bytes counts as `echo_equal`; the same JSON value in other bytes (spacing, key order, escapes) passes too, as `echo_reserialized`, or as `echo_rewritten:pi_json_reserialize` where the profile declares that the harness re-serializes; on Messages the echo is an object, so its JSON value is compared (`echo_equal`). A value that differs passes only under the rewrites the profile lists in `echo.tool_input`, each counted as `echo_rewritten:<rewrite>` (`echo_rewritten` is the total; the record's `echo_rewritten_by` breaks it down) (`schema_defaults`: an argument at its declared default may be added or left out; `cd_cwd_prefix`: a leading `cd <dir> &&` we delivered may be cut; `pi_edit_normalize`: the arguments we delivered for pi's `edit` tool are taken as pi keeps them, `edits` given as one edit object becoming `[edit]` and given as a JSON string becoming what it parses to; `pi_json_reserialize`: the arguments we delivered are read as pi reads them before it sends them back, with JavaScript's JSON.parse (a duplicate key keeps its last value), pi's repair of a raw control character or stray backslash in a string, and `{}` for arguments it cannot read; `contracts/pi.md` section 3). A value pi did not change in that reading still has to come back unchanged. An unknown rewrite name fails the run. A request answered more than once (re-sent after a cut, or retried by a proxy) is judged on its last completed answer; the earlier ones count as `superseded_answers`. So does an answer the harness asked again for: a later request at the same position with blocks appended to its last user message and nothing else changed (e.g. Claude Code after an empty turn; also counted as `reasked_answers`) | all |
| C-04 | G03 | Exactly one terminal per reply, nothing after it | FAIL, except client disconnects, which are reported under H-03 | all |
| C-05 | G03 | Ids stable across stream, final, raw and echo | FAIL | all (present and unique) |
| C-06 | G04 | Calls against the request's own declarations, serving-shape classes: `type_mismatch`, `namespace_mismatch`, `prefixed_name_of_declared` | FAIL at more than 0 | all |
| C-07 | G04 | Parallel policy: `parallel_tool_calls=false` or `disable_parallel_tool_use` gives at most one call | FAIL at more than 0; still FAIL if the model omitted a close tag (the serving policy for that case is still open). NOT_APPLICABLE when no request asked for at most one call | all |
| C-08 | G04 | `tool_choice` honored: none, required/any, named | FAIL at more than 0. NOT_APPLICABLE when no request set one of them | all |
| C-09 | G08 | Reasoning and content channels separated: no think markup in content, no answer text in the reasoning channel | Flagged, then proven against raw | all (flag) |
| C-10 | G19 | Over-length input answered the way the harness profile's error mapping requires. `errors.overlength.shape`: `in_stream` (a failed terminal event with the profile's `code`, e.g. Codex), `http_400` (an HTTP 400 with the profile's error type and wording, e.g. Claude Code), or `any_error`: an error the harness reads by its text, whatever its status or whether it came in the stream (an HTTP error, or an error or failed terminal event). The text is the status, the error's code and type and its message; it must match one of the profile's `matches` regexes and none of its `unless` regexes, compiled case-insensitively (e.g. pi). A regex that does not compile is a usage error | FAIL on the wrong shape (e.g. HTTP 400 where Codex needs an in-stream `response.failed`, or a wording pi does not read) | all |
| C-11 | G03 | Coverage: every protocol element in the reply has a rule (WIRE-05) | An unchecked element makes conformance BLOCKED, never PASS | all |
| C-12 | G03 | Independent reading: the official SDK (`anthropic`, `openai`) reads the same bytes, and its text, reasoning, calls, order, stop, ids and usage equal the adapter's (`protocols.md` section 5) | Any disagreement is FAIL. An SDK that is not installed makes it BLOCKED | all |
| C-14 | G19 | Usage (`protocols.md` section 6): present and typed, totals add up, details within their wholes, output reported when output was delivered (USG-01 to 04); `count_tokens` equals the input the same request billed (USG-05); the reported output equals the engine's count (USG-06) and the input the server's prompt count less the offset the model profile declares for the reply's protocol (USG-07; e.g. Kimi-K3's 3-token opener is left out on the Chat and Messages routes, not on Responses); the request that sends an echoed call back (found by the call ids, not the next request in time: a side request of the session can come first) bills no less input (USG-08), unless its history is shorter than the turn's (the harness compacted: `next_input_shrank_by_compaction`), its `context_management` can remove history (clearing thinking with `keep: all` cannot), or it declares strictly fewer of the turn's tools and bills less by no more than they could take (`next_input_shrank_by_tools_change`; e.g. Codex asks for its compaction summary with the whole history and no tools). Strictly fewer: its tool names (a namespace's tools as `<namespace>.<name>`, a declaration without a name by its type) are a strict subset of the turn's, none at all included; other tools are not fewer ones. What they could take, in tokens: the UTF-8 bytes of the JSON of every turn declaration the request does not carry as it was (dropped, or changed: a changed one may have shrunk), each namespace tool counted with the namespace's own fields (e.g. DeepSeek V4.1 repeats its description per tool), since a template renders a declaration from its own fields and every token spans at least one byte; plus 512 when no tool is left, for the template's tools preamble (heading, instructions, call format), which goes with the last tool (e.g. GLM-5.3's is 96 tokens, DeepSeek V4.1's 211). A larger shrink is USG-08 (`next_input_shrank_beyond_removed_tools`) | FAIL on any violation. Rules with no data are listed in the record's `not_exercised` | all (USG-05 needs count_tokens traffic, USG-06 `--raw`, USG-07 a trace's prompt count, USG-08 a next turn) |
| C-15 | G03 | Protocol differential: one harness and task over several wires, greedy decoding, requests made equal (`smoke.md` section 3a). Per turn: the same request must bill the same input and get the same reply on every protocol path | FAIL on `server_render` or `server_output`; a turn whose requests differ is not judged | `protocol_diff.py` (separate exit code), smoke runs over at least two APIs |
| C-16 | G03 | The server accepts what the harness sends, `count_tokens` included: a 4xx other than over-length (C-10) and capacity (429) is a rejection | FAIL. A rejection whose wording the harness profile declares in `errors.capability_retries` (the harness drops that capability and retries) is FLAGGED with its field instead: each costs a round trip | all |
| C-13 | G03 | Raw view: what was delivered is what the model wrote, its raw record read in its own format (`turn-ir.md` section 4). RAW-01 calls, RAW-02 arguments, RAW-03 content, RAW-04 reasoning, RAW-05 a stopped reply that is not a prefix of what was written. The raw record of a generation the server stopped (the hook's `partial: true` with its `end_reason`, for a request that ended in an error, or `aborted: true`) holds only what the model wrote until then, so a reply joined to one is held to RAW-05 whatever its terminal says; such joins are counted as `stopped_raw`. Accepted: the first call under `parallel_tool_calls=false`, and repairs the model profile declares (`native.repairs`, each noted as `repair:<name>`). For example, GLM declares `qualified_name_suffix`, and `owed_think_close`: a call opened inside the reasoning ends it there, and the first `</think>` outside a call, the one the reasoning still owed, is dropped; a profile that does not declare it reads that shape as the model's anomaly `call_in_reasoning`. A record is read in the mode its request asked for: a model profile with `native.prompt_opens_without_thinking` (e.g. V4.1: `content`) has a Messages request without `thinking` read from the content, not as an unclosed think (`read_without_thinking`) | FAIL on any other difference. NOT_APPLICABLE without raw output; replies without a raw record are counted as not covered. The denominator is the replies compared: a joined reply whose reading has a model format anomaly (M-07) is not compared (`not_judged`, next to `joined`), and when no joined reply was compared C-13 is BLOCKED. The record places every reply: `joined` (compared, or `not_judged`), `unjoined`, or never held against a raw record: `client_cut` (cut by the client before its end, so incomplete by construction; the hook may write no raw record for its aborted generation; the reason names them), `http_error` (nothing generated was delivered), `recorder_lost` (R-06) and `pending` (at the capture's end). With raw output they add up to `replies`. A reply is joined to its raw record by its ids, in this order: the orchestrator's `disagg_request_id` as the trace recorded it; the one a disaggregated server names a Responses reply after (`resp_<id>_<hex>`; read only as a `disagg_request_id`, since no server names a Responses reply after an engine id, and main's `resp_<hex>` names none); the engine id the reply header `x-trtllm-engine-request-id` names, which the recorder keeps as `engine_request_id`; the engine id in a Chat or Messages reply's id (`chatcmpl-<id>`, `msg_<id>_<hex>`), and then that number as a `disagg_request_id`. A `disagg_request_id` comes before any engine id: it is unique across the instance, and an engine id is not. The first of these ids that names exactly one raw record joins it, whatever its times. An engine id counts the requests of one server process, so with several generation workers it names a record on each; the reply then joins the one record not ruled out by evidence, and only if that record's times are known: its start and finish must lie in the reply's window, from its request's `recorded_at` to its response's `finished_at` (or the arrival plus `timing.total_s`), widened on each side by `RAW_CLOCK_SKEW_S` (2 s, clock skew between hosts; a whole-second stamp counts as its whole second), and its output token count must equal the reply's where both are known. Such joins are counted as `resolved_ambiguous_id`, and USG-06 does not compare them again. None or several left, a request for several choices, or a `disagg_request_id` naming two records: nothing joins (`unjoined_ambiguous_id`, part of `unjoined`) | all (needs `--raw`) |
| R-01 | G19 | First-event latency against the harness's first-byte cutoff. A client that gave up before the first byte after waiting at least the cutoff counts too: the cutoff passed. The profile gives the cutoff exactly, or as a measured lower bound (`{at_least: s}`: that long is known to survive, in runs before the one judged; a bound taken from the judged run itself would pass it by construction) | Rate over replies. FAIL if any reply exceeds an exact cutoff without a keepalive. Against a lower bound, a reply that waits at least that long makes R-01 BLOCKED: the cut itself is unknown | all (needs timing) |
| R-02 | G19 | Longest silent gap between events against the idle watchdog, given exactly or as a measured lower bound. Needs per-frame timestamps. Keepalive pings count as events, so a stream that pings more often than the bound stays under it whatever the cut is | Same as R-01 | all (needs per-chunk timing) |
| R-03 | G19 | Error-mapping controls: overload, unsupported field, invalid tool choice, mid-stream failure | Per the acceptance cases | planned |
| R-04 | G19 | A degenerate run does not reach the client as an answer | FAIL if the run reached the client: in what was delivered, or in the generation of a reply delivered as an answer (completed, or cut at its token budget: decoding drops the run's special tokens, and the client keeps a truncated reply). A run only the raw record shows, in a reply that ended without an answer (an error, a failure, no terminal event), did not reach the client: M-02 counts it, and the R-04 record counts it under `stopped`, not as a failure (a reply without a terminal event fails C-04 on its own) | all |
| R-05 | G19 | Process-level identity of the deployment | Missing gives BLOCKED | all (build_id presence) |
| R-06 | G19 | The server answered every request it was sent: the recorder reached it and read each reply to its end (no record with `status: upstream_error`) | FAIL at more than 0. Such a reply is the recorder's own error, or the server's bytes up to the loss, so no conformance check judges it | all (recorder captures) |
| M-01 | G04 | Model call classes: `undeclared`, `nested_tool_called_directly`, `payload_invalid`. Arguments that are not strict JSON (a repeated key, `NaN`, `Infinity`) are `payload_invalid` (`arguments_not_strict_json`), never tolerated. A payload invalid only in ways the harness profile tolerates (`tools.tolerated_payload`, JSON-schema validator names, e.g. Codex ignores an undeclared argument, so its profiles list `additionalProperties`) is `payload_tolerated`: counted in the M-01:payload_invalid record as `tolerated`, not rated, since the harness runs the call and what it tolerates is not serving's to control; the harness still judges such a call through H-01 when it rejects it. `tools.tolerated_payload` may also name a repair the harness makes before it runs a call, `pi_edit_normalize` (pi's preparation of its `edit` tool's arguments, which also folds a legacy `oldText`/`newText` pair into `edits`): a payload invalid as written is tolerated only when the repaired one is valid | Rate, after a raw check | all |
| M-02 | G12 | Degenerate runs (model profile `degenerate`) | Flag; rate. BLOCKED, never PASS, when the model profile names no degenerate tokens: only the default pattern, 20 or more `[PAD]` in a row, is looked for. With `--raw`, the raw text is read too, and only replies that could show the tokens count: a raw record that kept special tokens (its stop marker is there), or one where a run was found; the others are reported as `raw_unobservable`. A raw record that carries `token_ids` is read on the ids (entries with an `id`, `last_vocab_id` resolved from the profile's `vocab_size`), which show a run whether or not decoding kept the special tokens (`id_range: [a, b]` names a block of ids). A profile that names its tokens only by id is BLOCKED without raw records, since no text can show them. A run counts whether or not it reached the client: the model wrote it (R-04 judges where it went) | all (profile) |
| M-03 | G12 | Markup leaks (model profile `markers`), quoted vs bare | Flag; bare hits proven against raw | all (profile) |
| M-04 | G08 | Reasoning round-trip: the harness echoes our carrier (encrypted content, signature) and the next render uses it per the model profile | Flag. The policy is per cell | planned |
| M-05 | G12 | Repetitive reasoning: zlib ratio below 0.10 over at least 2,000 chars | Flag | all |
| M-06 | G12 | Empty turn: no text, no call, no error. EOS-only is not an action | Rate | all |
| M-07 | G12 | The model broke its own format in the raw generation: an unclosed call or channel, a malformed name, a value not of its declared type, a second think close, a call opened inside the reasoning that the model profile declares no repair for, the format's own argument markup inside an argument value (e.g. GLM: `</arg_key>` or `<arg_value>` in a value closed once, `markup_in_arg_value`; with a second `</arg_value>` it is `text_between_args`) (`native.py` anomalies) | Flag; the reply's differences are the model's, not serving's, and C-13 does not compare it | all (needs `--raw`) |
| H-01 | G05 | Outcome of each call from the next request (harness profile `outcomes`). A profile's regexes and substrings are read in a result's first and last 4,096 characters (`profiles.py` `OUTCOME_WINDOW`), where harnesses write them: an error's own message at the start, a status line at the end of a command's output (e.g. pi's bash status line, after output pi keeps up to 50 KB). So they are read whatever the result's length, and a long result is not scanned whole; prefixes and exact strings are read on the whole result | Rate of `aborted` and `arguments_rejected` against the threshold. Each case traced to raw. A rated outcome the profile does not name is BLOCKED (such a call would land in another class); one declared absent with `{none: reason}` is NOT_APPLICABLE. The other outcomes are flags. A flagged outcome the profile does not name cannot fire: NOT_APPLICABLE when the profile is `measured` (it names every outcome its harness reports), BLOCKED when it is not | all (profile outcomes) |
| H-02 | G05 | Identical resends | Hard zero for resends caused by our error mapping: after an over-length answer or another client error. A capacity answer (429) asks the client to come back, so a resend after it is a retry, counted with those after a server error or a failed stream as `H-02:retry_after_failure`, not judged | all |
| H-03 | G05 | Disconnects, each attributed to the client, a timeout or the server | Unattributed gives BLOCKED. Client-side cuts are flagged with their status; the server's are R-06. A reply its client cut before the terminal event, a disconnect before any byte included, is incomplete by construction: no wire, call, content or usage rule judges it, and its timing still counts (R-01, R-02). A client that closed after the terminal event cut nothing, and its reply is judged | all (by status; a flag record) |
| H-04 | G07 | Over-length reaches the harness's compaction path | FAIL if the harness resends instead. NOT_APPLICABLE when no over-length reply was followed by a next turn | all |
| H-05 | G07 | Compaction requests answered with substantive summaries | Sampled semantic review | planned |
| V-01 | G20 | The validator tree hash has a passing control receipt (`acceptance-cases.md`) | Without one, no verdict counts | all (controls.py receipt) |
| V-02 | G20 | The report binds the input manifest, profile hashes and validator hash | Missing gives BLOCKED | partial (profiles, tree hash, the evidence read, the matrix and the declared `--build-id`; a `MANIFEST.sha256` beside the evidence is recorded, not verified) |
| V-03 | G20 | Invalidations consumed before a saved PASS is trusted | — | planned |

Gates that do not apply to a cell:
- G09–G11 apply only to `task-outcome` and performance claims;
- G13–G15 are about training;
- G16 applies only to replay-corpus de-duplication and held-out L4 tasks;
- G18 applies only when a `comparison` claim is requested.

## 5. Tiers

| | Fast | Full |
|---|---|---|
| Who runs it | Adapter (pre-check) and the loop, every iteration | Validator, at stage exits |
| Input | The iteration's evidence directory, live | Frozen copy with `MANIFEST.sha256` |
| Runs | `scripts/validate/*` with the cell's profiles | Its own re-derivation of key numbers from raw bytes, plus the scripts, plus controls |
| Time | Minutes | Hours |
| Output | `verdict.json`, exit code | A report: `run-manifest.json`, `validation-report.json`, `findings.jsonl`, `SUMMARY.md`, and a `verdict.json` |
| Counts for stage exit | No | Yes |

An independent re-derivation writes fresh scripts for each batch of evidence, and its reports
can run to gigabytes. That belongs in the full tier, and only at stage exits.

## 6. Outputs

**`verdict.json`** has one file per cell and population:

```json
{
  "schema": "trtllm-serve-model-harness-validation/verdict@1",
  "cell": {"model": "deepseek-v41", "harness": "claude-code", "protocol": "anthropic-messages"},
  "tier": "fast",
  "population": {"kind": "smoke", "paths": ["<snap>/trace"],
                 "evidence_sha256": "...", "evidence_bytes": 2048,
                 "evidence_shards": [{"role": "population", "path": "<snap>/trace/requests-x.jsonl",
                                      "bytes": 1024, "sha256": "..."}],
                 "manifests": [{"path": "<snap>/MANIFEST.sha256", "sha256": "..."}]},
  "revision": {"declared_build_id": "1.4.0rc2+0123456789", "checked": true,
               "replies_with_build_id": 1834, "replies_without_build_id": 0,
               "replies_with_other_build_id": 0, "build_ids_seen": {"1.4.0rc2+0123456789": 1834}},
  "claims": {"conformance": "PASS", "model-behavior": "BLOCKED", "harness-outcome": "NOT_APPLICABLE"},
  "checks": [
    {"id": "C-07", "gate": "G04", "claim": "conformance", "status": "PASS",
     "count": 0, "unit": "reply", "denominator": 1834, "interval": null,
     "evidence": ["calls.json#ptc_violation"], "not_applicable_reason": null}
  ],
  "validator": {"tree_sha256": "...", "sdk_versions": {"anthropic": "2.1.0", "openai": "2.48.0"},
                "controls_receipt": "controls.json", "controls_passed": true},
  "profiles": {"model_sha256": "...", "harness_sha256": "..."},
  "thresholds_from": "matrix.yaml@<sha256>"
}
```

- **What it judged (V-02).** `evidence_shards` lists every shard the passes read, with its role
  (`population`; `requests` and `responses`, the boundary hours; `raw`), size and SHA-256 as
  stored. `evidence_sha256` is the SHA-256 of one line `<role> <shard sha256>` per shard, sorted,
  each ending in a newline: it names the bytes judged in their roles, wherever they lie, and
  changes with any byte. A `MANIFEST.sha256` in a given directory or the one above it (a frozen
  snapshot's root) is recorded with its own SHA-256; it is not checked against the shards (I-03).
- **Which thresholds.** `thresholds_from` is the `--matrix` file as given, `@` the SHA-256 of the
  bytes its thresholds were read from (null without `--matrix`).
- **Which code it checked (I-05).** `revision` holds `--build-id` as given (null when not given,
  "" when empty), whether the replies were checked against it, and the build ids they carry.
  Replies that carry one while none is declared make inventory BLOCKED (section 8 says what to
  pass).

**Exit codes** (`verdict.py`):

| Code | Meaning |
|---|---|
| 0 | Every requested claim is PASS or NOT_APPLICABLE |
| 1 | Any requested claim is FAIL |
| 2 | Otherwise, any requested claim is BLOCKED |
| 3 | Usage or internal error; no verdict |

**`report.json`** holds the counts, rule breakdowns and, per check, up to five examples: the five
that come first by file, line and trace id, so the same evidence gives the same report however its
shards were split among worker processes.

**Gate records** (full tier) carry the gate catalogue's fields: `gate_id`, `catalogue_version`,
`evaluated_population`, `required_for_claim`, `applicability_reason`, `status`,
`check_revision`, `evidence_refs`, `failure_count`/`unit`/`denominator`, `unresolved_evidence`
and `limitations`. A missing count is unknown, not zero.

## 7. Evidence discipline

- **Freeze before you look.**
  - Use closed UTC hours, plus the preceding hour as join-only context.
  - Copy the writer sidecars, then the shards, then hash the copies.
  - Never hash live files and copy later.
- **Late partners.** A record whose partner landed after the window gets a line-level look-up
  in later closed hours: file, line, line SHA-256, file size and file SHA-256. Mark it as
  receipt-only.
- **Membership filters fail closed.** An unknown, missing or conflicting tag excludes the
  record.
- **No content in reports.**
  - Locators are `(file, line, trace_id[, item_id])`.
  - Texts are compared by length, first differing offset and SHA-256.
  - Excerpts stay off by default.
- **Regression evidence.** Only an executable failure on the old code followed by a pass on the
  new code counts. Name tests by node id, and run each `-k` selection separately.
- **Test the seam, not the halves.** A fix can pass the tests of both halves and break their
  join: a stable-call-id fix once broke nearly every stream at the join between stream assembly
  and the snapshot rebuild.

## 8. Implementation

**Entry point: `scripts/validate/verdict.py`.** It handles all three protocols and runs the
three passes in `pipeline.py`:
1. **requests:** one small summary per request, plus the first appearance per thread of every
   echoed item and tool result;
2. **replies:** each reply joined to its request and parsed by its adapter (`proto_responses.py`,
   `proto_anthropic.py`, `proto_chat.py`, built on `turnlib.py`), then read again by the
   protocol's official SDK (`sdk_diff.py`, C-12), and held against its raw record when
   `--raw` is given (`native.py`, C-13);
3. **threads:** echo, outcomes, resends and over-length handling.

Constants (markers, degenerate ids, identity keys, outcome strings, timeouts, error mapping)
come from the profiles (`profiles.py`). Memory grows with requests and calls, not with history
length.

```bash
python scripts/validate/verdict.py --harness profiles/harness/<h>.yaml --model profiles/model/<m>.yaml \
    --matrix <run>/matrix.yaml --population replay|smoke|window [--build-id ID] [--raw <raw dirs>] \
    [--tier full --controls-receipt controls.json] [--requests <boundary hour dirs>] \
    [--responses <next hour dirs>] --out-dir <out> <trace or replay dirs>
```

`--build-id` names the build under test as it was deployed: the value given to the recorder's
`--build-id` or to the trace writer's `TRTLLM_BUILD_ID`, never one read back from the records,
which are checked against it. Records that carry a `build_id` while none is declared make
inventory BLOCKED (I-05).

`--raw` names the raw-output records. Pass it whenever the served build recorded raw output:
without it C-13 reads NOT_APPLICABLE, which can turn a claim that should be BLOCKED into a PASS.
NOT_APPLICABLE is right only for a build without the raw-output hook.

**Qualification: `scripts/validate/controls.py`.**
- It starts `fake_responses_server.py` and `fake_agent_server.py` and replays a synthetic corpus
  once per fault mode.
- Each mode must trip exactly its expected set.
- It also runs the cross cases X-01 to X-128: echo, outcomes, the raw view, text echo, declared
  echo rewrites, usage, the protocol differential, the recorder's own replies, lost requests, rejections, probes, cuts, boundary hours, blind degenerate checks, echo and usage pairing by call ids, the raw view per thinking mode, checks that tested nothing, the unit for rates, client cuts known only from below, the raw view's declared repairs and the replies it did not compare, replies not echoed by design, outcome flags a profile does not declare, engine ids that several generation workers share, degenerate runs stopped or delivered, raw records of stopped generations, the revision and evidence a verdict binds, over-length read by its text, resends after a capacity answer, the measured pi profiles, tools the echo request no longer declares, argument markup inside a value, compaction summaries on Responses, the place of every reply in the raw view's record, Codex's compaction under its measured profile, pi's preparation of its `edit` tool's arguments, the prompt offset per protocol, the order of the raw join's ids, pi's re-serialization of a call's arguments, outcome patterns at either end of a long result, Chat's `stop_reason`, the examples a report lists, what an echo of a call is equal by, and NaN in call arguments
  (some ids are not implemented yet, see `acceptance-cases.md` section 4).
- It writes `controls.json`, bound to the validator tree SHA-256 and to the installed SDK
  versions. The receipt passes only when every case passes.
- The tests in `tests/test_validator.py` wrap it (`MH_RUN_CONTROLS=1` for the full matrix).
- **Properties.** `tests/test_validator_properties.py` states the re-ask rule (C-03), the
  declared-wording match (C-16), the raw join (C-13: ids, the window, the output count), the
  degenerate-run rule (M-02, R-04), the stopped raw record (C-13), the revision check (I-05), the
  over-length read by text (C-10 `any_error`), the resend after a capacity answer (H-02), the
  shrink allowed for tools the echo request no longer declares (USG-08) and GLM's argument values
  (M-07) over generated inputs. `tests/validator_properties.mutations` gates them with
  `scripts/mutate.py` (the method is `references/verify-logic.md`): every planted bug must be
  detected. A property counts only with its gate.

**Environment.** The validator runs from its own virtualenv built from
`scripts/validate/requirements.txt`, which pins `anthropic` and `openai` (C-12) besides
`zstandard`, `jsonschema`, `lark`, `PyYAML` and `aiohttp`. TensorRT-LLM's build environment
also installs `openai`, at whatever version its own requirements say, so it is not the
validator's environment.

**How the validator is qualified.**
- **The control matrix** (above), bound to the validator tree and the SDK versions (V-01), each
  case implemented to the stimulus and decision `acceptance-cases.md` documents.
- **The mutation gate on its properties** (above).
- **Calibration on recorded traces of real deployments (Step 3).** On a frozen window of
  production traffic:
  - `verdict.py` and the reference checkers (Responses) agree on every count they share, or the
    difference is explained and one of them is fixed;
  - the normal controls trip nothing, and C-11 finds no element without a rule;
  - the SDK reading (C-12) agrees with the adapters on every reply;
  - seeded faults on copies of its replies are each caught by their own check
    (`mutate_traces.py`, `acceptance-cases.md` section 5).

  Calibration also decides what is a rule and what is a flag. USG-06 (the engine's output count)
  and USG-08 (the next turn bills no less) held on every case there, so they FAIL.
- **Full tiers that re-derive the numbers independently** (section 5). At a stage exit an
  independent session verifies the manifest, runs the control matrix for the exact validator
  tree, runs `verdict.py --tier full`, and re-derives the key numbers from the raw bytes with its
  own scripts, e.g. a raw join that does not use the output count, holding each reply's
  reasoning, text and calls against every raw record. A number it cannot reproduce is a finding,
  against the run or against the validator.

**A validator defect** that calibration or a full tier finds is fixed as a policy change
(`SKILL.md` section 2), with controls for the decision it changes:
- on the validator before the change, each new control either fails by the decision it checks,
  or passes because the old validator already made that decision; the iteration log says which;
- bugs planted in the new code, one at a time, are each caught by a control or a property;
- the affected verdicts are computed again with the fixed validator on the same frozen evidence.
  The earlier ones are invalidated, never edited (section 2).

**Reference implementation, kept for Step 3 (calibration).** `check_stream.py`,
`check_tool_calls.py`, `harness_signals.py` and `run_controls.sh` are the Responses-only
checkers that judged production windows before `verdict.py`. They stay as an independent second
implementation for calibration, and are retired once Step 3 has reproduced their results.

**Not implemented yet:**
- I-02 (sidecar reconciliation), I-03 (manifest verification), I-07 (capture-boundary
  declaration), I-10 (strata coverage);
- R-03 (error-mapping controls beyond over-length), M-04 (reasoning round-trip policy), H-05
  (compaction substance), V-03 (invalidation log);
- the raw view on main: main does not have the raw-output hook yet, so a main build gives C-13
  NOT_APPLICABLE unless its trial build adds the hook (`smoke.md` sections 3c and 3d);
- per-frame timing in the trtllm-serve request trace. Without it, R-02 is BLOCKED on production
  windows; replay and recorder captures carry chunk times;
- Step 3 with code-level seeded bugs (`acceptance-cases.md` section 5).

**Known limits.**
- Records of two generation workers under one engine id that both ran inside a reply's window
  and wrote its output count leave the reply unjoined (fail closed). The 2 s margin assumes the
  hosts' clocks agree within it.
- USG-06 compares only the replies whose id named one raw record: a resolved join already used
  the output count, so comparing it again proves nothing. Covering every reply needs a join key
  that does not use the count: a digest of what the reply carries, or the orchestrator's id in
  the reply.
- A reply with an error status is not joined to a raw record. So M-02 does not see a degenerate
  run that the server stopped in a non-streamed request and answered with an HTTP error; a
  streamed one ends in an error event, and its partial raw record is read.
