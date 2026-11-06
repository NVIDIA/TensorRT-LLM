<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S1: adapt trtllm-serve to a model's chat format

**Inputs:**
- the harness contracts of the cells in play;
- the checkpoint, with its own encoder or chat template.

**Outputs:**
- rendering, parsers and stop rules that pass the fault matrix;
- regression tests that failed on the old code;
- `profiles/model/<id>.yaml`: format card, markers, degenerate ids, parsers, renderer source.

One adaptation serves every harness. Both output paths (Responses, and Chat with Messages on top)
use the same parsers.

## 0. Invariants (each one was a production bug)

1. **One generation, one story.** The streaming path (`parse_streaming_increment` per chunk, then
   the end-of-stream drain) and the whole-text path (`detect_and_parse`: non-streaming replies,
   and any snapshot rebuild) yield the same calls and the same text.
2. **Chunking never changes the result.** Real streams split text anywhere between tokens.
3. **Nothing generated is silently dropped.** Unterminated markup reaches the client as text;
   whitespace is text.
4. **Names resolve the same way on both paths:** bare and qualified spellings of a declared tool,
   and never a name recovered from prose.
5. **A single-call request yields at most one call.** That covers `parallel_tool_calls=false`
   and `disable_parallel_tool_use`. The parser exposes a `single_call_stop`.
6. **Degenerate ids are declared, not stopped.** The model profile names them, so the validator
   sees a run on the raw token ids (M-02) and fails a reply that delivers one (R-04). A run is a
   numerical or kernel bug: fix it at its source. Serving never stops or hides it.

## 1. Inventory what exists (do not rebuild it)

```bash
grep -n "<model_type>" tensorrt_llm/serve/tool_parser/tool_parser_factory.py \
    tensorrt_llm/llmapi/reasoning_parser.py tensorrt_llm/tokenizer/tokenizer.py
ls tensorrt_llm/serve/tool_parser/ tensorrt_llm/tokenizer/ tensorrt_llm/serve/extensions/
git log --oneline origin/main -- tensorrt_llm/serve/tool_parser/<model>*
```

Each model profile records what main has for the model and what it still lacks: `renderer`,
`parsers` and `upstream`. Main moves: re-run the inventory before relying on a profile's account.

## 2. Take the format from the model, not from memory

- **Source of truth.** Use the checkpoint's own encoder (e.g. DeepSeek V4.1 `encoding/encoding.py`
  with `test_encoding.py`; Kimi-K3 `encoding_k3.py`) or its `chat_template.jinja` (e.g. GLM).
- **Golden conversations.** Render them with the reference AND with trtllm-serve's renderer, and
  require byte equality. At minimum:
  - developer and system messages, including a mid-conversation system message;
  - a tool call with its result;
  - two calls with results given in a different order;
  - a tool result as content parts, and an image;
  - reasoning from a previous turn: does the model keep, drop or re-render it? Record the answer
    as `reasoning.history_policy`;
  - a multi-turn history in each harness's real shape.
- **The format card.** Write it into the model profile (`format_card`): `block_open` /
  `block_close`, `call_open`, `name_end`, `call_close`, `value_open` / `value_close`, exactly as
  the encoder writes them, with `null` for a delimiter the format does not have. The cards of the
  supported models are in `profiles/model/`.
- **Special and degenerate ids.** List the stop tokens, the think open/close tokens, and the
  degenerate ids (pad, last vocab id). The validator's leak and degeneration screens read them
  from the profile.

## 3. Parser checklist (`tensorrt_llm/serve/tool_parser/base_tool_parser.py`)

- **`bot_token` / `eot_token`, `has_tool_call`, `needs_raw_special_tokens`.** Set the last one to
  True when the markup consists of special tokens; otherwise detokenization strips them.
- **`detect_and_parse` and `parse_streaming_increment` share one grammar.** Never a regex on one
  side and a state machine on the other without the fault matrix proving them equal.
- **`finish()`** emits, or releases as text, whatever the stream left buffered. A no-op `finish`
  is how a dangling close tag reaches the client.
- **`single_call_stop`:** the text that completes exactly one call.
- **`markup_tokens`:** delivered arguments containing them are flagged in traces.
- **`structure_info` / structural tags:** for strict or forced tool choice.
- **`resolve_tool_name`** on both paths.
- **No text-rewriting idioms:** no `.strip()`, no `replace(token, "")` on generated text.

Some of these hooks (`single_call_stop`, `markup_tokens`, the stream-assembly helpers) are not on
main yet. Port them with the fixes (`smoke.md` section 3b).

## 4. Run the fault matrix

```bash
python scripts/fault_mutations.py --card card.json --samples canonical.jsonl --out mutated.jsonl
python scripts/chunk_equivalence.py --parser <name> --samples mutated.jsonl --tools tools.json \
    --marker '<block open>' --marker '<call close>' --ignorable '<eos token>' --out report.json
```

- **The card** is the profile's `format_card` written out as JSON.
- **Canonical samples** are rendered with the model's own encoder (section 2), never by hand.
- **What `chunk_equivalence.py` compares per sample:**
  - the whole-text parse;
  - streams at every split point;
  - per-character streams;
  - random chunkings.
- **Split at token boundaries for special-token formats.** E.g. Kimi-K3's `<|...|>` control
  tokens never arrive split per character. Character-level splits produced 23 false findings in
  28 samples.
- **Target:**
  - zero `calls_differ`, `streams_disagree` and `text_lost`;
  - `normal_text_differs` only where documented;
  - whitespace-only differences understood.
- **Run it on the whole parser family.** Siblings inherit defects (e.g. glm4/glm47, and DeepSeek
  V3.2 → V4 → V4.1). The model profile lists the family under `parsers.family`.
- **Then state the parser's rule as a property** (`references/verify-logic.md`). For a parser,
  the property is stream equivalence: take the seed texts of the parser's test class and let
  Hypothesis vary where the chunk boundaries fall, the one dimension the example tests hold
  fixed. Gate it with `scripts/mutate.py`, which plants plausible bugs and requires the property
  to catch each one. A property counts only after its gate: one that no planted bug breaks proves
  nothing.
- **Without the C++ build:** run inside the TRT-LLM container. A stub loader that imports the
  serving modules without the build breaks as main moves: one written for an older branch
  crashes in `responses_utils.py` on main.

## 5. Fault catalog: what production models emitted, and what serving must do

GLM-5.3 and Kimi-K3 emitted these shapes in production agent traffic under Codex. The Evidence
column is what went wrong (for DeepSeek: what its parser does with the mutation). The Fix column
says what the fix does; most of these fixes are not on main yet, so look for them first
(`smoke.md` section 3b).

| Shape | Mutation | Evidence | Serving must | Fix |
|---|---|---|---|---|
| Stream ends inside a call | `truncate@N` | The stream showed nothing; the final snapshot showed the call's markup | Release unterminated markup as text on BOTH views; never deliver a half call | `_flush_tool_parser`, `_assembled_tool_calls` |
| A call's close tag is missing | `drop_call_close` | Kimi-K3; under `parallel_tool_calls=false` two calls were delivered | Recover identically on both paths, or treat as text on both | The K3 parser reads a call whose close tags the model left out; the single-call policy for this case is open |
| The wrapper is never closed | `drop_block_close` | DeepSeek V3.2/V4 whole-text parse returns no call; streaming emits them | Same calls on both paths | Open for DeepSeek |
| Empty name | `empty_name` | A nameless call; `output_item.done` and `response.completed` carried texts of different lengths | Never a nameless call; identical text on both views | The GLM parsers release a nameless call as text on both views |
| Bare vs qualified name | `qualified_name`, `bare_name` | Models wrote namespaced tools bare more often than qualified; Codex aborted calls to a custom tool delivered as a function | Resolve unambiguous spellings on BOTH paths; custom tool as `custom_tool_call` | `resolve_tool_name` maps a bare spelling onto the one qualified tool that answers to it |
| Mangled names | (manual) | GLM: markup fused onto a name by an unbalanced tag, or a group qualifier on a tool declared unqualified | Strip balanced markup from names; never recover a name from prose | Map a mangled name back onto the tool it names; the GLM parsers end a name at its first tag |
| Extra calls where one is allowed | `duplicate_call` | Kimi-K3: replies to `parallel_tool_calls=false` carried several calls | Stop at the end of the first call (`single_call_stop`) | Stop generation on the text that completes the first call (`single_call_stop`; GLM, then K3) |
| A new opener inside an unclosed call | `nested_open` | GLM: the model abandons a call and starts over | Do not deliver the abandoned half call; both views agree | GLM47: a fresh opener inside an open call's arguments abandons that call, and parsing re-anchors on the opener |
| Close markup inside a value | `markup_in_value` | GLM's own; also GLM markup reaching Kimi-K3 through cross-model history | Deliver the value byte-exact or reject the call, the same on both views; flag it | The request trace flags calls whose arguments carry the parser's markup (`markup_tokens`); a gateway keeps each conversation in its model's pool |
| Prose before or after the calls | `leading_text`, `trailing_text` | DeepSeek whole-text parse drops text after the block | Deliver all prose, identically | Open for DeepSeek |
| Whitespace | `blank_lines` | `.strip()` deleted generated bytes, close tags included | Whitespace is generated text | The Responses snapshot repeats the streamed bytes, edge whitespace included |
| Separator whitespace streamed as text | (streaming) | DeepSeek's DSML parser streamed the newline after each invoke, and the blank line before the calls block, as content. On Messages they became text blocks of only whitespace, which harnesses drop when they send the turn back | The separator the encoder writes between text and calls is markup, not text; never deliver a whitespace-only text block | Drop the DSML separators while streaming (#20873); no whitespace-only text blocks on Messages (#20874) |
| Non-ASCII, quotes, backslashes | `unicode_value` | Escaping errors | Arguments round-trip exactly; types follow the schema | The request trace keeps escapes lossless; the GLM parsers deliver arguments as written, typed by the schema |
| Degenerate runaway (`[PAD]` repeated) | (separate) | Kimi-K3 runaway loops held batch slots (root cause: a missing fence in the Helix all-to-all) | Nothing: no stop and no rewrite, so the run stays visible to M-02 and R-04 | At the source: fence the Helix all-to-all's shared-memory writes before the receiver's bulk copy. No run since |
| `</think>` twice; leading `</think>` residue | (separate) | GLM: a literal `</think>` reached the client | A dedicated reasoning parser; drop the residue, not the content | GLM's own reasoning parser; a close before any content is residue, dropped on both views |

**Reading the results:**
- `calls_differ`, `streams_disagree` and `text_lost` must be zero.
- `normal_text_differs` on `trailing_text` or `markup_in_value` means generated text was
  dropped or rewritten.
- Every finding is parser-level until L2 confirms it with real token boundaries.

## 6. Tests and registration

1. **Turn every failing sample into a test.**
   - Inherit `BaseToolParserTestClass` (`tests/unittest/llmapi/apps/test_tool_parsers.py`), with
     the catalog cases parametrized.
   - Each new test must FAIL on the old parser and PASS on the new one. Keep the node id and
     both results.
   - Where the fix changes a rule of pure Python logic (a parser, a protocol path, request
     validation), keep a property for that rule too (`references/verify-logic.md`), with its
     `.mutations` file for `scripts/mutate.py` beside the test.
2. **Register the parsers.**
   - Map them in `tool_parser_factory.py` (and `MODEL_TYPE_TO_TOOL_PARSER`) and in
     `reasoning_parser.py`.
   - Document the flags: `--tool_parser`, `--reasoning_parser`, `--custom_tokenizer`.
3. **Map the harness's reasoning controls into the template where needed.** For example, Codex
   sends only `reasoning.effort`, Claude Code sends `thinking` and `output_config.effort`, and pi
   sends whatever its `thinkingFormat` dictates. Compare `serve/extensions/kimi_k3.py`.
4. **Move on to S2** (`replay.md`).

## 7. Traps

- **Recovering a tool name from prose fabricates calls.** Only bare dotted identifiers, and only
  unambiguous tails.
- **Reasoning history from another model** (cross-pool routing) puts foreign markup into the
  prompt, and the model copies it into its own calls. Keep a conversation on one model.
- **The Messages adapter must read the chat stream past `[DONE]`.** Stopping at `[DONE]` closes
  the generator early: it logs a `GeneratorExit` and loses the request's metrics (#20867).
- **A model writing its own markup inside code is not a leak.** Quoted hits are usually the model
  explaining syntax. Report screens; do not change serving to hide model behavior.
