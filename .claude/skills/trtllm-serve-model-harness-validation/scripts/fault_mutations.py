#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Turn canonical tool-call outputs into the malformed shapes models really emit.

Every mutation below is a shape a model emitted at least once under an agent
harness, and each one broke or nearly broke serving (see the fault catalog,
references/adapt.md section 5). The shapes are format-independent once the
format's markers are known, so a new model's parser can be tested against all
of them before it serves a single request.

Format card (JSON), literal strings of the model's tool-call markup:

  call_open     text that opens one call, immediately before the tool name
                (DeepSeek V4: '<｜DSML｜invoke name="';  GLM: '<tool_call>')
  name_end      text that ends the name ('">' for DSML; '\n' or '<arg_key>' for GLM)
  call_close    text that closes one call ('</｜DSML｜invoke>', '</tool_call>')
  block_open / block_close   optional wrapper around all calls ('<｜DSML｜tool_calls>' ...)
  value_open / value_close   optional markers around one argument value
  qualified_prefix           namespace prefix models add or drop (default 'functions.')

Canonical samples (JSONL): {"id", "text", "tools"?}. Render them with the
model's OWN encoder or chat template (the checkpoint's encoding/ directory or
tokenizer chat template) so the markup is exactly what the model was trained on.

Mutations (each output line carries "mutation" and the catalog's "expect"):
  truncate@N       the generation ends inside or right after a marker (stream cut)
  drop_call_close  the first call is never closed
  drop_block_close the wrapper is never closed
  empty_name       a call with an empty name (e.g. a nameless <tool_call></tool_call>)
  qualified_name   'exec' written as 'functions.exec' (bare vs qualified spelling)
  bare_name        'functions.exec' written as 'exec'
  duplicate_call   the first call twice (parallel_tool_calls=false must stop at one)
  nested_open      a fresh call opener inside the first call's arguments (the model
                   abandons a call and starts over)
  markup_in_value  the call/block close markup inside an argument value
  leading_text / trailing_text   prose before / after the calls
  blank_lines      blank lines between and after calls
  unicode_value    non-ASCII, quotes and backslashes inside a value

usage: fault_mutations.py --card card.json --samples canonical.jsonl --out mutated.jsonl [--max-truncations 12]
"""

import argparse
import json

EXPECT = {
    "canonical": "every view parses the same calls; nothing else in the text changes",
    "truncate": "stream and whole text agree; an unfinished call is not delivered as a call, and its raw "
    "text is not silently dropped",
    "drop_call_close": "stream and whole text agree (recover the call on both paths, or on neither); no text lost",
    "drop_block_close": "stream and whole text agree; no text lost",
    "empty_name": "never a call without a name; both views deliver the same text",
    "qualified_name": "resolved to the declared tool when unambiguous (both views)",
    "bare_name": "resolved to the declared tool when unambiguous (both views)",
    "duplicate_call": "two calls parsed identically on both views; with parallel_tool_calls=false the server's "
    "single_call_stop must end generation after the first",
    "nested_open": "the abandoned outer call is not delivered with half its arguments; both views agree",
    "markup_in_value": "the value is delivered byte-exact or the call is rejected -- the same on both views",
    "leading_text": "the prose is delivered as text on both views, byte-exact",
    "trailing_text": "the prose after the calls is delivered, not dropped (DeepSeek V3.2/V4 whole-text parse "
    "drops it)",
    "blank_lines": "whitespace is generated text: both views keep it identically",
    "unicode_value": "the value round-trips exactly through JSON",
}


def first_call(text, card):
    """(start, name_start, name_end, end) of the first complete call, or None."""
    start = text.find(card["call_open"])
    if start < 0:
        return None
    name_start = start + len(card["call_open"])
    name_end = text.find(card["name_end"], name_start) if card.get("name_end") else -1
    if name_end < 0:
        return None
    close = text.find(card["call_close"], name_end)
    if close < 0:
        return None
    return start, name_start, name_end, close + len(card["call_close"])


def first_value(text, card, after):
    if not card.get("value_open"):
        return None
    i = text.find(card["value_open"], after)
    if i < 0:
        return None
    return i + len(card["value_open"])


def mutations(text, card, max_truncations):
    prefix = card.get("qualified_prefix", "functions.")
    call = first_call(text, card)
    markers = [
        m
        for m in (
            card.get("block_open"),
            card["call_open"],
            card.get("name_end"),
            card.get("value_open"),
            card.get("value_close"),
            card["call_close"],
            card.get("block_close"),
        )
        if m
    ]
    cuts = sorted(
        {
            p
            for m in markers
            for i in _find_all(text, m)
            for p in (i, i + len(m) // 2, i + len(m))
            if 0 < p < len(text)
        }
    )
    step = max(1, len(cuts) // max_truncations) if cuts else 1
    for p in cuts[::step][:max_truncations]:
        yield f"truncate@{p}", "truncate", text[:p]
    if card["call_close"] in text:
        yield "drop_call_close", "drop_call_close", text.replace(card["call_close"], "", 1)
    if card.get("block_close") and card["block_close"] in text:
        yield "drop_block_close", "drop_block_close", text.replace(card["block_close"], "", 1)
    if call:
        start, ns, ne, end = call
        name = text[ns:ne]
        yield "empty_name", "empty_name", text[:ns] + text[ne:]
        if "." in name:
            yield "bare_name", "bare_name", text[:ns] + name.rsplit(".", 1)[1] + text[ne:]
        else:
            yield "qualified_name", "qualified_name", text[:ns] + prefix + name + text[ne:]
        yield "duplicate_call", "duplicate_call", text[:end] + text[start:end] + text[end:]
        opener = card["call_open"] + name + card["name_end"]
        vstart = first_value(text, card, ne)
        at = vstart if vstart is not None and vstart < end else ne + len(card["name_end"])
        yield "nested_open", "nested_open", text[:at] + opener + text[at:]
        if vstart is not None and vstart < end:
            closing = card["call_close"] + (card.get("block_close") or "")
            yield (
                "markup_in_value",
                "markup_in_value",
                text[:vstart] + "see " + closing + " here " + text[vstart:],
            )
            yield "unicode_value", "unicode_value", text[:vstart] + 'ü€😀 "q" \\ ' + text[vstart:]
    yield "leading_text", "leading_text", "Let me check that first.\n" + text
    yield "trailing_text", "trailing_text", text + "\nThat should do it."
    yield (
        "blank_lines",
        "blank_lines",
        text.replace(card["call_close"], card["call_close"] + "\n\n"),
    )


def _find_all(text, needle):
    i = text.find(needle)
    while needle and i >= 0:
        yield i
        i = text.find(needle, i + 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--card", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-truncations", type=int, default=12)
    args = ap.parse_args()
    card = json.load(open(args.card))
    for key in ("call_open", "name_end", "call_close"):
        if not card.get(key):
            ap.error(f"format card needs {key}")
    written = 0
    with open(args.samples) as fh, open(args.out, "w") as out:
        for line in fh:
            if not line.strip():
                continue
            sample = json.loads(line)
            base = {k: v for k, v in sample.items() if k not in ("text", "id")}
            out.write(
                json.dumps(
                    dict(
                        base,
                        id=f"{sample['id']}:canonical",
                        base_id=sample["id"],
                        mutation="canonical",
                        expect=EXPECT["canonical"],
                        text=sample["text"],
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            for label, kind, text in mutations(sample["text"], card, args.max_truncations):
                out.write(
                    json.dumps(
                        dict(
                            base,
                            id=f"{sample['id']}:{label}",
                            base_id=sample["id"],
                            mutation=label,
                            expect=EXPECT[kind],
                            text=text,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    print(json.dumps({"written": written}))


if __name__ == "__main__":
    main()
