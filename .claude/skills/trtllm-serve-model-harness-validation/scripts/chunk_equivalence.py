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
"""Streaming-vs-whole-text equivalence of a trtllm-serve tool parser (CPU, no model).

The serving layer reads one generation twice: incrementally while streaming
(``parse_streaming_increment`` per chunk, then ``finish``) and as a whole
(``detect_and_parse`` -- the snapshot, the non-streaming reply). Whenever the two
disagree, the client's stream and its final snapshot disagree, and a harness that
keeps the stream's version of the conversation (e.g. Codex) goes on with it. Every
such divergence seen in agent traffic was a real bug; the chunking at which it
shows is often a single split point.

For each sample (a model output with tool-call markup), with fresh parser
instances throughout:

  whole      detect_and_parse(text)
  streams    the same text split every way that matters:
               char     one character per chunk
               split2   every two-way split point (all of them up to --max-splits,
                        else a seeded sample that always includes positions next to
                        the markers named with --marker)
               random   --random seeded chunkings with chunk sizes 1..16
  findings   normal_text_differs        stream text != whole-text text (whitespace-
                                        only differences reported separately)
             calls_differ               names or JSON arguments differ
             streams_disagree           two chunkings of the same text disagree (calls or
                                        non-whitespace text; whitespace-only reported separately)
             text_lost                  no call parsed, yet the text the client gets
                                        is shorter than the generation (markup and
                                        prose silently dropped)
             exception                  the parser raised

Streams are assembled the way the server does. Where the tree has them, the
serving layer's own helpers are used:
``_accumulate_tool_call_fragments``, the end-of-stream drain ``_flush_tool_parser``
(a parser may report one finished call per increment, and unterminated markup is
released as text) and ``_assembled_tool_calls`` (unfinished and abandoned calls
dropped). Elsewhere a generic drain approximates them (--assembly generic).

Samples: JSONL lines {"id", "text", "tools": [{"name", "parameters"?}]?}; tools
default to --tools FILE (same list shape). The parser comes from
ToolParserFactory (--parser NAME) or --parser-class module:Class.

Runs where tensorrt_llm.serve imports (the TRT-LLM container). On a host without
the C++ build, set TRT_LLM_NO_LIB_INIT=1 or load the serve modules through a stub
loader.

usage: chunk_equivalence.py --parser deepseek_v4 --samples samples.jsonl --out report.json
"""

import argparse
import importlib
import json
import random
import sys


def load_parser_factory(args):
    if args.parser_class:
        module_name, _, class_name = args.parser_class.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

    return lambda: ToolParserFactory.create_tool_parser(args.parser)


def make_tools(specs):
    from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam, FunctionDefinition

    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name=s["name"],
                description=s.get("description", ""),
                parameters=s.get("parameters") or {"type": "object", "properties": {}},
            ),
        )
        for s in specs
    ]


def norm_args(parameters):
    try:
        return json.loads(parameters) if parameters else {}
    except (json.JSONDecodeError, TypeError):
        return {"__unparsed__": parameters}


def run_whole(new_parser, text, tools):
    result = new_parser().detect_and_parse(text, tools)
    calls = [(c.name, norm_args(c.parameters)) for c in result.calls if c.name]
    return result.normal_text or "", calls


def server_helpers():
    """The serving layer's own stream assembly, when this tree has every helper it needs."""
    try:
        from tensorrt_llm.serve import responses_utils as ru
    except ImportError:
        return None
    names = ("_flush_tool_parser", "_accumulate_tool_call_fragments", "_assembled_tool_calls")
    if not all(hasattr(ru, n) for n in names):
        return None
    return ru


def run_stream(new_parser, chunks, tools, helpers):
    """What a streaming client receives: text and assembled calls.

    With ``helpers`` (the server's responses_utils) the stream is assembled exactly
    as ``_generate_streaming_event`` does: fragments per call, then the parser is
    drained at the end of the stream (it may report at most one finished call per
    increment), unterminated markup is released as text, and unfinished or
    abandoned calls are dropped. Without them, a generic drain approximates it.
    """
    parser = new_parser()
    text = []
    if helpers is not None:
        fragments = {}
        for chunk in chunks:
            result = parser.parse_streaming_increment(chunk, tools)
            text.append(result.normal_text or "")
            helpers._accumulate_tool_call_fragments(fragments, result.calls)
        released, flushed, unfinished = helpers._flush_tool_parser(
            tools=tools, output_index=0, tool_parser_dict={0: parser}
        )
        text.append(released or "")
        helpers._accumulate_tool_call_fragments(fragments, flushed)
        abandoned = (
            helpers._abandoned_tool_indices(parser)
            if hasattr(helpers, "_abandoned_tool_indices")
            else ()
        )
        assembled = helpers._assembled_tool_calls(fragments, unfinished, abandoned)
        calls = [(c.name, norm_args(c.parameters)) for c in assembled if c.name]
        return "".join(text), calls

    fragments, order = {}, []

    def take(result):
        text.append(result.normal_text or "")
        for call in result.calls:
            if call.tool_index not in fragments:
                fragments[call.tool_index] = {"name": None, "parameters": ""}
                order.append(call.tool_index)
            if call.name:
                fragments[call.tool_index]["name"] = call.name
            fragments[call.tool_index]["parameters"] += call.parameters or ""

    for chunk in chunks:
        take(parser.parse_streaming_increment(chunk, tools))
    for _ in range(64):  # drain: a parser may report one finished call per increment
        result = parser.parse_streaming_increment("", tools)
        if not result.calls and not result.normal_text:
            break
        take(result)
    finish = getattr(parser, "finish", None)
    if callable(finish):
        take(finish(tools))
    leftover = getattr(parser, "_buffer", "") or ""
    text.append(leftover)  # unterminated markup is released as text, never dropped
    calls = [
        (fragments[i]["name"], norm_args(fragments[i]["parameters"]))
        for i in order
        if fragments[i]["name"]
    ]
    return "".join(text), calls


def chunkings(text, args, rng):
    yield "char", list(text)
    n = len(text)
    points = list(range(1, n))
    if len(points) > args.max_splits:
        near = {
            p
            for m in args.marker
            for i in _find_all(text, m)
            for p in (i - 1, i, i + 1, i + len(m) - 1, i + len(m), i + len(m) + 1)
        }
        keep = sorted(p for p in near if 0 < p < n)
        rest = [p for p in points if p not in set(keep)]
        points = sorted(set(keep) | set(rng.sample(rest, max(0, args.max_splits - len(keep)))))
    for p in points:
        yield f"split2@{p}", [text[:p], text[p:]]
    for k in range(args.random):
        chunks, i = [], 0
        while i < n:
            step = rng.randint(1, 16)
            chunks.append(text[i : i + step])
            i += step
        yield f"random#{k}", chunks


def _find_all(text, needle):
    i = text.find(needle)
    while needle and i >= 0:
        yield i
        i = text.find(needle, i + 1)


def strip_ignorable(text, args):
    """Text without the tokens a parser may legitimately drop (end-of-sentence and the like)."""
    for token in args.ignorable:
        text = text.replace(token, "")
    return text.strip()


def check_sample(sample, new_parser, default_tools, args, rng, helpers):
    tools = make_tools(sample["tools"]) if sample.get("tools") else default_tools
    text = sample["text"]
    findings = []
    try:
        whole_text, whole_calls = run_whole(new_parser, text, tools)
    except Exception as error:  # noqa: BLE001 - a parser raising is itself the finding
        return {
            "whole": None,
            "findings": [
                {
                    "kind": "exception",
                    "where": "detect_and_parse",
                    "error": f"{type(error).__name__}: {str(error)[:200]}",
                }
            ],
        }
    first_stream = None
    for name, chunks in chunkings(text, args, rng):
        try:
            stream_text, stream_calls = run_stream(new_parser, chunks, tools, helpers)
        except Exception as error:  # noqa: BLE001
            findings.append(
                {
                    "kind": "exception",
                    "where": name,
                    "error": f"{type(error).__name__}: {str(error)[:200]}",
                }
            )
            continue
        if stream_calls != whole_calls:
            findings.append(
                {
                    "kind": "calls_differ",
                    "where": name,
                    "stream": [c[0] for c in stream_calls],
                    "whole": [c[0] for c in whole_calls],
                    "same_names": [c[0] for c in stream_calls] == [c[0] for c in whole_calls],
                }
            )
        if stream_text != whole_text:
            kind = (
                "normal_text_differs_whitespace"
                if stream_text.split() == whole_text.split()
                else "normal_text_differs"
            )
            findings.append(
                {
                    "kind": kind,
                    "where": name,
                    "stream_chars": len(stream_text),
                    "whole_chars": len(whole_text),
                }
            )
        if first_stream is None:
            first_stream = (name, stream_text, stream_calls)
        elif (stream_text, stream_calls) != first_stream[1:]:
            whitespace_only = (
                stream_calls == first_stream[2] and stream_text.split() == first_stream[1].split()
            )
            findings.append(
                {
                    "kind": "streams_disagree_whitespace"
                    if whitespace_only
                    else "streams_disagree",
                    "where": name,
                    "versus": first_stream[0],
                }
            )
        if not stream_calls and len(strip_ignorable(stream_text, args)) < len(
            strip_ignorable(text, args)
        ):
            findings.append(
                {
                    "kind": "text_lost",
                    "where": name,
                    "generated_chars": len(text),
                    "delivered_chars": len(stream_text),
                }
            )
        if len(findings) >= args.max_findings:
            break
    if not whole_calls and len(strip_ignorable(whole_text, args)) < len(
        strip_ignorable(text, args)
    ):
        findings.append(
            {
                "kind": "text_lost",
                "where": "detect_and_parse",
                "generated_chars": len(text),
                "delivered_chars": len(whole_text),
            }
        )
    return {
        "whole": {"calls": [c[0] for c in whole_calls], "normal_text_chars": len(whole_text)},
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parser", help="ToolParserFactory name, e.g. deepseek_v4, kimi_k3, glm47")
    ap.add_argument("--parser-class", help="module:Class, for a parser not yet registered")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--tools", help="JSON list of {name, parameters} used when a sample has none")
    ap.add_argument(
        "--marker",
        action="append",
        default=[],
        help="literal markup; split points next to it are always tried",
    )
    ap.add_argument(
        "--ignorable",
        action="append",
        default=[],
        help="literal token a parser may drop from text (e.g. an end-of-sentence token)",
    )
    ap.add_argument("--max-splits", type=int, default=400)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--max-findings", type=int, default=50, help="per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--assembly",
        choices=("auto", "server", "generic"),
        default="auto",
        help="assemble streams with the server's responses_utils helpers (auto: when present)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.parser and not args.parser_class:
        ap.error("--parser or --parser-class is required")
    new_parser = load_parser_factory(args)
    default_tools = make_tools(json.load(open(args.tools))) if args.tools else []
    rng = random.Random(args.seed)
    helpers = None if args.assembly == "generic" else server_helpers()
    if args.assembly == "server" and helpers is None:
        ap.error(
            "--assembly server: tensorrt_llm.serve.responses_utils has no stream assembly helpers here"
        )
    results, totals = [], {}
    with open(args.samples) as fh:
        for line in fh:
            if not line.strip():
                continue
            sample = json.loads(line)
            res = check_sample(sample, new_parser, default_tools, args, rng, helpers)
            kinds = sorted({f["kind"] for f in res["findings"]})
            for kind in kinds:
                totals[kind] = totals.get(kind, 0) + 1
            results.append(
                {
                    "id": sample.get("id"),
                    "mutation": sample.get("mutation"),
                    "expect": sample.get("expect"),
                    "whole": res["whole"],
                    "finding_kinds": kinds,
                    "findings": res["findings"][:10],
                }
            )
    report = {
        "tool": "chunk_equivalence",
        "parser": args.parser or args.parser_class,
        "assembly": "server" if helpers is not None else "generic",
        "samples": len(results),
        "samples_with_findings": sum(1 for r in results if r["finding_kinds"]),
        "by_kind": totals,
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("parser", "assembly", "samples", "samples_with_findings", "by_kind")
            }
        )
    )
    sys.exit(1 if report["samples_with_findings"] else 0)


if __name__ == "__main__":
    main()
