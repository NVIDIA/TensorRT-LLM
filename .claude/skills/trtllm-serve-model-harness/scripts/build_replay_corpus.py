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
"""Build a per-turn replay corpus from recorded request traces (read-only on the sources).

Each recorded request of an agent harness is a complete, self-contained prompt:
the harness resends the whole conversation on every turn (e.g. Codex: instructions,
tools, input items). Replaying such a request against another build or another
model asks "what would this server/model answer at this exact point of a real
session" -- without running the harness or its tools.

Selection is per conversation, because the turns of one session overlap almost
entirely (request bodies make up most of a trace's bytes):

  thread key   record.thread.thread_id > client_metadata thread_id (Codex turn
               metadata) > Claude Code's x-claude-code-session-id (+ agent id)
               > record.session > body.prompt_cache_key > trace_id
  --per-thread K turns spread over the session (first and last included);
               --pick first|last|random changes that
  --max-records N caps the corpus after per-thread selection (seeded)

Exclusions, each counted in the manifest:
  credential    the request line carries a strong credential signature
                (the record is dropped whole; never copied into a corpus)
  degenerate    an input reasoning item ends with a degenerate token
                (--degenerate-suffix, default "[PAD]"): the cut-off reasoning of
                a runaway generation that the client replays every turn
  previous_id   the body uses previous_response_id (not self-contained)
  route / status / body   not the requested route, not accepted, no JSON body

Transforms (recorded per entry):
  --model NAME               rewrite body.model
  --reasoning drop|strip-encrypted|keep
                             input reasoning items were produced by ANOTHER model;
                             drop them (default), keep only their visible text, or
                             keep them verbatim (same-model replay only)
  --max-output-tokens N      cap generation length
  --stream true|false|keep   force streaming on or off

Output: corpus JSONL (one entry per selected request, with its source locator
and body SHA-256) and a manifest JSON with every count and parameter. Prints
counts only.

usage: build_replay_corpus.py --out corpus.jsonl --manifest manifest.json [--workers N] [options] PATH [PATH ...]
PATH: request shards (requests-*.jsonl[.zst|.gz]) or directories holding them.
A shard that cannot be read to its end stops the run with a message naming it.
"""

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate"))
from scan_credentials import matches as credential_matches  # noqa: E402
from tracelib import ShardError, iter_lines  # noqa: E402

SHARD = re.compile(r"(^|/)requests-[^/]*\.jsonl(\.zst|\.gz)?$")


def expand(paths):
    found = []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, names in os.walk(path, followlinks=True):
                found += [
                    os.path.join(dirpath, n)
                    for n in names
                    if SHARD.search(os.path.join(dirpath, n))
                ]
        else:
            found.append(path)
    return sorted(set(found))


def header(rec, name):
    headers = rec.get("headers")
    pairs = (
        headers.items()
        if isinstance(headers, dict)
        else (
            [h for h in headers if isinstance(h, (list, tuple)) and len(h) == 2]
            if isinstance(headers, list)
            else []
        )
    )
    for key, value in pairs:
        if isinstance(key, str) and key.lower() == name:
            return value
    return None


def thread_key(rec, body):
    thread = rec.get("thread")
    if isinstance(thread, dict) and thread.get("thread_id"):
        return "thread:" + str(thread["thread_id"])
    session = header(rec, "x-claude-code-session-id")
    if session:
        agent = header(rec, "x-claude-code-agent-id")
        return "thread:cc:" + str(session) + (":" + str(agent) if agent else "")
    meta = body.get("client_metadata") if isinstance(body.get("client_metadata"), dict) else {}
    turn = meta.get("x-codex-turn-metadata")
    if isinstance(turn, str):
        try:
            turn = json.loads(turn)
        except json.JSONDecodeError:
            turn = None
    if isinstance(turn, dict) and turn.get("thread_id"):
        return "thread:" + str(turn["thread_id"])
    if meta.get("thread_id"):
        return "thread:" + str(meta["thread_id"])
    if rec.get("session"):
        return "session:" + str(rec["session"])
    if body.get("prompt_cache_key"):
        return "cache:" + str(body["prompt_cache_key"])
    return "trace:" + str(rec.get("trace_id"))


def reasoning_texts(item):
    texts = []
    for field in ("content", "summary"):
        value = item.get(field)
        if isinstance(value, list):
            texts += [
                p.get("text")
                for p in value
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
        elif isinstance(value, str):
            texts.append(value)
    return texts


def pick(indices, k, how, rng):
    if len(indices) <= k:
        return indices
    if how == "first":
        return indices[:k]
    if how == "last":
        return indices[-k:]
    if how == "random":
        return sorted(rng.sample(indices, k))
    if k == 1:
        return [indices[-1]]
    step = (len(indices) - 1) / (k - 1)
    return sorted({indices[round(i * step)] for i in range(k)})


def drop_reasoning_messages(body, mode):
    """Anthropic thinking blocks / Chat reasoning fields written by ANOTHER model.

    ``drop`` removes them all. ``strip-encrypted`` keeps the visible reasoning and removes only
    what the other model's server issued: thinking signatures and redacted thinking. Chat
    reasoning fields are visible text, so they stay.
    """
    changed = False
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for key in ("reasoning_content", "reasoning"):  # chat completions
            if key in message and mode == "drop":
                del message[key]
                changed = True
        blocks = message.get("content")
        if isinstance(blocks, list):  # anthropic messages
            kept = []
            for block in blocks:
                kind = block.get("type") if isinstance(block, dict) else None
                if kind == "redacted_thinking" or (kind == "thinking" and mode == "drop"):
                    changed = True
                    continue
                if kind == "thinking" and "signature" in block:
                    block = {k: v for k, v in block.items() if k != "signature"}
                    changed = True
                kept.append(block)
            message["content"] = kept
    return changed


def transform(body, args):
    body = copy.deepcopy(body)
    done = []
    if args.model:
        body["model"] = args.model
        done.append("model")
    if args.reasoning != "keep" and isinstance(body.get("messages"), list):
        if drop_reasoning_messages(body, args.reasoning):
            done.append("reasoning:" + args.reasoning)
    items = body.get("input")
    if isinstance(items, list) and args.reasoning != "keep":
        kept = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                if args.reasoning == "drop":
                    continue
                item = {k: v for k, v in item.items() if k != "encrypted_content"}
            kept.append(item)
        if len(kept) != len(items) or args.reasoning == "strip-encrypted":
            done.append("reasoning:" + args.reasoning)
        body["input"] = kept
        include = body.get("include")
        if isinstance(include, list) and "reasoning.encrypted_content" in include:
            body["include"] = [i for i in include if i != "reasoning.encrypted_content"]
            done.append("include:-reasoning.encrypted_content")
    if args.max_output_tokens:
        # The field that bounds output differs by protocol.
        if args.route == "/v1/messages":
            fields = ("max_tokens",)
        elif args.route == "/v1/chat/completions":
            fields = ("max_completion_tokens", "max_tokens")
        else:
            fields = ("max_output_tokens",)
        field = next((f for f in fields if f in body), fields[0])
        current = body.get(field)
        cap = (
            args.max_output_tokens
            if not isinstance(current, int)
            else min(current, args.max_output_tokens)
        )
        body[field] = cap
        done.append("max_output_tokens:" + field)
    if args.stream != "keep":
        body["stream"] = args.stream == "true"
        done.append("stream")
    return body, done


CFG = {}


def scan_shard(path):
    """Pass 1 over one shard: (counts, [(thread key, (recorded_at, path, line, line sha256, trace_id))])."""
    counts, rows = Counter(), []
    for line_no, raw in iter_lines(path):
        counts["lines"] += 1
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            counts["drop:unparsable"] += 1
            continue
        if not isinstance(rec, dict) or rec.get("event", "request") != "request":
            continue
        body = rec.get("body")
        if rec.get("route") != CFG["route"]:
            counts["drop:route"] += 1
            continue
        if rec.get("status", "accepted") != "accepted":
            counts["drop:status"] += 1
            continue
        if not isinstance(body, dict):
            counts["drop:body"] += 1
            continue
        if any(credential_matches(raw)):  # the strong-credential patterns of scan_credentials.py
            counts["drop:credential"] += 1
            continue
        if body.get("previous_response_id"):
            counts["drop:previous_id"] += 1
            continue
        items = body.get("input") if isinstance(body.get("input"), list) else []
        if any(
            isinstance(i, dict)
            and i.get("type") == "reasoning"
            and any(t.rstrip().endswith(s) for t in reasoning_texts(i) for s in CFG["suffixes"])
            for i in items
        ):
            counts["drop:degenerate"] += 1
            continue
        key = thread_key(rec, body)
        if CFG["require_thread"] and not key.startswith("thread:"):
            counts["drop:no_thread"] += 1
            continue
        counts["eligible"] += 1
        rows.append(
            (
                key,
                (
                    rec.get("recorded_at") or "",
                    path,
                    line_no,
                    hashlib.sha256(raw).hexdigest(),
                    rec.get("trace_id"),
                ),
            )
        )
    return counts, rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--route", default="/v1/responses")
    ap.add_argument("--per-thread", type=int, default=3)
    ap.add_argument("--pick", choices=("spread", "first", "last", "random"), default="spread")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--model")
    ap.add_argument("--reasoning", choices=("drop", "strip-encrypted", "keep"), default="drop")
    ap.add_argument("--max-output-tokens", type=int, default=0)
    ap.add_argument("--stream", choices=("true", "false", "keep"), default="keep")
    ap.add_argument("--degenerate-suffix", action="append", default=None)
    ap.add_argument(
        "--require-thread",
        action="store_true",
        help="skip records without harness thread identity (probes, hand-written curls)",
    )
    ap.add_argument("--workers", type=int, default=1, help="processes (fork) for the first pass")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    suffixes = args.degenerate_suffix or ["[PAD]"]
    rng = random.Random(args.seed)

    shards = expand(args.paths)
    counts = Counter()
    by_thread = defaultdict(list)
    CFG.update(route=args.route, suffixes=suffixes, require_thread=args.require_thread)
    # Pass 1: eligibility, with only locators kept in memory.
    if args.workers > 1 and len(shards) > 1:
        pool = multiprocessing.get_context("fork").Pool(min(args.workers, len(shards)))
        parts = pool.imap(scan_shard, shards)
    else:
        pool, parts = None, map(scan_shard, shards)
    for part_counts, rows in parts:
        counts.update(part_counts)
        for key, row in rows:
            by_thread[key].append(row)
    if pool is not None:
        pool.close()
        pool.join()

    selected = []
    for key in sorted(by_thread):
        turns = sorted(by_thread[key])
        for i in pick(list(range(len(turns))), args.per_thread, args.pick, rng):
            selected.append((key, i, len(turns), turns[i]))
    if args.max_records and len(selected) > args.max_records:
        selected = sorted(rng.sample(selected, args.max_records), key=lambda s: (s[3][1], s[3][2]))
    wanted = defaultdict(dict)
    for entry in selected:
        wanted[entry[3][1]][entry[3][2]] = entry

    # Pass 2: copy and transform only the selected lines.
    out_sha = hashlib.sha256()
    written = 0
    # The corpus holds whole request bodies (user code, tool output): owner only. It is written
    # under a temporary name and renamed once complete, so a run that stops leaves no corpus that
    # a replay could take for a whole one.
    partial = args.out + ".partial"
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    try:
        with os.fdopen(fd, "w") as out:
            for path in sorted(wanted):
                lines = wanted[path]
                last = max(lines)
                for line_no, raw in iter_lines(path):
                    if line_no > last:  # what a live shard appended after pass 1 is not read
                        break
                    entry = lines.get(line_no)
                    if entry is None:
                        continue
                    key, turn_index, turns_in_thread, (_, _, _, line_sha, trace_id) = entry
                    if hashlib.sha256(raw).hexdigest() != line_sha:
                        counts["drop:source_changed"] += 1
                        continue
                    rec = json.loads(raw)
                    body, transforms = transform(rec["body"], args)
                    written += 1
                    line = json.dumps(
                        {
                            "corpus_id": f"c{written:07d}",
                            "route": args.route,
                            "thread": key,
                            "turn_index": turn_index,
                            "turns_in_thread": turns_in_thread,
                            "source": {
                                "file": path,
                                "line": line_no,
                                "line_sha256": line_sha,
                                "trace_id": trace_id,
                            },
                            "original_body_sha256": hashlib.sha256(
                                json.dumps(rec["body"], sort_keys=True, ensure_ascii=False).encode()
                            ).hexdigest(),
                            "transforms": transforms,
                            "body": body,
                        },
                        ensure_ascii=False,
                    )
                    out.write(line + "\n")
                    out_sha.update((line + "\n").encode())
    except BaseException:
        os.unlink(partial)
        raise
    os.replace(partial, args.out)
    counts["threads"] = len(by_thread)
    counts["selected"] = len(selected)
    counts["written"] = written
    manifest = {
        "tool": "build_replay_corpus",
        "parameters": {k: v for k, v in vars(args).items() if k != "paths"},
        "sources": [
            {"file": p, "bytes": os.path.getsize(p), "mtime": int(os.path.getmtime(p))}
            for p in shards
        ],
        "counts": dict(counts),
        "output": {"file": args.out, "sha256": out_sha.hexdigest()},
    }
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    try:
        main()
    except ShardError as error:  # a corpus from part of a shard would read as all of it
        sys.exit(str(error))
