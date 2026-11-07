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
"""Replay a corpus (build_replay_corpus.py) against a serving endpoint and record what came back.

Every corpus entry is POSTed to ``<base-url><route>`` as recorded (after the
corpus transforms). Streamed replies are kept byte-exact, the way trtllm-serve's
request trace keeps them, and the output uses the request-trace record shapes:

    <out-dir>/requests-replay.jsonl    {"event": "request", "trace_id", "route", "body", "replay": {...}}
    <out-dir>/responses-replay.jsonl   {"event": "response", "trace_id", "status", "http_status",
                                        "response": {"kind": "sse_text"|"json"|"text", "body"}, "timing"}

``timing`` holds ``headers_s``, ``first_byte_s``, ``first_event_s``, ``total_s`` and,
for streams, ``chunks``: ``[seconds since the request was sent, bytes]`` per
received chunk, so the validator can measure the longest silence (check R-02).
The conformance checks (scripts/validate/) read replay output exactly like
recorded traffic.

Client behavior mirrors what agent harnesses do to a slow server:
  --first-event-timeout S   give up when no SSE frame arrived within S seconds
                            (default 120; use the harness profile's timeouts.first_event_s)
  --timeout S               overall cap per request
Status values: completed, client_first_event_timeout, client_timeout,
connection_error, http_error (a non-2xx answer; its body is recorded).

--build-id ID writes the served build's identity into every record, request and response, as
record_requests.py does: pass the same value to the verdict, or it cannot prove which build
answered the replay (I-05).

The API key, when needed, is read from the environment variable named by
--api-key-env and is never written to any output.

usage: replay_requests.py --base-url URL --out-dir DIR [--concurrency N] [--build-id ID]
                          corpus.jsonl
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from collections import Counter

try:
    import aiohttp
except ImportError:
    sys.exit("replay_requests.py needs aiohttp (pip install aiohttp)")


def open_private(path):
    """``path`` opened for appending and readable by its owner only, whether new or not."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "a")


def utc_now():
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        + f".{int(time.time() * 1e6) % 1000000:06d}Z"
    )


async def replay_one(session, entry, args, headers):
    url = args.base_url.rstrip("/") + entry.get("route", "/v1/responses")
    headers = dict(headers)
    if entry.get("route", "").startswith("/v1/messages"):
        # Anthropic clients send a version header, and their key as x-api-key.
        headers.setdefault("anthropic-version", "2023-06-01")
        if "authorization" in headers:
            headers.setdefault("x-api-key", headers["authorization"].split(" ", 1)[-1])
    body = entry["body"]
    trace_id = "rp_" + uuid.uuid4().hex
    request_line = {
        "event": "request",
        "trace_id": trace_id,
        "recorded_at": utc_now(),
        "route": entry.get("route", "/v1/responses"),
        "status": "accepted",
        "body": body,
        "replay": {
            "corpus_id": entry.get("corpus_id"),
            "source": entry.get("source"),
            "thread": entry.get("thread"),
            "transforms": entry.get("transforms"),
        },
    }
    timing, status, http_status, payload, kind = {}, "completed", None, None, "text"
    start = time.monotonic()
    try:
        async with session.post(
            url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as resp:
            http_status = resp.status
            timing["headers_s"] = round(time.monotonic() - start, 4)
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                kind, chunks, buffered = "sse_text", [], b""
                first_event = None
                iterator = resp.content.iter_any().__aiter__()
                while True:
                    remaining = None
                    if first_event is None:
                        remaining = args.first_event_timeout - (time.monotonic() - start)
                        if remaining <= 0:
                            status = "client_first_event_timeout"
                            break
                    try:
                        chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        # Two limits run here: --first-event-timeout (wait_for) until the first
                        # event, and --timeout (aiohttp's total for the request) throughout.
                        # Before the first event, the one that fired is the earlier of the two.
                        first = first_event is None and args.first_event_timeout <= args.timeout
                        status = "client_first_event_timeout" if first else "client_timeout"
                        break
                    if not chunk:
                        continue
                    now = round(time.monotonic() - start, 4)
                    if "first_byte_s" not in timing:
                        timing["first_byte_s"] = now
                    timing.setdefault("chunks", []).append([now, len(chunk)])
                    chunks.append(chunk)
                    if first_event is None:
                        buffered += chunk
                        if b"\n\n" in buffered.replace(b"\r\n", b"\n"):  # LF or CRLF framing
                            first_event = time.monotonic() - start
                            timing["first_event_s"] = round(first_event, 4)
                payload = b"".join(chunks).decode("utf-8", "replace")
            else:
                raw = await resp.read()
                text = raw.decode("utf-8", "replace")
                try:
                    payload, kind = json.loads(text), "json"
                except json.JSONDecodeError:
                    payload = text
            if http_status >= 400 and status == "completed":
                status = "http_error"
    except asyncio.TimeoutError:
        status = "client_timeout"
    except aiohttp.ClientError as error:
        status = "connection_error"
        payload, kind = type(error).__name__, "text"
    timing["total_s"] = round(time.monotonic() - start, 4)
    response_line = {
        "event": "response",
        "trace_id": trace_id,
        "finished_at": utc_now(),
        "status": status,
        "http_status": http_status,
        "response": {"kind": kind, "body": payload},
        "timing": timing,
        "replay": {"corpus_id": entry.get("corpus_id")},
    }
    if args.build_id:
        request_line["build_id"] = response_line["build_id"] = args.build_id
    return request_line, response_line


async def run(args):
    with open(args.corpus) as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        entries = entries[: args.limit]
    headers = {"content-type": "application/json"}
    for header in args.header or []:
        name, _, value = header.partition(":")
        headers[name.strip()] = value.strip()
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            sys.exit(f"environment variable {args.api_key_env} is not set")
        headers["authorization"] = "Bearer " + key
    if args.dry_run:
        routes = Counter(e.get("route") for e in entries)
        print(json.dumps({"entries": len(entries), "routes": dict(routes)}))
        return
    # The replies can hold credentials and user code, as the corpus can: owner only.
    os.makedirs(args.out_dir, mode=0o700, exist_ok=True)
    req_path = os.path.join(args.out_dir, "requests-replay.jsonl")
    resp_path = os.path.join(args.out_dir, "responses-replay.jsonl")
    statuses, totals, firsts = Counter(), [], []
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        with open_private(req_path) as req_out, open_private(resp_path) as resp_out:

            async def worker(entry):
                async with semaphore:
                    req_line, resp_line = await replay_one(session, entry, args, headers)
                req_out.write(json.dumps(req_line, ensure_ascii=False) + "\n")
                resp_out.write(json.dumps(resp_line, ensure_ascii=False) + "\n")
                statuses[f"{resp_line['status']}:{resp_line['http_status']}"] += 1
                totals.append(resp_line["timing"]["total_s"])
                if "first_event_s" in resp_line["timing"]:
                    firsts.append(resp_line["timing"]["first_event_s"])

            await asyncio.gather(*(worker(e) for e in entries))

    def pct(values, q):
        return round(statistics.quantiles(values, n=100)[q - 1], 3) if len(values) >= 2 else None

    summary = {
        "entries": len(entries),
        "statuses": dict(statuses),
        "first_event_s": {"p50": pct(firsts, 50), "p90": pct(firsts, 90), "p99": pct(firsts, 99)},
        "total_s": {"p50": pct(totals, 50), "p90": pct(totals, 90), "p99": pct(totals, 99)},
        "out": {"requests": req_path, "responses": resp_path},
    }
    with open(os.path.join(args.out_dir, "replay_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--first-event-timeout", type=float, default=120.0)
    ap.add_argument("--api-key-env", help="name of the env var holding the API key (never logged)")
    ap.add_argument("--header", action="append", help="extra 'Name: value' header (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build-id", help="the served build's identity, written into every record")
    ap.add_argument("corpus")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
