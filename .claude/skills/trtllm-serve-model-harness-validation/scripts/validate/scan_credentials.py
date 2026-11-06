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
"""Strong-credential scan of trace shards or any text files. Never prints a value.

Agent traffic leaks credentials into prompts: a sandboxed agent runs ``curl -v``
with a token from its environment (the expanded ``Authorization: Bearer`` header
lands in the tool output), prints its environment, or prints a helper's
auth-header dict, and every later turn of the session replays that output.
Training and replay corpora must not carry them.

Per file and pattern: the number of matches and the distinct values, each
identified only by the first 12 hex digits of its SHA-256 and its length. With
``--known FILE`` (one SHA-256 hex digest per line, '#' comments allowed) each
value is marked KNOWN or NEW, and full digests of NEW values are listed so they
can be added to the known list -- a digest, never the value.

Provenance still needs a human or a validator: a model can invent a Bearer-shaped
string or write ``sk-`` as shorthand. Report hits as suspicions until the value
is shown to be a real credential.

A file that cannot be read to its end gets the error in its row, next to the
hits of the lines read before the damage; those hits count like any others.

``--restrict`` chmods files with hits to 0400. Only use it on CLOSED shards: a
writer that re-opens a 0400 file it still owns fails with EACCES.

usage: scan_credentials.py [--known digests.txt] [--restrict] --out report.json PATH [PATH ...]
"""

import argparse
import hashlib
import json
import os
import re
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import iter_lines  # noqa: E402

# What a file that cannot be read raises: it is reported in the file's row, and the scan goes on.
FILE_ERRORS = (OSError, EOFError, ValueError)
try:
    import zstandard

    FILE_ERRORS += (zstandard.ZstdError,)  # a .zst shard that is not zstd
except ImportError:  # without zstandard no .zst shard is read at all
    pass

# (name, pattern, word boundary required before the match). A pattern with a
# named group "value" is digested on that group, else on the whole match.
PATTERNS = (
    ("bearer", re.compile(rb"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/-]{20,})"), False),
    ("openai_sk", re.compile(rb"sk-(?:proj-|live-|svcacct-)?[A-Za-z0-9_-]{20,}"), True),
    ("nvapi", re.compile(rb"nvapi-[A-Za-z0-9_-]{20,}"), True),
    ("github_token", re.compile(rb"(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"), True),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}\b"), True),
    ("slack_token", re.compile(rb"xox[abprs]-[A-Za-z0-9-]{10,}"), True),
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"), False),
    ("jwt", re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), True),
    ("hf_token", re.compile(rb"hf_[A-Za-z0-9]{30,}"), True),
    ("google_api_key", re.compile(rb"AIza[0-9A-Za-z_-]{35}\b"), True),
    # A secret in an environment dump (`env` in a tool result): NAME=value for token-like names.
    # The name may follow a JSON-escaped newline ("\\nAPI_TOKEN="), so no word boundary.
    (
        "secret_assignment",
        re.compile(
            rb"(?<![A-Z0-9_])[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)[A-Z0-9_]*"
            rb"=(?P<value>[A-Za-z0-9._~+/=-]{16,})"
        ),
        False,
    ),
)


def matches(raw):
    """Yield (pattern_name, value_bytes) for every strong-credential match in one line."""
    for name, rx, boundary in PATTERNS:
        for m in rx.finditer(raw):
            i = m.start()
            if boundary and i > 0 and (raw[i - 1 : i].isalnum() or raw[i - 1 : i] == b"_"):
                continue
            yield name, m.group("value") if "value" in rx.groupindex else m.group(0)


def expand(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            for dirpath, _, names in os.walk(path, followlinks=True):
                files += [os.path.join(dirpath, n) for n in names]
        else:
            files.append(path)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--known", help="file of known SHA-256 digests, one per line")
    ap.add_argument(
        "--min-bearer-chars",
        type=int,
        default=20,
        help="ignore Bearer values shorter than this (model-invented examples are often short)",
    )
    ap.add_argument(
        "--restrict", action="store_true", help="chmod 0400 files with hits (closed files only)"
    )
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    known = set()
    if args.known:
        with open(args.known) as fh:
            known = {line.split()[0] for line in fh if line.strip() and not line.startswith("#")}

    files = expand(args.paths)
    report_files, new_digests, totals = [], {}, {}
    for path in files:
        per, error = {}, None
        try:
            for _, raw in iter_lines(path):
                for name, value in matches(raw):
                    if name == "bearer" and len(value) < args.min_bearer_chars:
                        continue
                    digest = hashlib.sha256(value).hexdigest()
                    entry = per.setdefault(name, {})
                    slot = entry.setdefault(digest, {"count": 0, "chars": len(value)})
                    slot["count"] += 1
        except FILE_ERRORS as exc:
            # The lines before the damage were read: their hits count as any file's do.
            error = type(exc).__name__
        if not per:
            if error:
                report_files.append({"file": path, "error": error})
            continue
        row = {"file": path, "patterns": {}}
        if error:
            row["error"] = error
        for name, values in per.items():
            row["patterns"][name] = [
                {
                    "sha256_12": d[:12],
                    "chars": v["chars"],
                    "count": v["count"],
                    "known": (d in known) if known else None,
                }
                for d, v in sorted(values.items(), key=lambda kv: -kv[1]["count"])
            ]
            totals[name] = totals.get(name, 0) + sum(v["count"] for v in values.values())
            for d in values:
                if known and d not in known:
                    new_digests.setdefault(d, {"pattern": name, "first_file": path})
        if args.restrict:
            before = stat.S_IMODE(os.stat(path).st_mode)
            os.chmod(path, 0o400)
            row["mode"] = f"{before:o}->400"
        report_files.append(row)

    report = {
        "tool": "scan_credentials",
        "files_scanned": len(files),
        "files_with_hits": sum(1 for r in report_files if "patterns" in r),
        "matches_by_pattern": totals,
        "new_digests": new_digests if known else None,
        "files": report_files,
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(
        json.dumps(
            {
                "files_scanned": len(files),
                "files_with_hits": report["files_with_hits"],
                "matches_by_pattern": totals,
                "new_values": len(new_digests) if known else None,
            }
        )
    )


if __name__ == "__main__":
    main()
