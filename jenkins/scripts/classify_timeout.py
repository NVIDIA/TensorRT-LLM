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
r"""Scan a pytest output log and emit confirmed pytest-timeout records as NDJSON.

For each test that was killed by pytest-timeout (``--timeout-method=thread``),
the timer thread prints two banner lines before calling ``os._exit(1)``:

    <stage>/<file>::[Class::]<method>[params] <- <file.py>  +++...Timeout...+++
    +++...Timeout...+++

This script identifies those banner lines, extracts the surrounding snippet
(first banner through second banner inclusive), and appends one JSON record
per confirmed timeout to an NDJSON file:

    {"nodeid": "<stage-prefixed nodeid>", "snippet": "<log excerpt>"}

The nodeid exactly matches the format used in ``unfinished_test.txt``, so the
caller can intersect the two files to produce the final classification.

Usage::

    python3 classify_timeout.py \\
        --log   <pytest_output.log> \\
        --out   <timeout_data.jsonl> \\
        --unfinished <unfinished_test.txt>
"""

import argparse
import json
import os
import re
import sys
import time

# Regex patterns for banner line detection.
# The first banner contains the nodeid; the second is a plain separator.
_ARROW_RE = re.compile(r" <- ")
_TIMEOUT_MARKER_RE = re.compile(r"\+{5,}.*Timeout.*\+{5,}")

# Snippet size limits.
_MAX_SNIPPET_LINES = 5000
_MAX_SNIPPET_BYTES = 1 * 1024 * 1024  # 1 MiB
_TRUNCATED_MARKER = "\n[truncated]"


def _is_first_banner(line, unfinished):
    """Return (True, nodeid) if *line* is a first timeout banner for a known nodeid.

    A first banner contains a ``+++...Timeout...+++`` marker and a known
    nodeid.  With the CI's normal ``pytest -vv`` invocation, pytest's terminal
    reporter includes the ``<-`` separator, so the nodeid is extracted from
    its left-hand side.  At lower verbosity pytest omits that separator; fall
    back to a bare ``<nodeid> `` prefix so a verbosity change does not silently
    disable timeout classification.
    """
    if not _TIMEOUT_MARKER_RE.search(line):
        return False, ""

    if _ARROW_RE.search(line):
        # The nodeid is the part before " <- " when pytest runs with -vv.
        nodeid = _ARROW_RE.split(line, maxsplit=1)[0].strip()
        if nodeid in unfinished:
            return True, nodeid

    # pytest -v omits " <- ".  A whitespace boundary prevents a shorter
    # parameterized nodeid from matching the prefix of a longer one.
    for nodeid in unfinished:
        if line.startswith(nodeid) and len(line) > len(nodeid) and line[len(nodeid)].isspace():
            return True, nodeid
    return False, ""


def _is_second_banner(line):
    """Return True if *line* is a plain timeout separator (no ``<-``)."""
    return bool(_TIMEOUT_MARKER_RE.search(line)) and not _ARROW_RE.search(line)


def _load_unfinished(path):
    """Load nodeid set from *path*; return empty set if the file is missing."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


def _scan_log(log_path, unfinished):
    """Scan *log_path* line-by-line and return a list of ``{nodeid, snippet}`` dicts.

    The file is read as a stream so that arbitrarily large logs do not cause
    memory pressure; only the current snippet window (≤ 5000 lines / 1 MiB) is
    kept in memory at any time.
    """
    try:
        f = open(log_path, encoding="utf-8", errors="replace")
    except OSError:
        return []

    records = []
    with f:
        for line in f:
            matched, nodeid = _is_first_banner(line, unfinished)
            if not matched:
                continue

            # Collect snippet starting from the first banner line.
            snippet_lines = [line]
            total_bytes = len(line.encode("utf-8", errors="replace"))
            truncated = False

            for next_line in f:
                if _is_second_banner(next_line):
                    snippet_lines.append(next_line)
                    break
                # Check size limits before appending.
                line_bytes = len(next_line.encode("utf-8", errors="replace"))
                if (
                    len(snippet_lines) >= _MAX_SNIPPET_LINES
                    or total_bytes + line_bytes > _MAX_SNIPPET_BYTES
                ):
                    truncated = True
                    # Drain until the second banner so the outer loop resumes
                    # correctly after this block.
                    for skip_line in f:
                        if _is_second_banner(skip_line):
                            break
                    break
                snippet_lines.append(next_line)
                total_bytes += line_bytes

            snippet = "".join(snippet_lines)
            if truncated:
                snippet += _TRUNCATED_MARKER

            records.append({"type": "timeout", "nodeid": nodeid, "snippet": snippet})

    return records


def _append_records(out_path, records):
    """Append *records* to *out_path* in NDJSON format.

    Creates parent directories if they do not exist.  If the file cannot be
    opened after that, a warning is printed and the function returns without
    raising so that the wrapper script can still exit with pytest's original
    exit code.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"classify_timeout: WARNING: could not write {out_path}: {exc}", file=sys.stderr)


def _append_unfinished_end_times(out_path, unfinished):
    """Record the wall-clock end time for tests left unfinished by pytest."""
    if not out_path:
        return
    end_time = time.time()
    _append_records(
        out_path,
        [{"type": "end", "nodeid": nodeid, "end_time": end_time} for nodeid in sorted(unfinished)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify pytest-timeout kills from a captured pytest output log."
    )
    parser.add_argument("--log", required=True, help="Path to pytest_output.log")
    parser.add_argument(
        "--out", required=True, help="Path to timeout_data.jsonl (NDJSON, appended)"
    )
    parser.add_argument(
        "--unfinished",
        required=True,
        help="Path to unfinished_test.txt (nodeid set for intersection)",
    )
    args = parser.parse_args()

    unfinished = _load_unfinished(args.unfinished)
    if not unfinished:
        # Nothing to match against; exit cleanly without touching --out.
        return

    _append_unfinished_end_times(args.out, unfinished)
    records = _scan_log(args.log, unfinished)
    if not records:
        return

    _append_records(args.out, records)
    print(
        f"classify_timeout: wrote {len(records)} record(s) to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
