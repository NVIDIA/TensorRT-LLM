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
r"""Extract pytest-timeout banner blocks for unfinished tests as NDJSON."""

import argparse
import json
import os
import re
import sys

_ARROW_RE = re.compile(r" <- ")
_TIMEOUT_MARKER_RE = re.compile(r"\+{5,}.*Timeout.*\+{5,}")
_MAX_SNIPPET_LINES = 5000
_MAX_SNIPPET_BYTES = 1024 * 1024
_TRUNCATED_MARKER = "\n[truncated]"


def _is_first_banner(line: str, unfinished: set[str]) -> tuple[bool, str]:
    """Match the first timeout banner and return its unfinished nodeid."""
    if not _TIMEOUT_MARKER_RE.search(line):
        return False, ""

    if _ARROW_RE.search(line):
        nodeid = _ARROW_RE.split(line, maxsplit=1)[0].strip()
        if nodeid in unfinished:
            return True, nodeid

    # pytest -v omits " <- "; require a boundary after a bare nodeid.
    for nodeid in unfinished:
        if line.startswith(nodeid) and len(line) > len(nodeid) and line[len(nodeid)].isspace():
            return True, nodeid
    return False, ""


def _is_second_banner(line: str) -> bool:
    return bool(_TIMEOUT_MARKER_RE.search(line)) and not _ARROW_RE.search(line)


def _load_unfinished(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


def _scan_log(log_path: str, unfinished: set[str]) -> list[dict[str, str]]:
    """Return bounded timeout snippets while streaming the log once."""
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

            snippet_lines = [line]
            total_bytes = len(line.encode("utf-8", errors="replace"))
            truncated = False

            for next_line in f:
                if _is_second_banner(next_line):
                    snippet_lines.append(next_line)
                    break
                line_bytes = len(next_line.encode("utf-8", errors="replace"))
                if (
                    len(snippet_lines) >= _MAX_SNIPPET_LINES
                    or total_bytes + line_bytes > _MAX_SNIPPET_BYTES
                ):
                    truncated = True
                    # Drain this block so the outer loop resumes at the next test.
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


def _append_records(out_path: str, records: list[dict[str, str]]) -> None:
    """Append records best-effort so classification cannot fail pytest."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"classify_timeout: WARNING: could not write {out_path}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify pytest-timeout kills from a captured pytest output log."
    )
    parser.add_argument("--log", required=True, help="Captured pytest log")
    parser.add_argument("--out", required=True, help="Appended timeout NDJSON")
    parser.add_argument(
        "--unfinished",
        required=True,
        help="unfinished_test.txt used to confirm nodeids",
    )
    args = parser.parse_args()

    unfinished = _load_unfinished(args.unfinished)
    if not unfinished:
        return

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
