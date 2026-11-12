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

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterable

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TIMEOUT_EVIDENCE_MARKER = "TRTLLM_PYTEST_TIMEOUT "
_TIMEOUT_ATTEMPT_MARKER = "TRTLLM_PYTEST_ATTEMPT "


def _nodeid_set(nodeids: Iterable[str]) -> set[str]:
    """Normalize non-empty nodeids for exact matching."""
    return {nodeid.strip() for nodeid in nodeids if nodeid.strip()}


def _json_record_from_line(raw_line: str, marker: str) -> dict[str, object] | None:
    """Decode one structured evidence record with an optional log prefix."""
    line = _ANSI_ESCAPE_RE.sub("", raw_line)
    marker_index = line.find(marker)
    if marker_index < 0:
        return None

    payload = line[marker_index + len(marker) :].lstrip()
    try:
        record, _ = json.JSONDecoder().raw_decode(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _timeout_event_from_line(raw_line: str) -> tuple[str, str, int, float | None] | None:
    """Return a validated timeout or in-process attempt event."""
    record = _json_record_from_line(raw_line, _TIMEOUT_EVIDENCE_MARKER)
    event_type = "timeout"
    if record is None:
        record = _json_record_from_line(raw_line, _TIMEOUT_ATTEMPT_MARKER)
        event_type = "attempt"
    if record is None:
        return None

    nodeid = record.get("nodeid")
    if not isinstance(nodeid, str):
        return None
    attempt = record.get("attempt", 1)
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
        return None
    if event_type == "attempt":
        return event_type, nodeid, attempt, None

    timeout = record.get("timeout")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        return None
    try:
        timeout_value = float(timeout)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        return None
    return event_type, nodeid, attempt, timeout_value


def timeout_evidence_from_lines(
    lines: Iterable[str], unfinished_nodeids: Iterable[str]
) -> dict[str, float]:
    """Return timeout durations for unfinished nodeids in structured records."""
    candidates = _nodeid_set(unfinished_nodeids)
    latest_attempts: dict[str, int] = {}
    timeouts: dict[tuple[str, int], float] = {}

    for raw_line in lines:
        event = _timeout_event_from_line(raw_line)
        if event is None:
            continue
        event_type, nodeid, attempt, timeout = event
        if nodeid not in candidates:
            continue
        latest_attempts[nodeid] = max(latest_attempts.get(nodeid, 0), attempt)
        if event_type == "timeout" and timeout is not None:
            timeouts[(nodeid, attempt)] = timeout

    return {
        nodeid: timeouts[(nodeid, attempt)]
        for nodeid, attempt in latest_attempts.items()
        if (nodeid, attempt) in timeouts
    }


def load_timeout_evidence(path: str | None, unfinished_nodeids: Iterable[str]) -> dict[str, float]:
    """Load structured pytest-timeout evidence for unfinished nodeids."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return timeout_evidence_from_lines(f, unfinished_nodeids)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        print(
            f"WARNING: cannot read timeout evidence file {path}: {exc}",
            file=sys.stderr,
        )
        return {}


def retain_unfinished_timeout_evidence(path: str, unfinished_nodeids: Iterable[str]) -> None:
    """Discard records from completed attempts before a rerun starts."""
    candidates = _nodeid_set(unfinished_nodeids)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"WARNING: cannot read timeout evidence file {path}: {exc}", file=sys.stderr)
        return

    retained_lines = []
    for line in lines:
        event = _timeout_event_from_line(line)
        if event is not None and event[1] in candidates:
            retained_lines.append(line if line.endswith("\n") else line + "\n")

    temporary_path = f"{path}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            f.writelines(retained_lines)
        os.replace(temporary_path, path)
    except OSError as exc:
        print(f"WARNING: cannot prune timeout evidence file {path}: {exc}", file=sys.stderr)
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep timeout evidence only for tests still unfinished before a rerun"
    )
    parser.add_argument("--evidence-file", required=True)
    parser.add_argument("--unfinished-test-file", required=True)
    args = parser.parse_args()

    try:
        with open(args.unfinished_test_file, encoding="utf-8", errors="replace") as f:
            unfinished_nodeids = f.readlines()
    except FileNotFoundError:
        unfinished_nodeids = []
    except OSError as exc:
        print(
            f"WARNING: cannot read unfinished test file {args.unfinished_test_file}: {exc}",
            file=sys.stderr,
        )
        return

    retain_unfinished_timeout_evidence(args.evidence_file, unfinished_nodeids)


if __name__ == "__main__":
    main()
