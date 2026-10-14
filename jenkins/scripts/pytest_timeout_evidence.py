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

import json
import re
import sys

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TIMEOUT_EVIDENCE_MARKER = "TRTLLM_PYTEST_TIMEOUT "


def timeout_evidence_from_lines(lines, unfinished_nodeids):
    """Return timeout durations for unfinished nodeids in structured records."""
    candidates = {nodeid.strip() for nodeid in unfinished_nodeids if nodeid.strip()}
    confirmed = {}
    decoder = json.JSONDecoder()

    for raw_line in lines:
        line = _ANSI_ESCAPE_RE.sub("", raw_line)
        marker_index = line.find(_TIMEOUT_EVIDENCE_MARKER)
        if marker_index < 0:
            continue

        payload = line[marker_index + len(_TIMEOUT_EVIDENCE_MARKER) :].lstrip()
        try:
            record, _ = decoder.raw_decode(payload)
            nodeid = record.get("nodeid") if isinstance(record, dict) else None
            timeout = record.get("timeout") if isinstance(record, dict) else None
            if nodeid not in candidates:
                continue
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
                continue
            # A rerun may emit another record for the same nodeid. The final
            # record represents the attempt that remains unfinished.
            confirmed[nodeid] = float(timeout)
        except (json.JSONDecodeError, ValueError):
            continue

    return confirmed


def load_timeout_evidence(path, unfinished_nodeids):
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
