# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record format a rank prints: ``<MARKER> <json>``, the JSON on one line.

A check reads ONLY those lines, never the engine's log, so a prefix added by a launcher
or a logger does not hide a record, and nothing else in a log is taken for one.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: What one rank's executor worker runs, printed by that worker as it sets its
#: engine up, once per rank (see ``rankhook/sitecustomize.py``).
RANK_ENGINE_MARKER = "MV2FLOW_RANK_ENGINE"


def parse_marked(output: str, marker: str) -> list[dict[str, Any]]:
    """Every ``<marker> <json>`` record in ``output``, in order.

    Anything may precede the marker on its line (a rank tag, a logger
    prefix). A line whose JSON does not parse is skipped rather than raised:
    a torn last line of a killed job must not hide the records before it.
    """
    pattern = re.compile(rf"{re.escape(marker)} (\{{.*\}})\s*$")
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        found = pattern.search(line)
        if not found:
            continue
        try:
            data = json.loads(found.group(1))
        except ValueError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records
