# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join finished performance runs; publish meta.json last as the completion marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

PARTS_FILE = "parts.json"


def write_index(
    out_root: Path,
    label: str,
    parts: Sequence[str],
    constraints: Mapping[str, Mapping[str, float]],
) -> None:
    """Keep every member's host; a cross-host measurement is no host's control."""
    root = Path(out_root) / label
    records = {name: json.loads((root / name / "meta.json").read_text()) for name in parts}
    hosts = {name: str(meta.get("hostname") or "") for name, meta in records.items()}
    meta = dict(records[parts[0]])
    shared = set(hosts.values())
    meta.update(
        label=label,
        parts=list(hosts),
        hosts=hosts,
        hostname=shared.pop() if len(shared) == 1 else "",
    )
    index = {"parts": list(hosts), "constraints": dict(constraints)}
    (root / PARTS_FILE).write_text(json.dumps(index))
    (root / "summary.json").write_text("[]")
    (root / "meta.json").write_text(json.dumps(meta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--index", required=True, help="JSON parts and constraints")
    args = parser.parse_args()
    index = json.loads(args.index)
    write_index(args.out_root, args.label, index["parts"], index["constraints"])


if __name__ == "__main__":
    main()
