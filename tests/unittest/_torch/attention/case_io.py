# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialization of :class:`BackendCase` to/from JSON / JSONL.

A captured real-workload case is just a serialized ``BackendCase``. The capture
writer (``tensorrt_llm/_torch/attention_backend/_capture.py``) appends one JSON
object per line to a ``.jsonl``; the minimizer and replay test read them back
here. Committed fixtures may be single-object ``.json`` files (one minimized
case) or multi-line ``.jsonl``.
"""

import json
from pathlib import Path
from typing import Iterator, List, Union

from attention_test_harness import BackendCase


def dump_case_line(case: BackendCase, path: Union[str, Path]) -> None:
    """Append one case as a JSON line (used to build fixtures from captures)."""
    with open(path, "a") as f:
        f.write(json.dumps(case.to_dict()) + "\n")


def dump_case(case: BackendCase, path: Union[str, Path]) -> None:
    """Write a single case as a pretty JSON object (committed fixture form)."""
    Path(path).write_text(json.dumps(case.to_dict(), indent=2))


def _load_text(text: str) -> List[dict]:
    text = text.strip()
    if not text:
        return []
    # A single JSON object or a JSON list?
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    # Otherwise treat as JSONL.
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def iter_case_specs(path: Union[str, Path]) -> Iterator[BackendCase]:
    """Yield BackendCase from a ``.json``/``.jsonl`` file or a directory of them."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl"))
    elif p.exists():
        files = [p]
    else:
        files = []
    for f in files:
        for d in _load_text(f.read_text()):
            yield BackendCase.from_dict(d)
