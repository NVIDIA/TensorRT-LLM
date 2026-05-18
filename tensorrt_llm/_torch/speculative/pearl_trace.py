# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small JSONL trace helper for PEARL / draft-offload debugging."""

import json
import os
import time
from typing import Any


def trace_path(role: str) -> str:
    role = str(role).upper()
    return os.environ.get(f"PEARL_{role}_TRACE_PATH") or os.environ.get("PEARL_TRACE_PATH") or ""


def enabled(role: str) -> bool:
    return bool(trace_path(role))


def to_int_list(value: Any, *, limit: int | None = None) -> list[int]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().reshape(-1).cpu().tolist()
        elif hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
        else:
            value = list(value)
    except Exception:
        return []
    if limit is not None:
        value = value[: max(0, int(limit))]
    out = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            pass
    return out


def tensor_rows(value: Any, rows: int, *, width: int | None = None) -> list[list[int]]:
    if value is None:
        return []
    out = []
    try:
        tensor = value.detach().cpu() if hasattr(value, "detach") else value
        for row in range(int(rows)):
            row_value = tensor[row]
            if width is not None:
                row_value = row_value.reshape(-1)[: int(width)]
            out.append(to_int_list(row_value))
    except Exception:
        return []
    return out


def log(role: str, event: str, **fields: Any) -> None:
    path = trace_path(role)
    if not path:
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        "role": str(role).lower(),
        "event": str(event),
    }
    record.update(fields)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass
