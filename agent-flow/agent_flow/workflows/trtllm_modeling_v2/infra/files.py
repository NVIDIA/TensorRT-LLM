# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-file reads and atomic replacement for controllers and standalone jobs."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Mapping


def write_text_atomic(path: Path, text: str, *, mode: int = 0o666) -> None:
    """Replace a file whole, preserving its old contents if writing fails.

    Each writer owns a unique sibling temporary file, including concurrent
    threads in one process. The caller owns creation of the parent directory.
    ``mode`` is filtered by the process umask, as for an ordinary file write.
    """
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    """Replace a JSON record whole, with deterministic formatting."""
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict | None:
    """Read a mapping record; missing, partial or non-mapping records have no value."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


__all__ = ["read_json", "write_json_atomic", "write_text_atomic"]
