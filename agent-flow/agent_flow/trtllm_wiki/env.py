# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TLLM_*`` / ``TRTLLM_*`` environment variables the checkout reads, and where.

A read is a string literal naming the variable inside ``os.environ``,
``os.getenv``, ``getenv``/``std::getenv`` or a ``get_env*`` helper call on the
same line. Python under ``tensorrt_llm/`` and C++ under ``cpp/`` (when the
checkout has it) are scanned.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_NAME = r"(?P<name>(?:TLLM|TRTLLM)_[A-Z0-9_]+)"
_READ = re.compile(
    r"(?:environ(?:\.get)?\s*[\(\[]|getenv\s*\(|get_env[a-z_]*\s*\(|getEnv[A-Za-z]*\s*\()\s*"
    rf"[\"']{_NAME}[\"']"
)
_ROOTS = (("tensorrt_llm", ("*.py",)), ("cpp", ("*.cpp", "*.h", "*.cu")))


def extract_env_vars(repo: Path) -> list[dict[str, Any]]:
    repo = Path(repo)
    reads: dict[str, list[str]] = {}
    for root, patterns in _ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if "TLLM_" not in text:
                    continue
                rel = path.relative_to(repo).as_posix()
                for number, line in enumerate(text.splitlines(), 1):
                    for match in _READ.finditer(line):
                        reads.setdefault(match["name"], []).append(f"{rel}:{number}")
    return [{"name": name, "reads": sorted(set(where))} for name, where in sorted(reads.items())]
