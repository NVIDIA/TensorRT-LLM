# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the CUDA-graph capture lists a survey's dispatch carries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from ...checks import check_config
from ._render import render_template

#: The heading of the dispatch's block, which the `survey` tag's section points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"


@dataclass(frozen=True)
class _CapturePreview:
    name: str
    unit: str
    description: str | None
    source: str


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """Resolve each unit's gate preview, differing variants and defaults."""
    error: str | None = None
    try:
        captures = check_config.launch_captures(task_data)
    except (ValueError, yaml.YAMLError) as exc:
        error = str(exc)
        captures = []
    units = {c.name: c for c in captures if c.name == c.unit}
    inherited: list[str] = []
    listed: list[_CapturePreview] = []
    for capture in captures:
        base = units.get(capture.unit)
        if capture.name != capture.unit and base is not None and base.resolved == capture.resolved:
            inherited.append(capture.name)
        else:
            listed.append(
                _CapturePreview(
                    capture.name,
                    capture.unit,
                    capture.resolved.describe() if capture.resolved is not None else None,
                    capture.resolved.source if capture.resolved is not None else "",
                )
            )
    return render_template(
        "survey.md.j2",
        heading=CAPTURE_LISTS_HEADING,
        error=error,
        captures=listed,
        inherited=inherited,
        default_capture=check_config.LLM_API_DEFAULT_CAPTURE.describe(),
    )


__all__ = ["CAPTURE_LISTS_HEADING", "capture_lists_block"]
