# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the CUDA-graph capture lists a survey's dispatch carries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ...checks import check_config
from ...tune.serving_config import derive_serving_config
from ._render import render_template

#: The heading of the dispatch's block, which the `survey` tag's section points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"


@dataclass(frozen=True)
class _CapturePreview:
    name: str
    unit: str
    description: str | None
    source: str


@dataclass(frozen=True)
class _TuneCapture:
    name: str
    description: str | None
    path: str = ""
    readable: bool = True
    overlay: bool = False
    max_concurrency: int = 0
    error: str = ""


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """Resolve each unit's gate preview, differing variants, defaults and tune configurations."""
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
    perf_enabled = bool((task_data.get("perf") or {}).get("enabled"))
    return render_template(
        "survey.md.j2",
        heading=CAPTURE_LISTS_HEADING,
        error=error,
        captures=listed,
        inherited=inherited,
        default_capture=check_config.LLM_API_DEFAULT_CAPTURE.describe(),
        perf_enabled=perf_enabled,
        tune=_tune_captures(task_data) if perf_enabled else [],
    )


def _tune_captures(task_data: Mapping[str, Any]) -> list[_TuneCapture]:
    """Resolve serving configurations as the tune stage does, leaving prose to the template."""
    perf = task_data.get("perf") or {}
    if perf.get("configs"):
        entries = [(str(e.get("name")), e, [1]) for e in perf["configs"]]
    else:
        entries = [("serving", perf, [1, 128])]
    captures: list[_TuneCapture] = []
    for name, entry, default_concurrency in entries:
        try:
            captures.extend(_one_tune_capture(name, entry, default_concurrency))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            captures.append(_TuneCapture(name, None, error=str(exc)))
    return captures


def _one_tune_capture(
    name: str, entry: Mapping[str, Any], default_concurrency: list[int]
) -> list[_TuneCapture]:
    captures: list[_TuneCapture] = []
    identity = entry.get("identity_configs") or []
    if identity:
        for path in identity:
            if not Path(str(path)).is_file():
                captures.append(_TuneCapture(name, None, path=str(path), readable=False))
                continue
            options = check_config.load_llm_api_options(path)
            capture = check_config.options_capture(options, str(path))
            captures.append(
                _TuneCapture(
                    name, capture.describe() if capture is not None else None, path=str(path)
                )
            )
        return captures
    workload = entry.get("workload") or {}
    concurrency = [int(c) for c in workload.get("concurrency", default_concurrency)]
    config = derive_serving_config(
        isl=int(workload.get("isl", 1024)),
        osl=int(workload.get("osl", 1024)),
        concurrency=concurrency,
    )
    overlay = entry.get("llm_api_options") or {}
    config.update(overlay)
    capture = check_config.options_capture(config, f"perf.configs `{name}`")
    return [
        _TuneCapture(
            name,
            capture.describe() if capture is not None else None,
            overlay="cuda_graph_config" in overlay,
            max_concurrency=max(concurrency),
        )
    ]


__all__ = ["CAPTURE_LISTS_HEADING", "capture_lists_block"]
