# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CUDA-graph capture lists a survey's dispatch carries.

The survey judges every vocabulary row against the batches the engines will
capture, and those are read, never recalled: each unit's list as its gates
resolve it, the LLM API's default and each list the tune campaign measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ...checks import check_config
from ...tune.serving_config import derive_serving_config

#: The heading of the dispatch's block, which the `survey` tag's section points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """The decode CUDA-graph capture lists the survey turn plans against.

    Each unit's lists are resolved as its gates resolve them
    (``check_config.launch_captures``); a configuration variant is listed only where
    its lists differ from its unit's. Then the LLM API's default and the list
    of each serving config tune measures.
    """
    lines = [
        CAPTURE_LISTS_HEADING,
        "",
        "Planning preview of the shared public LLM configuration (`checks/check_config.py`, "
        "`launch_captures`). Request checks and trtllm-eval receive the same options. "
        "The engine can filter batches by launch limits; these previews are not actual "
        "capture evidence. Worker observations remain necessary.",
        "",
    ]
    try:
        captures = check_config.launch_captures(task_data)
    except (ValueError, yaml.YAMLError) as exc:
        lines.append(f"- the units' lists cannot be resolved ({exc}); preflight names the cause")
        captures = []
    units = {c.name: c for c in captures if c.name == c.unit}
    inherited: list[str] = []
    for capture in captures:
        base = units.get(capture.unit)
        if capture.name != capture.unit and base is not None:
            if base.resolved == capture.resolved:
                inherited.append(capture.name)
                continue
            lines.append(f"- variant `{capture.name}` (unit `{capture.unit}`): {_by_path(capture)}")
        else:
            lines.append(f"- `{capture.name}`: {_by_path(capture)}")
    if inherited:
        lines.append(f"- variants {', '.join(inherited)}: their unit's list")
    lines.append(
        f"- LLM API default, `CudaGraphConfig()`, what a user who sets no "
        f"`cuda_graph_config` runs: {check_config.LLM_API_DEFAULT_CAPTURE.describe()}"
    )
    lines.extend(_tune_lines(task_data))
    return "\n".join(lines) + "\n"


def _by_path(capture: check_config.LaunchCapture) -> str:
    """The shared configuration's planning preview."""
    if capture.resolved is None:
        return "no CUDA graphs at any gate"
    return (
        f"{capture.resolved.describe()}, at boot, acceptance and accuracy, "
        f"from {capture.resolved.source}"
    )


def _tune_lines(task_data: Mapping[str, Any]) -> list[str]:
    """Each serving config tune measures and its capture list, as ``tune/stage.py`` derives it."""
    perf = task_data.get("perf") or {}
    if not perf.get("enabled"):
        return ["- tune: `perf` is off, so no serving config is measured"]
    if perf.get("configs"):
        entries = [(str(e.get("name")), e, [1]) for e in perf["configs"]]
    else:
        entries = [("serving", perf, [1, 128])]
    lines: list[str] = []
    for name, entry, default_concurrency in entries:
        identity = entry.get("identity_configs") or []
        for path in identity:
            if not Path(str(path)).is_file():
                lines.append(f"- tune `{name}`: {path} (not readable here)")
                continue
            options = check_config.load_llm_api_options(path)
            capture = check_config.options_capture(options, str(path))
            listed = capture.describe() if capture else "no CUDA graphs"
            lines.append(f"- tune `{name}`: {listed}, from {path}")
        if identity:
            continue
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
        listed = capture.describe() if capture else "no CUDA graphs"
        source = (
            "its `llm_api_options`"
            if "cuda_graph_config" in overlay
            else f"`capture_batch_sizes` up to concurrency {max(concurrency)}"
        )
        lines.append(f"- tune `{name}`: {listed}, from {source}")
    return lines


__all__ = ["CAPTURE_LISTS_HEADING", "capture_lists_block"]
