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

from ...checks import gates
from ...tune.serving_config import derive_serving_config

#: The heading of the dispatch's block, which the `survey` tag's section points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """The decode CUDA-graph capture lists the survey turn plans against.

    Each unit's lists are resolved as its gates resolve them
    (``gates.launch_captures``); a configuration variant is listed only where
    its lists differ from its unit's. Then the LLM API's default and the list
    of each serving config tune measures.
    """
    lines = [
        CAPTURE_LISTS_HEADING,
        "",
        "The decode batch sizes each unit's gates capture, resolved once per unit from its "
        "options file's `cuda_graph_config`, else from the LLM API's rule over its runtime "
        "flags (`checks/gates.py`, `resolve_capture_list`). Boot, acceptance and envelope "
        "receive it as `--cuda_graph_batch_sizes` / `--cuda_graph_padding_enabled`; accuracy "
        "reads the options file. The engine then drops the sizes above "
        "`min(max_batch_size, max_num_tokens // (1 + draft))`.",
        "",
    ]
    try:
        captures = gates.launch_captures(task_data)
    except (ValueError, yaml.YAMLError) as exc:
        lines.append(f"- the units' lists cannot be resolved ({exc}); preflight names the cause")
        captures = []
    units = {c.name: c for c in captures if c.name == c.unit}
    inherited: list[str] = []
    for capture in captures:
        base = units.get(capture.unit)
        if capture.name != capture.unit and base is not None:
            if (base.quickstart, base.llm_api) == (capture.quickstart, capture.llm_api):
                inherited.append(capture.name)
                continue
            lines.append(f"- variant `{capture.name}` (unit `{capture.unit}`): {_by_path(capture)}")
        else:
            lines.append(f"- `{capture.name}`: {_by_path(capture)}")
    if inherited:
        lines.append(f"- variants {', '.join(inherited)}: their unit's list")
    lines.append(
        f"- LLM API default, `CudaGraphConfig()`, what a user who sets no "
        f"`cuda_graph_config` runs: {gates.LLM_API_DEFAULT_CAPTURE.describe()}"
    )
    lines.extend(_tune_lines(task_data))
    return "\n".join(lines) + "\n"


def _by_path(capture: gates.LaunchCapture) -> str:
    """One unit's lists, said once when every gate captures the same."""
    quick, llm_api = capture.quickstart, capture.llm_api
    if quick is None and llm_api is None:
        return "no CUDA graphs at any gate"
    if quick is None:
        return (
            f"boot, acceptance and envelope run eager (runtime `cuda_graph` off); accuracy "
            f"captures {llm_api.describe()}, from {llm_api.source}"
        )
    if llm_api is None:
        return (
            f"accuracy runs eager (`cuda_graph_config: null` in {capture.options}); boot, "
            f"acceptance and envelope capture {quick.describe()}"
        )
    if capture.conflicting:
        return (
            f"boot, acceptance and envelope capture {quick.describe()} but accuracy "
            f"{llm_api.describe()}; preflight refuses this unit"
        )
    source = capture.resolved.source if capture.resolved is not None else llm_api.source
    return f"{quick.describe()}, at boot, acceptance, envelope and accuracy, from {source}"


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
            options = gates.load_llm_api_options(path)
            capture = gates.options_capture(options, str(path))
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
        capture = gates.options_capture(config, f"perf.configs `{name}`")
        listed = capture.describe() if capture else "no CUDA graphs"
        source = (
            "its `llm_api_options`"
            if "cuda_graph_config" in overlay
            else f"`capture_batch_sizes` up to concurrency {max(concurrency)}"
        )
        lines.append(f"- tune `{name}`: {listed}, from {source}")
    return lines


__all__ = ["CAPTURE_LISTS_HEADING", "capture_lists_block"]
