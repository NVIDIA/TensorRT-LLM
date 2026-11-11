# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CUDA-graph settings a survey's dispatch carries: what each deployment's options state."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...spec import deployments as deployments_mod

#: The heading of the dispatch's block, which the `survey` tag's section points at.
CAPTURE_LISTS_HEADING = "## CUDA-graph capture lists"
_UNSAID = object()


def capture_lists_block(task_data: Mapping[str, Any]) -> str:
    """Each deployment's ``cuda_graph_config``, as its options state it."""
    lines = [
        CAPTURE_LISTS_HEADING,
        "",
        "What each deployment's options state for `cuda_graph_config`. Every check and "
        "`trtllm-eval` run the deployment with exactly its options; the engine can still "
        "filter batches by its launch limits, and what a rank really runs is in its boot's "
        "record.",
        "",
    ]
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            who = deployment.name if role == "server" else f"{deployment.name}.{role}"
            graphs = server.options.get("cuda_graph_config", _UNSAID)
            if graphs is _UNSAID:
                said = "not stated: the engine's default"
            elif graphs is None:
                said = "no CUDA graphs"
            else:
                said = f"`{json.dumps(graphs, sort_keys=True)}`" + (
                    "" if graphs else " (the engine's default list)"
                )
            lines.append(f"- `{who}`: {said}")
    return "\n".join(lines)


__all__ = ["CAPTURE_LISTS_HEADING", "capture_lists_block"]
