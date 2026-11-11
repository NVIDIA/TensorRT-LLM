# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one in-process check hands the engine: a deployment's options, as they are."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ..spec.deployments import Deployment

#: How every check samples unless a shorthand spec's runtime flags say otherwise: the
#: check's own protocol, never an option of the engine.
GREEDY_SAMPLING: dict[str, Any] = {"temperature": 0.0, "top_k": 1}


def check_configuration(
    task_data: Mapping[str, Any], deployment: Deployment, *, max_tokens: int
) -> dict[str, Any]:
    """The public API configuration one in-process check runs ``deployment`` under.

    The deployment's options as they are, with the checkpoint as the model: nothing is
    defaulted here, so a deployment runs exactly what it states. A disaggregated
    deployment's in-process engine is its context server's.
    """
    server = deployment.server or deployment.ctx
    options = copy.deepcopy(dict(server.options))
    options["model"] = str(task_data["checkpoint_path"])
    sampling = {**GREEDY_SAMPLING, **deployment.sampling}
    sampling.setdefault("max_tokens", int(max_tokens))
    width = int(options.get("max_beam_width", 1))
    if width > 1:
        sampling.setdefault("n", width)
        sampling["use_beam_search"] = True
    return {"llm": options, "sampling": sampling, "apply_chat_template": deployment.chat_template}


__all__ = ["GREEDY_SAMPLING", "check_configuration"]
