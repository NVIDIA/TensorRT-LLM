# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trtllm_modeling_v2: the ModelingV2 onboarding workflow, built on ``agent_flow.AgentLayer``.

Brings one HuggingFace checkpoint up as a
``tensorrt_llm/_torch/_experimental/modeling_v2`` target, its first deployment first::

    preflight -> the tasks in a fixed order, the onboard check after `core` -> the
    final pass -> delivery_report.md

See ``README.md`` for usage and architecture, and ``task.example.yaml`` for configuration.
"""

from typing import Any

__all__ = ["TrtllmModelingV2Workflow", "main"]


def __getattr__(name: str) -> Any:
    if name == "TrtllmModelingV2Workflow":
        from .run.workflow import TrtllmModelingV2Workflow

        return TrtllmModelingV2Workflow
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
