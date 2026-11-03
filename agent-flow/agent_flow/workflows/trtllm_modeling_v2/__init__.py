# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trtllm_modeling_v2: the ModelingV2 onboarding and tuning workflow, built on ``agent_flow.AgentLayer``.

Brings one deployment target — a ``(checkpoint, GPU arch, parallel
config)`` triple — from a HuggingFace checkpoint to a gated
``tensorrt_llm/_torch/_experimental/modeling_v2`` target::

    preflight -> execution stage (planner + workers; a tune task joins) -> delivery

See ``README.md`` for usage and architecture, and ``task.example.yaml`` for configuration.
"""

from typing import Any

from .agents.prompts import DEFAULT_PROMPTS, PromptBundle, build_trtllm_modeling_v2_prompts
from .run.records.state import STAGE_EXECUTION, STAGE_PREFLIGHT, STAGE_TUNE

__all__ = [
    "DEFAULT_PROMPTS",
    "PromptBundle",
    "STAGE_EXECUTION",
    "STAGE_PREFLIGHT",
    "STAGE_TUNE",
    "TrtllmModelingV2Workflow",
    "build_trtllm_modeling_v2_prompts",
    "main",
]


def __getattr__(name: str) -> Any:
    if name == "TrtllmModelingV2Workflow":
        from .run.workflow import TrtllmModelingV2Workflow

        return TrtllmModelingV2Workflow
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
