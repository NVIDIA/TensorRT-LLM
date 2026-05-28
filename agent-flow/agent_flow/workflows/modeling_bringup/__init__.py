"""Domain specialization of :mod:`agent_flow.workflows.agent_team` for
TensorRT-LLM model bring-up.

This subpackage layers modeling-bringup guidance (source boundary,
validation policies, attention/MoE/full-model scope, accuracy gate, …)
onto the generic ``agent_team`` system prompts. The orchestrator, MCP
tools, checkpoint format, and CLI flags are reused unchanged — only
the prompt bundle differs.

Public surface:

- :data:`MODELING_BRINGUP_PROMPTS` — the default local-host
  :class:`PromptBundle`.
- :func:`build_modeling_bringup_prompts` — builds a task-scoped bundle,
  optionally including Slurm/container guidance.
- :func:`main` — console-script entry point (also reachable via
  ``modeling-bringup`` and ``python -m
  agent_flow.workflows.modeling_bringup.cli``).
"""
from typing import Any

from .prompts import MODELING_BRINGUP_PROMPTS, build_modeling_bringup_prompts

__all__ = [
    "MODELING_BRINGUP_PROMPTS",
    "build_modeling_bringup_prompts",
    "main",
]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
