"""Domain specialization of :mod:`agent_flow.workflows.agent_team` for staircase.

A staircase target is one self-contained modeling codebase per (checkpoint, GPU
arch, parallel topology) triple under
``tensorrt_llm/_torch/staircase/models/<family>/targets/``, assembled only from
entries in the shared ``catalog/`` vocabulary and trusted through accuracy
gates rather than shared abstractions.

This subpackage layers staircase guidance (closed-vocabulary rule, catalog
entry spec and receipts, target product spec, the gate ladder) onto the generic
``agent_team`` system prompts. The orchestrator, MCP tools, checkpoint format,
and CLI flags are reused unchanged — only the prompt bundle differs, with one
exception: QA's base prompt is replaced rather than extended, because its
evaluation rubric is part of that base prompt. See :mod:`.prompts.qa`.

One run is one Stage-1 bring-up: catalog Goals (onboard or certify one op
each, closing on a receipt) and target Goals (modeling / weights / smoke),
all converging on the Stage's exit criterion — ``measured >= reference - tol``
on the accuracy gate. A second Stage is for a capability variant that changes
the forward path, such as speculative decoding, whose exit is the same bar
plus its own third signal.

Public surface:

- :data:`STAIRCASE_PROMPTS` — the default local-host :class:`PromptBundle`.
- :func:`build_staircase_prompts` — builds a task-scoped bundle, optionally
  including Slurm/container guidance.
- :func:`main` — console entry point, also reachable via
  ``python -m agent_flow.workflows.staircase_bringup.cli``.
"""

from typing import Any

from .prompts import STAIRCASE_PROMPTS, build_staircase_prompts

__all__ = [
    "STAIRCASE_PROMPTS",
    "build_staircase_prompts",
    "main",
]


def __getattr__(name: str) -> Any:
    """Defer the CLI import so ``import ...staircase_bringup`` stays cheap."""
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
