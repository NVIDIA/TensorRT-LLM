# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staircase onboarding workflow built on ``agent_flow.AgentLayer``.

Brings one deployment target — a ``(checkpoint, GPU arch, parallel
config)`` triple — from a HuggingFace checkpoint to a gated
``tensorrt_llm/_torch/_experimental/modeling_v2`` target: a flat,
self-contained forward assembled only from catalog entries, trusted
through accuracy gates rather than through shared abstractions.

The pipeline is five phases, and the first and last are mostly not agents
at all::

    preflight -> plan -> smith x N (concurrent) -> assemble x M (serial) -> deliver

- **preflight** is pure Python and runs in three widening rings. The cheap
  ring is seconds: the checkpoint parses, the arch matches the host, the
  parallel segment is well formed, the device set is claimed by
  ``memory.free`` and a lone-process probe, ``TMPDIR`` is sane, and a
  multi-rank run's NCCL/MPI substrate answers. Then the **substrate gate**
  generates a token on the *stock* modeling path under the run's exact
  runtime block — minutes, because it loads the real weights, and worth
  every one of them: the three failures it catches
  (``MPI_ERR_SPAWN``, ``Impossible to fit ... in kvCache``, NVRTC's
  ``cannot open "cuda.h"``) each used to surface hours in, inside an
  agent's turn, reading as though the target were at fault. Last is the
  **surface survey**, which measures what the checkpoint structurally
  demands and what this build supplies, so the plan phase reads an
  inventory instead of guessing one. No agent runs in any of it, so a
  broken environment costs minutes instead of a whole run.
- **plan** is PlanDrafter <-> PlanReviewer, capped at ``max_plan_rounds``
  (default 1) and **non-blocking**: when the cap is reached the reviewer's
  unresolved items are written into ``plan.md`` as open risks and the run
  proceeds. Products are ``plan.md`` and ``vocabulary.yaml``, the typed
  step-to-catalog mapping that becomes the dispatch queue.
- **smith** walks that queue, one agent per catalog entry, concurrently
  within a dependency batch. Each smith onboards or verifies exactly one
  entry and commits it through the workspace's commit helper, which takes a
  lock and refuses a commit whose contents are not exactly what was asked
  for — concurrency into one worktree, made safe in code rather than in
  prose.
- **assemble** runs one assembler per unit, serially — core first, then one
  increment per feature — with the boot gate after every turn and the
  accuracy gate at each unit boundary. Gates are commands with exit codes,
  not agent judgements.
- **deliver** verifies the agents' claims against disk across the in-tree
  trees, spends one stateless auditor turn on the single check that is
  irreducibly prose-reading (the runtime-value dump against the catalog's
  certified columns), lints once, and merges.

Public surface:

- :class:`StaircaseOnboardWorkflow` — the orchestrator.
- :class:`PromptBundle`, :data:`DEFAULT_PROMPTS`, and
  :func:`build_staircase_onboard_prompts` — the five agents' system
  prompts and the builder that scopes them to one task.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.staircase_onboard_state.json``).
"""

from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle, build_staircase_onboard_prompts
from .state import STAGE_ASSEMBLE, STAGE_DELIVER, STAGE_PLAN, STAGE_PREFLIGHT, STAGE_SMITH

__all__ = [
    "DEFAULT_PROMPTS",
    "PromptBundle",
    "STAGE_ASSEMBLE",
    "STAGE_DELIVER",
    "STAGE_PLAN",
    "STAGE_PREFLIGHT",
    "STAGE_SMITH",
    "StaircaseOnboardWorkflow",
    "build_staircase_onboard_prompts",
    "main",
]


def __getattr__(name: str) -> Any:
    if name == "StaircaseOnboardWorkflow":
        from .workflow import StaircaseOnboardWorkflow

        return StaircaseOnboardWorkflow
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
