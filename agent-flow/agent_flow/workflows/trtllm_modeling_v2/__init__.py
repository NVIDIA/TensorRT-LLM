# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trtllm_modeling_v2: the ModelingV2 onboarding and tuning workflow, built on ``agent_flow.AgentLayer``.

Brings one deployment target — a ``(checkpoint, GPU arch, parallel
config)`` triple — from a HuggingFace checkpoint to a gated
``tensorrt_llm/_torch/_experimental/modeling_v2`` target: a flat,
self-contained forward assembled only from catalog entries, trusted
through accuracy gates rather than through shared abstractions.

The pipeline has three steps, and the first and last are not agents::

    preflight -> planned run (planner + workers; a tune task joins) -> delivery

- **preflight** is pure Python and runs in three widening rings. The cheap
  ring is seconds: the checkpoint parses, the arch matches the host, the
  parallel segment is well formed, the device set is claimed by
  ``memory.free`` and a lone-process probe, ``TMPDIR`` is sane, and a
  multi-rank run's NCCL/MPI substrate answers. Then the **substrate gate**
  generates a token on the *stock* modeling path under the run's exact
  runtime block, beside the run's first turns — minutes, because it loads
  the real weights, and worth
  every one of them: the three failures it catches
  (``MPI_ERR_SPAWN``, ``Impossible to fit ... in kvCache``, NVRTC's
  ``cannot open "cuda.h"``) would otherwise surface hours in, inside an
  agent's turn, reading as though the target were at fault. Last is the
  **surface survey**, which measures what the checkpoint structurally
  demands and what this build supplies, so the planner reads an
  inventory instead of guessing one. No agent runs in any of it, so a
  broken environment costs minutes instead of a whole run.
- **planned** is the run itself. A planner agent writes the task graph
  (``graph.yaml``): survey, kernel entries, assembly, features, reviews --
  decomposed and ordered as it sees fit. The orchestrator admits each
  revision, starts tasks as devices and paths allow, runs every task's
  contract checks itself (boot, accuracy, the feature's witness or
  acceptance gate, the envelope boot), and wakes the planner on every
  failure. Checks are commands with exit codes, not agent judgements. When
  the contract holds on the target as it is, delivery verifies the agents'
  claims against disk across the in-tree trees.
- **tune** is a task of the planned run when ``perf.enabled`` asks for it,
  or ``trtllm-modeling-v2 tune`` on a delivered target; it never gates
  delivery.

Public surface:

- :class:`TrtllmModelingV2Workflow` — the orchestrator.
- :class:`PromptBundle`, :data:`DEFAULT_PROMPTS`, and
  :func:`build_trtllm_modeling_v2_prompts` — the roles' system prompts
  and the builder that scopes them to one task.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.trtllm_modeling_v2_state.json``).
"""

from typing import Any

from .agents.prompts import DEFAULT_PROMPTS, PromptBundle, build_trtllm_modeling_v2_prompts
from .run.state import STAGE_PLANNED, STAGE_PREFLIGHT, STAGE_TUNE

__all__ = [
    "DEFAULT_PROMPTS",
    "PromptBundle",
    "STAGE_PLANNED",
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
