# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared doctrine and run-scoped prompt context."""

from __future__ import annotations

import re
from typing import Sequence

from ...infra.git import CommitPolicy
from ...operators.coverage import (
    CATALOG_TESTS_DIR,
    MODELING_V2_PACKAGE,
    TIER_INTREE_CALLABLE,
    TIER_KERNEL_LIBRARY,
    TIER_NATIVE_TORCH,
    TIER_TRITON,
    TIER_TRTLLM_OP,
)
from ._render import render_template


def tidy(text: str) -> str:
    """Collapse the blank-line runs that composing blocks leaves behind."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


MODELING_V2_LAYOUT = render_template(
    "common/modeling_v2_layout.md.j2",
    MODELING_V2_PACKAGE=MODELING_V2_PACKAGE,
    CATALOG_TESTS_DIR=CATALOG_TESTS_DIR,
)
MODELING_V2_SWITCH = render_template("common/modeling_v2_switch.md.j2")
SOURCE_BOUNDARY = render_template("common/source_boundary.md.j2")
EVIDENCE_DISCIPLINE = render_template("common/evidence_discipline.md.j2")
CITATION_DISCIPLINE = render_template("common/citation_discipline.md.j2")


def decision_file_block(agent: str, extra: Sequence[str] = ()) -> str:
    """Render the decision-file protocol; decisions come from the role's ending section."""
    return render_template("common/decision_file.md.j2", agent=agent, extra=extra)


def commit_block(policy: CommitPolicy) -> str:
    """Render the commit rules every committing role's prompt carries."""
    own = ", ".join(f"`{p}`" for p in policy.own_paths) or "the target's family and its tests"
    judge = ", ".join(f"`{p}`" for p in policy.judge_paths)
    return render_template("common/committing.md.j2", own=own, judge=judge)


def notes_block(notes: str) -> str:
    """Render the task's notes: the user's own guidance, verbatim, to every role."""
    if not notes or not notes.strip():
        return ""
    return render_template("common/notes.md.j2", notes=notes.strip())


def scope_block(scope: str) -> str:
    """Render the target's scope when it is narrower than the checkpoint."""
    if scope != "text_only":
        return ""
    return render_template("common/text_only.md.j2")


#: What each rung means to the agent that has to build it.
TIER_DESCRIPTION: dict[int, str] = {
    TIER_TRTLLM_OP: "an op the engine already exports as torch.ops.trtllm.*",
    TIER_INTREE_CALLABLE: (
        "a callable already in this repository: the engine's modules, its attention backends "
        "(sparse ones included), custom ops and kernels, or another modeling_v2 target's "
        "`ops/` (reused by copying it into this target's own `ops/`); the kernel plane is "
        "reused, not rewritten"
    ),
    TIER_KERNEL_LIBRARY: (
        "a kernel from a library the engine already pins (FlashInfer), called through its "
        "public Python API because no trtllm op wraps it; the kernel is reused, not "
        "rewritten. Read its API in the run's interpreter"
    ),
    TIER_NATIVE_TORCH: (
        "no engine implementation exists, so compose one from torch ops. Correct first and "
        "slow is acceptable for a functionality onboarding; the tuning phase replaces it"
    ),
    TIER_TRITON: (
        "native torch cannot express it, or does so at a cost that breaks the gate's budget, "
        "so write a Triton kernel, adapted from an existing one (FlashInfer, vLLM, SGLang) "
        "where there is one"
    ),
}
