# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts of the op primer: the session every op worker is forked from.

Every worker that writes or speeds up an entry (``new_op`` and ``perf``
tasks) starts by reading the same things: the entry contract, a few existing
entries and their tests, how to run a GPU test. The primer reads them once,
while the planner plans; each such worker then starts as a fork of the
primer's session, with that reading already in its history.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._common import CITATION_DISCIPLINE, MODELING_V2_LAYOUT, tidy

__all__ = [
    "PRIMED_TAGS",
    "build_primer_system_prompt",
    "primer_prompt",
    "primer_version",
    "primed_preamble",
]

# Tasks with any of these tags start from the op primer.
PRIMED_TAGS = frozenset({"new_op", "perf"})

_PRIMER = """\
You are preparing a **briefing session** for a ModelingV2 onboarding run.
You write no code. When you are done, your session is copied to start every
worker that writes a new ModelingV2 entry or makes one faster, so what you
read now is what each of them already knows when it starts. Read what every
such worker needs and no single task owns; leave out anything specific to
one op.

**Until you are told otherwise, you only read.** Do not edit, create or
delete any file except the summary file named in your prompt, and run no
command that changes the checkout. These limits hold for this briefing only;
a worker started from your session is told what it may do.

{modeling_v2_layout}
{citation_discipline}
"""

_PRIMER_TURN = """\
Read, in the TensorRT-LLM checkout `{repo}` (commit `{commit}`):

1. `tensorrt_llm/_torch/_experimental/modeling_v2/README.md`, whole.
2. `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py`, whole: the
   `OpWrapper` contract, `Cell`, `Arch`, how `CELLS` and `ARCHS` are checked.
3. The validation helpers the entry tests use (the `certifying` fixture, the
   tolerance helpers, `true_fp32_matmul`) — find them from an entry test's
   imports.
4. Three existing entries from different categories in `catalog/` — one GEMM,
   one normalization or elementwise op, one attention or MoE op — each with
   its GPU test under `tests/unittest/_torch/modeling_v2/`. Read entry and
   test whole.
5. One target's own `ops/` directory, to see what an entry only one target
   calls looks like and how its test is placed.
6. How GPU work runs in this run: `{task_yaml}` (its `slurm-environment` and
   `implements_tips`) and, if it has any entries, `{test_commands}`.

Keep the reading selective — whole files where named, otherwise only what
the next worker needs — so the briefing stays well under 60k tokens.

Then write `{summary}`: what an entry worker needs to know, in at most 150
lines — the contract's rules, the shape of an entry and of its test (naming
the exemplars you read, by file and symbol), the validation helpers and when
to use each, how to run one entry's GPU test here, and the pitfalls you saw.
Workers that do not start from your session read it instead.

End your reply with the single word `primed`.
"""


def build_primer_system_prompt() -> str:
    """The op primer's system prompt."""
    return tidy(
        _PRIMER.format(
            modeling_v2_layout=MODELING_V2_LAYOUT,
            citation_discipline=CITATION_DISCIPLINE,
        )
    )


def primer_prompt(
    *, repo: Path, commit: str, task_yaml: Path, test_commands: Path, summary: Path
) -> str:
    """The op primer's one turn."""
    return _PRIMER_TURN.format(
        repo=repo, commit=commit, task_yaml=task_yaml, test_commands=test_commands, summary=summary
    )


def primer_version() -> str:
    """A digest of the primer's prompts: a primer built from others is rebuilt."""
    text = build_primer_system_prompt() + _PRIMER_TURN
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def primed_preamble(*, repo: Path, commit: str) -> str:
    """What a worker forked from the primer is told before its task."""
    return (
        "**You are now a worker, not the briefing session.** The reading above was "
        f"done in `{repo}` at commit `{commit}` and is yours to use. Its limits are "
        "over: you edit, build and test as your system prompt and your task say. "
        "Your worktree may hold newer code than that commit (the branches of the "
        "tasks you build on are merged into it): before you rely on or edit a file "
        f"you read above, check it with `git diff {commit} -- <file>` and read it "
        "again if it changed.\n\n"
    )
