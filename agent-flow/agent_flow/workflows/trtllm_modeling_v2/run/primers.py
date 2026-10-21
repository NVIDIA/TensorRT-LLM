# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primers: one briefing session per role, which that role's turns start as forks of.

A primer reads once, as a turn of its own, what every planner, kernel writer
and assembler turn would otherwise read first, and writes a summary. A later
turn of the role forks the primer's session, or, when it cannot fork (the
external transport, a backend such as Codex), reads the summary; either way it is told the commit the
reading was done at. A fork that fails drops the primer, and the turn runs
again as a plain one. A primer read under other prompts is built again
(:func:`version`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_flow.agent_runtime import AgentConfig

#: The roles whose turns start from a primer.
PRIMED_ROLES: tuple[str, ...] = ("planner", "kernel_writer", "assembler")
#: Where primers are kept, under the workspace: ``<role>.md`` summaries and the record.
PRIMERS_DIRNAME = "primers"
RECORD_FILENAME = "primers.json"
#: The backends whose sessions a turn can start as a copy of.
FORKING_BACKENDS: tuple[str, ...] = ("claude-code",)

_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"

_READING: dict[str, str] = {
    "planner": f"""\
1. `{_PACKAGE}/README.md`, whole.
2. The run's spec, `task.yaml`, and its contract, `contract.yaml`, in the workspace.
3. `surface.md` and `runtime_map.md` in the workspace, when they exist.
4. The existing targets under `{_PACKAGE}/models/`: how each is laid out, by
   listing, not reading every file.
""",
    "kernel_writer": f"""\
1. `{_PACKAGE}/README.md` and `{_PACKAGE}/catalog/_op.py`, whole.
2. `tests/unittest/_torch/modeling_v2/_validating.py`, whole: the fixtures
   every entry test uses.
3. Three entries of different categories in `{_PACKAGE}/catalog/`, each with
   its GPU test under `tests/unittest/_torch/modeling_v2/`, whole.
4. One target's own `ops/`, to see an entry only one target calls and where
   its test sits.
5. The `trtllm-modeling-v2-entry` skill.
""",
    "assembler": f"""\
1. `{_PACKAGE}/README.md` and `{_PACKAGE}/catalog/_op.py`, whole.
2. One existing target end to end: its `modeling.py`, `weights.py`, its
   family's `routing.py` and its tests.
3. The source-level gates, `tests/unittest/_torch/modeling_v2/test_modeling_v2_*.py`:
   what each checks of a target.
4. The run's spec, `task.yaml`, in the workspace.
""",
}


@dataclass(frozen=True)
class Primer:
    """One role's briefing session: its id, the commit it read at, and its summary."""

    role: str
    session_id: str
    commit: str
    version: str
    summary: str


def version(role: str, system_prompt: str) -> str:
    """A digest of what a primer of ``role`` is built from: a primer from others is stale."""
    text = system_prompt + _READING.get(role, "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def can_fork(agent: AgentConfig) -> bool:
    """Whether a turn on ``agent`` can start as a copy of a primer's session."""
    return agent.backend in FORKING_BACKENDS


def summary_path(workspace: Path, role: str) -> Path:
    return Path(workspace) / PRIMERS_DIRNAME / f"{role}.md"


def primer_prompt(role: str, *, repo: str, commit: str, summary: Path, forks: bool = True) -> str:
    """The primer's one turn: what to read, and the summary to write.

    ``forks`` says whether the role's turns can start as copies of this
    session; on a transport that cannot fork them, the summary is all they get.
    """
    if forks:
        handed_on = (
            f"your session is copied to start every later {role} turn, so what you "
            "read now is what each of them already knows"
        )
        summary_use = "Turns that do not start from your session read it instead."
    else:
        handed_on = f"every later {role} turn starts by reading the summary you write"
        summary_use = "It is all a later turn gets of your reading."
    return f"""\
You are preparing a **briefing session** for the {role} turns of this run. You
write no code and change nothing: when you are done, {handed_on}. Read what
every such turn needs and no single task owns.

**Until you are told otherwise, you only read**: edit, create or delete no
file but the summary below, and run no command that changes the checkout.

Read, in the TensorRT-LLM checkout `{repo}` (commit `{commit[:12]}`):

{_READING.get(role, "")}
Keep the reading selective: whole files where named, otherwise only what the
next turn needs.

Then write `{summary}`: what a {role} turn needs to know, in at most 150
lines, naming each file and symbol it comes from. {summary_use} Your decision
file is not needed for this turn.

End your reply with the single word `primed`.
"""


def primed_preamble(commit: str) -> str:
    """What a turn forked from a primer is told before its own dispatch."""
    return (
        "**You are now working your own turn, not the briefing session.** The reading "
        f"above was done at commit `{commit[:12]}` and is yours to use; its limits are "
        "over. Your worktree may hold newer code: before you rely on or edit a file "
        f"read above, check it with `git diff {commit[:12]} -- <file>` and read it again "
        "if it changed.\n\n"
    )


def summary_note(summary: Path, commit: str) -> str:
    """What a turn that does not start from the primer's session is told."""
    if commit:
        check = (
            f"It was read at commit `{commit[:12]}`, and your worktree may hold newer code: "
            f"before you rely on or edit a file it names, check it with "
            f"`git diff {commit[:12]} -- <file>` and read it again if it changed."
        )
    else:
        check = "Your worktree may hold newer code: read a file again before you rely on it."
    return (
        f"Read `{summary}` first: what this role's briefing session found, so you need "
        f"not read it again. {check}\n\n"
    )


def load(workspace: Path) -> dict[str, Primer]:
    """The primers recorded in ``workspace``, by role."""
    path = Path(workspace) / PRIMERS_DIRNAME / RECORD_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for role, raw in (data or {}).items():
        try:
            out[role] = Primer(**raw)
        except TypeError:
            continue
    return out


def save(workspace: Path, primers: dict[str, Primer]) -> None:
    directory = Path(workspace) / PRIMERS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECORD_FILENAME
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps({role: asdict(p) for role, p in primers.items()}, indent=1), encoding="utf-8"
    )
    os.replace(partial, path)


__all__ = [
    "FORKING_BACKENDS",
    "PRIMED_ROLES",
    "PRIMERS_DIRNAME",
    "Primer",
    "can_fork",
    "load",
    "primed_preamble",
    "primer_prompt",
    "save",
    "summary_note",
    "summary_path",
    "version",
]
