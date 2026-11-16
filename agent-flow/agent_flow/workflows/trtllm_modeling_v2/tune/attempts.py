# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What earlier attempts found: a failed attempt is a result the next turn starts from.

One entry per finished candidate: its outcome and the numbers the stage measured (the local
speedups, the projected e2e gain, the serving deltas), the stage's reason, and the turn's own
``findings`` (what it learned and what to try next). The stage writes the whole list to
``attempts.md`` and puts the latest entries in every candidate, revise and roadmap request.
The verdicts are the stage's, not the turns' own claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .campaign import CandidateState

ATTEMPTS_FILE = "attempts.md"
#: Entries a request carries; the file holds them all.
REQUEST_ENTRIES = 12
#: Statuses still in flight: no result yet.
_OPEN = ("issued", "ready", "micro", "e2e", "gate")


def outcome(st: CandidateState) -> str:
    if st.status == "kept":
        return "kept"
    if st.status == "reverted" and st.reason.startswith("the checkpoint at"):
        return "rolled back at a checkpoint"
    if st.status == "reverted":
        return {"LOSS": "lost", "NEUTRAL": "no gain"}.get(st.verdict, "reverted")
    return {"revising": "being revised", "split": "split by member"}.get(st.status, st.status)


def _percent(values: dict, suffix: str = "") -> str:
    return ", ".join(
        f"{key.removesuffix(suffix)} {float(v):+.1%}"
        for key, v in sorted(values.items())
        if isinstance(v, (int, float))
    )


def entry(st: CandidateState) -> str:
    """One attempt: outcome and measured numbers, the stage's reason, the turn's findings."""
    what = f"{st.kind}, " if st.kind else ""
    lines = [f"- **`{st.slug}`** ({what}{st.scope}): **{outcome(st)}**"]
    if st.hypothesis:
        lines.append(f"  - hypothesis: {st.hypothesis[:240]}")
    if st.scope == "e2e" and st.deltas:
        lines.append(f"  - served: {_percent(st.deltas)}")
    elif st.deltas:
        lines.append(f"  - local speedup: {_percent(st.deltas, '.speedup')}")
    if st.projected:
        lines.append(f"  - projected e2e: {_percent(st.projected)}")
    if st.reason:
        lines.append(f"  - stage: {st.reason[:400]}")
    for row in st.findings:
        found = row["what"] + (f" ({row['evidence']})" if row.get("evidence") else "")
        lines.append(f"  - found: {found}" + (f"; next: {row['next']}" if row.get("next") else ""))
    return "\n".join(lines)


def digest(candidates: Iterable[CandidateState], path: Path) -> str:
    """Write every finished attempt to ``path``; the latest :data:`REQUEST_ENTRIES` as text."""
    done = [st for st in candidates if st.status not in _OPEN]
    if not done:
        return ""
    entries = [entry(st) for st in reversed(done)]  # the latest first
    path.write_text("# Attempts so far (latest first)\n\n" + "\n".join(entries) + "\n")
    text = "\n".join(entries[:REQUEST_ENTRIES])
    if len(entries) > REQUEST_ENTRIES:
        text += f"\n- ... {len(entries) - REQUEST_ENTRIES} earlier attempts in `{path}`"
    return text


__all__ = ["ATTEMPTS_FILE", "REQUEST_ENTRIES", "digest", "entry", "outcome"]
