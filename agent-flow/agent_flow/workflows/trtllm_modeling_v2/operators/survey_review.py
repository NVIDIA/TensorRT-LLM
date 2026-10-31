# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The survey review: a drafted vocabulary held to the tree's prior art and the engine's limits.

The costliest survey mistake is a step mapped to a GAP that writes new code
(the native-torch or Triton rung) while the tree already implements something
that shares its words, which nothing in the row names: every hour after it goes
to rewriting what exists. The second is a plan that does not say how it gets
past a limit the engine's kernels set for this geometry. Both are read against
the same prior art and limits the survey's dispatch carried
(:func:`~.prior_art.survey_block`), so the review asks for nothing the
surveyor was not shown. It never judges a reason: a row that names a lead and
says why it does not serve passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import prior_art as prior_art_mod
from .coverage import TIER_NATIVE_TORCH

#: A lead counts for a step when the two share at least this many of the config's words.
SHARED_WORDS = 2
#: The leads of each kind a step is held to, best first.
LEADS_PER_KIND = 5
#: The leads one finding names at most.
NAMED_PER_STEP = 3
#: Where the review that sent a survey back is kept, under the workspace: once is enough.
REVIEW_FILENAME = "survey_review.md"


def _tier(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("tier"))
    except (TypeError, ValueError):
        return None


def _named(text: str, candidate: prior_art_mod.Candidate) -> bool:
    """Whether ``text`` names ``candidate``: its ``<dir>/<file>``, a name it defines, or its file."""
    lowered = text.lower()
    path = Path(candidate.path)
    tail = "/".join(path.parts[-2:]).lower()
    if tail in lowered or any(n.lower() in lowered for n in candidate.names):
        return True
    # A compound file name (`qsa_paged_sparse_gqa`) names it; `kernels` would name anything.
    return "_" in path.stem and path.stem.lower() in lowered


def _step_words(row: Mapping[str, Any], wanted: set[str]) -> set[str]:
    values = row.get("values")
    text = " ".join(
        [str(row.get("step") or ""), *(f"{k} {v}" for k, v in (values or {}).items())]
        if isinstance(values, Mapping)
        else [str(row.get("step") or "")]
    )
    return prior_art_mod.words(text) & wanted


def unconsidered_leads(
    vocabulary: Mapping[str, Any], found: prior_art_mod.PriorArt | None
) -> list[str]:
    """Each GAP row that writes new code beside a lead it does not name, as a finding."""
    if found is None or not found.candidates:
        return []
    wanted = set(found.words)
    leads = [
        c
        for kind in dict.fromkeys(c.kind for c in found.candidates)
        for c in found.by_kind(kind)[:LEADS_PER_KIND]
    ]
    findings = []
    for index, row in enumerate(vocabulary.get("steps") or ()):
        if not isinstance(row, Mapping) or str(row.get("state")) != "GAP":
            continue
        tier = _tier(row)
        if tier is None or tier < TIER_NATIVE_TORCH:
            continue
        said = " ".join(str(row.get(k) or "") for k in ("tried", "candidate", "contract_line"))
        shared = _step_words(row, wanted)
        missed = [
            (c, sorted(set(c.matched) & shared))
            for c in leads
            if len(set(c.matched) & shared) >= SHARED_WORDS and not _named(said, c)
        ]
        if not missed:
            continue
        named = "; ".join(
            f"`{c.path}`"
            + (f" ({', '.join(c.names[:2])})" if c.names else "")
            + f" shares {', '.join(words)}"
            for c, words in missed[:NAMED_PER_STEP]
        )
        findings.append(
            f"steps[{index}] `{str(row.get('step'))[:90]}` is a GAP at tier {tier}, new code, "
            f"while the tree has {named}. Reuse it (a CERT row, a promotion, or a tier 1-3 "
            "`candidate`), or name it in `tried` with why it does not serve."
        )
    return findings


def unaddressed_limits(plan: str, limits: Sequence[prior_art_mod.EngineLimit]) -> list[str]:
    """Each engine limit no line of ``plan.md`` states a way out for, as a finding."""
    lines = [line.lower() for line in re.split(r"[\n]|(?<=\.)\s", plan)]
    findings = []
    for limit in limits:
        subject = limit.subject.lower()
        if any(
            subject in line and any(way.lower() in line for way in limit.ways_out) for line in lines
        ):
            continue
        findings.append(
            f"`plan.md` states no design for an engine limit: {limit.text.split('. ')[0]}. "
            f"Say which way out the target takes ({' or '.join(limit.ways_out)}) on a line "
            f"naming {limit.subject}, and plan its rows for it."
        )
    return findings


def review_survey(
    vocabulary: Mapping[str, Any],
    plan: str,
    found: prior_art_mod.PriorArt | None,
    limits: Iterable[prior_art_mod.EngineLimit],
) -> list[str]:
    """What the drafted survey left unconsidered; empty when it holds."""
    return unconsidered_leads(vocabulary, found) + unaddressed_limits(plan, list(limits))


def render_review(findings: Sequence[str]) -> str:
    """The section a re-drafted survey's dispatch carries."""
    return "\n".join(
        [
            "## Survey review: re-draft",
            "",
            "Your draft validated, and the review of it against this dispatch's prior art and "
            "engine limits found what it leaves unconsidered. Re-draft `plan.md` and "
            "`vocabulary.yaml` for each finding (reuse it, or say in the row why it does not "
            "serve), then write your decision file again. This review runs once.",
            "",
            *(f"- {finding}" for finding in findings),
        ]
    )


__all__ = [
    "LEADS_PER_KIND",
    "REVIEW_FILENAME",
    "SHARED_WORDS",
    "render_review",
    "review_survey",
    "unaddressed_limits",
    "unconsidered_leads",
]
