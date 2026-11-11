# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How far a target's next-token distribution is from a reference's, recorded rather than gated.

The model-level acceptance: the target answers its prompts reasonably (the boot gate's
keywords), its first token is the reference's (a flip within ``tie_margin`` nats of the
reference's own top two is a near-tie, not a disagreement), and its distribution has not
diverged from the reference's (the Jensen-Shannon divergence of the two top-k distributions,
in nats, and the centered cosine of their log-probs). The golden rule stays the benchmark
score against its anchor.

A divergence past the bounds is **recorded** with the verdict (its numbers and the prompts
that crossed them), never a failure by itself: a check that blocks a run over a
low-precision difference costs more than the difference. Only output that is wrong
(the boot's keywords missing, nothing generated) or a benchmark under its floor fails.
The bounds can be loosened per run (``accuracy.reference_logits``); the record stays.

The reference is a file of first-token top-k log-probs per prompt, made once by a run that
can load the checkpoint natively (``checks/harness/hf_reference.py``: HuggingFace
``transformers``), and named by ``accuracy.reference_logits.path``::

    {"source": "transformers 4.57 bf16", "top_k": 20,
     "prompts": {"p0": "The capital of France is", ...},
     "first_tokens": {"p0": {"12366": -0.01, ...}, ...}}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Nats between the reference's top token and the target's that still read as a near-tie.
DEFAULT_TIE_MARGIN = 0.15
#: The Jensen-Shannon divergence (nats, at most ln 2) of the two top-k distributions.
DEFAULT_MAX_JS = 0.20
#: The centered cosine of the two top-k log-prob vectors.
DEFAULT_MIN_COSINE = 0.98
#: ``accuracy.reference_logits`` keys.
REFERENCE_KEYS: tuple[str, ...] = ("path", "tie_margin", "max_js", "min_cosine")
#: A token one side's top-k lacks gets this much below that side's lowest listed log-prob.
_MISSING_BELOW = 1.0

TopK = Mapping[int, float]


@dataclass(frozen=True)
class Bounds:
    """Where a divergence starts to be recorded as one."""

    tie_margin: float = DEFAULT_TIE_MARGIN
    max_js: float = DEFAULT_MAX_JS
    min_cosine: float = DEFAULT_MIN_COSINE

    @classmethod
    def from_block(cls, block: Mapping[str, Any] | None) -> "Bounds":
        block = block or {}
        return cls(
            tie_margin=float(block.get("tie_margin", DEFAULT_TIE_MARGIN)),
            max_js=float(block.get("max_js", DEFAULT_MAX_JS)),
            min_cosine=float(block.get("min_cosine", DEFAULT_MIN_COSINE)),
        )


def _aligned(reference: TopK, measured: TopK) -> tuple[list[float], list[float]]:
    """Both sides' log-probs on the union of their tokens; a missing one just below its side's."""
    tokens = sorted(set(reference) | set(measured))
    low_r = min(reference.values(), default=-30.0) - _MISSING_BELOW
    low_m = min(measured.values(), default=-30.0) - _MISSING_BELOW
    return [reference.get(t, low_r) for t in tokens], [measured.get(t, low_m) for t in tokens]


def first_token_agrees(reference: TopK, measured: TopK, tie_margin: float) -> bool:
    """The target's top token is the reference's, or within ``tie_margin`` of it in the reference."""
    if not reference or not measured:
        return False
    top, mine = max(reference, key=reference.get), max(measured, key=measured.get)
    return top == mine or (mine in reference and reference[top] - reference[mine] <= tie_margin)


def js_divergence(reference: TopK, measured: TopK) -> float:
    """Jensen-Shannon divergence, in nats, of the two top-k distributions, each renormalized."""
    r, m = _aligned(reference, measured)
    p = [math.exp(v) for v in r]
    q = [math.exp(v) for v in m]
    sp, sq = sum(p) or 1.0, sum(q) or 1.0
    p, q = [v / sp for v in p], [v / sq for v in q]
    mid = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x: list[float]) -> float:
        return sum(a * math.log(a / c) for a, c in zip(x, mid) if a > 0 and c > 0)

    return 0.5 * kl(p) + 0.5 * kl(q)


def centered_cosine(reference: TopK, measured: TopK) -> float:
    """The cosine of the two top-k log-prob vectors, each less its mean."""
    r, m = _aligned(reference, measured)
    if len(r) < 2:
        return 1.0
    mr, mm = sum(r) / len(r), sum(m) / len(m)
    a, b = [v - mr for v in r], [v - mm for v in m]
    na, nb = math.sqrt(sum(v * v for v in a)), math.sqrt(sum(v * v for v in b))
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@dataclass(frozen=True)
class Divergence:
    """The target against the reference over its prompts."""

    prompts: int
    answered: int
    #: The share of prompts whose first token agrees (near-ties count as agreeing).
    top1: float
    max_js: float
    min_cosine: float
    bounds: Bounds
    #: The prompts past a bound, and how.
    crossed: tuple[str, ...] = field(default=())

    @property
    def within(self) -> bool:
        return not self.crossed and self.answered == self.prompts

    def describe(self) -> str:
        said = (
            f"against the reference, first tokens agree on {self.top1:.0%} of {self.prompts} "
            f"prompt(s), max JS {self.max_js:.3f} nats (bound {self.bounds.max_js:g}), "
            f"min centered cosine {self.min_cosine:.3f} (bound {self.bounds.min_cosine:g})"
        )
        if self.answered < self.prompts:
            said += f", {self.prompts - self.answered} unanswered"
        if self.crossed:
            said += f"; divergence recorded: {'; '.join(self.crossed[:6])}"
        return said


def assess(
    reference: Mapping[str, TopK], measured: Mapping[str, TopK], bounds: Bounds
) -> Divergence:
    """``measured`` against ``reference``, prompt by prompt (both ``{id: {token: log-prob}}``)."""
    asked = [key for key in reference if reference[key]]
    agree, answered = 0, 0
    max_js, min_cos = 0.0, 1.0
    crossed = []
    for key in asked:
        ours = measured.get(key)
        if not ours:
            crossed.append(f"{key} unanswered")
            continue
        answered += 1
        same = first_token_agrees(reference[key], ours, bounds.tie_margin)
        js = js_divergence(reference[key], ours)
        cos = centered_cosine(reference[key], ours)
        agree += same
        max_js, min_cos = max(max_js, js), min(min_cos, cos)
        why = [
            *(["first token differs"] if not same else []),
            *([f"JS {js:.3f}"] if js > bounds.max_js else []),
            *([f"cosine {cos:.3f}"] if cos < bounds.min_cosine else []),
        ]
        if why:
            crossed.append(f"{key} ({', '.join(why)})")
    return Divergence(
        prompts=len(asked),
        answered=answered,
        top1=agree / len(asked) if asked else 0.0,
        max_js=max_js,
        min_cosine=min_cos,
        bounds=bounds,
        crossed=tuple(crossed),
    )


@dataclass(frozen=True)
class Reference:
    """``accuracy.reference_logits``, read: the prompts, their first tokens and the bounds."""

    path: Path
    prompts: Mapping[str, str]
    first_tokens: Mapping[str, TopK]
    top_k: int
    source: str
    bounds: Bounds


def reference_block(task_data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    block = (task_data.get("accuracy") or {}).get("reference_logits")
    return block if isinstance(block, Mapping) and block.get("path") else None


def load_reference(task_data: Mapping[str, Any]) -> Reference | None:
    """The run's reference, or ``None`` when it names none.

    Raises ``OSError`` or ``ValueError`` when it names one that cannot be read.
    """
    block = reference_block(task_data)
    if block is None:
        return None
    path = Path(str(block["path"]))
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = {str(k): str(v) for k, v in (data.get("prompts") or {}).items()}
    tokens = {
        str(k): {int(t): float(lp) for t, lp in (row or {}).items()}
        for k, row in (data.get("first_tokens") or {}).items()
    }
    if not prompts or not any(tokens.get(k) for k in prompts):
        raise ValueError(f"{path} holds no prompt with first-token log-probs")
    return Reference(
        path=path,
        prompts={k: v for k, v in prompts.items() if tokens.get(k)},
        first_tokens=tokens,
        top_k=int(data.get("top_k") or max(len(r) for r in tokens.values())),
        source=str(data.get("source") or ""),
        bounds=Bounds.from_block(block),
    )


__all__ = [
    "DEFAULT_MAX_JS",
    "DEFAULT_MIN_COSINE",
    "DEFAULT_TIE_MARGIN",
    "REFERENCE_KEYS",
    "Bounds",
    "Divergence",
    "Reference",
    "assess",
    "centered_cosine",
    "first_token_agrees",
    "js_divergence",
    "load_reference",
    "reference_block",
]
