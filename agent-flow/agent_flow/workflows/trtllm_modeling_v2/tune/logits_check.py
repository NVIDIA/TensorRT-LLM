# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tune campaign's quick accuracy watch: the first output token's distribution.

At each cumulative checkpoint the best tree answers a few prompts with one token each,
and its top-k log-probs are held to those of the last tree that held (the control),
served the same way. It passes when the first tokens mostly agree (a flip within
``tie_margin`` nats of the control's own top two is a near-tie, not a disagreement) and
the distributions are close (mean KL of the control's top-k against the candidate's).
Loose by design: it catches a change that breaks the model, not one that moves the
benchmark a point; the full benchmark runs once, on the final tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..checks.check_config import check_configuration
from ..checks.check_runtime import (
    MODE_REQUIRE,
    engine_driver_command,
    parse_driver_records,
    write_driver_spec,
)
from ..infra.compute.environment import RunEnvironment

#: Short prompts of different kinds; one token each.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in all?",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "Translate to French: Good morning, how are you?",
    "The three primary colors are",
    "If a train travels 60 miles per hour for 3 hours, it travels",
    "Once upon a time, in a small village by the sea,",
    "The chemical symbol for gold is",
    "Q: What is 17 times 23?\nA:",
    "import numpy as np\narr = np.zeros(",
    "The largest planet in our solar system is",
    "List three benefits of regular exercise:",
    "In 1969, the first humans to walk on the Moon were",
    "The derivative of x squared is",
    "Summarize in one sentence: The quick brown fox jumps over the lazy dog.",
)


@dataclass(frozen=True)
class LogitsCheck:
    """``perf.campaign.logits``: the prompts and how close the first tokens must stay."""

    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    top_k: int = 20
    #: Mean KL (nats) of the control's top-k distribution against the candidate's.
    max_kl: float = 0.1
    #: The share of prompts whose first token agrees (near-ties count as agreeing).
    min_top1: float = 0.9
    tie_margin: float = 0.15

    @classmethod
    def from_campaign(cls, campaign: Mapping[str, Any]) -> "LogitsCheck":
        block = campaign.get("logits") if isinstance(campaign.get("logits"), Mapping) else {}
        return cls(
            prompts=tuple(str(p) for p in block.get("prompts") or DEFAULT_PROMPTS),
            top_k=int(block.get("top_k", 20)),
            max_kl=float(block.get("max_kl", 0.1)),
            min_top1=float(block.get("min_top1", 0.9)),
            tie_margin=float(block.get("tie_margin", 0.15)),
        )

    def requests(self) -> list[dict[str, Any]]:
        return [
            {"id": f"p{i}", "prompt": prompt, "max_tokens": 1, "logprobs": self.top_k}
            for i, prompt in enumerate(self.prompts)
        ]


def job_script(
    check: LogitsCheck,
    spec: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str,
    member: str,
    tree: Path,
    out: Path,
) -> str:
    """The engine driver at ``member`` on ``tree``, one token per prompt with its top-k."""
    config = check_configuration(
        {**spec, "trtllm_repo_path": str(tree)},
        environment,
        unit=unit,
        features=None,
        max_tokens=1,
        member=member,
    )
    config["llm"]["modeling_v2"] = MODE_REQUIRE
    out.mkdir(parents=True, exist_ok=True)
    path = write_driver_spec(
        out / "logits.json", config=config, requests=check.requests(), max_tokens=1
    )
    return engine_driver_command(environment, path)


def first_tokens(output: str) -> dict[str, dict[int, float]]:
    """Each prompt's first-token top-k from a driver's output: ``{id: {token: log-prob}}``."""
    rows: dict[str, dict[int, float]] = {}
    for record in parse_driver_records(output):
        steps = record.get("top_logprobs") or [[]]
        row = {int(e[0]): float(e[1]) for e in steps[0] or () if e and e[0] is not None}
        if row:
            rows[str(record.get("id"))] = row
    return rows


def _kl(control: Mapping[int, float], candidate: Mapping[int, float]) -> float:
    """KL of the control's top-k, renormalized, against the candidate's on the same tokens.

    A token the candidate's top-k lacks gets its lowest listed log-prob: at most that.
    """
    floor = min(candidate.values(), default=-30.0)
    p = {t: math.exp(lp) for t, lp in control.items()}
    q = {t: math.exp(candidate.get(t, floor)) for t in control}
    sp, sq = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    return sum((v / sp) * math.log((v / sp) / (q[t] / sq)) for t, v in p.items() if v > 0)


def compare(
    control: Mapping[str, Mapping[int, float]],
    candidate: Mapping[str, Mapping[int, float]],
    check: LogitsCheck,
) -> tuple[bool, str, dict[str, Any]]:
    """Whether the candidate's first tokens stay with the control's, why, and the numbers.

    A prompt the control answered and the candidate did not is a disagreement.
    """
    asked = [key for key in control if control[key]]
    if not asked:
        return False, "the control answered no prompt", {}
    agree, kls = 0, []
    for key in asked:
        ours, theirs = candidate.get(key), control[key]
        if not ours:
            continue
        top, mine = max(theirs, key=theirs.get), max(ours, key=ours.get)
        tied = mine in theirs and theirs[top] - theirs[mine] <= check.tie_margin
        agree += top == mine or tied
        kls.append(_kl(theirs, ours))
    top1 = agree / len(asked)
    kl = sum(kls) / len(kls) if kls else math.inf
    numbers = {
        "prompts": len(asked),
        "answered": len(kls),
        "top1": round(top1, 4),
        "mean_kl": round(kl, 5),
    }
    passed = top1 >= check.min_top1 and kl <= check.max_kl
    detail = (
        f"first tokens agree on {top1:.0%} of {len(asked)} prompts (>= {check.min_top1:.0%}), "
        f"mean KL {kl:.4f} (<= {check.max_kl:g})"
        + (f", {len(asked) - len(kls)} unanswered" if len(kls) < len(asked) else "")
    )
    return passed, detail, numbers


__all__ = [
    "DEFAULT_PROMPTS",
    "LogitsCheck",
    "compare",
    "first_tokens",
    "job_script",
]
