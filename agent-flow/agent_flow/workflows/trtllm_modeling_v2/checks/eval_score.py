# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ``trtllm-eval`` score held to its statistical floor: how a stated disaggregated deployment is judged.

The served disagg gate (``checks/disagg.py``) runs ``trtllm-eval`` through the deployment's
proxy. It runs it under the unit's protocol, ``accuracy.evaluator_kwargs`` as ``trtllm-eval``
flags (:func:`eval_arguments`), reads the score the evaluator prints (:func:`parse_eval_score`)
and holds it to ``reference - k * SE``, with the binomial SE of the sample count
(:func:`statistical_floor`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

#: ``k`` in ``reference - k * SE``.
DEFAULT_FLOOR_K = 2.0

#: Questions in the split ``trtllm-eval`` scores for each benchmark it ships, for the
#: standard error when the eval's own output does not state the count.
KNOWN_BENCHMARK_SAMPLES: dict[str, int] = {
    "gsm8k": 1319,
    "mmlu": 14042,
    "mmmu": 900,
    "gpqa_diamond": 198,
    "gpqa_main": 448,
    "gpqa_extended": 546,
}

# ``lm-eval <task> <filter> accuracy: 90.67`` and ``lm-eval <task> average accuracy: 90.67``,
# the two shapes ``tensorrt_llm/evaluate/lm_eval.py`` prints at the end of a run.
_LM_EVAL_SCORE = re.compile(r"lm-eval\s+\S+\s+.*?accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
# ``MMLU weighted average accuracy: 74.98 (14042)``: MMLU does not go through lm-eval.
_MMLU_SCORE = re.compile(r"MMLU weighted average accuracy:\s*([0-9]+(?:\.[0-9]+)?)")
_MMLU_SAMPLES = re.compile(r"MMLU weighted average accuracy:\s*[0-9.]+\s*\((\d+)\)")
_PARTIAL_TOTAL = re.compile(r"Partial scores after\s+(\d+)/(\d+)\s+responses")
#: A line that may carry a benchmark's score: a metric's name, then a number.
_SCORE_LIKE = re.compile(r"(?i)\b(accuracy|score|rouge\w*|exact_match|f1|pass@\d+|bleu)\b[^\n]*?\d")
#: How many of those lines a verdict quotes, the last ones printed.
SCORE_LINES = 8


def binomial_se(score: float, samples: int) -> float:
    """The standard error, in points, of a percentage measured on ``samples`` items.

    ``score`` is on the 0-100 scale. The proportion is clamped half a sample away from 0
    and 1, so a score of exactly 100 still has a nonzero error.
    """
    if samples < 1:
        raise ValueError(f"a standard error needs at least one sample, got {samples}")
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return 100.0 * math.sqrt(p * (1.0 - p) / samples)


@dataclass(frozen=True)
class Floor:
    """``reference - k * SE``, with the SE taken at the reference score."""

    value: float
    reference: float
    se: float
    samples: int
    k: float

    def describe(self) -> str:
        """``reference 92.08 - 2.00 x SE 0.75 (n=1319) = 90.59``."""
        return (
            f"reference {self.reference:.4f} - {self.k:.2f} x SE {self.se:.2f} "
            f"(n={self.samples}) = {self.value:.4f}"
        )


def statistical_floor(reference: float, samples: int, k: float = DEFAULT_FLOOR_K) -> Floor:
    """The floor ``reference`` gets on ``samples`` items."""
    se = binomial_se(reference, samples)
    return Floor(value=reference - k * se, reference=reference, se=se, samples=samples, k=k)


def eval_arguments(evaluator_kwargs: Mapping[str, Any] | None) -> list[str]:
    """``trtllm-eval``'s flags for ``accuracy.evaluator_kwargs``, the protocol ``accuracy_core`` passes.

    A true value is ``--name``, a false one is left out, a mapping or a list is
    ``--name <JSON>``, anything else ``--name <value>``. A key ``trtllm-eval`` does not
    take fails the run, which the verdict then says.
    """
    out: list[str] = []
    for name, value in (evaluator_kwargs or {}).items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(f"--{name}")
        elif isinstance(value, (Mapping, list, tuple)):
            out += [f"--{name}", json.dumps(value, sort_keys=True)]
        else:
            out += [f"--{name}", str(value)]
    return out


def parse_eval_score(output: str) -> float | None:
    """The final score a ``trtllm-eval`` run printed; ``None`` when it printed none.

    MMLU prints its own weighted average; everything else comes through lm-eval. The
    **last** match wins: a long run also prints partial-score estimates.
    """
    for compiled in (_MMLU_SCORE, _LM_EVAL_SCORE):
        for match in reversed(list(compiled.finditer(output))):
            try:
                return float(match.group(1))
            except (IndexError, TypeError, ValueError):
                continue
    return None


def eval_samples(
    output: str, benchmark: str, evaluator_kwargs: Mapping[str, Any] | None = None
) -> tuple[int | None, str]:
    """How many samples the score was measured on, and from where; ``(None, reason)`` if unknown.

    Most specific first: the count the eval printed, then ``num_samples`` in the
    protocol, then the size of the benchmark's split.
    """
    mmlu = _MMLU_SAMPLES.findall(output)
    if mmlu:
        return int(mmlu[-1]), "the eval's own output"
    partial = _PARTIAL_TOTAL.findall(output)
    if partial:
        return int(partial[-1][1]), "the eval's own output"
    declared = (evaluator_kwargs or {}).get("num_samples")
    if declared is not None:
        try:
            return int(declared), "num_samples in accuracy.evaluator_kwargs"
        except (TypeError, ValueError):
            pass
    known = KNOWN_BENCHMARK_SAMPLES.get(benchmark)
    if known is not None:
        return known, f"the {benchmark} split size"
    return None, f"no sample count for {benchmark}: the eval printed none and the split is unknown"


def score_lines(output: str, limit: int = SCORE_LINES) -> list[str]:
    """The last ``limit`` distinct lines of ``output`` that look like a score, oldest first."""
    seen: list[str] = []
    for line in reversed(output.splitlines()):
        text = line.strip()[:240]
        if text and _SCORE_LIKE.search(text) and text not in seen:
            seen.append(text)
            if len(seen) == limit:
                break
    return seen[::-1]


def floor_fields(floor: Floor | None) -> dict[str, Any]:
    """A verdict's fields of ``floor``."""
    if floor is None:
        return {}
    return {
        "floor": floor.value,
        "se": floor.se,
        "samples": floor.samples,
        "floor_rule": floor.describe(),
    }


def judge_score(
    benchmark: str,
    reference: float,
    output: str,
    *,
    evaluator_kwargs: Mapping[str, Any] | None = None,
    said: str = "measured",
) -> dict[str, Any]:
    """A verdict's fields for a ``trtllm-eval`` run's ``output``: its score against the floor.

    The detail opens with ``said``. A run that printed no score, or whose sample count is
    unknown, does not pass.
    """
    measured = parse_eval_score(output)
    if measured is None:
        lines = score_lines(output)
        printed = (
            "; the lines it printed that look like a score:\n" + "\n".join(f"  {x}" for x in lines)
            if lines
            else "; it printed no line that looks like a score"
        )
        return dict(
            passed=False,
            detail="trtllm-eval printed no score in a form this gate reads (lm-eval's "
            f"`accuracy:` line or MMLU's weighted average){printed}",
        )
    samples, source = eval_samples(output, benchmark, evaluator_kwargs)
    if samples is None:
        return dict(
            passed=False,
            detail=f"{said} {measured:.4f}, but the statistical floor needs the sample count "
            f"and there is {source}. A floor that cannot be computed is not a pass",
            measured=measured,
        )
    floor = statistical_floor(reference, samples)
    passed = measured >= floor.value
    return dict(
        passed=passed,
        detail=f"{said} {measured:.4f} {'>=' if passed else '<'} {floor.value:.4f} "
        f"({floor.describe()}; count from {source})",
        measured=measured,
        **floor_fields(floor),
    )


__all__ = [
    "DEFAULT_FLOOR_K",
    "KNOWN_BENCHMARK_SAMPLES",
    "Floor",
    "binomial_se",
    "eval_arguments",
    "eval_samples",
    "floor_fields",
    "judge_score",
    "parse_eval_score",
    "score_lines",
    "statistical_floor",
]
