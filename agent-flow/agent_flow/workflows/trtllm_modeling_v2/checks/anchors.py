# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each benchmark of each deployment is judged against, frozen once per workspace.

For a deployment and one of its benchmarks, in this order:

1. ``declared``: the benchmark's own ``anchor``, with its ``source``;
2. ``reference``: the row of the tree's reference file for the benchmark's ``model_key``
   whose tags are exactly the deployment's (:func:`tags`);
3. ``relative``: the score of the deployment named by ``relative_to``, measured as this
   one is, so the floor allows for the error of both (``sqrt(2) x SE``);
4. ``none``: the score is reported and nothing judges it.

The floor under a score is ``anchor - 2 x SE``, the binomial error at the anchor over the
samples scored (:func:`floor`). A benchmark states the scale its scores are on (``unit``:
``percent``, or ``fraction``); nothing is guessed from a score's size. :func:`freeze`
writes the anchors to ``anchors.yaml`` in the workspace the first time a run starts, each
with its unit and the sample count its floor is computed over, and adds only what is
absent on a later start: no commit of the run and no later edit of the spec moves what
its checks are held to, and an anchor frozen as ``none`` stays none. A file that cannot
be read stops the start (:class:`AnchorError`): the user deletes it to freeze again. The
checks read the frozen copy (:func:`frozen`); :func:`differences` says where the tree's
row now differs from it, for the report.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.files import write_text_atomic
from ..spec import deployments as deployments_mod
from ..spec.deployments import Benchmark, Deployment

ANCHORS_FILENAME = "anchors.yaml"
DECLARED = "declared"
REFERENCE = "reference"
RELATIVE = "relative"
NONE = "none"
#: Where the tree keeps its reference rows, one file per benchmark.
REFERENCES_DIR = "tests/integration/defs/accuracy/references"
#: ``k`` in ``anchor - k x SE``.
FLOOR_K = 2.0
#: Questions in the split ``trtllm-eval`` scores for each benchmark it ships, for the
#: standard error when neither the run's own output nor the spec states the count.
KNOWN_SAMPLES: dict[str, int] = {
    "gsm8k": 1319,
    "mmlu": 14042,
    "mmmu": 900,
    "gpqa_diamond": 198,
    "gpqa_main": 448,
    "gpqa_extended": 546,
}
#: The reference rows' tag for each KV-cache dtype an options file can state.
_KV_TAGS: dict[str, str] = {"fp8": "FP8", "nvfp4": "NVFP4"}
# What a run prints of its own count: MMLU's total, and lm-eval's running score.
_MMLU_SAMPLES = re.compile(r"MMLU weighted average accuracy:\s*[0-9.]+\s*\((\d+)\)")
_PARTIAL_TOTAL = re.compile(r"Partial scores after\s+(\d+)/(\d+)\s+responses")
_NUM_SAMPLES_ARG = re.compile(r"^--num_samples(?:=(\d+))?$")


class AnchorError(ValueError):
    """The workspace's frozen anchors cannot be read: nothing is judged until they can."""


@dataclass(frozen=True)
class Anchor:
    """What one benchmark of one deployment is judged against."""

    deployment: str
    benchmark: str
    kind: str
    #: The score, on the scale the benchmark reports; ``None`` for ``relative`` and ``none``.
    value: float | None = None
    #: Where it comes from: the declared source, the reference file and its tags, the
    #: deployment it is relative to, or why there is none.
    source: str = ""
    relative_to: str | None = None
    #: The scale ``value`` and the benchmark's scores are on: ``percent`` or ``fraction``.
    unit: str = "percent"
    #: The sample count the floor is computed over, fixed when the anchor is resolved.
    samples: int | None = None

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.benchmark}"


@dataclass(frozen=True)
class Floor:
    """One accuracy floor and how it was derived."""

    value: float
    reference: float
    se: float
    samples: int
    #: The reference is another deployment's score, measured as this one is: the SE is
    #: that of the difference of two measurements.
    relative: bool = False

    def describe(self) -> str:
        """``reference 92.08 - 2.00 x SE 0.75 (n=1319) = 90.59``."""
        se = f"SE {self.se:.2f} (n={self.samples})"
        if self.relative:
            return (
                f"the other deployment's score {self.reference:.4f} - {FLOOR_K:.2f} x {se} "
                f"of the difference = {self.value:.4f}"
            )
        return f"reference {self.reference:.4f} - {FLOOR_K:.2f} x {se} = {self.value:.4f}"


def binomial_se(score: float, samples: int) -> float:
    """The standard error, in points, of a percentage measured on ``samples`` items.

    ``score`` is on the 0-100 scale. The proportion is clamped half a sample away from 0
    and 1, so a score of exactly 100 still has a nonzero error.
    """
    if samples < 1:
        raise ValueError(f"a standard error needs at least one sample, got {samples}")
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return 100.0 * math.sqrt(p * (1.0 - p) / samples)


def model_key(task_data: Mapping[str, Any], benchmark: Benchmark) -> str:
    """The name the benchmark's reference rows are keyed by: stated, else ``<org>/<model>``.

    Stated on the benchmark, else in the spec's ``accuracy``; the fallback is the
    checkpoint path's last two components.
    """
    stated = benchmark.model_key or (task_data.get("accuracy") or {}).get("model_key")
    if isinstance(stated, str) and stated:
        return stated
    parts = Path(str(task_data["checkpoint_path"]).rstrip("/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def tags(deployment: Deployment, benchmark: Benchmark) -> dict[str, Any]:
    """The tags of the reference row ``deployment`` is judged by on ``benchmark``.

    What the deployment's options imply (a quantized KV cache, a speculative algorithm)
    with the benchmark's own ``discriminators`` (the checkpoint's weight format) over
    them. A row matches only its exact tags: the unquantized row never stands in for a
    deployment that quantizes its cache.
    """
    # Of a disaggregated deployment, the generation server's: it produces what is scored.
    options = (deployment.gen or deployment.server).options
    implied: dict[str, Any] = {}
    dtype = str((options.get("kv_cache_config") or {}).get("dtype") or "").lower()
    if dtype in _KV_TAGS:
        implied["kv_cache_quant_algo"] = _KV_TAGS[dtype]
    speculative = options.get("speculative_config") or {}
    if speculative.get("decoding_type"):
        implied["spec_dec_algo"] = str(speculative["decoding_type"])
    stated = {k: v for k, v in (benchmark.discriminators or {}).items() if v is not None}
    return {**implied, **stated}


def load_reference_score(
    reference_yaml: str | Path, key: str, *, discriminators: Mapping[str, Any] | None = None
) -> float:
    """Read the anchor for one model from TensorRT-LLM's accuracy references.

    An entry matches when its tags (``quant_algo``, ``kv_cache_quant_algo``,
    ``spec_dec_algo``) are exactly the ones asked for, no more, no less. Raises
    ``KeyError`` when no entry, or more than one, matches: the anchor is never invented.
    """
    path = Path(reference_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"accuracy reference file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if key not in data:
        raise KeyError(
            f"{path} has no entry for {key!r}. The anchor comes from a written source or from "
            f"the user — it is never invented. Add the entry (citing where the score came "
            f"from) before the accuracy gate can run: `{key}:` then `  - accuracy: <score>`, "
            f"committed to the checkout. A score under the gate's own protocol comes from an "
            f"implementation that already serves the checkpoint, e.g. `trtllm-eval --model "
            f"<checkpoint> <benchmark>` on a TensorRT-LLM whose existing model class loads "
            f"it, on the same GPU type."
        )
    wanted = {k: v for k, v in (discriminators or {}).items() if v is not None}
    entries = [e for e in data[key] or [] if isinstance(e, dict)]
    matches = [e for e in entries if {k: v for k, v in e.items() if k != "accuracy"} == wanted]
    if not matches:
        available = [{k: v for k, v in e.items() if k != "accuracy"} for e in entries]
        raise KeyError(
            f"{path}:{key} has no entry matching {wanted or '{}'}. Available tag sets: {available}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"{path}:{key} has {len(matches)} entries matching {wanted or '{}'}; the anchor "
            f"must be unambiguous"
        )
    return float(matches[0]["accuracy"])


def resolve(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    repo: str | Path | None = None,
) -> Anchor:
    """The anchor of ``benchmark`` on ``deployment``, read now from the spec and the tree."""
    base = dict(
        deployment=deployment.name,
        benchmark=benchmark.name,
        unit=benchmark.unit,
        samples=samples_of(benchmark)[0],
    )
    if benchmark.anchor is not None:
        return Anchor(kind=DECLARED, value=benchmark.anchor, source=benchmark.source, **base)
    path = Path(str(benchmark.reference_yaml or f"{REFERENCES_DIR}/{benchmark.name}.yaml"))
    if not path.is_absolute():
        path = Path(str(repo or task_data.get("trtllm_repo_path") or ".")) / path
    key, wanted = model_key(task_data, benchmark), tags(deployment, benchmark)
    try:
        score = load_reference_score(path, key, discriminators=wanted)
    except (KeyError, FileNotFoundError, TypeError, ValueError, yaml.YAMLError) as exc:
        why = str(exc).strip("'\"").splitlines()[0][:300]
    else:
        shown = f" with {wanted}" if wanted else ""
        return Anchor(
            kind=REFERENCE, value=score, source=f"{path}: the row of `{key}`{shown}", **base
        )
    if deployment.relative_to:
        return Anchor(
            kind=RELATIVE,
            source=f"the score of `{deployment.relative_to}`",
            relative_to=deployment.relative_to,
            **base,
        )
    return Anchor(kind=NONE, source=f"no declared anchor and no reference row ({why})", **base)


def load(workspace: str | Path) -> dict[str, Anchor]:
    """The anchors frozen in ``workspace``, by ``<deployment>:<benchmark>``; none without a file.

    Raises :class:`AnchorError` for a file that is there and does not hold them.
    """
    path = Path(workspace) / ANCHORS_FILENAME
    if not path.exists():
        return {}
    known = set(Anchor.__dataclass_fields__)
    try:
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))
        found = [Anchor(**{k: v for k, v in row.items() if k in known}) for row in rows]
    except (OSError, yaml.YAMLError, TypeError, AttributeError) as exc:
        raise AnchorError(
            f"{path} does not hold the frozen anchors ({type(exc).__name__}: {exc}): nothing "
            f"is judged against a file that cannot be read; delete it to freeze them again"
        ) from exc
    return {anchor.key: anchor for anchor in found}


def freeze(
    workspace: str | Path, task_data: Mapping[str, Any], repo: str | Path | None = None
) -> dict[str, Anchor]:
    """Freeze the anchor of every benchmark of every deployment that has none frozen yet.

    What is frozen stays: a later start, after a spec edit or a commit to the reference
    file, adds the new deployments' and benchmarks' and changes no other.
    """
    held = load(workspace)
    added = False
    for deployment in deployments_mod.resolve(task_data):
        for benchmark in deployment.benchmarks:
            key = f"{deployment.name}:{benchmark.name}"
            if key not in held:
                held[key] = resolve(task_data, deployment, benchmark, repo)
                added = True
    if added:
        rows = [asdict(anchor) for anchor in held.values()]
        Path(workspace).mkdir(parents=True, exist_ok=True)
        write_text_atomic(Path(workspace) / ANCHORS_FILENAME, yaml.safe_dump(rows, sort_keys=False))
    return held


def frozen(workspace: str | Path | None, deployment: str, benchmark: str) -> Anchor | None:
    """The frozen anchor of ``benchmark`` on ``deployment``; ``None`` where nothing is frozen."""
    return load(workspace).get(f"{deployment}:{benchmark}") if workspace is not None else None


def differences(
    workspace: str | Path, task_data: Mapping[str, Any], repo: str | Path | None = None
) -> list[str]:
    """Where the tree's reference row now differs from the frozen one, one line each."""
    said = []
    by_name = {d.name: d for d in deployments_mod.resolve(task_data)}
    for anchor in load(workspace).values():
        deployment = by_name.get(anchor.deployment)
        benchmark = deployment.benchmark(anchor.benchmark) if deployment else None
        if anchor.kind != REFERENCE or benchmark is None:
            continue
        now = resolve(task_data, deployment, benchmark, repo)
        if now.kind != REFERENCE:
            said.append(f"{anchor.key}: frozen at {anchor.value:g}; the tree now has no such row")
        elif now.value != anchor.value:
            said.append(
                f"{anchor.key}: frozen at {anchor.value:g}; the tree's row now says {now.value:g}"
            )
    return said


def samples_of(benchmark: Benchmark, output: str = "") -> tuple[int | None, str]:
    """How many samples a score was (or will be) measured on, and where the count is from.

    Most specific first: the count the run itself printed (what was actually scored),
    ``--num_samples`` among the benchmark's options, the benchmark's ``samples``, the
    benchmark's known size. ``(None, why)`` when none of them answers.
    """
    printed = _MMLU_SAMPLES.findall(output) or [t for _, t in _PARTIAL_TOTAL.findall(output)]
    if printed:
        return int(printed[-1]), "the run's own output"
    for index, arg in enumerate(benchmark.args):
        matched = _NUM_SAMPLES_ARG.match(arg)
        value = matched and (matched.group(1) or "".join(benchmark.args[index + 1 : index + 2]))
        if value and value.isdigit():
            return int(value), "--num_samples in the benchmark's args"
    if benchmark.samples is not None:
        return int(benchmark.samples), "the benchmark's `samples`"
    if benchmark.name in KNOWN_SAMPLES:
        return KNOWN_SAMPLES[benchmark.name], f"the {benchmark.name} split size"
    return None, f"no sample count for `{benchmark.name}`: state its `samples`"


def floor(
    anchor: Anchor,
    benchmark: Benchmark,
    *,
    relative_score: float | None = None,
    output: str = "",
) -> tuple[Floor | None, str]:
    """The floor ``anchor`` puts under a score, and where its sample count came from.

    ``reference - 2 x SE`` for a declared or reference anchor; for a relative one,
    ``relative_score - 2 x sqrt(2) x SE``, the error of the difference of two scores.
    ``None`` for no anchor, a relative one whose deployment has no score, or an unknown
    sample count. The count is the one fixed with the anchor; an anchor resolved with
    none takes what the run printed.
    """
    relative = anchor.kind == RELATIVE
    reference = relative_score if relative else anchor.value
    if anchor.kind == NONE or reference is None:
        return None, "no anchor"
    samples, source = samples_of(benchmark, output)
    if anchor.samples:
        # Fixed with the anchor: what the run prints or the spec says later does not move it.
        samples, source = anchor.samples, "the count fixed with the anchor"
    if samples is None:
        return None, source
    # The binomial error is of a percentage; the floor is on the benchmark's own scale.
    scale = 100.0 if anchor.unit == "fraction" else 1.0
    se = binomial_se(reference * scale, samples) * (math.sqrt(2) if relative else 1.0) / scale
    return Floor(reference - FLOOR_K * se, reference, se, samples, relative), source


__all__ = [
    "ANCHORS_FILENAME",
    "DECLARED",
    "FLOOR_K",
    "KNOWN_SAMPLES",
    "NONE",
    "REFERENCE",
    "RELATIVE",
    "Anchor",
    "AnchorError",
    "Floor",
    "binomial_se",
    "differences",
    "floor",
    "freeze",
    "frozen",
    "load",
    "load_reference_score",
    "model_key",
    "resolve",
    "samples_of",
    "tags",
]
