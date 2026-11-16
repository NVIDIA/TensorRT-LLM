# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each benchmark of each deployment is judged against, frozen once per workspace.

For a deployment and one of its benchmarks, in this order:

1. ``declared``: the benchmark's own ``anchor``, with its ``source``;
2. ``reference``: the row of the tree's reference file for the benchmark's ``model_key``
   whose tags are exactly the deployment's (:func:`tags`);
3. ``relative``: the score of the deployment named by ``relative_to``, measured as this
   one is, so the floor allows for the error of both;
4. ``none``: nothing to hold a score to. The benchmark is blocked before its load, with
   the reason, and everything else of the deployment goes on.

The floor under a score is the spec's ``accuracy.floor`` rule (``accuracy_policy.py``:
``anchor - k x SE`` unless the spec says a fixed tolerance). A benchmark states the scale
its scores are on (``unit``: ``percent``, or ``fraction``); nothing is guessed from a
score's size. :func:`freeze` writes the anchors to ``anchors.yaml`` in the workspace the
first time a run starts and adds only what is absent on a later start: no commit of the
run and no later edit of the tree's reference file moves what its checks are held to.

No anchor is beyond change: one that is mistaken, or was frozen as ``none``, is changed by
an explicit change (:func:`change`, which ``resolve change-requirement`` calls), which
writes the new one and appends what changed, from what to what and why, to
``anchor_changes.jsonl``. Every start of a run journals the changes made (:func:`changes`).
"""

from __future__ import annotations

import datetime
import json
import re
import textwrap
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.files import write_text_atomic
from ..spec import deployments as deployments_mod
from ..spec.deployments import Benchmark, Deployment
from . import accuracy_policy

ANCHORS_FILENAME = "anchors.yaml"
#: One line per changed anchor: when, which, from what to what, and why.
CHANGES_FILENAME = "anchor_changes.jsonl"
DECLARED = "declared"
REFERENCE = "reference"
RELATIVE = "relative"
NONE = "none"
#: Where the tree keeps its reference rows, one file per benchmark, when the spec names none.
REFERENCES_DIR = "tests/integration/defs/accuracy/references"
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
    them. A row matches as the in-tree accuracy test selects it: the unquantized row never
    stands in for a deployment that quantizes its cache, and a row's other keys (its
    threshold, its sample count) do not select.
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


def reference_path(
    task_data: Mapping[str, Any], benchmark: Benchmark, repo: str | Path | None = None
) -> Path:
    """The reference file ``benchmark``'s rows are read from, resolved against the tree."""
    path = Path(str(benchmark.reference_yaml or f"{REFERENCES_DIR}/{benchmark.name}.yaml"))
    if not path.is_absolute():
        path = Path(str(repo or task_data.get("trtllm_repo_path") or ".")) / path
    return path


def resolve(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    repo: str | Path | None = None,
) -> Anchor:
    """The anchor of ``benchmark`` on ``deployment``, read now from the spec and the tree."""
    from .accuracy import load_reference_score  # that module reads this one's Anchor

    base = dict(
        deployment=deployment.name,
        benchmark=benchmark.name,
        unit=benchmark.unit,
        samples=samples_of(benchmark)[0],
    )
    if benchmark.anchor is not None:
        return Anchor(kind=DECLARED, value=benchmark.anchor, source=benchmark.source, **base)
    path = reference_path(task_data, benchmark, repo)
    key, wanted = model_key(task_data, benchmark), tags(deployment, benchmark)
    try:
        score = load_reference_score(path, key, discriminators=wanted)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        # A reference that is missing, unreadable or holds no such row is no anchor, said
        # with its reason: it never stops the deployments it does not judge.
        why = textwrap.shorten(str(exc).strip("'\"").splitlines()[0], 600, placeholder=" ...")
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


def _save(workspace: str | Path, held: Mapping[str, Anchor]) -> None:
    rows = [asdict(anchor) for anchor in held.values()]
    Path(workspace).mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(workspace) / ANCHORS_FILENAME, yaml.safe_dump(rows, sort_keys=False))


def freeze(
    workspace: str | Path, task_data: Mapping[str, Any], repo: str | Path | None = None
) -> dict[str, Anchor]:
    """Freeze the anchor of every benchmark of every deployment that has none frozen yet.

    What is frozen stays until :func:`change` is asked to change it: a later start, after
    a spec edit or a commit to the reference file, adds the new deployments' and
    benchmarks' and moves no other.
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
        _save(workspace, held)
    return held


def frozen(workspace: str | Path | None, deployment: str, benchmark: str) -> Anchor | None:
    """The frozen anchor of ``benchmark`` on ``deployment``; ``None`` where nothing is frozen."""
    return load(workspace).get(f"{deployment}:{benchmark}") if workspace is not None else None


def change(
    workspace: str | Path,
    deployment: str,
    benchmark: str,
    *,
    reason: str,
    value: float | None = None,
    source: str = "",
    relative_to: str | None = None,
    unit: str | None = None,
    samples: int | None = None,
) -> tuple[Anchor | None, Anchor]:
    """Change one frozen anchor, and record what changed; ``(before, after)``.

    A ``value`` (with its ``source``) makes it a declared anchor; ``relative_to`` makes it
    the named deployment's own score; neither takes the anchor away. ``reason`` says why,
    and is required: every start of the run journals every change made. Nothing
    refuses a change; it is the caller's to make.
    """
    if not reason.strip():
        raise ValueError("a changed anchor needs its reason: say why it changes")
    if value is not None and relative_to:
        raise ValueError("an anchor is a score or another deployment's score, not both")
    held = load(workspace)
    key = f"{deployment}:{benchmark}"
    before = held.get(key)
    base = before or Anchor(deployment=deployment, benchmark=benchmark, kind=NONE)
    if value is not None:
        after = replace(
            base, kind=DECLARED, value=float(value), source=source or reason, relative_to=None
        )
    elif relative_to:
        after = replace(
            base,
            kind=RELATIVE,
            value=None,
            source=f"the score of `{relative_to}`",
            relative_to=relative_to,
        )
    else:
        after = replace(base, kind=NONE, value=None, source=source or reason, relative_to=None)
    after = replace(
        after,
        unit=unit or after.unit,
        samples=samples if samples is not None else after.samples,
    )
    held[key] = after
    _save(workspace, held)
    line = {
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "anchor": key,
        "from": asdict(before) if before is not None else None,
        "to": asdict(after),
        "reason": reason.strip(),
    }
    with (Path(workspace) / CHANGES_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    return before, after


def changes(workspace: str | Path) -> list[dict[str, Any]]:
    """Every anchor change recorded in ``workspace``, oldest first."""
    try:
        text = (Path(workspace) / CHANGES_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            found.append(data)
    return found


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
    known = accuracy_policy.KNOWN_BENCHMARK_SAMPLES.get(benchmark.name)
    if known is not None:
        return known, f"the {benchmark.name} split size"
    return None, f"no sample count for `{benchmark.name}`: state its `samples`"


def floor(
    anchor: Anchor,
    benchmark: Benchmark,
    policy: accuracy_policy.FloorPolicy,
    *,
    relative_score: float | None = None,
    output: str = "",
) -> tuple[accuracy_policy.Floor | None, str]:
    """The floor ``anchor`` puts under a score, and where its sample count came from.

    The spec's floor rule over a declared or reference anchor; for a relative one, over
    ``relative_score`` with the error of the difference of two scores. ``None`` for no
    anchor, a relative one whose deployment has no score, or a rule that needs an unknown
    sample count. The count is the one fixed with the anchor; an anchor resolved with
    none takes what the run printed.
    """
    relative = anchor.kind == RELATIVE
    reference = relative_score if relative else anchor.value
    if anchor.kind == NONE or reference is None:
        return (
            None,
            f"`{anchor.relative_to}` has no score to hold it to" if relative else "no anchor",
        )
    samples, source = samples_of(benchmark, output)
    if anchor.samples:
        # Fixed with the anchor: what the run prints or the spec says later does not move it.
        samples, source = anchor.samples, "the count fixed with the anchor"
    # The binomial error is of a percentage; the floor is on the benchmark's own scale.
    scale = 100.0 if anchor.unit == "fraction" else 1.0
    found = accuracy_policy.floor_for(policy, reference * scale, samples, relative)
    if found is None:
        return None, source
    if scale != 1.0:
        found = replace(
            found,
            value=found.value / scale,
            reference=found.reference / scale,
            se=None if found.se is None else found.se / scale,
            tolerance=None if found.tolerance is None else found.tolerance / scale,
        )
    return found, source


__all__ = [
    "ANCHORS_FILENAME",
    "CHANGES_FILENAME",
    "DECLARED",
    "NONE",
    "REFERENCE",
    "RELATIVE",
    "Anchor",
    "AnchorError",
    "change",
    "changes",
    "floor",
    "freeze",
    "frozen",
    "load",
    "model_key",
    "reference_path",
    "resolve",
    "samples_of",
    "tags",
]
