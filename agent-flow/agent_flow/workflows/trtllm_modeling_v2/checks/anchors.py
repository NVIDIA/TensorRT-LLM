# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What each benchmark of each deployment is judged against, frozen once per workspace.

For a deployment and one of its benchmarks, in this order:

1. ``declared``: the benchmark's own ``anchor``, with its ``source``;
2. ``reference``: the row of the tree's reference file for ``accuracy.model_key`` whose
   tags are exactly the deployment's;
3. ``relative``: the score of the deployment named by ``relative_to``, measured as this
   one is, so the floor allows for the error of both (``sqrt(2) x SE``);
4. ``none``: the score is reported and nothing judges it.

:func:`freeze` writes them to ``anchors.yaml`` in the workspace the first time a run
starts and adds only what is absent on a later start, so no commit of the run, to the
reference file or anywhere else, can move what its checks are held to. The gates read
the frozen copy (:func:`frozen`); :func:`differences` says where the tree's row now
differs from it, for the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..infra.files import write_text_atomic
from ..spec import deployments as deployments_mod
from ..spec.deployments import Benchmark, Deployment
from ..spec.shorthand import unit_of, variant_spec
from ..spec.task_schema import resolve_per_unit, resolve_samples
from ..spec.variants import is_variant
from . import accuracy_policy
from .accuracy import load_reference_score, resolve_discriminators, resolve_model_key

ANCHORS_FILENAME = "anchors.yaml"
DECLARED = "declared"
REFERENCE = "reference"
RELATIVE = "relative"
NONE = "none"
#: Where the tree keeps its reference rows, one file per benchmark.
REFERENCES_DIR = "tests/integration/defs/accuracy/references"
#: The reference rows' tag for each KV-cache dtype and speculative algorithm an options
#: file can state.
_KV_TAGS: dict[str, str] = {"fp8": "FP8", "nvfp4": "NVFP4"}


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

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.benchmark}"


def _tags(task_data: Mapping[str, Any], deployment: Deployment) -> dict[str, Any]:
    """The reference row's tags for a deployment a spec states itself.

    ``accuracy.discriminators`` (the checkpoint's own, such as its weight format) with
    what the deployment's options add: a KV-cache dtype, a speculative algorithm.
    """
    tags = dict((task_data.get("accuracy") or {}).get("discriminators") or {})
    options = next(iter(deployment.servers.values())).options
    dtype = str((options.get("kv_cache_config") or {}).get("dtype") or "").lower()
    if dtype in _KV_TAGS:
        tags["kv_cache_quant_algo"] = _KV_TAGS[dtype]
    speculative = options.get("speculative_config") or {}
    if speculative.get("decoding_type"):
        tags["spec_dec_algo"] = str(speculative["decoding_type"])
    return tags


def _reference_row(
    task_data: Mapping[str, Any], deployment: Deployment, benchmark: str, repo: Path
) -> tuple[float, str]:
    """The reference row for ``deployment`` on ``benchmark``: its score and where it was read.

    Raises ``KeyError`` or ``FileNotFoundError`` when no row, or more than one, fits.
    """
    accuracy = task_data.get("accuracy") or {}
    if deployments_mod.is_native(task_data):
        key = accuracy.get("model_key")
        if not key:
            raise KeyError("the spec states no accuracy.model_key")
        stated, tags = accuracy.get("reference_yaml"), _tags(task_data, deployment)
    else:
        # A shorthand spec names its rows per unit; a variant's are its unit's with its own.
        unit = unit_of(deployment.name)
        spec = (
            variant_spec(task_data, unit, world_size=1)
            if is_variant(task_data, unit)
            else task_data
        )
        accuracy = spec.get("accuracy") or {}
        key = resolve_model_key(spec, unit)
        stated = resolve_per_unit(accuracy, "reference_yaml", unit)
        tags = dict(resolve_discriminators(accuracy, unit) or {})
    path = Path(str(stated or f"{REFERENCES_DIR}/{benchmark}.yaml"))
    if not path.is_absolute():
        path = repo / path
    score = load_reference_score(path, str(key), discriminators=tags or None)
    shown = f" with {tags}" if tags else ""
    return score, f"{path}: the row of `{key}`{shown}"


def resolve(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    repo: str | Path | None = None,
) -> Anchor:
    """The anchor of ``benchmark`` on ``deployment``, read now from the spec and the tree."""
    base = dict(deployment=deployment.name, benchmark=benchmark.name)
    if benchmark.anchor is not None:
        return Anchor(kind=DECLARED, value=benchmark.anchor, source=benchmark.source, **base)
    root = Path(str(repo or task_data.get("trtllm_repo_path") or "."))
    why = ""
    if not (deployment.relative_to and not deployments_mod.is_native(task_data)):
        # A shorthand variant judged against its unit names no row of its own.
        try:
            score, where = _reference_row(task_data, deployment, benchmark.name, root)
            return Anchor(kind=REFERENCE, value=score, source=where, **base)
        except (KeyError, FileNotFoundError, TypeError, ValueError) as exc:
            why = str(exc).strip("'\"").splitlines()[0][:200]
    if deployment.relative_to:
        return Anchor(
            kind=RELATIVE,
            source=f"the score of `{deployment.relative_to}`",
            relative_to=deployment.relative_to,
            **base,
        )
    return Anchor(kind=NONE, source=f"no declared anchor and no reference row ({why})", **base)


def load(workspace: str | Path) -> dict[str, Anchor]:
    """The anchors frozen in ``workspace``, by ``<deployment>:<benchmark>``."""
    try:
        rows = yaml.safe_load((Path(workspace) / ANCHORS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    known = set(Anchor.__dataclass_fields__)
    found = [Anchor(**{k: v for k, v in row.items() if k in known}) for row in rows or ()]
    return {anchor.key: anchor for anchor in found}


def freeze(
    workspace: str | Path, task_data: Mapping[str, Any], repo: str | Path | None = None
) -> dict[str, Anchor]:
    """Freeze the anchor of every benchmark of every deployment that has none frozen yet.

    What is frozen stays: a later start, after a spec edit or a commit to the reference
    file, adds the new deployments' and benchmarks' and changes no other.
    """
    frozen = load(workspace)
    added = False
    for deployment in deployments_mod.resolve(task_data):
        for benchmark in deployment.benchmarks:
            key = f"{deployment.name}:{benchmark.name}"
            if key not in frozen:
                frozen[key] = resolve(task_data, deployment, benchmark, repo)
                added = True
    if added:
        rows = [asdict(anchor) for anchor in frozen.values()]
        Path(workspace).mkdir(parents=True, exist_ok=True)
        write_text_atomic(Path(workspace) / ANCHORS_FILENAME, yaml.safe_dump(rows, sort_keys=False))
    return frozen


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
        if anchor.kind != REFERENCE or deployment is None:
            continue
        now = resolve(task_data, deployment, Benchmark(name=anchor.benchmark), repo)
        if now.kind != REFERENCE:
            said.append(f"{anchor.key}: frozen at {anchor.value:g}; the tree now has no such row")
        elif now.value != anchor.value:
            said.append(
                f"{anchor.key}: frozen at {anchor.value:g}; the tree's row now says {now.value:g}"
            )
    return said


def floor(
    anchor: Anchor,
    accuracy: Mapping[str, Any],
    benchmark: Benchmark,
    *,
    relative_score: float | None = None,
    output: str = "",
) -> tuple[accuracy_policy.Floor | None, str]:
    """The floor ``anchor`` puts under a score, and where its sample count came from.

    ``reference - k x SE`` for a declared or reference anchor; for a relative one,
    ``relative_score - k x sqrt(2) x SE``, the error of the difference of two scores.
    ``None`` for no anchor, a relative one whose deployment has no score, or an unknown
    sample count. The count is, in this order: what the run printed, ``--num_samples``
    among the benchmark's options, ``accuracy.samples``, the benchmark's known size.
    """
    reference = relative_score if anchor.kind == RELATIVE else anchor.value
    if anchor.kind == NONE or reference is None:
        return None, "no anchor"
    counted = {**accuracy, "benchmark_args": list(benchmark.args)}
    samples, source = resolve_samples(counted, None, benchmark.name, output)
    policy = accuracy_policy.resolve_floor_policy(accuracy)
    # The binomial error is of a percentage: a score reported as a fraction is scaled.
    scale = 100.0 if 0 <= reference <= 1 else 1.0
    found = accuracy_policy.floor_for(policy, reference * scale, samples, anchor.kind == RELATIVE)
    if found is None or scale == 1.0:
        return found, source
    from dataclasses import replace

    return (
        replace(
            found,
            value=found.value / scale,
            reference=reference,
            se=None if found.se is None else found.se / scale,
        ),
        source,
    )


__all__ = [
    "ANCHORS_FILENAME",
    "DECLARED",
    "NONE",
    "REFERENCE",
    "RELATIVE",
    "Anchor",
    "differences",
    "floor",
    "freeze",
    "frozen",
    "load",
    "resolve",
]
