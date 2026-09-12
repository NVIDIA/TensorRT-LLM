"""Schema validation for staircase-bringup's ``task.yaml`` input.

On top of the three source paths every bring-up needs, a staircase run is
identified by a **target triple** — family, checkpoint, GPU arch, and parallel
topology — which decides where the target is written and how many GPUs the
run claims. Getting that wrong is expensive in a way the gates cannot catch:
every gate is topology-blind, so a `tep4` target actually served at tp1 passes
smoke, passes accuracy, and produces a self-consistently wrong comparison
while the directory name lies. Validating it at the CLI boundary is the cheap
place to catch it.

Validation runs before any agent is constructed, and batches every detected
problem into one message so the user sees all gaps at once.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "reference_code_path",
    "checkpoint_path",
    "trtllm_repo_path",
)

TARGET_FIELD = "target"
TARGET_REQUIRED_KEYS: tuple[str, ...] = ("family", "checkpoint", "arch", "parallel")

OPTIONAL_LIST_FIELDS: tuple[str, ...] = (
    "completion_criteria",
    "implements_tips",
)

ACCURACY_ANCHOR_FIELD = "accuracy_anchor"
ACCURACY_ANCHOR_REQUIRED_KEYS: tuple[str, ...] = ("benchmark", "score", "tol", "source")

SLURM_ENVIRONMENT_FIELD = "slurm-environment"
SLURM_REQUIRED_FIELDS: tuple[str, ...] = ("slurm_partition", "docker_image")

# `tp1`, `tep4`, `dep8` — the trailing integer is the run's world size, hence
# how many GPUs it claims. "single" is deliberately not a spelling: a topology
# segment always carries its rank count.
_PARALLEL_RE = re.compile(r"^(tp|tep|dep)([1-9][0-9]*)$")
_ARCH_RE = re.compile(r"^sm_[0-9]{2,3}$")


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails staircase-bringup schema validation."""


def _validate_target(target: Any, errors: list[str]) -> None:
    if not isinstance(target, dict):
        errors.append(f"'{TARGET_FIELD}' must be a mapping, got {type(target).__name__}")
        return
    for key in TARGET_REQUIRED_KEYS:
        value = target.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{TARGET_FIELD}.{key}' must be a non-empty string")
    arch = target.get("arch")
    if isinstance(arch, str) and not _ARCH_RE.match(arch):
        errors.append(f"'{TARGET_FIELD}.arch' must look like 'sm_100' or 'sm_103', got {arch!r}")
    parallel = target.get("parallel")
    if isinstance(parallel, str) and not _PARALLEL_RE.match(parallel):
        errors.append(
            f"'{TARGET_FIELD}.parallel' must be tp<N>, tep<N>, or dep<N> with N >= 1, "
            f"got {parallel!r}"
        )


def _validate_accuracy_anchor(anchor: Any, errors: list[str]) -> None:
    if not isinstance(anchor, dict):
        errors.append(f"'{ACCURACY_ANCHOR_FIELD}' must be a mapping, got {type(anchor).__name__}")
        return
    for key in ACCURACY_ANCHOR_REQUIRED_KEYS:
        if key not in anchor:
            errors.append(f"'{ACCURACY_ANCHOR_FIELD}.{key}' is required")
    for key in ("score", "tol"):
        value = anchor.get(key)
        if key in anchor and not isinstance(value, (int, float)):
            errors.append(f"'{ACCURACY_ANCHOR_FIELD}.{key}' must be a number, got {value!r}")
    source = anchor.get("source")
    if "source" in anchor and (not isinstance(source, str) or not source.strip()):
        errors.append(
            f"'{ACCURACY_ANCHOR_FIELD}.source' must name where the anchor came from "
            "(an accuracy-suite entry, a model card, or a stock measurement) — "
            "an anchor without a written source is not usable as a release criterion"
        )


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the staircase-bringup schema.

    Returns the parsed mapping with optional list fields normalized to ``[]``
    when absent. Raises :class:`TaskSchemaError` with **every** detected
    problem batched into a single message, so the user fixes them in one pass
    rather than one-by-one.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    for field in REQUIRED_PATH_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")
            continue
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{field}' must be a non-empty string, got {type(value).__name__}")
            continue
        if not Path(value).exists():
            errors.append(f"'{field}' points to a non-existent path: {value}")

    if TARGET_FIELD not in data:
        errors.append(
            f"missing required field '{TARGET_FIELD}' (family, checkpoint, arch, parallel)"
        )
    else:
        _validate_target(data[TARGET_FIELD], errors)

    if ACCURACY_ANCHOR_FIELD in data:
        _validate_accuracy_anchor(data[ACCURACY_ANCHOR_FIELD], errors)

    for field in OPTIONAL_LIST_FIELDS:
        if field not in data or data[field] is None:
            data[field] = []
            continue
        value = data[field]
        if not isinstance(value, list):
            errors.append(f"'{field}' must be a list of strings, got {type(value).__name__}")
            continue
        for i, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(f"'{field}[{i}]' must be a string, got {type(item).__name__}")

    if SLURM_ENVIRONMENT_FIELD in data:
        slurm_environment = data[SLURM_ENVIRONMENT_FIELD]
        if not isinstance(slurm_environment, dict):
            errors.append(
                f"'{SLURM_ENVIRONMENT_FIELD}' must be a mapping, "
                f"got {type(slurm_environment).__name__}"
            )
        else:
            for field in SLURM_REQUIRED_FIELDS:
                if field not in slurm_environment:
                    errors.append(f"'{SLURM_ENVIRONMENT_FIELD}.{field}' is required")
                    continue
                value = slurm_environment[field]
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"'{SLURM_ENVIRONMENT_FIELD}.{field}' must be a "
                        f"non-empty string, got {type(value).__name__}"
                    )

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed staircase-bringup schema validation:{bullet}{bullet.join(errors)}"
        )

    return data


def has_slurm_environment(data: Mapping[str, Any]) -> bool:
    """Return whether a validated task spec requested Slurm guidance."""
    return SLURM_ENVIRONMENT_FIELD in data


def world_size(data: Mapping[str, Any]) -> int:
    """Return the run's world size, read from the parallel segment's trailing int.

    ``tp1`` -> 1, ``tep4`` -> 4, ``dep8`` -> 8. This is how many GPUs the run
    claims, and passing it to a multi-rank preflight is what rejects a device
    claim that does not match the directory the target will be written to.
    """
    match = _PARALLEL_RE.match(data[TARGET_FIELD]["parallel"])
    if match is None:  # pragma: no cover - validation rejects this earlier
        raise TaskSchemaError(f"unparsable parallel segment: {data[TARGET_FIELD]['parallel']!r}")
    return int(match.group(2))


def target_relpath(data: Mapping[str, Any]) -> str:
    """Return the in-tree path the target will be written to, relative to the repo."""
    t = data[TARGET_FIELD]
    return (
        f"tensorrt_llm/_torch/staircase/models/{t['family']}/targets/"
        f"{t['checkpoint']}/{t['arch']}/{t['parallel']}"
    )


__all__ = [
    "ACCURACY_ANCHOR_FIELD",
    "ACCURACY_ANCHOR_REQUIRED_KEYS",
    "OPTIONAL_LIST_FIELDS",
    "REQUIRED_PATH_FIELDS",
    "SLURM_ENVIRONMENT_FIELD",
    "SLURM_REQUIRED_FIELDS",
    "TARGET_FIELD",
    "TARGET_REQUIRED_KEYS",
    "TaskSchemaError",
    "has_slurm_environment",
    "load_and_validate_task_yaml",
    "target_relpath",
    "world_size",
]
