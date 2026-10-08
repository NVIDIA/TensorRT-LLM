"""Schema validation for modeling-bringup's ``task.yaml`` input.

The modeling-bringup workflow requires three path fields up front so the
agents always know where to find the reference HuggingFace implementation,
the validation checkpoint, and the TensorRT-LLM checkout. Two optional list
fields carry pass/fail completion criteria and free-form implementation
tips. Anything else the user puts in the YAML is preserved on disk for the
agents to read. The optional `model_express` block (`enabled`,
`existing_model`) controls the ModelExpress (MX) qualification step and is
validated here, so a typo cannot silently fall back to the default.

Validation runs at the CLI boundary so an invalid spec aborts before any
agent is constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "reference_code_path",
    "checkpoint_path",
    "trtllm_repo_path",
)
OPTIONAL_LIST_FIELDS: tuple[str, ...] = (
    "completion_criteria",
    "implements_tips",
)
SLURM_ENVIRONMENT_FIELD = "slurm-environment"
SLURM_REQUIRED_FIELDS: tuple[str, ...] = (
    "slurm_partition",
    "docker_image",
)
MODEL_EXPRESS_FIELD = "model_express"
MODEL_EXPRESS_ENABLED_FIELD = "enabled"
MODEL_EXPRESS_EXISTING_MODEL_FIELD = "existing_model"
MODEL_EXPRESS_DEFAULTS: dict[str, bool] = {
    MODEL_EXPRESS_ENABLED_FIELD: True,
    MODEL_EXPRESS_EXISTING_MODEL_FIELD: False,
}


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails modeling-bringup schema validation."""


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the modeling-bringup schema.

    Returns the parsed mapping with optional list fields normalized to ``[]``
    when absent and `model_express` resolved against its defaults. Raises
    :class:`TaskSchemaError` with **every** detected problem batched into a
    single message so the user sees all gaps at once instead of fixing them
    one-by-one.
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

    model_express = data.get(MODEL_EXPRESS_FIELD)
    if model_express is None:
        data[MODEL_EXPRESS_FIELD] = dict(MODEL_EXPRESS_DEFAULTS)
    elif not isinstance(model_express, dict):
        errors.append(
            f"'{MODEL_EXPRESS_FIELD}' must be a mapping, got {type(model_express).__name__}"
        )
    else:
        unknown = sorted(map(str, set(model_express) - set(MODEL_EXPRESS_DEFAULTS)))
        if unknown:
            errors.append(f"'{MODEL_EXPRESS_FIELD}' has unknown field(s) {unknown}")
        for field in MODEL_EXPRESS_DEFAULTS:
            if field in model_express and not isinstance(model_express[field], bool):
                errors.append(f"'{MODEL_EXPRESS_FIELD}.{field}' must be a boolean")
        resolved = {**MODEL_EXPRESS_DEFAULTS, **model_express}
        if (
            resolved[MODEL_EXPRESS_EXISTING_MODEL_FIELD] is True
            and resolved[MODEL_EXPRESS_ENABLED_FIELD] is False
        ):
            errors.append(
                f"'{MODEL_EXPRESS_FIELD}.{MODEL_EXPRESS_EXISTING_MODEL_FIELD}' requires "
                f"'{MODEL_EXPRESS_FIELD}.{MODEL_EXPRESS_ENABLED_FIELD}' to be true"
            )
        data[MODEL_EXPRESS_FIELD] = resolved

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed modeling-bringup schema validation:{bullet}{bullet.join(errors)}"
        )

    return data


def has_slurm_environment(data: Mapping[str, Any]) -> bool:
    """Return whether a validated task spec requested Slurm guidance."""
    return SLURM_ENVIRONMENT_FIELD in data


def model_express_enabled(data: Mapping[str, Any]) -> bool:
    """Return whether the task keeps the ModelExpress (MX) qualification step."""
    block = data.get(MODEL_EXPRESS_FIELD)
    if isinstance(block, Mapping):
        return block.get(MODEL_EXPRESS_ENABLED_FIELD, True) is not False
    return True


def model_express_existing_model(data: Mapping[str, Any]) -> bool:
    """Return whether the whole task is MX qualification of an existing model."""
    block = data.get(MODEL_EXPRESS_FIELD)
    if isinstance(block, Mapping):
        return block.get(MODEL_EXPRESS_EXISTING_MODEL_FIELD, False) is True
    return False


__all__ = [
    "REQUIRED_PATH_FIELDS",
    "OPTIONAL_LIST_FIELDS",
    "SLURM_ENVIRONMENT_FIELD",
    "SLURM_REQUIRED_FIELDS",
    "MODEL_EXPRESS_FIELD",
    "MODEL_EXPRESS_ENABLED_FIELD",
    "MODEL_EXPRESS_EXISTING_MODEL_FIELD",
    "MODEL_EXPRESS_DEFAULTS",
    "TaskSchemaError",
    "has_slurm_environment",
    "load_and_validate_task_yaml",
    "model_express_enabled",
    "model_express_existing_model",
]
