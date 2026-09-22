# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parse and apply canonical ``LlmArgs`` command-line overrides."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel

_PATH_COMPONENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_MAX_VALUE_DEPTH = 64
_MAX_VALUE_NODES = 10_000


class ConfigOverride(NamedTuple):
    """One parsed assignment to a canonical configuration path."""

    path: tuple[str, ...]
    value: Any


class ConfigOverrideError(ValueError):
    """A safe-to-display error in a configuration override."""


_RESERVED_PATH_REASONS = {
    ("model",): "use the MODEL positional argument",
    ("backend",): "use --backend",
    ("telemetry_config",): "use --telemetry or --no-telemetry",
    ("env_overrides",): "set environment overrides in the YAML config",
    ("internal_request_auth_key",): "keep authentication keys in the YAML config",
    ("allow_request_chat_template",): "use --allow_request_chat_template",
    ("disagg_cluster",): "use --disagg_cluster_uri or the YAML config",
    ("batched_logits_processor",): "use the Python LLM API for live objects",
    ("checkpoint_loader",): "use the Python LLM API for live objects",
    ("mpi_session",): "use the Python LLM API for live objects",
    ("ray_placement_config", "placement_groups"): "use the Python LLM API for live objects",
    ("speculative_config", "drafter"): "use the Python LLM API for live objects",
    ("speculative_config", "resource_manager"): "use the Python LLM API for live objects",
}


def _format_path(path: Sequence[str]) -> str:
    return ".".join(path)


def _raise_reserved_path(path: Sequence[str], reason: str) -> None:
    raise ConfigOverrideError(
        f"Configuration path '{_format_path(path)}' is not supported by --set; {reason}."
    )


def _validate_path(path_text: str) -> tuple[str, ...]:
    if not path_text:
        raise ConfigOverrideError("The override path must not be empty.")

    path = tuple(path_text.split("."))
    if any(not _PATH_COMPONENT_RE.fullmatch(component) for component in path):
        raise ConfigOverrideError(
            "Override paths must use dot-separated canonical snake_case "
            "field names without empty components."
        )
    return path


def _validate_path_policy(path: tuple[str, ...], allowed_roots: Collection[str]) -> None:
    for reserved_path, reason in _RESERVED_PATH_REASONS.items():
        if path[: len(reserved_path)] == reserved_path:
            _raise_reserved_path(path, reason)

    if path[0] not in allowed_roots:
        raise ConfigOverrideError(
            f"Configuration path '{_format_path(path)}' is not a supported "
            "LlmArgs field for the selected backend."
        )


def _validate_embedded_reserved_paths(path: tuple[str, ...], value: Any) -> None:
    """Reject reserved descendants supplied through an ancestor mapping."""
    for reserved_path, reason in _RESERVED_PATH_REASONS.items():
        if len(path) >= len(reserved_path) or reserved_path[: len(path)] != path:
            continue

        nested_value = value
        for component in reserved_path[len(path) :]:
            if not isinstance(nested_value, dict) or component not in nested_value:
                break
            nested_value = nested_value[component]
        else:
            _raise_reserved_path(reserved_path, reason)


def _validate_json_like_value(
    value: Any, active_ids: set[int], depth: int, node_count: list[int]
) -> None:
    node_count[0] += 1
    if node_count[0] > _MAX_VALUE_NODES:
        raise ConfigOverrideError("The override value exceeds the supported size limit.")
    if depth > _MAX_VALUE_DEPTH:
        raise ConfigOverrideError("The override value exceeds the supported nesting depth.")

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigOverrideError("Override values must not contain non-finite numbers.")
        return

    if isinstance(value, list):
        value_id = id(value)
        if value_id in active_ids:
            raise ConfigOverrideError("Override values must not contain recursive YAML aliases.")
        active_ids.add(value_id)
        try:
            for item in value:
                _validate_json_like_value(item, active_ids, depth + 1, node_count)
        finally:
            active_ids.remove(value_id)
        return

    if isinstance(value, dict):
        value_id = id(value)
        if value_id in active_ids:
            raise ConfigOverrideError("Override values must not contain recursive YAML aliases.")
        active_ids.add(value_id)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ConfigOverrideError("Override mappings must use string keys.")
                _validate_json_like_value(item, active_ids, depth + 1, node_count)
        finally:
            active_ids.remove(value_id)
        return

    raise ConfigOverrideError(
        "Override values must be null, booleans, finite numbers, strings, "
        "lists, or string-keyed mappings."
    )


def _parse_value(value_text: str) -> Any:
    if not value_text.strip():
        raise ConfigOverrideError("The override value must not be empty; use null explicitly.")
    try:
        value = yaml.safe_load(value_text)
    except (yaml.YAMLError, RecursionError, ValueError, OverflowError):
        raise ConfigOverrideError("The override value is not valid YAML.") from None

    _validate_json_like_value(value, set(), 0, [0])
    return value


def _paths_conflict(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    shorter_length = min(len(first), len(second))
    return first[:shorter_length] == second[:shorter_length] and first != second


def parse_config_overrides(
    assignments: Sequence[str], *, allowed_roots: Collection[str]
) -> tuple[ConfigOverride, ...]:
    """Parse repeatable ``PATH=YAML_VALUE`` assignments.

    Exact duplicate paths use the last value. Assignments to an ancestor and
    descendant path in the same invocation are rejected because their result
    would otherwise depend on ordering.
    """
    parsed_by_path: dict[tuple[str, ...], ConfigOverride] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ConfigOverrideError("Each override must use PATH=YAML_VALUE syntax.")

        path_text, value_text = assignment.split("=", 1)
        path = _validate_path(path_text)
        _validate_path_policy(path, allowed_roots)

        for previous_path in parsed_by_path:
            if _paths_conflict(path, previous_path):
                raise ConfigOverrideError(
                    "Override paths "
                    f"'{_format_path(previous_path)}' and "
                    f"'{_format_path(path)}' cannot be used together because "
                    "one is an ancestor of the other."
                )

        value = _parse_value(value_text)
        _validate_embedded_reserved_paths(path, value)
        parsed_by_path[path] = ConfigOverride(path=path, value=value)

    return tuple(parsed_by_path.values())


def _as_mutable_mapping(value: Any, path: Sequence[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", exclude_unset=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise ConfigOverrideError(
        f"Cannot set a child of configuration path '{_format_path(path)}' "
        "because its effective value is not a mapping or null."
    )


def apply_config_overrides(
    config: Mapping[str, Any], overrides: Sequence[ConfigOverride]
) -> dict[str, Any]:
    """Return ``config`` with assignments applied at their exact paths."""
    result = dict(config)
    for override in overrides:
        target = result
        for index, component in enumerate(override.path[:-1]):
            existing = target.get(component)
            child = _as_mutable_mapping(existing, override.path[: index + 1])
            target[component] = child
            target = child
        target[override.path[-1]] = copy.deepcopy(override.value)
    return result
