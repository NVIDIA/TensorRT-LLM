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

import pytest
from pydantic import BaseModel

from tensorrt_llm.commands._config_overrides import (
    ConfigOverride,
    ConfigOverrideError,
    apply_config_overrides,
    parse_config_overrides,
)

pytestmark = pytest.mark.cpu_only

_ALLOWED_ROOTS = {
    "batched_logits_processor",
    "checkpoint_loader",
    "dtype",
    "kv_cache_config",
    "max_batch_size",
    "model_kwargs",
    "mpi_session",
    "ray_placement_config",
    "scheduler_config",
    "speculative_config",
    "telemetry_configuration",
}


def _parse(*assignments: str) -> tuple[ConfigOverride, ...]:
    return parse_config_overrides(assignments, allowed_roots=_ALLOWED_ROOTS)


@pytest.mark.parametrize(
    ("assignment", "expected"),
    [
        ("max_batch_size=16", 16),
        ("dtype=null", None),
        ("scheduler_config={capacity: 4}", {"capacity": 4}),
        ("kv_cache_config.batch_sizes=[1, 2, 4]", [1, 2, 4]),
        ("model_kwargs.enabled=true", True),
        ("dtype='value=with=equals'", "value=with=equals"),
    ],
)
def test_parse_config_overrides_supports_yaml_values(assignment: str, expected: object) -> None:
    assert _parse(assignment)[0].value == expected


@pytest.mark.parametrize(
    "assignment",
    [
        "max_batch_size",
        "=1",
        "max_batch_size=",
        "scheduler_config..capacity=1",
        "SchedulerConfig.capacity=1",
        "scheduler-config.capacity=1",
    ],
)
def test_parse_config_overrides_rejects_malformed_assignments(assignment: str) -> None:
    with pytest.raises(ConfigOverrideError):
        _parse(assignment)


@pytest.mark.parametrize(
    "assignment",
    [
        "dtype=2026-09-22",
        "dtype=!!binary aGVsbG8=",
        "model_kwargs=!!set {secret: null}",
        "model_kwargs={1: value}",
        "dtype=.inf",
        "model_kwargs=&recursive [*recursive]",
    ],
)
def test_parse_config_overrides_rejects_non_json_like_yaml(assignment: str) -> None:
    with pytest.raises(ConfigOverrideError):
        _parse(assignment)


def test_parse_config_overrides_rejects_non_finite_nested_float() -> None:
    with pytest.raises(ConfigOverrideError, match="non-finite"):
        _parse("model_kwargs={value: .nan}")


def test_parse_config_overrides_rejects_oversized_integer() -> None:
    with pytest.raises(ConfigOverrideError, match="valid YAML"):
        _parse(f"max_batch_size={'9' * 10_000}")


def test_parse_config_overrides_exact_duplicate_uses_last_value() -> None:
    overrides = _parse("max_batch_size=8", "max_batch_size=16")
    assert overrides == (ConfigOverride(("max_batch_size",), 16),)


@pytest.mark.parametrize(
    "assignments",
    [
        ("scheduler_config={capacity: 4}", "scheduler_config.capacity=8"),
        ("scheduler_config.capacity=8", "scheduler_config={capacity: 4}"),
    ],
)
def test_parse_config_overrides_rejects_ancestor_collisions(assignments: tuple[str, str]) -> None:
    with pytest.raises(ConfigOverrideError, match="ancestor"):
        _parse(*assignments)


@pytest.mark.parametrize(
    "assignment",
    [
        "model=do-not-print-value",
        "backend=do-not-print-value",
        "telemetry_config.disabled=do-not-print-value",
        "env_overrides.api_key=do-not-print-value",
        "internal_request_auth_key=do-not-print-value",
        "allow_request_chat_template=do-not-print-value",
        "disagg_cluster.cluster_uri=do-not-print-value",
        "batched_logits_processor=do-not-print-value",
        "checkpoint_loader=do-not-print-value",
        "mpi_session=do-not-print-value",
        "ray_placement_config.placement_groups=do-not-print-value",
        "speculative_config.drafter=do-not-print-value",
        "speculative_config.resource_manager=do-not-print-value",
        "ray_placement_config={placement_groups: [do-not-print-value]}",
        "speculative_config={decoding_type: User_Provided, drafter: do-not-print-value}",
        "speculative_config={decoding_type: User_Provided, resource_manager: do-not-print-value}",
    ],
)
def test_parse_config_overrides_rejects_reserved_paths_without_values_in_error(
    assignment: str,
) -> None:
    with pytest.raises(ConfigOverrideError, match="not supported by --set") as raised:
        _parse(assignment)
    assert "do-not-print-value" not in str(raised.value)


def test_reserved_prefix_matching_is_structural() -> None:
    overrides = _parse("telemetry_configuration.enabled=true")
    assert overrides[0].path == ("telemetry_configuration", "enabled")


def test_parse_config_overrides_rejects_unknown_root() -> None:
    with pytest.raises(ConfigOverrideError, match="not a supported LlmArgs field"):
        _parse("unknown_config.enabled=true")


class _NestedConfig(BaseModel):
    first: int = 1
    second: int = 2


def test_apply_config_overrides_preserves_nested_siblings() -> None:
    original = {"scheduler_config": _NestedConfig(first=3, second=4)}
    overrides = (ConfigOverride(("scheduler_config", "first"), 8),)

    result = apply_config_overrides(original, overrides)

    assert result["scheduler_config"] == {"first": 8, "second": 4}
    assert original["scheduler_config"] == _NestedConfig(first=3, second=4)


def test_apply_config_overrides_preserves_unset_defaults() -> None:
    original = {"scheduler_config": _NestedConfig(first=3)}
    overrides = (ConfigOverride(("scheduler_config", "first"), 8),)

    result = apply_config_overrides(original, overrides)
    reconstructed = _NestedConfig(**result["scheduler_config"])

    assert reconstructed == _NestedConfig(first=8)
    assert reconstructed.model_fields_set == {"first"}


def test_apply_config_overrides_replaces_mapping_at_exact_path() -> None:
    original = {"scheduler_config": {"first": 1, "second": 2}}
    overrides = (ConfigOverride(("scheduler_config",), {"first": 8}),)

    result = apply_config_overrides(original, overrides)

    assert result["scheduler_config"] == {"first": 8}


@pytest.mark.parametrize("original", [{}, {"scheduler_config": None}])
def test_apply_config_overrides_creates_missing_or_null_ancestors(
    original: dict[str, object],
) -> None:
    overrides = (ConfigOverride(("scheduler_config", "first"), 8),)

    result = apply_config_overrides(original, overrides)

    assert result["scheduler_config"] == {"first": 8}


@pytest.mark.parametrize("existing", [1, "scalar", [1, 2]])
def test_apply_config_overrides_rejects_non_mapping_ancestor(existing: object) -> None:
    overrides = (ConfigOverride(("scheduler_config", "first"), 8),)
    with pytest.raises(ConfigOverrideError, match="not a mapping or null"):
        apply_config_overrides({"scheduler_config": existing}, overrides)
