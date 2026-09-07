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
"""Authenticated DSpark verification costs and exact ``(G, V)`` selection."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_MAX_EXACT_COMPACT_CELLS_PER_G = 4
_MAX_EXACT_COMPACT_CELLS_TOTAL = 32

__all__ = [
    "ExactSpsDrainGuard",
    "ExactSpsCostRow",
    "ExactSpsCostTable",
    "load_runtime_sps_cost_table",
    "select_exact_sps_candidate",
    "validate_sps_cost_table_payload",
]


@dataclass(frozen=True)
class ExactSpsCostRow:
    """Direct whole-step measurements for one rank-local graph size ``G``."""

    token_counts: Sequence[int]
    step_time_ms: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.token_counts) != len(self.step_time_ms):
            raise ValueError(
                f"token_counts ({len(self.token_counts)}) and step_time_ms "
                f"({len(self.step_time_ms)}) must have the same length"
            )
        if not self.token_counts:
            raise ValueError("ExactSpsCostRow requires at least one measured point")
        token_counts = tuple(
            _require_exact_int(value, field="measured verifier budget", minimum=0)
            for value in self.token_counts
        )
        step_time_ms = tuple(
            _require_positive_finite_number(value, field="SPS step time")
            for value in self.step_time_ms
        )
        if any(b <= a for a, b in zip(token_counts, token_counts[1:])):
            raise ValueError("token_counts must be strictly increasing")
        object.__setattr__(self, "token_counts", token_counts)
        object.__setattr__(self, "step_time_ms", step_time_ms)


@dataclass(frozen=True)
class ExactSpsDrainGuard:
    """Measured policy metadata for conservative group-E2E admission.

    ``mean_output_tokens_per_request_iteration`` must come from a matched
    workload trace, while ``tail_graph_batch_size`` identifies the measured
    native ``T(G, 0)`` tail cell in the same exact cost table.  Keeping both
    values in the authenticated table metadata prevents serving code from
    silently reusing trace-specific constants on another workload.
    """

    loss_multiplier: float
    mean_output_tokens_per_request_iteration: float
    minimum_group_value_ms: float
    tail_graph_batch_size: int
    source_result_sha256: str

    def __post_init__(self) -> None:
        loss_multiplier = _require_positive_finite_number(
            self.loss_multiplier, field="iteration drain loss_multiplier"
        )
        mean_output_tokens = _require_positive_finite_number(
            self.mean_output_tokens_per_request_iteration,
            field="iteration drain mean_output_tokens_per_request_iteration",
        )
        minimum_group_value_ms = _require_nonnegative_finite_number(
            self.minimum_group_value_ms,
            field="iteration drain minimum_group_value_ms",
        )
        tail_graph_batch_size = _require_exact_int(
            self.tail_graph_batch_size,
            field="iteration drain tail_graph_batch_size",
            minimum=1,
        )
        source_result_sha256 = _require_sha256(
            self.source_result_sha256,
            field="iteration drain source_result_sha256",
        )
        object.__setattr__(self, "loss_multiplier", loss_multiplier)
        object.__setattr__(
            self,
            "mean_output_tokens_per_request_iteration",
            mean_output_tokens,
        )
        object.__setattr__(self, "minimum_group_value_ms", minimum_group_value_ms)
        object.__setattr__(self, "tail_graph_batch_size", tail_graph_batch_size)
        object.__setattr__(self, "source_result_sha256", source_result_sha256)

    def identity_payload(self) -> dict[str, object]:
        """Canonical fields included in all-rank table agreement."""
        return {
            "loss_multiplier": self.loss_multiplier,
            "mean_output_tokens_per_request_iteration": (
                self.mean_output_tokens_per_request_iteration
            ),
            "minimum_group_value_ms": self.minimum_group_value_ms,
            "source_result_sha256": self.source_result_sha256,
            "tail_graph_batch_size": self.tail_graph_batch_size,
        }


@dataclass(frozen=True)
class ExactSpsCostTable:
    """Directly measured whole-step costs keyed by exact ``(G, V)``.

    ``G`` is the rank-local padded CUDA-graph batch size and ``V`` is the
    rank-local submitted ragged verifier-token count, including anchor and
    pad-row tokens. ``V=0`` is a sentinel for the native static K5 path: it is
    the mandatory fallback comparator and is never a ragged capture bucket.
    Every positive cell must fit the physical verifier window
    ``G <= V <= G * (max_draft_len + 1)``.
    No interpolation is permitted on either axis: an unmeasured graph shape
    is a configuration error rather than an estimate.
    """

    tables: dict[int, ExactSpsCostRow]
    max_draft_len: int
    minimum_predicted_gain: float = 0.01
    iteration_drain_guard: Optional[ExactSpsDrainGuard] = None
    identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        max_draft_len = _require_exact_int(self.max_draft_len, field="max_draft_len", minimum=1)
        normalized: dict[int, ExactSpsCostRow] = {}
        for graph_batch_size, table in self.tables.items():
            canonical_graph_batch_size = _require_exact_int(
                graph_batch_size, field="graph batch size", minimum=1
            )
            if canonical_graph_batch_size in normalized:
                raise ValueError("ExactSpsCostTable contains duplicate canonical graph batch sizes")
            normalized[canonical_graph_batch_size] = table
        if not normalized:
            raise ValueError("ExactSpsCostTable requires at least one graph batch size")
        if any(not isinstance(table, ExactSpsCostRow) for table in normalized.values()):
            raise TypeError("ExactSpsCostTable values must be ExactSpsCostRow instances")
        for graph_batch_size, table in normalized.items():
            budgets = tuple(
                _require_exact_int(value, field="measured verifier budget", minimum=0)
                for value in table.token_counts
            )
            if budgets[0] != 0:
                raise ValueError(
                    "Every exact SPS G requires V=0 native static K5 "
                    f"fallback; missing for G={graph_batch_size}"
                )
            max_verifier_budget = graph_batch_size * (max_draft_len + 1)
            invalid_positive_budgets = [
                verifier_budget
                for verifier_budget in budgets[1:]
                if (verifier_budget < graph_batch_size or verifier_budget > max_verifier_budget)
            ]
            if invalid_positive_budgets:
                raise ValueError(
                    "Exact SPS positive verifier budgets must satisfy "
                    "G <= V <= G*(K+1); "
                    f"G={graph_batch_size}, K={max_draft_len}, "
                    f"invalid V={invalid_positive_budgets}"
                )
            production_budgets = [
                verifier_budget
                for verifier_budget in budgets[1:]
                if verifier_budget < max_verifier_budget
            ]
            if len(production_budgets) > _MAX_EXACT_COMPACT_CELLS_PER_G:
                raise ValueError(
                    "Exact SPS graph budget exceeds the production limit of "
                    f"{_MAX_EXACT_COMPACT_CELLS_PER_G} compact V cells per G; "
                    f"G={graph_batch_size} has {len(production_budgets)}"
                )
        production_cell_count = sum(
            len(
                [
                    verifier_budget
                    for verifier_budget in table.token_counts[1:]
                    if verifier_budget < graph_batch_size * (max_draft_len + 1)
                ]
            )
            for graph_batch_size, table in normalized.items()
        )
        if production_cell_count > _MAX_EXACT_COMPACT_CELLS_TOTAL:
            raise ValueError(
                "Exact SPS graph budget exceeds the production limit of "
                f"{_MAX_EXACT_COMPACT_CELLS_TOTAL} compact (G,V) cells; "
                f"artifact has {production_cell_count}"
            )
        object.__setattr__(self, "tables", normalized)
        object.__setattr__(self, "max_draft_len", max_draft_len)
        minimum_predicted_gain = _require_nonnegative_finite_number(
            self.minimum_predicted_gain, field="minimum_predicted_gain"
        )
        object.__setattr__(self, "minimum_predicted_gain", minimum_predicted_gain)
        iteration_drain_guard = self.iteration_drain_guard
        if iteration_drain_guard is not None:
            if not isinstance(iteration_drain_guard, ExactSpsDrainGuard):
                raise TypeError("iteration_drain_guard must be an ExactSpsDrainGuard or None")
            if iteration_drain_guard.tail_graph_batch_size not in normalized:
                raise ValueError(
                    "Iteration drain guard requires a measured native tail table for "
                    f"G={iteration_drain_guard.tail_graph_batch_size}"
                )
        # This is the rank-agreement identity used on every exact scheduling
        # step. It covers the canonical ordered grid, every measured cost, K,
        # and the minimum-gain policy. Two artifacts that merely have the same
        # number of cells must not be allowed to index each other's yields.
        identity_payload = {
            "schema": "exact-sps-runtime-v1",
            "max_draft_len": max_draft_len,
            "minimum_predicted_gain": minimum_predicted_gain,
            "iteration_drain_guard": (
                iteration_drain_guard.identity_payload()
                if iteration_drain_guard is not None
                else None
            ),
            "tables": [
                {
                    "graph_batch_size": graph_batch_size,
                    "token_counts": [int(value) for value in table.token_counts],
                    "step_time_ms": [float(value) for value in table.step_time_ms],
                }
                for graph_batch_size, table in sorted(normalized.items())
            ],
        }
        object.__setattr__(self, "identity_sha256", _canonical_json_sha256(identity_payload))

    @property
    def collective_identity_words(self) -> tuple[int, ...]:
        """Full SHA256 as eight transport-safe unsigned 32-bit integers."""
        return tuple(
            int(self.identity_sha256[offset : offset + 8], 16)
            for offset in range(0, len(self.identity_sha256), 8)
        )

    def for_graph_batch_size(self, num_requests: int) -> ExactSpsCostRow:
        """Return the directly measured V cells for one exact G."""
        graph_batch_size = _require_exact_int(num_requests, field="graph batch size", minimum=1)
        try:
            return self.tables[graph_batch_size]
        except KeyError as error:
            raise ValueError(
                f"SPS cost artifact has no direct measurements for G={graph_batch_size}"
            ) from error

    def step_times(self, num_tokens: np.ndarray, num_requests: int) -> np.ndarray:
        """Return direct whole-step measurements without interpolation."""
        table = self.for_graph_batch_size(num_requests)
        measured = {
            _require_exact_int(token_count, field="measured verifier budget", minimum=0): float(
                step_time
            )
            for token_count, step_time in zip(table.token_counts, table.step_time_ms)
        }
        requested = np.asarray(num_tokens)
        requested_budgets = [
            _require_exact_int(value, field="requested verifier budget", minimum=0)
            for value in requested.reshape(-1)
        ]
        missing = sorted(set(requested_budgets) - measured.keys())
        if missing:
            raise ValueError(
                f"SPS cost artifact has no direct measurements for G={num_requests}, V={missing}"
            )
        return np.asarray(
            [measured[value] for value in requested_budgets],
            dtype=np.float64,
        ).reshape(requested.shape)

    def step_time(self, num_tokens: int, num_requests: int) -> float:
        """Return one directly measured ``T(G, V)`` value."""
        return float(self.step_times(np.asarray([num_tokens]), num_requests)[0])

    def candidate_budgets(
        self, num_requests: int, *, include_native: bool = False
    ) -> tuple[int, ...]:
        """Return exact per-G V candidates, excluding native by default."""
        budgets = tuple(
            _require_exact_int(value, field="measured verifier budget", minimum=0)
            for value in self.for_graph_batch_size(num_requests).token_counts
        )
        if include_native:
            return budgets
        return tuple(value for value in budgets if value != 0)

    def candidate_cells(self) -> tuple[tuple[int, int], ...]:
        """Stable positive ``(G,V)`` ordering for collective payloads."""
        return tuple(
            (graph_batch_size, verifier_budget)
            for graph_batch_size in sorted(self.tables)
            for verifier_budget in self.production_candidate_budgets(graph_batch_size)
        )

    def production_candidate_budgets(self, num_requests: int) -> tuple[int, ...]:
        """Compact serving cells, excluding native and full-token controls."""
        graph_batch_size = _require_exact_int(num_requests, field="graph batch size", minimum=1)
        full_budget = graph_batch_size * (self.max_draft_len + 1)
        return tuple(
            verifier_budget
            for verifier_budget in self.candidate_budgets(graph_batch_size)
            if verifier_budget < full_budget
        )


def _read_sps_cost_payload(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("SPS cost artifact must contain a JSON object")
    return payload


def load_runtime_sps_cost_table(
    path: str | Path,
    *,
    graph_batch_sizes: Sequence[int],
    max_draft_len: int,
    live_engine_fingerprint_path: str | Path | None = None,
) -> tuple[ExactSpsCostTable, dict[str, object]]:
    """Load an authenticated schema-v2 exact ``T(G,V)`` table."""
    payload = _read_sps_cost_payload(path)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("Exact SPS cost artifacts require schema_version=2")
    if live_engine_fingerprint_path is None:
        raise ValueError(
            "Schema-v2 exact SPS artifacts require an independently generated "
            "live engine fingerprint path"
        )
    live_engine_fingerprint = _read_sps_cost_payload(live_engine_fingerprint_path)
    validate_sps_cost_table_payload(
        payload,
        graph_batch_sizes=graph_batch_sizes,
        max_draft_len=max_draft_len,
        live_engine_fingerprint=live_engine_fingerprint,
    )
    return _build_exact_sps_cost_table(payload, max_draft_len=max_draft_len), payload


def _build_exact_sps_cost_table(
    payload: dict[str, object],
    *,
    max_draft_len: int,
) -> ExactSpsCostTable:
    exact_tables = payload["cost_tables"]
    assert isinstance(exact_tables, dict)
    parsed_tables: dict[int, ExactSpsCostRow] = {}
    for graph_batch_size, exact_payload in exact_tables.items():
        canonical_graph_batch_size = _parse_graph_batch_size_key(graph_batch_size)
        assert isinstance(exact_payload, dict)
        parsed_tables[canonical_graph_batch_size] = ExactSpsCostRow(
            token_counts=tuple(
                _require_json_int(value, field="rank-local verifier budget", minimum=0)
                for value in exact_payload["token_counts"]
            ),
            step_time_ms=tuple(
                _require_positive_finite_number(value, field="SPS step time")
                for value in exact_payload["step_time_ms"]
            ),
        )
    drain_guard_payload = payload.get("iteration_drain_guard")
    iteration_drain_guard = None
    if drain_guard_payload is not None:
        assert isinstance(drain_guard_payload, dict)
        iteration_drain_guard = ExactSpsDrainGuard(
            loss_multiplier=_require_positive_finite_number(
                drain_guard_payload["loss_multiplier"],
                field="iteration drain loss_multiplier",
            ),
            mean_output_tokens_per_request_iteration=_require_positive_finite_number(
                drain_guard_payload["mean_output_tokens_per_request_iteration"],
                field="iteration drain mean_output_tokens_per_request_iteration",
            ),
            minimum_group_value_ms=_require_nonnegative_finite_number(
                drain_guard_payload["minimum_group_value_ms"],
                field="iteration drain minimum_group_value_ms",
            ),
            tail_graph_batch_size=_require_json_int(
                drain_guard_payload["tail_graph_batch_size"],
                field="iteration drain tail_graph_batch_size",
                minimum=1,
            ),
            source_result_sha256=_require_sha256(
                drain_guard_payload["source_result_sha256"],
                field="iteration drain source_result_sha256",
            ),
        )
    return ExactSpsCostTable(
        tables=parsed_tables,
        max_draft_len=max_draft_len,
        minimum_predicted_gain=_require_nonnegative_finite_number(
            payload["minimum_predicted_gain"], field="minimum_predicted_gain"
        ),
        iteration_drain_guard=iteration_drain_guard,
    )


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


_V2_TOP_LEVEL_FIELDS = {
    "cost_tables",
    "engine_fingerprint",
    "engine_fingerprint_sha256",
    "measurements",
    "minimum_predicted_gain",
    "schema_version",
}
_V2_OPTIONAL_TOP_LEVEL_FIELDS = {"iteration_drain_guard"}
_V2_TABLE_FIELDS = {"step_time_ms", "token_counts"}
_V2_DRAIN_GUARD_FIELDS = {
    "loss_multiplier",
    "mean_output_tokens_per_request_iteration",
    "minimum_group_value_ms",
    "source_result_sha256",
    "tail_graph_batch_size",
}
_V2_MEASUREMENT_FIELDS = {
    "rank_local_graph_batch_size",
    "rank_local_verifier_budget",
    "source_result_sha256",
    "step_time_ms",
}
_V2_FINGERPRINT_FIELDS = {
    "global_graph_batch_sizes",
    "gpu",
    "gpu_count",
    "gpu_snapshot_sha256",
    "max_draft_len",
    "rank_local_graph_batch_sizes",
    "runtime_snapshot",
    "source_diff_sha256",
    "source_head",
    "topology",
}


def _validate_exact_fields(
    value: object,
    *,
    name: str,
    fields: set[str],
    optional_fields: Optional[set[str]] = None,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"{name} field names must be strings")
    missing = sorted(fields - value.keys())
    if missing:
        raise ValueError(f"{name} is missing required fields: " + ", ".join(missing))
    unknown = sorted(value.keys() - fields - (optional_fields or set()))
    if unknown:
        raise ValueError(f"{name} has unknown fields: " + ", ".join(unknown))
    null_fields = sorted(key for key, item in value.items() if item is None)
    if null_fields:
        raise ValueError(f"{name} has null fields: " + ", ".join(null_fields))
    return value


def _validate_v2_shape(payload: dict[str, object]) -> None:
    _validate_exact_fields(
        payload,
        name="schema-v2 SPS artifact",
        fields=_V2_TOP_LEVEL_FIELDS,
        optional_fields=_V2_OPTIONAL_TOP_LEVEL_FIELDS,
    )
    exact_tables = payload["cost_tables"]
    if not isinstance(exact_tables, dict) or not exact_tables:
        raise TypeError("SPS cost_tables must be a non-empty object keyed by graph batch size")
    for graph_batch_size, table_payload in exact_tables.items():
        _parse_graph_batch_size_key(graph_batch_size)
        table_payload = _validate_exact_fields(
            table_payload, name=f"SPS cost table for G={graph_batch_size}", fields=_V2_TABLE_FIELDS
        )
        token_counts = table_payload["token_counts"]
        step_time_ms = table_payload["step_time_ms"]
        if not isinstance(token_counts, list) or not token_counts:
            raise TypeError(
                f"SPS cost table for G={graph_batch_size} token_counts must be a non-empty list"
            )
        if not isinstance(step_time_ms, list) or not step_time_ms:
            raise TypeError(
                f"SPS cost table for G={graph_batch_size} step_time_ms must be a non-empty list"
            )
        if len(token_counts) != len(step_time_ms):
            raise ValueError(f"SPS cost table for G={graph_batch_size} has mismatched cells")
        for value in token_counts:
            _require_json_int(value, field="rank-local verifier budget", minimum=0)
        for value in step_time_ms:
            _require_positive_finite_number(value, field="SPS step time")
    fingerprint = _validate_exact_fields(
        payload["engine_fingerprint"],
        name="SPS engine_fingerprint",
        fields=_V2_FINGERPRINT_FIELDS,
    )
    for key in ("gpu", "runtime_snapshot", "source_head", "topology"):
        if not isinstance(fingerprint[key], str) or not fingerprint[key]:
            raise TypeError(f"SPS engine_fingerprint {key} must be a non-empty string")
    for key in ("gpu_snapshot_sha256", "source_diff_sha256"):
        _require_sha256(fingerprint[key], field=f"engine_fingerprint {key}")
    _require_json_int(fingerprint["gpu_count"], field="engine_fingerprint gpu_count", minimum=1)
    _require_json_int(
        fingerprint["max_draft_len"], field="engine_fingerprint max_draft_len", minimum=1
    )
    for key in ("rank_local_graph_batch_sizes", "global_graph_batch_sizes"):
        values = fingerprint[key]
        if not isinstance(values, list) or not values:
            raise TypeError(f"engine_fingerprint {key} must be a non-empty list")
        for value in values:
            _require_json_int(value, field=f"engine_fingerprint {key}", minimum=1)
        if len(set(values)) != len(values):
            raise ValueError(f"engine_fingerprint {key} must not contain duplicates")
    _require_sha256(payload["engine_fingerprint_sha256"], field="engine_fingerprint_sha256")
    _require_nonnegative_finite_number(
        payload["minimum_predicted_gain"], field="minimum_predicted_gain"
    )
    drain_guard = payload.get("iteration_drain_guard")
    if drain_guard is not None:
        drain_guard = _validate_exact_fields(
            drain_guard,
            name="SPS iteration_drain_guard",
            fields=_V2_DRAIN_GUARD_FIELDS,
        )
        _require_positive_finite_number(
            drain_guard["loss_multiplier"],
            field="iteration drain loss_multiplier",
        )
        _require_positive_finite_number(
            drain_guard["mean_output_tokens_per_request_iteration"],
            field="iteration drain mean_output_tokens_per_request_iteration",
        )
        _require_nonnegative_finite_number(
            drain_guard["minimum_group_value_ms"],
            field="iteration drain minimum_group_value_ms",
        )
        _require_json_int(
            drain_guard["tail_graph_batch_size"],
            field="iteration drain tail_graph_batch_size",
            minimum=1,
        )
        _require_sha256(
            drain_guard["source_result_sha256"],
            field="iteration drain source_result_sha256",
        )
    measurements = payload["measurements"]
    if not isinstance(measurements, list) or not measurements:
        raise TypeError("SPS measurements must be a non-empty list")
    for index, measurement in enumerate(measurements):
        measurement = _validate_exact_fields(
            measurement,
            name=f"SPS measurement {index}",
            fields=_V2_MEASUREMENT_FIELDS,
        )
        _require_json_int(
            measurement["rank_local_graph_batch_size"],
            field="measurement graph batch size",
            minimum=1,
        )
        _require_json_int(
            measurement["rank_local_verifier_budget"],
            field="measurement verifier budget",
            minimum=0,
        )
        _require_positive_finite_number(measurement["step_time_ms"], field="measurement step time")
        _require_sha256(
            measurement["source_result_sha256"], field="measurement source_result_sha256"
        )


def _parse_graph_batch_size_key(value: object) -> int:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or value.startswith("0")
    ):
        raise TypeError(
            f"SPS graph batch size key {value!r} must be a canonical positive integer string"
        )
    parsed = int(value)
    if parsed < 1 or str(parsed) != value:
        raise ValueError(f"SPS graph batch size key {value!r} must be positive and canonical")
    return parsed


def _require_json_int(value: object, *, field: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a JSON integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def _require_exact_int(value: object, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {parsed}")
    return parsed


def _require_positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a JSON number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be positive and finite, got {value!r}")
    return parsed


def _require_nonnegative_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a JSON number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be non-negative and finite, got {value!r}")
    return parsed


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def validate_sps_cost_table_payload(
    payload: dict[str, object],
    *,
    graph_batch_sizes: Sequence[int],
    max_draft_len: int,
    live_engine_fingerprint: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Validate exact schema-v2 coverage and engine provenance.

    Schema-v2 candidate budgets come directly from each per-G table. ``V=0``
    is required for every G and denotes the measured native static K5 path;
    positive V cells are arbitrary ragged candidates and need not be uniform
    length multiples. Other schemas are rejected.

    Args:
        payload: Parsed schema-v2 SPS artifact.
        graph_batch_sizes: Rank-local CUDA-graph row counts captured by the
            active engine.
        max_draft_len: Physical full draft-block length K.
        live_engine_fingerprint: Independently supplied runtime fingerprint.
            Validation refuses to use an artifact without this comparison.

    Returns:
        The artifact's authenticated engine fingerprint.

    Raises:
        TypeError: If the artifact has malformed container types.
        ValueError: If its schema, fingerprint, coverage, or provenance does
            not exactly describe the configured graph ladder.
    """
    if not isinstance(payload, dict):
        raise TypeError("SPS cost artifact must contain a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("Exact SPS cost artifacts require schema_version=2")
    _validate_v2_shape(payload)
    exact_tables = payload["cost_tables"]
    fingerprint = payload["engine_fingerprint"]
    assert isinstance(exact_tables, dict)
    assert isinstance(fingerprint, dict)

    if live_engine_fingerprint is None:
        raise ValueError(
            "Schema-v2 SPS validation requires an independently supplied live engine fingerprint"
        )
    if live_engine_fingerprint is fingerprint:
        raise ValueError(
            "Live engine fingerprint must be supplied independently of the SPS artifact"
        )
    live_engine_fingerprint = _validate_exact_fields(
        live_engine_fingerprint,
        name="live engine fingerprint",
        fields=_V2_FINGERPRINT_FIELDS,
    )
    _validate_live_fingerprint_values(live_engine_fingerprint)

    if payload["engine_fingerprint_sha256"] != _canonical_json_sha256(fingerprint):
        raise ValueError("SPS engine_fingerprint SHA256 does not match payload")
    mismatches = sorted(
        key for key in _V2_FINGERPRINT_FIELDS if fingerprint[key] != live_engine_fingerprint[key]
    )
    if mismatches:
        raise ValueError(
            "SPS engine_fingerprint does not match active runtime: " + ", ".join(mismatches)
        )

    configured_graph_batch_sizes = {
        _require_json_int(value, field="runtime graph batch size", minimum=1)
        for value in graph_batch_sizes
    }
    if not configured_graph_batch_sizes:
        raise ValueError("graph_batch_sizes must not be empty")
    if len(configured_graph_batch_sizes) != len(graph_batch_sizes):
        raise ValueError("runtime graph_batch_sizes must not contain duplicates")
    measured_tables = {
        _parse_graph_batch_size_key(graph_batch_size): table_payload
        for graph_batch_size, table_payload in exact_tables.items()
    }
    drain_guard = payload.get("iteration_drain_guard")
    if drain_guard is not None:
        assert isinstance(drain_guard, dict)
        tail_graph_batch_size = _require_json_int(
            drain_guard["tail_graph_batch_size"],
            field="iteration drain tail_graph_batch_size",
            minimum=1,
        )
        if tail_graph_batch_size not in measured_tables:
            raise ValueError(
                "Iteration drain guard requires a measured native tail table for "
                f"G={tail_graph_batch_size}"
            )
    fingerprint_graph_batch_sizes = set(fingerprint["rank_local_graph_batch_sizes"])
    if (
        configured_graph_batch_sizes != set(measured_tables)
        or configured_graph_batch_sizes != fingerprint_graph_batch_sizes
    ):
        raise ValueError(
            "Multi-G SPS graph batch sizes must match exactly across runtime "
            "graphs, cost_tables, and engine_fingerprint"
        )
    runtime_max_draft_len = _require_json_int(
        max_draft_len, field="runtime max_draft_len", minimum=1
    )
    if fingerprint["max_draft_len"] != runtime_max_draft_len:
        raise ValueError(
            f"SPS engine_fingerprint max_draft_len does not match runtime K={max_draft_len}"
        )

    table_cells: dict[tuple[int, int], float] = {}
    for graph_batch_size, table_payload in measured_tables.items():
        assert isinstance(table_payload, dict)
        token_counts = [
            _require_json_int(value, field="rank-local verifier budget", minimum=0)
            for value in table_payload["token_counts"]
        ]
        step_times = [
            _require_positive_finite_number(value, field="SPS step time")
            for value in table_payload["step_time_ms"]
        ]
        if len(token_counts) != len(step_times):
            raise ValueError(f"SPS cost table for G={graph_batch_size} has mismatched cells")
        if not token_counts or any(
            right <= left for left, right in zip(token_counts, token_counts[1:])
        ):
            raise ValueError(
                f"SPS cost table for G={graph_batch_size} budgets must be strictly increasing"
            )
        if token_counts[0] != 0:
            raise ValueError(
                "Every schema-v2 G requires V=0 native static K5 fallback; "
                f"missing for G={graph_batch_size}"
            )
        max_verifier_budget = graph_batch_size * (runtime_max_draft_len + 1)
        invalid_positive_budgets = [
            verifier_tokens
            for verifier_tokens in token_counts[1:]
            if (verifier_tokens < graph_batch_size or verifier_tokens > max_verifier_budget)
        ]
        if invalid_positive_budgets:
            raise ValueError(
                "Schema-v2 positive verifier budgets must satisfy "
                "G <= V <= G*(K+1); "
                f"G={graph_batch_size}, K={runtime_max_draft_len}, "
                f"invalid V={invalid_positive_budgets}"
            )
        for verifier_tokens, step_time in zip(token_counts, step_times):
            table_cells[(graph_batch_size, verifier_tokens)] = step_time

    measurements = payload["measurements"]
    assert isinstance(measurements, list)
    measurement_cells: dict[tuple[int, int], float] = {}
    for item in measurements:
        assert isinstance(item, dict)
        pair = (
            _require_json_int(
                item["rank_local_graph_batch_size"], field="measurement graph batch size", minimum=1
            ),
            _require_json_int(
                item["rank_local_verifier_budget"], field="measurement verifier budget", minimum=0
            ),
        )
        if pair in measurement_cells:
            raise ValueError(f"duplicate SPS measurement provenance for {pair}")
        measurement_cells[pair] = _require_positive_finite_number(
            item["step_time_ms"], field="measurement step time"
        )
    if set(measurement_cells) != set(table_cells):
        missing = sorted(set(table_cells) - set(measurement_cells))
        extras = sorted(set(measurement_cells) - set(table_cells))
        raise ValueError(
            "SPS cost table and measurement provenance cells must match exactly; "
            f"missing={missing}, extras={extras}"
        )
    mismatched_cells = sorted(
        pair for pair, step_time in table_cells.items() if measurement_cells[pair] != step_time
    )
    if mismatched_cells:
        raise ValueError(
            f"SPS cost table cells do not match measurement provenance: {mismatched_cells}"
        )

    gpu_count = fingerprint["gpu_count"]
    global_graph_batch_sizes = set(fingerprint["global_graph_batch_sizes"])
    if global_graph_batch_sizes != {
        graph_batch_size * gpu_count for graph_batch_size in configured_graph_batch_sizes
    }:
        raise ValueError(
            "SPS engine_fingerprint global_graph_batch_sizes are inconsistent "
            "with rank-local G values and gpu_count"
        )
    return fingerprint


def _validate_live_fingerprint_values(fingerprint: dict[str, object]) -> None:
    for key in ("gpu", "runtime_snapshot", "source_head", "topology"):
        if not isinstance(fingerprint[key], str) or not fingerprint[key]:
            raise TypeError(f"Live engine fingerprint {key} must be a non-empty string")
    for key in ("gpu_snapshot_sha256", "source_diff_sha256"):
        _require_sha256(fingerprint[key], field=f"live engine fingerprint {key}")
    _require_json_int(
        fingerprint["gpu_count"], field="live engine fingerprint gpu_count", minimum=1
    )
    _require_json_int(
        fingerprint["max_draft_len"], field="live engine fingerprint max_draft_len", minimum=1
    )
    for key in ("rank_local_graph_batch_sizes", "global_graph_batch_sizes"):
        values = fingerprint[key]
        if not isinstance(values, list) or not values:
            raise TypeError(f"live engine fingerprint {key} must be a non-empty list")
        for value in values:
            _require_json_int(value, field=f"live engine fingerprint {key}", minimum=1)
        if len(set(values)) != len(values):
            raise ValueError(f"live engine fingerprint {key} must not contain duplicates")


def select_exact_sps_candidate(
    *,
    graph_batch_size: int,
    native_expected_yield: float,
    compact_expected_yields: dict[int, float],
    compact_max_yield_losses_per_request: dict[int, float],
    cost_table: ExactSpsCostTable,
) -> int:
    """Choose a measured V, retaining native V=0 on weak group-E2E value.

    The caller supplies globally agreed expected yields after the phase-2
    allgather. This pure layer only compares exact measured cells; it cannot
    create a ragged budget or synchronize ranks. Compact serving fails closed
    until the exact table carries matched iteration/drain metadata. Every
    measured tier that clears the aggregate immediate-goodput guard is reranked
    by its guarded group-E2E value. Drain loss is the maximum expected loss per
    real request on any active attention-DP rank, derived from the same
    allgather as the aggregate yields. A selected tier must satisfy, strictly::

        T(G, 0) - T(G, V)
        - loss_multiplier * max_rank((Y_native - Y_V) / N_real)
          * T(tail_G, 0) / mean_output_tokens_per_request_iteration
        > minimum_group_value_ms
    """
    graph_batch_size = _require_exact_int(graph_batch_size, field="graph_batch_size", minimum=1)
    measured_budgets = cost_table.production_candidate_budgets(graph_batch_size)
    native_budgets = cost_table.candidate_budgets(graph_batch_size, include_native=True)
    if not native_budgets or native_budgets[0] != 0:
        raise ValueError(f"Exact SPS G={graph_batch_size} has no V=0 native fallback")
    if any(type(value) is not int for value in compact_expected_yields):
        raise TypeError("compact_expected_yields keys must be integer V values")
    if 0 in compact_expected_yields:
        raise ValueError("compact_expected_yields must not include native V=0")
    if set(compact_expected_yields) != set(measured_budgets):
        raise ValueError(
            f"compact_expected_yields must cover every positive measured V for G={graph_batch_size}"
        )
    if any(type(value) is not int for value in compact_max_yield_losses_per_request):
        raise TypeError("compact_max_yield_losses_per_request keys must be integer V values")
    if 0 in compact_max_yield_losses_per_request:
        raise ValueError("compact_max_yield_losses_per_request must not include native V=0")
    if set(compact_max_yield_losses_per_request) != set(measured_budgets):
        raise ValueError(
            "compact_max_yield_losses_per_request must cover every positive "
            f"measured V for G={graph_batch_size}"
        )
    compact_yields = {
        budget: _require_nonnegative_finite_number(
            compact_expected_yields[budget], field=f"expected yield for compact V={budget}"
        )
        for budget in measured_budgets
    }
    max_yield_losses_per_request = {
        budget: _require_nonnegative_finite_number(
            compact_max_yield_losses_per_request[budget],
            field=(f"maximum expected yield loss per request for compact V={budget}"),
        )
        for budget in measured_budgets
    }
    native_yield = _require_nonnegative_finite_number(
        native_expected_yield, field="native_expected_yield"
    )
    if any(value > native_yield for value in compact_yields.values()):
        raise ValueError("compact expected yield must not exceed native expected yield")
    native_score = native_yield / cost_table.step_time(0, graph_batch_size)
    compact_scores = {
        budget: compact_yields[budget] / cost_table.step_time(budget, graph_batch_size)
        for budget in measured_budgets
    }
    if not compact_scores:
        return 0
    required_score = native_score * (1.0 + cost_table.minimum_predicted_gain)
    goodput_budgets = [
        budget
        for budget in measured_budgets
        if compact_scores[budget] > native_score and compact_scores[budget] >= required_score
    ]
    if not goodput_budgets:
        return 0
    drain_guard = cost_table.iteration_drain_guard
    if drain_guard is None:
        # The matched mean-output statistic is workload-specific and cannot be
        # inferred from T(G,V). An older exact artifact therefore remains a
        # valid native-static control but is not allowed to enable compact V.
        return 0
    native_step_ms = cost_table.step_time(0, graph_batch_size)
    tail_step_ms = cost_table.step_time(0, drain_guard.tail_graph_batch_size)
    group_values_ms = {}
    for budget in goodput_budgets:
        compact_step_ms = cost_table.step_time(budget, graph_batch_size)
        group_value_ms = (
            native_step_ms
            - compact_step_ms
            - drain_guard.loss_multiplier
            * max_yield_losses_per_request[budget]
            * tail_step_ms
            / drain_guard.mean_output_tokens_per_request_iteration
        )
        if group_value_ms > drain_guard.minimum_group_value_ms:
            group_values_ms[budget] = group_value_ms
    if not group_values_ms:
        return 0
    return int(
        max(
            group_values_ms,
            key=lambda budget: (
                group_values_ms[budget],
                compact_scores[budget],
                budget,
            ),
        )
    )
