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
"""DSpark verification cost tables and exact ``(G, V)`` selection.

A cost table holds measured whole-step decode times per physical draft length
``K``, rank-local CUDA-graph batch size ``G`` and verifier-token budget ``V``.
One file can hold several ``K`` values; the runtime loads the entry matching
``max_draft_len``. ``tensorrt_llm.tools.dspark_profile_costs`` generates it.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tensorrt_llm.logger import logger

_MAX_EXACT_COMPACT_CELLS_PER_G = 4
_MAX_EXACT_COMPACT_CELLS_TOTAL = 32

COST_TABLE_SCHEMA_VERSION = 3
PROFILE_COMMAND_HINT = (
    "python -m tensorrt_llm.tools.dspark_profile_costs --model <checkpoint> "
    "--config <runtime YAML> --k {k} --output {path}"
)

__all__ = [
    "COST_TABLE_SCHEMA_VERSION",
    "CostTableSetup",
    "ExactSpsCostRow",
    "ExactSpsCostTable",
    "build_cost_table_payload",
    "load_runtime_sps_cost_table",
    "merge_cost_table_payloads",
    "read_cost_table_rows",
    "select_exact_sps_candidate",
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
            _require_positive_finite_number(value, field="step time") for value in self.step_time_ms
        )
        if any(b <= a for a, b in zip(token_counts, token_counts[1:])):
            raise ValueError("token_counts must be strictly increasing")
        object.__setattr__(self, "token_counts", token_counts)
        object.__setattr__(self, "step_time_ms", step_time_ms)


@dataclass(frozen=True)
class ExactSpsCostTable:
    """Directly measured whole-step costs keyed by exact ``(G, V)``.

    ``G`` is the rank-local padded CUDA-graph batch size and ``V`` is the
    rank-local submitted ragged verifier-token count, including anchor and
    pad-row tokens. ``V=0`` is a sentinel for the native static physical-K path: it is
    the mandatory fallback comparator and is never a ragged capture bucket.
    Every positive cell must fit the physical verifier window
    ``G <= V <= G * (max_draft_len + 1)``.
    No interpolation is permitted on either axis: an unmeasured graph shape
    is a configuration error rather than an estimate.
    """

    tables: dict[int, ExactSpsCostRow]
    max_draft_len: int
    minimum_predicted_gain: float = 0.04
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
            budgets = table.token_counts
            if budgets[0] != 0:
                raise ValueError(
                    "Every G requires a V=0 native static physical-K measurement; "
                    f"missing for G={graph_batch_size}"
                )
            max_verifier_budget = graph_batch_size * (max_draft_len + 1)
            invalid_positive_budgets = [
                verifier_budget
                for verifier_budget in budgets[1:]
                if (verifier_budget < graph_batch_size or verifier_budget > max_verifier_budget)
            ]
            if invalid_positive_budgets:
                raise ValueError(
                    "Positive verifier budgets must satisfy G <= V <= G*(K+1); "
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
                    "Cost table exceeds the limit of "
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
                "Cost table exceeds the limit of "
                f"{_MAX_EXACT_COMPACT_CELLS_TOTAL} compact (G,V) cells across captured "
                f"graph sizes; it has {production_cell_count}"
            )
        object.__setattr__(self, "tables", normalized)
        object.__setattr__(self, "max_draft_len", max_draft_len)
        minimum_predicted_gain = _require_nonnegative_finite_number(
            self.minimum_predicted_gain, field="minimum_predicted_gain"
        )
        object.__setattr__(self, "minimum_predicted_gain", minimum_predicted_gain)
        # This is the rank-agreement identity used on every exact scheduling
        # step. It covers the canonical ordered grid, every measured cost, K,
        # and the minimum-gain policy. Two tables that merely have the same
        # number of cells must not be allowed to index each other's yields.
        identity_payload = {
            "schema": "exact-sps-runtime-v2",
            "max_draft_len": max_draft_len,
            "minimum_predicted_gain": minimum_predicted_gain,
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
                f"Cost table has no direct measurements for G={graph_batch_size}"
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
                f"Cost table has no direct measurements for G={num_requests}, V={missing}"
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
        budgets = self.for_graph_batch_size(num_requests).token_counts
        if include_native:
            return tuple(budgets)
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


@dataclass(frozen=True)
class CostTableSetup:
    """Hardware and parallel layout a cost table was measured on.

    Step costs from another GPU model or parallel layout do not describe the
    running engine, so the loader rejects them. A different TensorRT-LLM
    version only warns: kernels drift by a few percent, but the table stays a
    reasonable price list until it is regenerated.
    """

    gpu: str
    world_size: int
    tensor_parallel_size: int
    moe_expert_parallel_size: int
    enable_attention_dp: bool
    tensorrt_llm_version: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "gpu": self.gpu,
            "world_size": self.world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "moe_expert_parallel_size": self.moe_expert_parallel_size,
            "enable_attention_dp": self.enable_attention_dp,
            "tensorrt_llm_version": self.tensorrt_llm_version,
        }


_SETUP_ERROR_FIELDS = (
    "gpu",
    "world_size",
    "tensor_parallel_size",
    "moe_expert_parallel_size",
    "enable_attention_dp",
)


def _read_cost_table_file(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DSpark cost table {path} does not exist. Generate one with: "
            + PROFILE_COMMAND_HINT.format(k="<K>", path=path)
        )
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"DSpark cost table {path} must contain a JSON object")
    return payload


def _parse_rows(rows: object, *, where: str) -> dict[int, ExactSpsCostRow]:
    if not isinstance(rows, dict) or not rows:
        raise TypeError(f"{where} must be a non-empty object keyed by graph batch size")
    parsed: dict[int, ExactSpsCostRow] = {}
    for key, row in rows.items():
        graph_batch_size = _parse_graph_batch_size_key(key)
        if not isinstance(row, dict):
            raise TypeError(f"{where} G={key} must be an object")
        missing = sorted({"token_counts", "step_time_ms"} - row.keys())
        if missing:
            raise ValueError(f"{where} G={key} is missing {', '.join(missing)}")
        token_counts = row["token_counts"]
        step_time_ms = row["step_time_ms"]
        if not isinstance(token_counts, list) or not isinstance(step_time_ms, list):
            raise TypeError(f"{where} G={key} token_counts and step_time_ms must be lists")
        parsed[graph_batch_size] = ExactSpsCostRow(
            token_counts=tuple(
                _require_json_int(value, field=f"{where} G={key} verifier budget", minimum=0)
                for value in token_counts
            ),
            step_time_ms=tuple(
                _require_positive_finite_number(value, field=f"{where} G={key} step time")
                for value in step_time_ms
            ),
        )
    return parsed


def _legacy_v2_setup(fingerprint: object) -> dict[str, object]:
    """Map a schema-v2 engine fingerprint onto the schema-v3 setup fields."""
    if not isinstance(fingerprint, dict):
        return {}
    setup: dict[str, object] = {}
    if isinstance(fingerprint.get("gpu"), str):
        setup["gpu"] = fingerprint["gpu"]
    if type(fingerprint.get("gpu_count")) is int:
        setup["world_size"] = fingerprint["gpu_count"]
    topology = fingerprint.get("topology")
    if isinstance(topology, str):
        try:
            topology = json.loads(topology)
        except ValueError:
            topology = None
    if isinstance(topology, dict):
        for source, target in (
            ("tp", "tensor_parallel_size"),
            ("ep", "moe_expert_parallel_size"),
            ("attention_dp", "enable_attention_dp"),
        ):
            if source in topology:
                setup[target] = topology[source]
    return setup


def read_cost_table_rows(
    payload: Mapping[str, object], *, max_draft_len: int, path: str | Path = "<cost table>"
) -> tuple[dict[int, ExactSpsCostRow], dict[str, object]]:
    """Return ``({G: row}, recorded setup)`` for one physical K.

    Schema 3 keys costs by K, then G. Schema 2 (one K per file, written by
    the earlier offline exporter) is still read so existing tables keep
    working; its provenance digests are not re-verified.
    """
    k = _require_exact_int(max_draft_len, field="max_draft_len", minimum=1)
    schema_version = payload.get("schema_version")
    if schema_version == COST_TABLE_SCHEMA_VERSION:
        costs = payload.get("costs")
        if not isinstance(costs, dict) or not costs:
            raise TypeError(f"DSpark cost table {path} has no 'costs' object")
        if str(k) not in costs:
            available = sorted(int(key) for key in costs if str(key).isdecimal())
            raise ValueError(
                f"DSpark cost table {path} has no costs for max_draft_len={k} "
                f"(it has K={available}). Add them with: "
                + PROFILE_COMMAND_HINT.format(k=k, path=path)
            )
        setup = payload.get("setup")
        return (
            _parse_rows(costs[str(k)], where=f"{path} K={k}"),
            dict(setup) if isinstance(setup, dict) else {},
        )
    if schema_version == 2:
        fingerprint = payload.get("engine_fingerprint")
        recorded_k = fingerprint.get("max_draft_len") if isinstance(fingerprint, dict) else None
        if recorded_k != k:
            raise ValueError(
                f"DSpark cost table {path} was measured for max_draft_len={recorded_k}, "
                f"but the runtime uses max_draft_len={k}. Generate a table for K={k} with: "
                + PROFILE_COMMAND_HINT.format(k=k, path=path)
            )
        return (
            _parse_rows(payload.get("cost_tables"), where=f"{path} K={k}"),
            _legacy_v2_setup(fingerprint),
        )
    raise ValueError(
        f"DSpark cost table {path} has unsupported schema_version={schema_version!r}; "
        f"expected {COST_TABLE_SCHEMA_VERSION} (or legacy 2)"
    )


def _check_setup(
    recorded: Mapping[str, object], live: CostTableSetup | None, *, path: str | Path
) -> None:
    if live is None:
        return
    live_values = live.to_json()
    mismatches = [
        f"{name}: table={recorded[name]!r}, runtime={live_values[name]!r}"
        for name in _SETUP_ERROR_FIELDS
        if name in recorded and recorded[name] != live_values[name]
    ]
    if mismatches:
        raise ValueError(
            f"DSpark cost table {path} was measured on a different setup ("
            + "; ".join(mismatches)
            + "). Step costs do not transfer across GPU models or parallel layouts; "
            "generate a table for this setup with: "
            + PROFILE_COMMAND_HINT.format(k="<K>", path=path)
        )
    recorded_version = recorded.get("tensorrt_llm_version")
    if (
        recorded_version
        and live.tensorrt_llm_version
        and recorded_version != live.tensorrt_llm_version
    ):
        logger.warning(
            f"DSpark cost table {path} was measured with TensorRT-LLM {recorded_version}; "
            f"running {live.tensorrt_llm_version}. Costs may have drifted; regenerate the "
            "table if confidence scheduling underperforms."
        )


def load_runtime_sps_cost_table(
    path: str | Path,
    *,
    graph_batch_sizes: Sequence[int],
    max_draft_len: int,
    minimum_predicted_gain: float,
    live_setup: CostTableSetup | None = None,
) -> ExactSpsCostTable:
    """Load the costs for ``max_draft_len`` and the captured graph ladder.

    Raises with a regenerate hint when the file is missing, has no entry for
    this K, lacks a row for a captured graph batch size, or was measured on
    another GPU model or parallel layout. Rows for graph sizes the engine
    does not capture are ignored.
    """
    payload = _read_cost_table_file(path)
    rows, recorded_setup = read_cost_table_rows(payload, max_draft_len=max_draft_len, path=path)
    _check_setup(recorded_setup, live_setup, path=path)
    captured = sorted(
        {
            _require_exact_int(value, field="captured graph batch size", minimum=1)
            for value in graph_batch_sizes
        }
    )
    if not captured:
        raise ValueError("DSpark confidence scheduling requires captured CUDA-graph batch sizes")
    missing = [graph_batch_size for graph_batch_size in captured if graph_batch_size not in rows]
    if missing:
        raise ValueError(
            f"DSpark cost table {path} (K={max_draft_len}) has no measurements for captured "
            f"graph batch sizes {missing}; it covers {sorted(rows)}. Either set "
            "cuda_graph_config.batch_sizes to covered sizes or regenerate the table with: "
            + PROFILE_COMMAND_HINT.format(k=max_draft_len, path=path)
        )
    return ExactSpsCostTable(
        tables={graph_batch_size: rows[graph_batch_size] for graph_batch_size in captured},
        max_draft_len=max_draft_len,
        minimum_predicted_gain=minimum_predicted_gain,
    )


def build_cost_table_payload(
    *,
    setup: CostTableSetup,
    costs: Mapping[int, Mapping[int, Sequence[tuple[int, float]]]],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize ``{K: {G: [(V, ms), ...]}}`` as a schema-3 cost table."""
    serialized: dict[str, object] = {}
    for k, rows in sorted(costs.items()):
        serialized_rows = {}
        for graph_batch_size, cells in sorted(rows.items()):
            ordered = sorted((int(v), float(ms)) for v, ms in cells)
            row = ExactSpsCostRow(
                token_counts=tuple(v for v, _ in ordered),
                step_time_ms=tuple(ms for _, ms in ordered),
            )
            serialized_rows[str(int(graph_batch_size))] = {
                "token_counts": list(row.token_counts),
                "step_time_ms": [round(ms, 4) for ms in row.step_time_ms],
            }
        # Validate the per-K grid with the same rules the runtime applies.
        ExactSpsCostTable(
            tables=_parse_rows(serialized_rows, where=f"K={k}"),
            max_draft_len=int(k),
        )
        serialized[str(int(k))] = serialized_rows
    payload: dict[str, object] = {
        "schema_version": COST_TABLE_SCHEMA_VERSION,
        "setup": setup.to_json(),
        "costs": serialized,
    }
    if extra:
        payload.update(extra)
    return payload


def merge_cost_table_payloads(
    existing: Mapping[str, object], new: Mapping[str, object]
) -> dict[str, object]:
    """Add or replace ``new``'s (K, G) rows in ``existing`` (same setup required).

    Rows of ``existing`` that ``new`` does not measure are kept, so a table can
    be extended one K or one graph batch size at a time.
    """
    if existing.get("schema_version") != COST_TABLE_SCHEMA_VERSION:
        raise ValueError(
            "Only schema-3 cost tables can be merged; write the new costs to a new file"
        )
    old_setup = dict(existing.get("setup") or {})
    new_setup = dict(new.get("setup") or {})
    conflicts = [
        name
        for name in _SETUP_ERROR_FIELDS
        if name in old_setup and name in new_setup and old_setup[name] != new_setup[name]
    ]
    if conflicts:
        raise ValueError(
            "Cannot merge cost tables measured on different setups: " + ", ".join(conflicts)
        )
    merged = dict(existing)
    merged["setup"] = new_setup or old_setup
    costs = _merge_nested(existing.get("costs") or {}, new.get("costs") or {})
    merged["costs"] = {
        k: dict(sorted(rows.items(), key=lambda item: int(item[0])))
        for k, rows in sorted(costs.items(), key=lambda item: int(item[0]))
    }
    for key, value in new.items():
        if key not in ("schema_version", "setup", "costs"):
            # Per-K metadata such as the profiler's per-(G, V) sample counts.
            merged[key] = _merge_nested(merged.get(key), value, depth=3)
    return merged


def _merge_nested(old: object, new: object, depth: int = 2) -> object:
    """``new`` over ``old``, merging dicts ``depth`` levels deep."""
    if depth == 0 or not isinstance(old, Mapping) or not isinstance(new, Mapping):
        return new
    merged = dict(old)
    for key, value in new.items():
        merged[key] = _merge_nested(old.get(key), value, depth - 1)
    return merged


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_graph_batch_size_key(value: object) -> int:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or value.startswith("0")
    ):
        raise TypeError(
            f"Cost table graph batch size key {value!r} must be a canonical positive integer string"
        )
    return int(value)


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


def select_exact_sps_candidate(
    *,
    graph_batch_size: int,
    native_expected_yield: float,
    compact_expected_yields: dict[int, float],
    cost_table: ExactSpsCostTable,
) -> int:
    """Choose a measured V by expected goodput, keeping native V=0 by default.

    The caller supplies globally agreed expected yields after the allgather;
    this pure layer only compares measured cells. A compact tier wins only
    when its expected tokens per millisecond beat native by at least
    ``cost_table.minimum_predicted_gain``; among winners the highest goodput
    is chosen.
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
    compact_yields = {
        budget: _require_nonnegative_finite_number(
            compact_expected_yields[budget], field=f"expected yield for compact V={budget}"
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
    required_score = native_score * (1.0 + cost_table.minimum_predicted_gain)
    goodput_budgets = [
        budget
        for budget in measured_budgets
        if compact_scores[budget] > native_score and compact_scores[budget] >= required_score
    ]
    if not goodput_budgets:
        return 0
    return int(max(goodput_budgets, key=lambda budget: (compact_scores[budget], budget)))
