# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-construction evidence binding for measured BREAKABLE mixed costs.

The optional mixed configuration binds measured costs and an admission receipt
to an independently observed live fingerprint. The runtime separately checks
the active route, request ownership and geometry before publishing row lengths.
Parsing authenticates evidence bytes and checks their claims; it does not run
the qualification performed by the evidence producer.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import dspark_planner

_BODY_BUCKETS = (96, 128, 192, 256, 384, 512)
_ACTION = "current-host-confidence-prefix-allocation-under-shared-body-capacity-v1"
_PRICE_SEMANTICS = "rank0_forward_pricing_proxy"
_GEOMETRY_FIELDS = (
    "context_rows",
    "context_chunk_squared",
    "context_position_product",
    "context_compressed_product",
    "generation_history_sum",
    "generation_history_max",
    "bootstrap_generations",
    "real_context_requests",
    "eligible_generations",
    "context_requests",
    "generation_requests",
    "fixed_generation_rows",
)
_MAX_BYTES = 16 * 1024 * 1024
_MAX_CELLS = 256


@dataclass(frozen=True)
class MixedRuntimeIdentity:
    """Independently observed live identity, not copied from the cost artifact.

    Support is limited to K3 and the eight-rank BREAKABLE body grid. The runner
    checks embedded-model, PP/CP/attention-DP and capture readiness separately;
    these properties are not inferred from a cost file.
    """

    source_tree_sha256: str
    runtime_snapshot_sha256: str
    model_config_sha256: str
    gpu_class_sha256: str
    topology_sha256: str
    selector_identity_sha256: str
    physical_k: int = 3
    world_size: int = 8
    body_buckets: tuple[int, ...] = _BODY_BUCKETS
    execution_mode: str = "BREAKABLE"

    def __post_init__(self) -> None:
        for field in (
            "source_tree_sha256",
            "runtime_snapshot_sha256",
            "model_config_sha256",
            "gpu_class_sha256",
            "topology_sha256",
            "selector_identity_sha256",
        ):
            dspark_planner._require_sha256(getattr(self, field), field=field)
        if (
            type(self.physical_k) is not int
            or self.physical_k != 3
            or type(self.world_size) is not int
            or self.world_size != 8
            or type(self.body_buckets) is not tuple
            or any(type(bucket) is not int for bucket in self.body_buckets)
            or self.body_buckets != _BODY_BUCKETS
            or self.execution_mode != "BREAKABLE"
        ):
            raise ValueError("unsupported mixed physical K, topology or BREAKABLE grid")

    def payload(self) -> dict[str, object]:
        return {
            **{
                field: getattr(self, field)
                for field in (
                    "source_tree_sha256",
                    "runtime_snapshot_sha256",
                    "model_config_sha256",
                    "gpu_class_sha256",
                    "topology_sha256",
                    "selector_identity_sha256",
                    "physical_k",
                    "world_size",
                    "execution_mode",
                )
            },
            "body_buckets": list(self.body_buckets),
        }


@dataclass(frozen=True)
class MixedGeometry:
    """One exact ordered-rank host-geometry/action key; never a decode ``(G,V)``.

    The twelve rank features match the accepted host price support dimensions.
    They summarize host geometry, not effective device KV lengths or per-request
    ownership. Ownership/incarnation/tickets must be checked separately at publish.
    """

    rank_features: tuple[tuple[int, ...], ...]
    capacity: int
    actual_rank_rows: tuple[int, ...]
    native_rank_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.rank_features) is not tuple
            or len(self.rank_features) != 8
            or type(self.actual_rank_rows) is not tuple
            or type(self.native_rank_rows) is not tuple
            or len(self.actual_rank_rows) != 8
            or len(self.native_rank_rows) != 8
            or type(self.capacity) is not int
            or self.capacity not in _BODY_BUCKETS
        ):
            raise ValueError("mixed geometry requires eight ordered ranks and a captured capacity")
        for rank, features in enumerate(self.rank_features):
            if type(features) is not tuple or len(features) != len(_GEOMETRY_FIELDS):
                raise ValueError("mixed rank feature dimensions differ")
            for value in (*features, self.actual_rank_rows[rank], self.native_rank_rows[rank]):
                dspark_planner._require_json_int(value, field="mixed geometry value", minimum=0)
                if value >= 2**63:
                    raise ValueError("mixed geometry exceeds the existing wire integer bound")
            (
                context,
                _,
                _,
                _,
                history_sum,
                history_max,
                bootstrap,
                real_context,
                eligible,
                contexts,
                generations,
                fixed,
            ) = features
            if (
                real_context > contexts
                or bootstrap > generations - eligible
                or eligible > generations
                or history_max > history_sum
                or fixed != (generations - eligible) * 4
                or self.native_rank_rows[rank] != context + fixed + eligible * 4
                or not context + fixed + eligible
                <= self.actual_rank_rows[rank]
                <= min(self.native_rank_rows[rank], self.capacity)
            ):
                raise ValueError("mixed geometry does not reserve context and full-K fixed rows")
        if (
            not any(row[7] for row in self.rank_features)
            or self.actual_rank_rows == self.native_rank_rows
            or _paid_bucket(self.native_rank_rows) is None
        ):
            raise ValueError("geometry is all-decode, noncompact or outside the captured grid")

    def payload(self) -> dict[str, object]:
        return {
            "rank_features": [dict(zip(_GEOMETRY_FIELDS, row)) for row in self.rank_features],
            "capacity": self.capacity,
            "actual_rank_rows": list(self.actual_rank_rows),
            "native_rank_rows": list(self.native_rank_rows),
        }


@dataclass(frozen=True)
class MixedEvidenceBinding:
    """Exact artifact pins supplied by the optional mixed configuration."""

    costs_sha256: str
    admission_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        for field in ("costs_sha256", "admission_sha256", "protocol_sha256"):
            dspark_planner._require_sha256(getattr(self, field), field=field)


@dataclass(frozen=True)
class MixedCostCell:
    geometry: MixedGeometry
    native_lower_ms: float
    candidate_upper_ms: float
    measurement_result_sha256: str


@dataclass(frozen=True)
class AdmittedMixedCosts:
    identity: MixedRuntimeIdentity
    binding: MixedEvidenceBinding
    minimum_predicted_gain: float
    cells: tuple[MixedCostCell, ...]

    def lookup(self, geometry: MixedGeometry) -> MixedCostCell | None:
        """Uncovered/invalid geometry stays native; no nearest-cell selection."""
        if type(geometry) is not MixedGeometry:
            return None
        return next((cell for cell in self.cells if cell.geometry == geometry), None)


@dataclass(frozen=True)
class MixedEvidenceResolution:
    costs: AdmittedMixedCosts | None
    reason: str


def load_mixed_evidence(config) -> MixedEvidenceResolution:
    """Read independently pinned deployment artifacts before model execution.

    The live fingerprint is a separately produced artifact, never the table's
    identity. Runtime route checks are additional to this byte binding. Invalid
    optional evidence leaves mixed verification native, not confidence-off.
    """
    if config is None:
        return MixedEvidenceResolution(None, "mixed_evidence_absent_native")
    try:
        binding = MixedEvidenceBinding(
            config.costs_sha256, config.admission_sha256, config.protocol_sha256
        )

        def bounded_file(path: str) -> bytes:
            with Path(path).open("rb") as handle:
                value = handle.read(_MAX_BYTES + 1)
            if not 0 < len(value) <= _MAX_BYTES:
                raise ValueError("Mixed artifact exceeds the bounded byte contract")
            return value

        live = dspark_planner._validate_exact_fields(
            _read_pinned(
                bounded_file(config.live_fingerprint_path),
                config.live_fingerprint_sha256,
                name="mixed live fingerprint",
            ),
            name="mixed live fingerprint",
            fields={"schema", "identity"},
        )
        if live["schema"] != "dspark-mixed-live-fingerprint-v1":
            raise ValueError("Unknown independently observed mixed fingerprint")
        identity = dspark_planner._validate_exact_fields(
            live["identity"],
            name="mixed live identity",
            fields={
                "source_tree_sha256",
                "runtime_snapshot_sha256",
                "model_config_sha256",
                "gpu_class_sha256",
                "topology_sha256",
                "selector_identity_sha256",
                "physical_k",
                "world_size",
                "body_buckets",
                "execution_mode",
            },
        )
        if type(identity["body_buckets"]) is not list:
            raise ValueError("Mixed fingerprint body buckets must be a list")
        observed = MixedRuntimeIdentity(
            **{**identity, "body_buckets": tuple(identity["body_buckets"])}
        )
        return resolve_mixed_evidence(
            binding,
            costs_raw=bounded_file(config.costs_path),
            admission_raw=bounded_file(config.admission_path),
            live_identity=observed,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        return MixedEvidenceResolution(
            None, f"mixed_evidence_invalid_native:{type(exc).__name__}:{exc}"
        )


def _paid_bucket(rows: tuple[int, ...]) -> int | None:
    return next((bucket for bucket in _BODY_BUCKETS if bucket >= max(rows)), None)


def _read_pinned(raw: bytes, pin: str, *, name: str) -> dict[str, object]:
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError(f"{name} must be bounded nonempty bytes")
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError(f"{name} byte digest differs")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} has duplicate fields")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} has nonfinite JSON: {value}")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _parse_geometry(value: object) -> MixedGeometry:
    row = dspark_planner._validate_exact_fields(
        value,
        name="mixed geometry",
        fields={
            "rank_features",
            "capacity",
            "actual_rank_rows",
            "native_rank_rows",
        },
    )
    if (
        type(row["rank_features"]) is not list
        or type(row["actual_rank_rows"]) is not list
        or type(row["native_rank_rows"]) is not list
    ):
        raise ValueError("mixed geometry arrays differ")
    features = tuple(
        tuple(
            dspark_planner._validate_exact_fields(
                rank, name="mixed rank", fields=set(_GEOMETRY_FIELDS)
            )[field]
            for field in _GEOMETRY_FIELDS
        )
        for rank in row["rank_features"]
    )
    return MixedGeometry(
        features, row["capacity"], tuple(row["actual_rank_rows"]), tuple(row["native_rank_rows"])
    )


def _validate(
    binding: MixedEvidenceBinding,
    costs_raw: bytes,
    admission_raw: bytes,
    identity: MixedRuntimeIdentity,
) -> AdmittedMixedCosts:
    table = dspark_planner._validate_exact_fields(
        _read_pinned(costs_raw, binding.costs_sha256, name="mixed costs"),
        name="mixed costs",
        fields={
            "schema",
            "identity",
            "protocol_sha256",
            "action_semantics",
            "pricing_semantics",
            "synthetic",
            "provenance",
            "minimum_predicted_gain",
            "cells",
        },
    )
    receipt = dspark_planner._validate_exact_fields(
        _read_pinned(admission_raw, binding.admission_sha256, name="mixed admission"),
        name="mixed admission",
        fields={
            "schema",
            "identity",
            "costs_sha256",
            "protocol_sha256",
            "action_semantics",
            "pricing_semantics",
            "production_admission",
            "source_and_measurements_verified",
            "accuracy_pass",
            "runtime_invariants_pass",
            "matched_workload_pass",
            "accuracy_result_sha256",
            "runtime_invariants_result_sha256",
            "matched_workload_result_sha256",
            "selector_replay_sha256",
            "measurement_results_sha256",
            "economics",
        },
    )
    if (
        table["schema"] != "dspark-mixed-breakable-costs-v1"
        or receipt["schema"] != "dspark-mixed-breakable-admission-v1"
        or table["identity"] != identity.payload()
        or receipt["identity"] != identity.payload()
        or table["protocol_sha256"] != binding.protocol_sha256
        or receipt["protocol_sha256"] != binding.protocol_sha256
        or receipt["costs_sha256"] != binding.costs_sha256
        or any(
            payload["action_semantics"] != _ACTION
            or payload["pricing_semantics"] != _PRICE_SEMANTICS
            for payload in (table, receipt)
        )
        or table["synthetic"] is not False
        or table["provenance"] != "closed_measured_breakable_mixed"
        or any(
            receipt[field] is not True
            for field in (
                "production_admission",
                "source_and_measurements_verified",
                "accuracy_pass",
                "runtime_invariants_pass",
                "matched_workload_pass",
            )
        )
    ):
        raise ValueError("mixed source, execution, action or production evidence differs")
    margin = dspark_planner._require_nonnegative_finite_number(
        table["minimum_predicted_gain"], field="mixed gain margin"
    )
    if margin < 0.01:
        raise ValueError("mixed evidence cannot reduce the incumbent 1% immediate-goodput gate")
    for field in (
        "accuracy_result_sha256",
        "runtime_invariants_result_sha256",
        "matched_workload_result_sha256",
        "selector_replay_sha256",
        "measurement_results_sha256",
    ):
        dspark_planner._require_sha256(receipt[field], field=field)
    economics = dspark_planner._validate_exact_fields(
        receipt["economics"],
        name="mixed economics",
        fields={
            "policy_steps",
            "compact_choices",
            "gross_compact_value_ms_lower_bound",
            "fixed_confidence_path_overhead_ms_upper_bound",
            "safety_margin_ms",
        },
    )
    admitted, _ = dspark_planner.evaluate_confidence_workload_admission(**economics)
    if not admitted:
        raise ValueError("mixed workload economics are not positive")
    if type(table["cells"]) is not list or not 0 < len(table["cells"]) <= _MAX_CELLS:
        raise ValueError("mixed cells must be a bounded nonempty list")
    cells = []
    for item in table["cells"]:
        row = dspark_planner._validate_exact_fields(
            item,
            name="mixed cell",
            fields={
                "geometry",
                "native_lower_ms",
                "candidate_upper_ms",
                "measurement_result_sha256",
            },
        )
        geometry = _parse_geometry(row["geometry"])
        if any(cell.geometry == geometry for cell in cells):
            raise ValueError("duplicate mixed geometry/action cell")
        cells.append(
            MixedCostCell(
                geometry,
                dspark_planner._require_positive_finite_number(
                    row["native_lower_ms"], field="mixed native lower time"
                ),
                dspark_planner._require_positive_finite_number(
                    row["candidate_upper_ms"], field="mixed candidate upper time"
                ),
                dspark_planner._require_sha256(
                    row["measurement_result_sha256"], field="mixed measurement result"
                ),
            )
        )
    if (
        dspark_planner._canonical_json_sha256([cell.measurement_result_sha256 for cell in cells])
        != receipt["measurement_results_sha256"]
    ):
        raise ValueError("mixed admission does not bind all ordered measurement results")
    return AdmittedMixedCosts(identity, binding, margin, tuple(cells))


def resolve_mixed_evidence(
    binding: MixedEvidenceBinding | None,
    *,
    costs_raw: bytes | None,
    admission_raw: bytes | None,
    live_identity: MixedRuntimeIdentity,
) -> MixedEvidenceResolution:
    """Resolve once before construction; no decode object/flag is read or mutated.

    Invalid optional mixed evidence fails to native mixed verification. It must
    never globally disable an already admitted decode policy, nor be upgraded by
    copying a validation-only fingerprint or merely asserting a production flag.
    """
    if binding is None:
        return MixedEvidenceResolution(None, "mixed_evidence_absent_native")
    try:
        if (
            type(binding) is not MixedEvidenceBinding
            or type(live_identity) is not MixedRuntimeIdentity
        ):
            raise ValueError("unknown mixed binding/live identity type")
        return MixedEvidenceResolution(
            _validate(binding, costs_raw, admission_raw, live_identity), "mixed_evidence_bound"
        )
    except (TypeError, ValueError, KeyError, UnicodeError, OverflowError, RecursionError) as exc:
        return MixedEvidenceResolution(
            None, f"mixed_evidence_invalid_native:{type(exc).__name__}:{exc}"
        )
