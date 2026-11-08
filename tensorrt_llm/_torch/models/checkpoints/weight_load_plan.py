# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral demand metadata for checkpoint loading."""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import CheckpointCatalog

WEIGHT_LOAD_PLAN_SCHEMA_VERSION = 1


class WeightLoadPlanCoverage(str, Enum):
    """Whether tensor names are safe to use as a selective source filter."""

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    OPAQUE = "opaque"


class WeightLoadOrderConfidence(str, Enum):
    """How strongly priority and predecessor metadata constrain loading."""

    EXACT = "exact"
    ADVISORY = "advisory"
    OPAQUE = "opaque"


@dataclass(frozen=True, slots=True)
class WeightDemand:
    """An atomic tensor group with partial-order and scheduling hints."""

    group_id: str
    source_names: tuple[str, ...]
    destination_ranks: tuple[int, ...]
    priority: int = 0
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("weight demand group_id must be a nonempty string")
        if not isinstance(self.source_names, tuple) or not self.source_names:
            raise ValueError("weight demand source_names must be a nonempty tuple")
        if any(not isinstance(name, str) or not name for name in self.source_names):
            raise ValueError("weight demand source_names must contain nonempty strings")
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("weight demand source_names must be unique within a group")
        if not isinstance(self.destination_ranks, tuple) or not self.destination_ranks:
            raise ValueError("weight demand destination_ranks must be a nonempty tuple")
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
            for rank in self.destination_ranks
        ):
            raise ValueError("weight demand destination_ranks must contain nonnegative integers")
        if len(set(self.destination_ranks)) != len(self.destination_ranks):
            raise ValueError("weight demand destination_ranks must be unique")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ValueError("weight demand priority must be an integer")
        if not isinstance(self.predecessors, tuple):
            raise ValueError("weight demand predecessors must be a tuple")
        if any(not isinstance(group_id, str) or not group_id for group_id in self.predecessors):
            raise ValueError("weight demand predecessors must contain nonempty strings")
        if len(set(self.predecessors)) != len(self.predecessors):
            raise ValueError("weight demand predecessors must be unique")
        if self.group_id in self.predecessors:
            raise ValueError("weight demand cannot depend on itself")


@dataclass(frozen=True, slots=True)
class WeightLoadPlan:
    """Validated shadow description of checkpoint tensors needed by ranks.

    Coverage controls filter safety; ordering independently controls whether
    priorities and predecessor edges are exact, advisory, or unavailable.
    Only ``EXACT`` coverage is safe for destination-aware selective I/O.
    ``CONSERVATIVE`` and ``OPAQUE`` coverage both instruct a source to load all
    tensors, although either may carry separately qualified ordering metadata.
    """

    catalog_id: str
    rank: int
    world_size: int
    coverage: WeightLoadPlanCoverage
    ordering: WeightLoadOrderConfidence
    demands: tuple[WeightDemand, ...]
    schema_version: int = WEIGHT_LOAD_PLAN_SCHEMA_VERSION
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_id, str) or len(self.catalog_id) != 64:
            raise ValueError("weight load plan catalog_id must be a SHA-256 hex digest")
        try:
            int(self.catalog_id, 16)
        except ValueError as error:
            raise ValueError("weight load plan catalog_id must be a SHA-256 hex digest") from error
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("weight load plan rank must be an integer")
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise ValueError("weight load plan world_size must be an integer")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("weight load plan rank and world_size must describe a valid group")
        if not isinstance(self.coverage, WeightLoadPlanCoverage):
            raise ValueError("weight load plan coverage must be a WeightLoadPlanCoverage")
        if not isinstance(self.ordering, WeightLoadOrderConfidence):
            raise ValueError("weight load plan ordering must be a WeightLoadOrderConfidence")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise ValueError("weight load plan schema_version must be an integer")
        if self.schema_version != WEIGHT_LOAD_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported weight load plan schema_version: {self.schema_version}")
        if not isinstance(self.demands, tuple) or not all(
            isinstance(demand, WeightDemand) for demand in self.demands
        ):
            raise ValueError("weight load plan demands must be a tuple of WeightDemand")

        if self.coverage is not WeightLoadPlanCoverage.OPAQUE and not self.demands:
            raise ValueError("exact and conservative weight load plans must expose demands")
        if self.ordering is not WeightLoadOrderConfidence.OPAQUE and not self.demands:
            raise ValueError("exact and advisory ordering require at least one demand")

        group_by_id = {demand.group_id: demand for demand in self.demands}
        if len(group_by_id) != len(self.demands):
            raise ValueError("weight demand group IDs must be unique")
        source_names = [name for demand in self.demands for name in demand.source_names]
        if len(set(source_names)) != len(source_names):
            raise ValueError("a source tensor cannot belong to multiple weight demand groups")
        for demand in self.demands:
            if any(rank >= self.world_size for rank in demand.destination_ranks):
                raise ValueError("weight demand destination rank is outside plan world_size")
            missing = set(demand.predecessors) - group_by_id.keys()
            if missing:
                raise ValueError(
                    f"weight demand {demand.group_id!r} has unknown predecessors: {sorted(missing)}"
                )
        self._validate_acyclic(group_by_id)

        payload = {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "coverage": self.coverage.value,
            "ordering": self.ordering.value,
            "demands": [
                {
                    "group_id": demand.group_id,
                    "source_names": sorted(demand.source_names),
                    "destination_ranks": sorted(demand.destination_ranks),
                    "priority": demand.priority,
                    "predecessors": sorted(demand.predecessors),
                }
                for demand in sorted(self.demands, key=lambda demand: demand.group_id)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "plan_id", hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    @staticmethod
    def _validate_acyclic(group_by_id: dict[str, WeightDemand]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(group_id: str) -> None:
            if group_id in visited:
                return
            if group_id in visiting:
                raise ValueError("weight demand predecessor graph must be acyclic")
            visiting.add(group_id)
            for predecessor in group_by_id[group_id].predecessors:
                visit(predecessor)
            visiting.remove(group_id)
            visited.add(group_id)

        for group_id in group_by_id:
            visit(group_id)

    @property
    def described_tensor_names(self) -> frozenset[str]:
        """Return every tensor named by demand metadata."""
        return frozenset(name for demand in self.demands for name in demand.source_names)

    @property
    def selected_tensor_names(self) -> frozenset[str] | None:
        """Return a safe set-valued source filter only for exact coverage."""
        return (
            self.described_tensor_names if self.coverage is WeightLoadPlanCoverage.EXACT else None
        )

    def validate_against(self, catalog: CheckpointCatalog) -> None:
        """Validate source names and coverage against a concrete catalog."""
        if catalog.catalog_id != self.catalog_id:
            raise ValueError("weight load plan catalog_id does not match checkpoint catalog")
        unknown = self.described_tensor_names - catalog.tensor_names
        if unknown:
            raise ValueError(f"weight load plan references unknown tensors: {sorted(unknown)}")
        if self.coverage is WeightLoadPlanCoverage.CONSERVATIVE:
            if self.described_tensor_names != catalog.tensor_names:
                raise ValueError("conservative weight load plan must include every catalog tensor")
