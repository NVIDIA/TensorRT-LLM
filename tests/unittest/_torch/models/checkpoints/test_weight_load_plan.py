# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointTensor,
)
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightDemand,
    WeightLoadOrderConfidence,
    WeightLoadPlan,
    WeightLoadPlanCoverage,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _catalog() -> CheckpointCatalog:
    return CheckpointCatalog(
        objects=(),
        tensors=(CheckpointTensor("a"), CheckpointTensor("b"), CheckpointTensor("c")),
    )


def _plan(
    catalog: CheckpointCatalog,
    *,
    coverage: WeightLoadPlanCoverage = WeightLoadPlanCoverage.EXACT,
    ordering: WeightLoadOrderConfidence = WeightLoadOrderConfidence.ADVISORY,
    demands: tuple[WeightDemand, ...] | None = None,
) -> WeightLoadPlan:
    if demands is None:
        demands = (WeightDemand("group", ("a",), (0,)),)
    return WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=2,
        coverage=coverage,
        ordering=ordering,
        demands=demands,
    )


def test_exact_plan_exposes_set_filter_separately_from_ordering() -> None:
    catalog = _catalog()
    plan = _plan(
        catalog,
        demands=(
            WeightDemand("later", ("b",), (0,), priority=2, predecessors=("first",)),
            WeightDemand("first", ("a",), (0,), priority=1),
        ),
    )

    plan.validate_against(catalog)
    assert plan.selected_tensor_names == frozenset({"a", "b"})


def test_conservative_plan_loads_all_without_exposing_filter() -> None:
    catalog = _catalog()
    plan = _plan(
        catalog,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        demands=(
            WeightDemand("early", ("a", "b"), (0,), priority=1),
            WeightDemand("late", ("c",), (0,), priority=2),
        ),
    )

    plan.validate_against(catalog)
    assert plan.selected_tensor_names is None


def test_plan_id_is_independent_of_demand_tuple_order() -> None:
    catalog = _catalog()
    first = WeightDemand("first", ("b", "a"), (1, 0))
    second = WeightDemand("second", ("c",), (0,), predecessors=("first",))

    assert (
        _plan(catalog, demands=(first, second)).plan_id
        == _plan(catalog, demands=(second, first)).plan_id
    )


def test_plan_rejects_unknown_tensors_and_dependency_cycles() -> None:
    catalog = _catalog()
    unknown = _plan(catalog, demands=(WeightDemand("group", ("missing",), (0,)),))
    with pytest.raises(ValueError, match="unknown tensors"):
        unknown.validate_against(catalog)

    with pytest.raises(ValueError, match="acyclic"):
        _plan(
            catalog,
            demands=(
                WeightDemand("a", ("a",), (0,), predecessors=("b",)),
                WeightDemand("b", ("b",), (0,), predecessors=("a",)),
            ),
        )


class _Mapper(BaseWeightMapper):
    def map_weights(self) -> None:
        pass

    def apply_callbacks(self, module, module_name, module_names_breakdown, weights):
        raise AssertionError("not used")


def test_base_mapper_emits_valid_conservative_plan() -> None:
    catalog = _catalog()
    model_config = ModelConfig(mapping=Mapping(world_size=2, rank=1))
    model = SimpleNamespace(model_config=model_config, config=object())
    mapper = _Mapper()
    mapper.init_model_and_config(model, model_config)

    plan = mapper.build_weight_load_plan(catalog)

    assert plan.coverage is WeightLoadPlanCoverage.CONSERVATIVE
    assert plan.ordering is WeightLoadOrderConfidence.OPAQUE
    assert plan.selected_tensor_names is None
    assert plan.demands[0].source_names == ("a", "b", "c")
    assert plan.demands[0].destination_ranks == (1,)
