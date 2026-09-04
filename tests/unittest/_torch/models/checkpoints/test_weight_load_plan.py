# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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


class _ManualModule(nn.Module):
    def __init__(self, value: float = 1.0, *, with_child: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value]))
        if with_child:
            self.child = _ManualModule(value + 1.0)


class _CustomLoadModule(_ManualModule):
    def __init__(self):
        super().__init__(3.0, with_child=True)

    def load_weights(self, *args, **kwargs):
        raise AssertionError("planning must not invoke custom weight loading")


class _SpecialInstanceModule(_ManualModule):
    def __init__(self):
        super().__init__(5.0, with_child=True)


class _MoeBackendModule(_CustomLoadModule):
    pass


class _StructuredModel(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = SimpleNamespace()
        self.preload_weight_modules = ["special", "qkv_proj"]
        self.layer = nn.Module()
        self.layer.qkv_proj = _ManualModule()
        self.layer.special = _SpecialInstanceModule()
        self.layer.custom = _CustomLoadModule()
        self.layer.manual = _ManualModule(with_child=True)


class _MoePathModel(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = SimpleNamespace()
        self.preload_weight_modules = ["experts"]
        self.layer = nn.Module()
        self.layer.experts = nn.Module()
        self.layer.experts.backend = _MoeBackendModule()


class _ManyCustomModulesModel(nn.Module):
    def __init__(self, model_config: ModelConfig, module_count: int):
        super().__init__()
        self.model_config = model_config
        self.config = SimpleNamespace()
        self.blocks = nn.ModuleList(_CustomLoadModule() for _ in range(module_count))


class _CountingName(str):
    startswith_calls = 0

    def startswith(self, prefix, *args):
        type(self).startswith_calls += 1
        return super().startswith(prefix, *args)


class _StructuredMapper(_Mapper):
    def map_weights(self) -> None:
        self.mapping.update({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})

    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, _SpecialInstanceModule)


def _catalog_with_names(*names: str) -> CheckpointCatalog:
    return CheckpointCatalog(objects=(), tensors=tuple(CheckpointTensor(name) for name in names))


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


def test_base_mapper_builds_deterministic_advisory_consumer_groups() -> None:
    catalog = _catalog_with_names(
        "unmatched.head",
        "layer.q_proj.bias",
        "layer.custom.child.weight",
        "layer.special.child.weight",
        "layer.manual.child.weight",
        "layer.q_proj.weight",
        "layer.special.weight",
        "layer.k_proj.weight",
        "layer.custom.weight",
        "layer.manual.weight",
        "layer.v_proj.weight",
        # These look similar to inferred names but are not structural prefix or
        # exact-parameter matches and must remain in the physical tail.
        "layer.q_proj_extra.weight",
        "layer.manual.weight.extra",
    )
    model_config = ModelConfig(mapping=Mapping(world_size=2, rank=1))
    model = _StructuredModel(model_config)
    mapper = _StructuredMapper()
    mapper.init_model_and_config(model, model_config)

    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_mapping = {name: tuple(source_names) for name, source_names in mapper.mapping.items()}
    original_preloads = tuple(model.preload_weight_modules)

    plan = mapper.build_weight_load_plan(catalog)
    repeated_plan = mapper.build_weight_load_plan(catalog)

    assert plan.coverage is WeightLoadPlanCoverage.CONSERVATIVE
    assert plan.ordering is WeightLoadOrderConfidence.ADVISORY
    assert plan.selected_tensor_names is None
    assert plan.plan_id == repeated_plan.plan_id
    assert plan.demands == repeated_plan.demands

    special, fusion = plan.demands[:2]
    assert special.source_names == (
        "layer.special.child.weight",
        "layer.special.weight",
    )
    assert fusion.source_names == (
        "layer.q_proj.bias",
        "layer.q_proj.weight",
        "layer.k_proj.weight",
        "layer.v_proj.weight",
    )
    assert special.priority == 0
    assert special.predecessors == ()
    assert fusion.priority == 1
    assert fusion.predecessors == (special.group_id,)

    nonpreload = [demand for demand in plan.demands if demand.group_id.startswith("module:")]
    assert [demand.source_names for demand in nonpreload] == [
        (
            "layer.custom.child.weight",
            "layer.custom.weight",
        ),
        ("layer.manual.weight",),
        ("layer.manual.child.weight",),
    ]
    assert {demand.priority for demand in nonpreload} == {2}
    assert all(not demand.predecessors for demand in nonpreload)

    assert plan.demands[-1].group_id == "unmatched_checkpoint_tensors"
    assert plan.demands[-1].source_names == (
        "unmatched.head",
        "layer.q_proj_extra.weight",
        "layer.manual.weight.extra",
    )

    described_names = [name for demand in plan.demands for name in demand.source_names]
    assert len(described_names) == len(set(described_names))
    assert set(described_names) == catalog.tensor_names
    assert all(demand.destination_ranks == (1,) for demand in plan.demands)

    assert mapper.mapping == {
        name: list(source_names) for name, source_names in original_mapping.items()
    }
    assert tuple(model.preload_weight_modules) == original_preloads
    assert model.state_dict().keys() == original_state.keys()
    assert all(
        torch.equal(value, original_state[name]) for name, value in model.state_dict().items()
    )


def test_base_mapper_uses_opaque_order_for_unmodeled_source_aliases() -> None:
    catalog = _catalog_with_names(
        # Some mappers strip source-only namespaces before materialization.
        # Generic planning must degrade conservatively until such aliases are
        # expressed through a pure mapper planning hook.
        "language_model.layer.manual.weight",
        "language_model.layer.q_proj.weight",
    )
    model_config = ModelConfig(mapping=Mapping(world_size=2, rank=1))
    model = _StructuredModel(model_config)
    mapper = _StructuredMapper()
    mapper.init_model_and_config(model, model_config)

    plan = mapper.build_weight_load_plan(catalog)

    assert plan.coverage is WeightLoadPlanCoverage.CONSERVATIVE
    assert plan.ordering is WeightLoadOrderConfidence.OPAQUE
    assert plan.selected_tensor_names is None
    assert len(plan.demands) == 1
    assert plan.demands[0].group_id == "all_checkpoint_tensors"
    assert plan.demands[0].source_names == (
        "language_model.layer.manual.weight",
        "language_model.layer.q_proj.weight",
    )


def test_base_mapper_normalizes_moe_backend_to_preloaded_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.checkpoints.base_weight_mapper.is_moe_weight_owner",
        lambda module: isinstance(module, _MoeBackendModule),
    )
    catalog = _catalog_with_names(
        "tail.first",
        "layer.experts.weight_scale",
        "layer.experts.weight",
        "tail.second",
    )
    model_config = ModelConfig(mapping=Mapping(world_size=2, rank=1))
    model = _MoePathModel(model_config)
    mapper = _Mapper()
    mapper.init_model_and_config(model, model_config)
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    plan = mapper.build_weight_load_plan(catalog)

    assert plan.coverage is WeightLoadPlanCoverage.CONSERVATIVE
    assert plan.ordering is WeightLoadOrderConfidence.ADVISORY
    assert plan.selected_tensor_names is None
    assert plan.demands[0].group_id.endswith("layer.experts.backend")
    assert plan.demands[0].priority == 0
    assert plan.demands[0].source_names == (
        "layer.experts.weight_scale",
        "layer.experts.weight",
    )
    assert plan.demands[-1].source_names == (
        "tail.first",
        "tail.second",
    )
    described_names = [name for demand in plan.demands for name in demand.source_names]
    assert len(described_names) == len(set(described_names))
    assert set(described_names) == catalog.tensor_names
    assert all(
        torch.equal(value, original_state[name]) for name, value in model.state_dict().items()
    )


def test_base_mapper_uses_indexed_structural_prefix_matching() -> None:
    module_count = 128
    catalog = _catalog_with_names(
        *(_CountingName(f"blocks.{index}.weight") for index in range(module_count)),
        _CountingName("unmatched.tail"),
    )
    model_config = ModelConfig(mapping=Mapping(world_size=2, rank=1))
    model = _ManyCustomModulesModel(model_config, module_count)
    mapper = _Mapper()
    mapper.init_model_and_config(model, model_config)
    _CountingName.startswith_calls = 0

    plan = mapper.build_weight_load_plan(catalog)

    # A full physical-name scan per custom module would make roughly
    # module_count**2 startswith calls. Bisected ranges inspect only matching
    # names plus each range boundary.
    assert _CountingName.startswith_calls < module_count * 4
    assert plan.coverage is WeightLoadPlanCoverage.CONSERVATIVE
    assert plan.ordering is WeightLoadOrderConfidence.ADVISORY
    assert plan.selected_tensor_names is None
    assert plan.demands[-1].source_names == ("unmatched.tail",)
    assert plan.described_tensor_names == catalog.tensor_names
