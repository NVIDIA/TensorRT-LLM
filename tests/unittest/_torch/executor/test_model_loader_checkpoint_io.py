# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Literal
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointTensor,
)
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightDemand,
    WeightLoadOrderConfidence,
    WeightLoadPlan,
    WeightLoadPlanCoverage,
)
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
from tensorrt_llm._torch.pyexecutor.model_loader import (
    ModelLoader,
    _construct_checkpoint_loader,
    _open_checkpoint_weight_session,
    _timed_checkpoint_weight_session,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("loader_cls", [MistralCheckpointLoader, MXCheckpointLoader])
def test_hf_subclasses_keep_polymorphic_load_weights(
    loader_cls: type[MistralCheckpointLoader] | type[MXCheckpointLoader],
) -> None:
    checkpoint_loader = loader_cls.__new__(loader_cls)
    checkpoint_loader.load_weights = MagicMock(return_value={"subclass.weight": object()})
    mapping = object()

    with checkpoint_loader.open_weight_session(
        "/checkpoint", mapping=mapping, model="model"
    ) as weights:
        assert list(weights) == ["subclass.weight"]

    checkpoint_loader.load_weights.assert_called_once_with(
        "/checkpoint", mapping=mapping, model="model"
    )


@pytest.mark.parametrize("requested", ["auto", "rank_striped_read_ahead"])
def test_construct_checkpoint_loader_selects_rank_striped_for_builtin_hf(
    requested: Literal["auto", "rank_striped_read_ahead"],
) -> None:
    loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy=requested,
    )

    assert isinstance(loader.weight_loader, HfWeightLoader)
    assert loader.weight_loader.checkpoint_io_policy == "rank_striped_read_ahead"
    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == requested
    assert status.selected == "rank_striped_read_ahead"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"checkpoint_format": "MX"}, "checkpoint_format='HF'"),
        ({"load_format": "dummy"}, "load_format='auto'"),
        ({"partial_model_loading": True}, "partial model loading"),
    ],
)
def test_construct_checkpoint_loader_falls_back_before_incompatible_request(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    reason: str,
) -> None:
    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    options = dict(kwargs)
    checkpoint_format = options.pop("checkpoint_format", "HF")

    loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        checkpoint_format,
        checkpoint_io_policy="rank_striped_read_ahead",
        **options,
    )

    assert loader.weight_loader.checkpoint_io_policy == "native"
    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == "rank_striped_read_ahead"
    assert status.selected == "native"
    assert reason in status.fallback_reason
    warning.assert_called_once()
    assert "selected=native" in warning.call_args.args[0]


def test_construct_checkpoint_loader_preserves_explicit_loader_on_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provided_loader = HfCheckpointLoader(
        weight_loader=HfWeightLoader(checkpoint_io_policy="rank_striped_read_ahead")
    )
    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)

    loader = _construct_checkpoint_loader(
        "pytorch",
        provided_loader,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
    )

    assert loader is provided_loader
    assert provided_loader.weight_loader.checkpoint_io_policy == "native"
    status = provided_loader.weight_loader.last_checkpoint_io_status
    assert status.requested == "rank_striped_read_ahead"
    assert status.selected == "native"
    assert "explicit checkpoint loader" in status.fallback_reason
    warning.assert_called_once()
    assert "explicit checkpoint loader" in warning.call_args.args[0]

    native_weights = {"native": object()}
    native_load = MagicMock(return_value=native_weights)
    active_communicator = MagicMock(side_effect=AssertionError("rank-striped setup must not run"))
    monkeypatch.setattr(provided_loader.weight_loader, "_load_weights_native", native_load)
    monkeypatch.setattr(provided_loader.weight_loader, "_active_communicator", active_communicator)
    with loader.open_weight_session("/checkpoint", mapping=Mapping()) as weights:
        assert weights is native_weights
    active_communicator.assert_not_called()


def test_auto_selects_native_for_incompatible_config_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    monkeypatch.setattr(model_loader_module.logger, "info", info)

    loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "MX",
        checkpoint_io_policy="auto",
    )

    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == "auto"
    assert status.selected == "native"
    warning.assert_not_called()
    assert any("selected=native" in call.args[0] for call in info.call_args_list)


def test_static_native_selection_skips_rank_striped_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
        partial_model_loading=True,
    )
    native_weights = {"native": object()}
    native_load = MagicMock(return_value=native_weights)
    active_communicator = MagicMock(side_effect=AssertionError("rank-striped setup must not run"))
    monkeypatch.setattr(loader.weight_loader, "_load_weights_native", native_load)
    monkeypatch.setattr(loader.weight_loader, "_active_communicator", active_communicator)

    weights = loader.weight_loader.load_weights("/checkpoint", mapping=Mapping())

    assert weights is native_weights
    native_load.assert_called_once()
    active_communicator.assert_not_called()


def test_construct_checkpoint_loader_preserves_custom_hf_weight_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CustomWeightLoader:
        def __init__(self) -> None:
            self.load_weights = MagicMock(return_value={"custom.weight": object()})

    monkeypatch.setattr(
        modeling_utils, "get_checkpoint_weight_loader", lambda _format: _CustomWeightLoader
    )

    loader = _construct_checkpoint_loader("pytorch", None, "HF")
    assert isinstance(loader.weight_loader, _CustomWeightLoader)
    mapping = Mapping()
    with loader.open_weight_session("/checkpoint", mapping=mapping) as weights:
        assert list(weights) == ["custom.weight"]
    loader.weight_loader.load_weights.assert_called_once_with("/checkpoint", mapping=mapping)

    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    fallback_loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
    )
    assert isinstance(fallback_loader.weight_loader, _CustomWeightLoader)
    warning.assert_called_once()
    assert "selected=native" in warning.call_args.args[0]


def test_construct_checkpoint_loader_detects_custom_hf_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CustomCheckpointLoader(HfCheckpointLoader):
        pass

    monkeypatch.setitem(
        modeling_utils.CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING,
        "HF",
        _CustomCheckpointLoader,
    )
    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)

    loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
    )

    assert isinstance(loader, _CustomCheckpointLoader)
    assert loader.weight_loader.checkpoint_io_policy == "native"
    warning.assert_called_once()
    assert "checkpoint loader is not the built-in" in warning.call_args.args[0]


class _SessionCheckpointLoader:
    checkpoint_format = "HF"

    def __init__(
        self,
        events: list[str],
        checkpoint_dir: str = "/checkpoint",
        weights: dict[str, object] | None = None,
    ) -> None:
        self.events = events
        self.checkpoint_dir = checkpoint_dir
        self.weights = {"model.weight": object()} if weights is None else weights

    @contextmanager
    def open_weight_session(
        self, checkpoint_dir: str, **kwargs: Any
    ) -> Iterator[dict[str, object]]:
        assert checkpoint_dir == self.checkpoint_dir
        assert "mapping" in kwargs
        self.events.append("session_enter")
        try:
            yield self.weights
        finally:
            self.events.append("session_exit")

    def is_weights_preloaded(self) -> bool:
        return False

    def get_initialized_weight_mapper(self, model: object, config: object) -> object:
        del model, config
        self.events.append("mapper_init")
        return object()


def test_legacy_weight_session_defers_load_until_enter() -> None:
    weights = {"model.weight": object()}
    mapping = object()
    checkpoint_loader = SimpleNamespace(load_weights=MagicMock(return_value=weights))

    session = _open_checkpoint_weight_session(checkpoint_loader, "/checkpoint", mapping=mapping)

    checkpoint_loader.load_weights.assert_not_called()
    with session as loaded_weights:
        assert loaded_weights is weights
    checkpoint_loader.load_weights.assert_called_once_with("/checkpoint", mapping=mapping)


def test_checkpoint_session_times_preparation_and_finalization_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    metrics = {}
    checkpoint_loader = _SessionCheckpointLoader(events)

    @contextmanager
    def record_timing(metric_name: str, metric_values: dict[str, float]) -> Iterator[None]:
        events.append(f"{metric_name}_start")
        try:
            yield
        finally:
            metric_values[metric_name] = 1.0
            events.append(f"{metric_name}_end")

    monkeypatch.setattr(model_loader_module, "timing_metric", record_timing)

    with _timed_checkpoint_weight_session(
        checkpoint_loader,
        "/checkpoint",
        metrics,
        "preparation",
        "finalization",
        mapping=object(),
    ):
        events.append("materialize")

    assert events == [
        "preparation_start",
        "session_enter",
        "preparation_end",
        "materialize",
        "finalization_start",
        "session_exit",
        "finalization_end",
    ]
    assert metrics == {"preparation": 1.0, "finalization": 1.0}


def test_model_loader_session_spans_mapper_and_materialization() -> None:
    events = []
    model = MagicMock()
    checkpoint_loader = _SessionCheckpointLoader(events)
    loader = ModelLoader.__new__(ModelLoader)
    loader._metrics = {}
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )
    weights_preloaded = loader._materialize_checkpoint_weights(
        checkpoint_loader,
        "/checkpoint",
        model,
        object(),
        {"mapping": object()},
    )

    assert not weights_preloaded
    assert events == [
        "session_enter",
        "mapper_init",
        "materialize",
        "session_exit",
    ]
    assert "checkpoint_preparation_seconds" in loader.metrics
    assert "weight_population_seconds" in loader.metrics
    assert "checkpoint_finalization_seconds" in loader.metrics


def test_shadow_plan_disabled_does_not_call_inspection_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", raising=False)
    checkpoint_loader = SimpleNamespace(build_checkpoint_catalog=MagicMock())
    weight_mapper = SimpleNamespace(build_weight_load_plan=MagicMock())

    result = model_loader_module._inspect_shadow_weight_load_plan(
        checkpoint_loader, "/checkpoint", weight_mapper, mapping=object()
    )

    assert result is None
    checkpoint_loader.build_checkpoint_catalog.assert_not_called()
    weight_mapper.build_weight_load_plan.assert_not_called()


def test_shadow_plan_is_advisory_and_preserves_materialization_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "1")
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("model.weight"),))
    plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.OPAQUE,
        demands=(WeightDemand("all", ("model.weight",), (0,)),),
    )
    events = []
    checkpoint_loader = _SessionCheckpointLoader(events)
    checkpoint_loader.build_checkpoint_catalog = MagicMock(
        side_effect=lambda *_args, **_kwargs: (events.append("catalog") or catalog)
    )
    weight_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(
            side_effect=lambda _catalog: (events.append("plan") or plan)
        )
    )
    checkpoint_loader.get_initialized_weight_mapper = MagicMock(
        side_effect=lambda *_args: (events.append("mapper_init") or weight_mapper)
    )
    model = MagicMock()
    loader = ModelLoader.__new__(ModelLoader)
    loader._metrics = {}
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )

    loader._materialize_checkpoint_weights(
        checkpoint_loader,
        "/checkpoint",
        model,
        object(),
        {"mapping": object()},
    )

    assert events == [
        "session_enter",
        "mapper_init",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    loader._call_load_weights.assert_called_once()


@pytest.mark.parametrize("failure_site", ["catalog", "plan"])
def test_shadow_plan_failure_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "true")
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("model.weight"),))
    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    checkpoint_loader = SimpleNamespace(
        build_checkpoint_catalog=(
            MagicMock(side_effect=ValueError("catalog failure"))
            if failure_site == "catalog"
            else MagicMock(return_value=catalog)
        )
    )
    weight_mapper = SimpleNamespace(
        build_weight_load_plan=(
            MagicMock(side_effect=ValueError("plan failure"))
            if failure_site == "plan"
            else MagicMock(side_effect=AssertionError("plan must not be called"))
        )
    )

    result = model_loader_module._inspect_shadow_weight_load_plan(
        checkpoint_loader, "/checkpoint", weight_mapper
    )

    assert result is None
    warning.assert_called_once()
    assert "continuing with the unchanged loader" in warning.call_args.args[0]
    if failure_site == "catalog":
        weight_mapper.build_weight_load_plan.assert_not_called()


def test_draft_session_spans_mapper_and_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "1")
    events = []
    checkpoint_loader = _SessionCheckpointLoader(events, "/draft")
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("draft.weight"),))
    plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.OPAQUE,
        demands=(WeightDemand("all", ("draft.weight",), (0,)),),
    )
    checkpoint_loader.build_checkpoint_catalog = MagicMock(
        side_effect=lambda *_args, **_kwargs: (events.append("catalog") or catalog)
    )
    model = MagicMock()
    model.draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["DraftForCausalLM"])
    )
    loader = ModelLoader.__new__(ModelLoader)
    loader._metrics = {}
    loader.spec_config = SimpleNamespace(speculative_model="/draft")
    loader.mapping = object()
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )
    draft_mapper = MagicMock()
    draft_mapper.init_model_and_config.side_effect = lambda *_args: events.append("mapper_init")
    draft_mapper.build_weight_load_plan.side_effect = lambda _catalog: (
        events.append("plan") or plan
    )
    monkeypatch.setattr(
        model_loader_module.AutoCheckpointMapper, "get", MagicMock(return_value=draft_mapper)
    )

    loader._materialize_draft_checkpoint_weights(checkpoint_loader, model)

    assert events == [
        "session_enter",
        "mapper_init",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    assert "draft_checkpoint_preparation_seconds" in loader.metrics
    assert "draft_weight_population_seconds" in loader.metrics
    assert "draft_checkpoint_finalization_seconds" in loader.metrics


def test_mtp_draft_session_reuses_target_mapper_during_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "1")
    events = []
    checkpoint_loader = _SessionCheckpointLoader(events, "/draft")
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("draft.weight"),))
    plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.OPAQUE,
        demands=(WeightDemand("all", ("draft.weight",), (0,)),),
    )
    checkpoint_loader.build_checkpoint_catalog = MagicMock(
        side_effect=lambda *_args, **_kwargs: (events.append("catalog") or catalog)
    )
    model = MagicMock()
    model.draft_config = None
    loader = ModelLoader.__new__(ModelLoader)
    loader._metrics = {}
    loader.spec_config = SimpleNamespace(speculative_model="/draft")
    loader.mapping = object()
    loader.weight_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(
            side_effect=lambda _catalog: (events.append("plan") or plan)
        )
    )
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )

    loader._materialize_draft_checkpoint_weights(checkpoint_loader, model)

    assert loader._call_load_weights.call_args.args[2] is loader.weight_mapper
    assert events == [
        "session_enter",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    assert "draft_checkpoint_preparation_seconds" in loader.metrics
    assert "draft_weight_population_seconds" in loader.metrics
    assert "draft_checkpoint_finalization_seconds" in loader.metrics
