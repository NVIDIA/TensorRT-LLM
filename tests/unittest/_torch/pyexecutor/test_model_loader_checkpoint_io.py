# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import sys
from contextlib import contextmanager, nullcontext
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import ANY, MagicMock

import pytest
import torch

from tensorrt_llm._torch.distributed import initialization as distributed_initialization
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints import base_checkpoint_loader, memory_init, model_hooks
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointTensor,
)
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import (
    HfCheckpointLoader,
    checkpoint_io_policy_metadata,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.model_streamer import (
    model_loading as model_streamer_loading,
)
from tensorrt_llm._torch.models.checkpoints.model_streamer.model_loading import (
    ModelStreamerMetricNames,
    ModelStreamerPlan,
    _gather_node_tensor_names,
    _get_weight_source_map,
    _run_model_streamer,
    _stream_batched_tensors,
    check_model_streamer_support,
    load_weights_with_model_streamer,
)
from tensorrt_llm._torch.models.checkpoints.mx import model_loading as mx_loading
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
    _checkpoint_startup_metadata,
    _load_checkpoint_into_model,
    _resolve_checkpoint_io_policy,
    construct_checkpoint_loader,
    load_model_from_auto_load_format,
)
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy, LoadFormat
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_checkpoint_startup_metadata_describes_actual_loader() -> None:
    checkpoint_loader = HfCheckpointLoader()

    metadata = _checkpoint_startup_metadata(checkpoint_loader, LoadFormat.AUTO)

    assert metadata == {
        "checkpoint_loader_kind": "HfCheckpointLoader",
        "checkpoint_weight_loader_kind": "HfWeightLoader",
        "checkpoint_source_kind": "HF",
        "load_format": "auto",
        "checkpoint_io_policy": None,
    }


@pytest.mark.parametrize("outcome", ["native", "activated", "fallback", "degraded"])
def test_model_loader_snapshots_final_policy_before_draft_reset(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    requested = "native" if outcome == "native" else "auto"
    checkpoint_loader = construct_checkpoint_loader(None, "HF", checkpoint_io_policy=requested)
    weight_loader = checkpoint_loader.weight_loader
    weights = {"model.weight": object()}
    monkeypatch.setattr(weight_loader, "_load_weights_native", MagicMock(return_value=weights))
    monkeypatch.setattr(weight_loader, "_active_communicator", lambda: None)
    monkeypatch.setattr(checkpoint_loader, "get_initialized_weight_mapper", MagicMock())
    monkeypatch.setattr(model_loader_module.torch.cuda, "is_available", lambda: False)
    _record_weight_copies(monkeypatch, [])

    startup_metadata = _checkpoint_startup_metadata(checkpoint_loader, LoadFormat.AUTO)
    fallback_reason = "read-ahead failure" if outcome in ("fallback", "degraded") else None
    activated = outcome in ("activated", "degraded")

    def finish(_body_error: BaseException | None) -> RuntimeError | None:
        assert startup_metadata["checkpoint_io_policy"] is None
        return RuntimeError(fallback_reason) if fallback_reason else None

    session = SimpleNamespace(finish=MagicMock(side_effect=finish))

    def start_read_ahead(*_args: Any, **_kwargs: Any) -> tuple[dict[str, object], Any]:
        status = weight_loader._last_checkpoint_io_status
        if outcome == "fallback":
            status.selected = "native"
            status.effective = "native"
            status.fallback_reason = fallback_reason
            return weights, None
        status.activated = True
        return weights, session

    monkeypatch.setattr(weight_loader, "_start_rank_striped_read_ahead", start_read_ahead)
    expected = {
        "requested": requested,
        "selected": "rank_striped_read_ahead" if activated else "native",
        "activated": activated,
        "effective": "rank_striped_read_ahead" if outcome == "activated" else "native",
        "fallback_reason": fallback_reason,
    }

    def get_draft_weight_mapper(*_args: Any) -> MagicMock:
        # Called between the target and the draft read: the target policy is
        # already recorded, and the draft read is made native.
        assert startup_metadata["checkpoint_io_policy"] == expected
        weight_loader._configure_checkpoint_io_policy("native")
        return MagicMock()

    monkeypatch.setattr(
        checkpoint_loader, "get_initialized_draft_weight_mapper", get_draft_weight_mapper
    )

    load_model_from_auto_load_format(
        checkpoint_loader,
        "/checkpoint",
        model_config=object(),
        model=SimpleNamespace(load_weights=object(), load_draft_weights=object()),
        mapping=Mapping(),
        llm_args=None,
        spec_config=SimpleNamespace(needs_separate_draft_weights=True, speculative_model="/draft"),
        source_identity=None,
        qualification=None,
        metrics={},
        startup_metadata=startup_metadata,
    )

    assert session.finish.call_count == int(activated)
    assert startup_metadata["checkpoint_io_policy"] == expected
    assert startup_metadata["draft_checkpoint_io_policy"] == {
        "requested": "native",
        "selected": "native",
        "activated": False,
        "effective": "native",
        "fallback_reason": None,
    }


@pytest.mark.parametrize("checkpoint_format", ["mistral", "MX"])
@pytest.mark.parametrize("requested", ["native", "auto", "rank_striped_read_ahead"])
def test_hf_subclass_disk_loads_report_final_native_policy(
    monkeypatch: pytest.MonkeyPatch, checkpoint_format: str, requested: str
) -> None:
    checkpoint_loader = construct_checkpoint_loader(
        None, checkpoint_format, checkpoint_io_policy=requested
    )
    weight_loader = checkpoint_loader.weight_loader
    native_load = MagicMock(return_value={})
    assert checkpoint_io_policy_metadata(checkpoint_loader, weights_preloaded=False) is None
    monkeypatch.setattr(weight_loader, "_load_weights_native", native_load)
    _record_weight_copies(monkeypatch, [])
    fallback_reason = weight_loader.last_checkpoint_io_status.fallback_reason

    checkpoint_io_policy = _load_checkpoint_into_model(
        checkpoint_loader,
        "/checkpoint",
        model=SimpleNamespace(load_weights=object()),
        is_draft_model=False,
        mapping=Mapping(),
        weight_mapper=None,
        extra_load_weights_kwargs=None,
        metrics={},
    )

    native_load.assert_called_once()
    assert checkpoint_io_policy == {
        "requested": requested,
        "selected": "native",
        "activated": False,
        "effective": "native",
        "fallback_reason": fallback_reason,
    }
    if requested == "native":
        assert fallback_reason is None
    else:
        assert "checkpoint_format='HF'" in fallback_reason


@pytest.mark.parametrize("loader_kind", ["unfinalized", "custom", "mx", "preloaded"])
def test_unobserved_checkpoint_policy_is_not_reported_as_native(loader_kind: str) -> None:
    if loader_kind == "custom":
        checkpoint_loader = HfCheckpointLoader(weight_loader=MagicMock())
    elif loader_kind == "mx":
        checkpoint_loader = MXCheckpointLoader()
    else:
        checkpoint_loader = HfCheckpointLoader()
    weights_preloaded = loader_kind in ("mx", "preloaded")
    if weights_preloaded:
        checkpoint_loader.weight_loader._last_checkpoint_io_status.effective = "native"

    assert (
        checkpoint_io_policy_metadata(checkpoint_loader, weights_preloaded=weights_preloaded)
        is None
    )


@pytest.mark.parametrize("loader_cls", [MistralCheckpointLoader, MXCheckpointLoader])
def test_hf_subclasses_keep_polymorphic_load_weights(
    monkeypatch: pytest.MonkeyPatch,
    loader_cls: type[MistralCheckpointLoader] | type[MXCheckpointLoader],
) -> None:
    checkpoint_loader = loader_cls.__new__(loader_cls)
    checkpoint_loader._weight_loader = HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead"
    )
    weights = {"subclass.weight": object()}
    checkpoint_loader.load_weights = MagicMock(return_value=weights)
    reader_session = MagicMock(side_effect=AssertionError("subclasses must not read ahead"))
    monkeypatch.setattr(checkpoint_loader.weight_loader, "open_weight_session", reader_session)
    call_load_weights = _record_weight_copies(monkeypatch, [])
    model = SimpleNamespace(load_weights=object())
    mapping = object()
    metrics = {}

    _load_checkpoint_into_model(
        checkpoint_loader,
        "/checkpoint",
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=None,
        extra_load_weights_kwargs=None,
        metrics=metrics,
    )

    checkpoint_loader.load_weights.assert_called_once_with(
        "/checkpoint", mapping=mapping, model=model
    )
    assert call_load_weights.call_args.args[1] is weights
    reader_session.assert_not_called()
    # The read is timed as preparation, with nothing to finalize.
    assert metrics["checkpoint_finalization_seconds"] == 0.0


@pytest.mark.parametrize("policy", ["native", "rank_striped_read_ahead"])
@pytest.mark.parametrize("via_model_loader", [False, True])
def test_context_only_loading_keeps_unused_checkpoint_tensors_lazy(
    monkeypatch: pytest.MonkeyPatch, policy: str, via_model_loader: bool
) -> None:
    loader = HfCheckpointLoader(weight_loader=HfWeightLoader(checkpoint_io_policy=policy))
    lazy_weights = {"layers.0.weight": object(), "layers.39.weight": object()}
    lazy_load = MagicMock(return_value=lazy_weights)
    eager_load = MagicMock(side_effect=AssertionError("must not read unused decoder tensors"))
    reader_session = MagicMock(side_effect=AssertionError("must not prefetch decoder tensors"))
    monkeypatch.setattr(loader.weight_loader, "_load_lazy_safetensors", lazy_load)
    monkeypatch.setattr(loader.weight_loader, "_load_weights_native", eager_load)
    monkeypatch.setattr(loader.weight_loader, "open_weight_session", reader_session)
    model = SimpleNamespace(model=SimpleNamespace(disagg_context_only=True), load_weights=object())
    options = dict(mapping=Mapping(), model=model, use_consolidated=True)

    if via_model_loader:
        # A context-only worker reads natively even when read-ahead is selected.
        call_load_weights = _record_weight_copies(monkeypatch, [])
        _load_checkpoint_into_model(
            loader,
            "/checkpoint",
            model=model,
            is_draft_model=False,
            mapping=options["mapping"],
            weight_mapper=None,
            extra_load_weights_kwargs={"use_consolidated": True},
            metrics={},
        )
        assert call_load_weights.call_args.args[1] is lazy_weights
    else:
        assert loader.load_weights("/checkpoint", **options) is lazy_weights

    lazy_load.assert_called_once_with("/checkpoint", True)
    eager_load.assert_not_called()
    reader_session.assert_not_called()
    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == policy
    assert status.selected == status.effective == "native"
    assert not status.activated
    if policy != "native":
        assert "context-only" in status.fallback_reason


@pytest.mark.parametrize("context_only", [False, None])
def test_full_model_loading_retains_existing_checkpoint_reader(
    monkeypatch: pytest.MonkeyPatch, context_only: bool | None
) -> None:
    loader = HfCheckpointLoader()
    expected = {"decoder.weight": object()}
    eager_load = MagicMock(return_value=expected)
    lazy_load = MagicMock(side_effect=AssertionError("must keep normal checkpoint loading"))
    monkeypatch.setattr(loader.weight_loader, "_load_weights_native", eager_load)
    monkeypatch.setattr(loader.weight_loader, "_load_lazy_safetensors", lazy_load)
    body = SimpleNamespace()
    if context_only is not None:
        body.disagg_context_only = context_only
    model = SimpleNamespace(model=body, load_weights=object())
    mapping = Mapping()
    call_load_weights = _record_weight_copies(monkeypatch, [])
    metrics = {}

    _load_checkpoint_into_model(
        loader,
        "/checkpoint",
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=None,
        extra_load_weights_kwargs=None,
        metrics=metrics,
    )

    assert call_load_weights.call_args.args[1] is expected
    eager_load.assert_called_once_with("/checkpoint", mapping, False, model=model)
    lazy_load.assert_not_called()
    # A native read is timed as preparation, with nothing to finalize.
    assert metrics["checkpoint_preparation_seconds"] > 0
    assert metrics["checkpoint_finalization_seconds"] == 0.0


@pytest.mark.parametrize("custom_wrapper", [False, True])
def test_context_only_flag_preserves_custom_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch, custom_wrapper: bool
) -> None:
    class _CustomCheckpointLoader(HfCheckpointLoader):
        pass

    class _CustomWeightLoader(HfWeightLoader):
        pass

    loader = (
        _CustomCheckpointLoader()
        if custom_wrapper
        else HfCheckpointLoader(weight_loader=_CustomWeightLoader())
    )
    expected = {"custom.weight": object()}
    custom_load = MagicMock(return_value=expected)
    lazy_load = MagicMock(side_effect=AssertionError("must not replace a custom loader"))
    monkeypatch.setattr(loader.weight_loader, "load_weights", custom_load)
    monkeypatch.setattr(loader.weight_loader, "_load_lazy_safetensors", lazy_load)
    model = SimpleNamespace(model=SimpleNamespace(disagg_context_only=True), load_weights=object())
    mapping = Mapping()
    call_load_weights = _record_weight_copies(monkeypatch, [])

    _load_checkpoint_into_model(
        loader,
        "/checkpoint",
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=None,
        extra_load_weights_kwargs=None,
        metrics={},
    )

    assert call_load_weights.call_args.args[1] is expected
    custom_load.assert_called_once_with("/checkpoint", mapping=mapping, model=model)
    lazy_load.assert_not_called()


@pytest.mark.parametrize(
    "requested", [CheckpointIOPolicy.AUTO, CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD]
)
def test_construct_checkpoint_loader_selects_rank_striped_for_builtin_hf(
    requested: CheckpointIOPolicy,
) -> None:
    loader = construct_checkpoint_loader(
        None,
        "HF",
        checkpoint_io_policy=requested,
    )

    assert isinstance(loader.weight_loader, HfWeightLoader)
    assert loader.weight_loader.checkpoint_io_policy == "rank_striped_read_ahead"
    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == requested
    assert status.selected == "rank_striped_read_ahead"


@pytest.mark.parametrize("requested", list(CheckpointIOPolicy))
def test_resolve_checkpoint_io_policy_returns_enum(requested: CheckpointIOPolicy) -> None:
    selected, reason = _resolve_checkpoint_io_policy(None, "HF", LoadFormat.AUTO, requested, False)
    expected = (
        CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD
        if requested is CheckpointIOPolicy.AUTO
        else requested
    )
    assert selected is expected
    assert reason is None


@pytest.mark.parametrize(
    "policy", [CheckpointIOPolicy.NATIVE, CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD]
)
def test_hf_reader_normalizes_policy_strings(policy: CheckpointIOPolicy) -> None:
    loader = HfWeightLoader(
        checkpoint_io_policy=policy.value, requested_checkpoint_io_policy="auto"
    )
    assert loader.checkpoint_io_policy is policy
    assert loader.last_checkpoint_io_status.requested is CheckpointIOPolicy.AUTO
    assert loader.last_checkpoint_io_status.selected is policy


def test_hf_reader_policy_error_shows_plain_values() -> None:
    with pytest.raises(
        ValueError, match=r"one of \('native', 'rank_striped_read_ahead'\), got 'auto'$"
    ):
        HfWeightLoader(checkpoint_io_policy=CheckpointIOPolicy.AUTO)


def test_model_streamer_policy_keeps_hf_reader_idle() -> None:
    loader = construct_checkpoint_loader(
        None,
        "HF",
        load_format=LoadFormat.AUTO,
        checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER,
    )
    assert type(loader) is HfCheckpointLoader
    assert loader.checkpoint_format == CheckpointFormat.HF
    assert type(loader.weight_loader) is HfWeightLoader
    assert loader.weight_loader.checkpoint_io_policy == "native"


class _StreamingModel:
    """Stands in for a model that plans its own Model Streamer load."""

    def __init__(self, plan: ModelStreamerPlan | None = None) -> None:
        self.plan = plan or ModelStreamerPlan(groups=[({"weight"}, lambda tensors: None)])
        self.planned_keys: list[set[str]] = []

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(())

    def plan_model_streamer_load(self, checkpoint_keys: set[str]) -> ModelStreamerPlan:
        self.planned_keys.append(checkpoint_keys)
        return self.plan


def test_model_streamer_streams_planned_tensors_into_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamed = []

    def run_model_streamer(
        model: _StreamingModel, checkpoint_dir: str, *, rank: int, world_size: int
    ) -> dict[str, float | int]:
        streamed.append((model, checkpoint_dir, rank, world_size))
        return {ModelStreamerMetricNames.SECONDS: 2.0, ModelStreamerMetricNames.BATCHES: 3}

    monkeypatch.setattr(model_streamer_loading, "_run_model_streamer", run_model_streamer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    logger = MagicMock(name="logger")
    monkeypatch.setattr(model_streamer_loading, "logger", logger)
    model = _StreamingModel()

    load_weights_with_model_streamer(model, "/checkpoint", rank=3, world_size=4)

    assert streamed == [(model, "/checkpoint", 3, 4)]
    # Times show four decimals, and counts show as integers.
    logger.info.assert_called_once_with(
        "Model Streamer metrics for rank 3: model_streamer_seconds=2.0000, model_streamer_batches=3"
    )


def test_model_streamer_streams_target_and_reads_draft_with_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A separate draft checkpoint, e.g. a DSpark drafter, is read as usual."""
    events = []
    target_mapper = object()
    checkpoint_loader = _RecordingCheckpointLoader(events, target_weight_mapper=target_mapper)
    _record_weight_copies(monkeypatch, events)
    streamed = []
    monkeypatch.setattr(
        model_streamer_loading,
        "load_weights_with_model_streamer",
        lambda model, checkpoint_dir, *, rank, world_size: streamed.append(
            (model, checkpoint_dir, rank, world_size)
        ),
    )
    model = SimpleNamespace(load_weights=object(), load_draft_weights=object(), draft_config=None)
    metrics = {}

    weight_mapper, weights_preloaded = load_model_from_auto_load_format(
        checkpoint_loader,
        "/checkpoint",
        model_config=object(),
        model=model,
        mapping=Mapping(world_size=4, tp_size=4, rank=3),
        llm_args=SimpleNamespace(checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER),
        spec_config=SimpleNamespace(needs_separate_draft_weights=True, speculative_model="/draft"),
        source_identity=None,
        qualification=None,
        metrics=metrics,
        startup_metadata={},
    )

    assert streamed == [(model, "/checkpoint", 3, 4)]
    assert weights_preloaded is True
    # Streaming does not use the weight mapper, but reload() and weight updates do.
    assert weight_mapper is target_mapper
    # Only the draft is read with the checkpoint loader.
    assert events == ["mapper_init", "read:/draft", "materialize"]
    assert "weight_population_seconds" in metrics
    assert "checkpoint_preparation_seconds" not in metrics


def test_model_streamer_rejects_models_that_cannot_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="object does not support .* plan_model_streamer_load"):
        check_model_streamer_support(object())

    model = _StreamingModel()
    monkeypatch.setattr(model_streamer_loading, "version", lambda _: "0.17.1")
    check_model_streamer_support(model)
    # 0.16 has no tensor_names or owned argument in stream_files().
    monkeypatch.setattr(model_streamer_loading, "version", lambda _: "0.16.1")
    with pytest.raises(ImportError, match=r">=0.17.0, but found 0.16.1"):
        check_model_streamer_support(model)

    def not_installed(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(model_streamer_loading, "version", not_installed)
    with pytest.raises(ImportError, match="it is not installed"):
        check_model_streamer_support(model)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"checkpoint_format": "MX"}, "checkpoint_format='HF'"),
        ({"load_format": LoadFormat.DUMMY}, "load_format='auto'"),
        ({"partial_model_loading": True}, "partial model loading"),
        ({"checkpoint_loader": object()}, "explicit checkpoint loader"),
    ],
)
def test_model_streamer_policy_rejects_incompatible_config(
    kwargs: dict[str, Any], reason: str
) -> None:
    options = dict(checkpoint_loader=None, checkpoint_format="HF")
    options.update(kwargs)
    with pytest.raises(ValueError, match=reason):
        construct_checkpoint_loader(
            **options, checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER
        )


@pytest.mark.parametrize("custom_weight_loader", [False, True])
def test_model_streamer_policy_rejects_custom_hf_registration(
    monkeypatch: pytest.MonkeyPatch, custom_weight_loader: bool
) -> None:
    if custom_weight_loader:
        monkeypatch.setattr(modeling_utils, "get_checkpoint_weight_loader", lambda _: object)
    else:
        monkeypatch.setitem(modeling_utils.CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING, "HF", object)
    with pytest.raises(ValueError, match="not the built-in loader"):
        construct_checkpoint_loader(
            None, "HF", checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER
        )


def _model_streamer_load(
    monkeypatch: pytest.MonkeyPatch, *, modules: tuple[Any, ...] = ()
) -> tuple[ModelLoader, BaseCheckpointLoader, MagicMock, list[str]]:
    """Prepare `ModelLoader.load()` for a Model Streamer load without CUDA or files.

    The model's plan names one tensor, and a fake streamer delivers it, so the
    load runs `_run_model_streamer()` for real. The returned list records the names
    that the fake streamer delivered.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    model = MagicMock(spec=KimiLinearForCausalLM)
    model.modules.return_value = list(modules)
    model.plan_model_streamer_load.return_value = ModelStreamerPlan(
        groups=[({"weight"}, MagicMock(name="load_weight"))]
    )
    loader = ModelLoader(
        llm_args=SimpleNamespace(
            load_format=LoadFormat.AUTO,
            lm_head_output_dtype="auto",
            checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER,
        ),
        mapping=Mapping(),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(
            pretrained_config=SimpleNamespace(
                architectures=["KimiLinearForCausalLM"], model_type="kimi_k3"
            )
        )
    )
    monkeypatch.setattr(memory_init, "MetaInitMode", nullcontext)
    monkeypatch.setattr(memory_init.AutoModelForCausalLM, "from_config", lambda _config: model)
    monkeypatch.setattr(memory_init, "get_rank_model_storage", lambda _model: 0)
    monkeypatch.setattr(
        model_loader_module, "maybe_create_moe_load_balancer", lambda *_: nullcontext()
    )
    # The model is a mock, so skip matching it against the MX post-transform profiles.
    monkeypatch.setattr(
        model_loader_module.mx_loading,
        "qualify_post_transform_profile",
        lambda *_args, **_kwargs: SimpleNamespace(qualified=False),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # The CI containers do not install runai-model-streamer.
    monkeypatch.setattr(model_streamer_loading, "version", lambda _: "0.17.1")
    monkeypatch.setattr(
        model_streamer_loading,
        "_get_weight_source_map",
        lambda _: {"weight": "/checkpoint/model.safetensors"},
    )
    streamed = []

    def stream(
        weight_map: dict[str, str],
        batches: list[set[str]],
        consume: Callable[[str, torch.Tensor], None],
        *,
        distributed: bool,
    ) -> dict[str, float | int]:
        for batch in batches:
            for name in sorted(batch):
                streamed.append(name)
                consume(name, torch.ones(1))
        return {ModelStreamerMetricNames.SECONDS: 1.0}

    monkeypatch.setattr(model_streamer_loading, "_stream_batched_tensors", stream)
    checkpoint_loader = construct_checkpoint_loader(
        None, "HF", checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER
    )
    checkpoint_loader.get_initialized_weight_mapper = MagicMock()
    return loader, checkpoint_loader, model, streamed


def test_model_loader_streams_through_model_streamer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, checkpoint_loader, model, streamed = _model_streamer_load(monkeypatch)

    loaded_model, _ = loader.load("/checkpoint", checkpoint_loader)

    assert loaded_model is model
    model.plan_model_streamer_load.assert_called_once_with({"weight"})
    assert streamed == ["weight"]
    [(_, load_weight)] = model.plan_model_streamer_load.return_value.groups
    load_weight.assert_called_once()
    model.load_weights.assert_not_called()
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once()
    assert loader.weight_mapper is checkpoint_loader.get_initialized_weight_mapper.return_value
    # The startup metrics keep their keys; the Model Streamer metrics are only logged.
    assert not set(ModelStreamerMetricNames) & loader.metrics.keys()
    assert "weight_population_seconds" in loader.metrics
    assert loader.startup_metadata["checkpoint_loader_kind"] == "HfCheckpointLoader"
    assert loader.startup_metadata["load_format"] == "auto"


def test_model_loader_rejects_loads_model_streamer_cannot_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, checkpoint_loader, model, streamed = _model_streamer_load(
        monkeypatch, modules=(SimpleNamespace(_requires_regular_checkpoint_loading=True),)
    )

    with pytest.raises(ValueError, match="Host-resident"):
        loader.load("/checkpoint", checkpoint_loader)

    model.plan_model_streamer_load.assert_not_called()
    assert streamed == []


def test_weight_source_map_reads_the_index_or_the_shards(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    shard1, shard2 = str(tmp_path / "model-1.safetensors"), str(tmp_path / "model-2.safetensors")
    save_file({"a": torch.zeros(1), "b": torch.zeros(1)}, shard1)
    save_file({"c": torch.zeros(1)}, shard2)
    # As the native reader does, a consolidated file next to the shards is skipped.
    save_file({"a": torch.zeros(1)}, str(tmp_path / "consolidated.safetensors"))
    assert _get_weight_source_map(str(tmp_path)) == {"a": shard1, "b": shard1, "c": shard2}

    # The index is the only source when it exists.
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "model-2.safetensors"}}))
    assert _get_weight_source_map(str(tmp_path)) == {"a": shard2}

    # Without an index, two shards that hold one name are an error.
    index.unlink()
    save_file({"c": torch.zeros(1)}, str(tmp_path / "model-3.safetensors"))
    with pytest.raises(ValueError, match="Duplicate safetensors keys"):
        _get_weight_source_map(str(tmp_path))


@pytest.mark.parametrize("use_mpi", [True, False])
def test_gather_tensor_names_covers_different_expert_layers(
    monkeypatch: pytest.MonkeyPatch, use_mpi: bool
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(model_streamer_loading, "mpi_disabled", lambda: not use_mpi)
    monkeypatch.setattr(socket, "gethostname", lambda: "node0")
    # Another rank shares node0, and a third rank is alone on node1.
    peers = [("node0", ("layer1.expert3",), None), ("node1", ("layer2.expert5",), None)]

    class Comm:
        def allgather(self, local: tuple[str, tuple[str, ...], None]) -> list[Any]:
            assert use_mpi, "Ray workers have no MPI communicator"
            return [local, *peers]

    def gather(entries: list[Any], local: tuple[str, tuple[str, ...], None]) -> None:
        # Under MPI, a default-group collective would keep NCCL buffers alive.
        assert not use_mpi, "MPI launches must not gather on the default process group"
        entries[:] = [local, *peers]

    monkeypatch.setattr(model_streamer_loading, "mpi_comm", lambda: Comm())
    monkeypatch.setattr(dist, "all_gather_object", gather)
    selected, max_ranks_per_node, ranks_span_nodes = _gather_node_tensor_names(
        {"layer0.expert0"}, None, world_size=3
    )
    assert selected == {"layer0.expert0", "layer1.expert3"}
    # The rank alone on node1 gets the same values, so it joins Model Streamer's
    # group setup like the ranks of node0.
    assert max_ranks_per_node == 2
    assert ranks_span_nodes


def test_model_streamer_group_bootstraps_from_mpi_and_ends_with_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    from tensorrt_llm import _utils

    # Launcher values may describe an outer job; they are neither used nor changed.
    monkeypatch.setenv("MASTER_ADDR", "10.9.9.9")
    monkeypatch.setenv("MASTER_PORT", "1")
    monkeypatch.setattr(_utils, "mpi_rank", lambda: 1)
    monkeypatch.setattr(_utils, "mpi_world_size", lambda: 2)

    class Comm:
        def bcast(self, value: None, root: int) -> tuple[str, int, bool, None]:
            assert value is None and root == 0
            # Rank 0 hosts the store on the port that the OS gave it.
            return "10.0.0.1", 41234, False, None

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    store = MagicMock(name="store")
    tcp_store = MagicMock(return_value=store)
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    excepthook = sys.excepthook
    initialized = []

    def init_torch_distributed(**kwargs: Any) -> None:
        initialized.append(kwargs)
        # init_process_group() replaces the hook, and destroy_process_group() keeps it.
        sys.excepthook = MagicMock(name="rank_excepthook")

    monkeypatch.setattr(
        distributed_initialization, "init_torch_distributed", init_torch_distributed
    )
    destroyed = []
    monkeypatch.setattr(dist, "destroy_process_group", lambda *args: destroyed.append(args))

    with distributed_initialization.model_streamer_torch_distributed(1, 2):
        tcp_store.assert_called_once_with("10.0.0.1", 41234, 2, is_master=False, timeout=ANY)
        assert initialized == [{"backend": "nccl", "rank": 1, "world_size": 2, "store": store}]
        assert destroyed == []
    # The code that runs after the load sees no process group and the old hook.
    assert destroyed == [()]
    assert sys.excepthook is excepthook
    assert os.environ["MASTER_ADDR"] == "10.9.9.9"
    assert os.environ["MASTER_PORT"] == "1"


@pytest.mark.parametrize(
    ("host_address", "error"),
    [
        # Ranks on other nodes reach rank 0 through its routable address.
        ("10.0.0.7", None),
        # A hostname label longer than 63 characters cannot be encoded, so rank 0
        # hosts the store on loopback, which the ranks of its own node can reach.
        (UnicodeError("label too long"), None),
        # When rank 0 cannot host the store, it broadcasts the reason.
        ("10.0.0.7", "address already in use"),
    ],
)
def test_rank_zero_hosts_the_rendezvous_store(
    monkeypatch: pytest.MonkeyPatch, host_address: str | Exception, error: str | None
) -> None:
    import torch.distributed as dist

    from tensorrt_llm import _utils

    monkeypatch.setenv("MASTER_ADDR", "10.9.9.9")
    monkeypatch.setenv("MASTER_PORT", "1")
    monkeypatch.setattr(_utils, "mpi_rank", lambda: 0)
    monkeypatch.setattr(_utils, "mpi_world_size", lambda: 2)
    monkeypatch.setattr(socket, "gethostname", lambda: "node0")

    def gethostbyname(name: str) -> str:
        if isinstance(host_address, Exception):
            raise host_address
        return host_address

    monkeypatch.setattr(socket, "gethostbyname", gethostbyname)
    broadcast = []

    class Comm:
        def bcast(self, value: tuple[Any, ...], root: int) -> tuple[Any, ...]:
            broadcast.append(value)
            return value

        def allgather(self, local: str) -> list[str]:
            # Only a loopback address needs the hostnames; both ranks are on node0.
            return [local, local]

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    store = MagicMock(name="store", port=41234)
    tcp_store = MagicMock(
        return_value=store, side_effect=dist.DistNetworkError(error) if error else None
    )
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    initialized = []
    monkeypatch.setattr(
        distributed_initialization,
        "init_torch_distributed",
        lambda **kwargs: initialized.append(kwargs),
    )
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            with distributed_initialization.model_streamer_torch_distributed(0, 2):
                pass
        # Rank 0 broadcast the reason, so the other rank raises too.
        ((_, _, _, reason),) = broadcast
        assert error in reason and initialized == []
        return

    with distributed_initialization.model_streamer_torch_distributed(0, 2):
        pass
    address = "127.0.0.1" if isinstance(host_address, Exception) else host_address
    tcp_store.assert_called_once_with(
        address, 0, 2, is_master=True, timeout=ANY, wait_for_workers=False
    )
    assert broadcast == [(address, 41234, address == "127.0.0.1", None)]
    assert initialized == [{"backend": "nccl", "rank": 0, "world_size": 2, "store": store}]
    assert os.environ["MASTER_ADDR"] == "10.9.9.9"
    assert os.environ["MASTER_PORT"] == "1"


def test_model_streamer_keeps_an_existing_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    # A Ray worker or a MegaMoE bootstrap created this group, and it outlives the load.
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend_config", lambda: "cuda:nccl,cpu:gloo")
    monkeypatch.delenv("RUNAI_STREAMER_DIST", raising=False)
    initialize, destroy = MagicMock(), MagicMock()
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)

    with distributed_initialization.model_streamer_torch_distributed(1, 2):
        # Model Streamer shares reads over a backend other than "nccl" only on request.
        assert os.environ["RUNAI_STREAMER_DIST"] == "1"
    assert "RUNAI_STREAMER_DIST" not in os.environ
    initialize.assert_not_called()
    destroy.assert_not_called()

    # A value that the user set stays.
    monkeypatch.setenv("RUNAI_STREAMER_DIST", "0")
    with distributed_initialization.model_streamer_torch_distributed(1, 2):
        assert os.environ["RUNAI_STREAMER_DIST"] == "0"
    assert os.environ["RUNAI_STREAMER_DIST"] == "0"

    with pytest.raises(RuntimeError, match="does not match the TRT-LLM mapping"):
        with distributed_initialization.model_streamer_torch_distributed(0, 2):
            pass

    # A group that does not send CUDA tensors through NCCL cannot share reads.
    monkeypatch.setattr(dist, "get_backend_config", lambda: "cpu:gloo,cuda:gloo")
    with pytest.raises(RuntimeError, match="only works with nccl"):
        with distributed_initialization.model_streamer_torch_distributed(1, 2):
            pass


def test_stream_batches_share_one_session(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = []
    calls = []

    class Streamer:
        shares_reads = True

        def __enter__(self) -> "Streamer":
            sessions.append(self)
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def stream_files(
            self,
            paths: list[str],
            *,
            device: str,
            is_distributed: bool,
            tensor_names: set[str],
            owned: bool,
        ) -> None:
            calls.append((paths, device, is_distributed, set(tensor_names), owned))
            # Model Streamer sets this once and never removes it.
            os.environ.setdefault("RUNAI_STREAMER_PROCESS_GROUP_SIZE", "2")
            self.file_streamer = SimpleNamespace(
                is_distributed=is_distributed and self.shares_reads
            )
            self.names = sorted(tensor_names)

        def get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
            for name in self.names:
                yield name, torch.zeros(2)

    monkeypatch.setitem(
        sys.modules, "runai_model_streamer", SimpleNamespace(SafetensorsStreamer=Streamer)
    )
    monkeypatch.delenv("RUNAI_STREAMER_PROCESS_GROUP_SIZE", raising=False)
    weight_map = {
        "a": "/ckpt/1.safetensors",
        "b": "/ckpt/2.safetensors",
        "c": "/ckpt/2.safetensors",
    }
    consumed = []
    metrics = _stream_batched_tensors(
        weight_map,
        [{"a"}, {"b", "c"}],
        lambda name, tensor: consumed.append(name),
        distributed=True,
    )
    assert len(sessions) == 1
    # Each batch reads only the shards that hold its tensors.
    assert calls == [
        (["/ckpt/1.safetensors"], "cuda", True, {"a"}, False),
        (["/ckpt/2.safetensors"], "cuda", True, {"b", "c"}, False),
    ]
    assert consumed == ["a", "b", "c"]
    assert metrics[ModelStreamerMetricNames.BATCHES] == 2
    assert metrics[ModelStreamerMetricNames.NODE_TENSORS] == 3
    # A later load with another number of ranks per node counts them again.
    assert "RUNAI_STREAMER_PROCESS_GROUP_SIZE" not in os.environ

    # The streamer can read alone on this rank, for example when it does not
    # accept the backend of the process group.
    Streamer.shares_reads = False
    with pytest.raises(RuntimeError, match="does not share the reads"):
        _stream_batched_tensors(weight_map, [{"a"}], lambda name, tensor: None, distributed=True)


@pytest.mark.parametrize("shares_reads", [True, False])
def test_run_model_streamer_loads_each_group_once_its_tensors_arrive(
    monkeypatch: pytest.MonkeyPatch, shares_reads: bool
) -> None:
    values = {"a": 1.0, "b": 2.0, "c": 3.0, "other": 4.0}
    weight_map = dict.fromkeys(values, "/ckpt/model.safetensors")
    monkeypatch.setattr(model_streamer_loading, "_get_weight_source_map", lambda _: weight_map)
    if shares_reads:
        monkeypatch.delenv("RUNAI_STREAMER_DIST", raising=False)
    else:
        # Each rank then streams only its own tensors.
        monkeypatch.setenv("RUNAI_STREAMER_DIST", "0")

    def gather(
        names: set[str], error: Exception | None, *, world_size: int
    ) -> tuple[set[str], int, bool]:
        assert error is None and world_size == 2
        # The other rank on this node also needs "other".
        return set(names) | {"other"}, 2, False

    group_events = []

    @contextmanager
    def process_group(rank: int, world_size: int) -> Iterator[None]:
        group_events.append(("create", rank, world_size))
        yield
        group_events.append("destroy")

    streamed_batches = []

    def stream(
        weight_map: dict[str, str],
        batches: list[set[str]],
        consume: Callable[[str, torch.Tensor], None],
        *,
        distributed: bool,
    ) -> dict[str, float | int]:
        assert distributed == shares_reads
        # The process group exists only while the tensors stream.
        assert group_events == [("create", 1, 2)]
        streamed_batches.extend(set(batch) for batch in batches)
        names = set().union(*batches)
        for name in ("b", "other", "c", "a"):
            if name in names:
                tensor = torch.full((1,), values[name])
                consume(name, tensor)
                # The streamer reuses its buffer, so a group must get a copy.
                tensor.zero_()
        return {ModelStreamerMetricNames.SECONDS: 1.0}

    monkeypatch.setattr(model_streamer_loading, "_gather_node_tensor_names", gather)
    monkeypatch.setattr(model_streamer_loading, "_stream_batched_tensors", stream)
    monkeypatch.setattr(model_streamer_loading, "model_streamer_torch_distributed", process_group)
    events = []

    def load(label: str) -> Callable[[dict[str, torch.Tensor]], None]:
        return lambda tensors: events.append(
            (label, {name: tensor.item() for name, tensor in tensors.items()})
        )

    model = _StreamingModel(
        ModelStreamerPlan(
            # A list can name a tensor twice; the group still loads once.
            groups=[({"a", "b"}, load("ab")), ({"b"}, load("b")), (["c", "c"], load("c"))],
            # "c" and "other" (which only the other rank needs) go in the first batch.
            batch_index=lambda name: 0 if name in ("c", "other") else 5,
            finish=lambda: events.append(("finish", {})),
        )
    )
    metrics = _run_model_streamer(model, "/ckpt", rank=1, world_size=2)

    assert model.planned_keys == [set(weight_map)]
    assert streamed_batches == [{"c", "other"} if shares_reads else {"c"}, {"a", "b"}]
    assert group_events == [("create", 1, 2), "destroy"]
    # "b" is in two groups; "ab" loads only when "a" arrives last.
    assert events == [
        ("b", {"b": 2.0}),
        ("c", {"c": 3.0}),
        ("ab", {"a": 1.0, "b": 2.0}),
        ("finish", {}),
    ]
    assert metrics == {
        ModelStreamerMetricNames.SECONDS: 1.0,
        ModelStreamerMetricNames.RANK_TENSORS: 3,
    }


def test_run_model_streamer_rejects_unsupported_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_streamer_loading, "_get_weight_source_map", lambda _: {"a": "/ckpt/model.safetensors"}
    )
    monkeypatch.setattr(model_streamer_loading, "mpi_disabled", lambda: False)
    monkeypatch.setattr(socket, "gethostname", lambda: "node0")
    stream = MagicMock(return_value={})
    monkeypatch.setattr(model_streamer_loading, "_stream_batched_tensors", stream)
    gathered = []

    def peer_comm(peer: tuple[str, tuple[str, ...], str | None]) -> SimpleNamespace:
        def allgather(local: tuple[str, tuple[str, ...], str | None]) -> list[Any]:
            gathered.append(local)
            return [local, peer]

        return SimpleNamespace(allgather=allgather)

    # This rank's plan names a tensor that the checkpoint lacks. The rank still
    # joins the gather, so that the other rank does not wait, and then raises.
    monkeypatch.setattr(model_streamer_loading, "mpi_comm", lambda: peer_comm(("node0", (), None)))
    model = _StreamingModel(ModelStreamerPlan(groups=[({"a", "absent"}, MagicMock())]))
    with pytest.raises(KeyError, match="absent"):
        _run_model_streamer(model, "/ckpt", rank=0, world_size=2)
    assert gathered[-1][2].startswith("KeyError: ")

    # The other rank failed, so this rank raises too, before any streaming.
    monkeypatch.setattr(
        model_streamer_loading,
        "mpi_comm",
        lambda: peer_comm(("node0", (), "KeyError: 'missing tensors'")),
    )
    model = _StreamingModel(ModelStreamerPlan(groups=[({"a"}, MagicMock())]))
    with pytest.raises(RuntimeError, match="rank 1: KeyError: 'missing tensors'"):
        _run_model_streamer(model, "/ckpt", rank=0, world_size=2)
    stream.assert_not_called()

    # A group that names its tensors as one str would stream single letters.
    model = _StreamingModel(ModelStreamerPlan(groups=[("a", MagicMock())]))
    with pytest.raises(TypeError, match="one str"):
        _run_model_streamer(model, "/ckpt", rank=0, world_size=1)

    # The ranks are on two nodes. Model Streamer's global mode would share reads
    # across them, although each node streams different tensors.
    monkeypatch.setattr(model_streamer_loading, "mpi_comm", lambda: peer_comm(("node1", (), None)))
    monkeypatch.setenv("RUNAI_STREAMER_DIST_GLOBAL", "1")
    model = _StreamingModel(ModelStreamerPlan(groups=[({"a"}, MagicMock())]))
    with pytest.raises(RuntimeError, match="RUNAI_STREAMER_DIST_GLOBAL"):
        _run_model_streamer(model, "/ckpt", rank=0, world_size=2)
    stream.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        # An enum format shows its plain value 'MX' in the reason, not the enum repr.
        ({"checkpoint_format": CheckpointFormat.MX}, "checkpoint_format='HF' (received 'MX')"),
        ({"load_format": LoadFormat.DUMMY}, "load_format='auto' (received 'DUMMY')"),
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

    loader = construct_checkpoint_loader(
        None,
        checkpoint_format,
        checkpoint_io_policy=CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD,
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

    loader = construct_checkpoint_loader(
        provided_loader,
        "HF",
        checkpoint_io_policy=CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD,
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
    call_load_weights = _record_weight_copies(monkeypatch, [])
    _load_checkpoint_into_model(
        loader,
        "/checkpoint",
        model=SimpleNamespace(load_weights=object()),
        is_draft_model=False,
        mapping=Mapping(),
        weight_mapper=None,
        extra_load_weights_kwargs=None,
        metrics={},
    )
    assert call_load_weights.call_args.args[1] is native_weights
    active_communicator.assert_not_called()


def test_auto_selects_native_for_incompatible_config_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    monkeypatch.setattr(model_loader_module.logger, "info", info)

    loader = construct_checkpoint_loader(
        None,
        "MX",
        checkpoint_io_policy=CheckpointIOPolicy.AUTO,
    )

    status = loader.weight_loader.last_checkpoint_io_status
    assert status.requested == "auto"
    assert status.selected == "native"
    warning.assert_not_called()
    assert any("selected=native" in call.args[0] for call in info.call_args_list)


def test_static_native_selection_skips_rank_striped_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = construct_checkpoint_loader(
        None,
        "HF",
        checkpoint_io_policy=CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD,
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

    loader = construct_checkpoint_loader(None, "HF")
    assert isinstance(loader.weight_loader, _CustomWeightLoader)
    mapping = Mapping()
    assert list(loader.load_weights("/checkpoint", mapping=mapping)) == ["custom.weight"]
    loader.weight_loader.load_weights.assert_called_once_with("/checkpoint", mapping=mapping)

    warning = MagicMock()
    monkeypatch.setattr(model_loader_module.logger, "warning", warning)
    fallback_loader = construct_checkpoint_loader(
        None,
        "HF",
        checkpoint_io_policy=CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD,
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

    loader = construct_checkpoint_loader(
        None,
        "HF",
        checkpoint_io_policy=CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD,
    )

    assert isinstance(loader, _CustomCheckpointLoader)
    assert loader.weight_loader.checkpoint_io_policy == "native"
    warning.assert_called_once()
    assert "checkpoint loader is not the built-in" in warning.call_args.args[0]


class _RecordingCheckpointLoader(HfCheckpointLoader):
    """HF checkpoint loader that records its reads instead of reading files."""

    def __init__(
        self,
        events: list[str],
        weights: dict[str, object] | None = None,
        *,
        target_weight_mapper: object | None = None,
    ) -> None:
        super().__init__()
        self.events = events
        self.weights = {"model.weight": object()} if weights is None else weights
        self.target_weight_mapper = (
            object() if target_weight_mapper is None else target_weight_mapper
        )
        # keyword arguments of each read, by checkpoint directory
        self.read_kwargs: dict[str, dict[str, Any]] = {}

    def load_weights(self, checkpoint_dir: str, mapping: Any, **kwargs: Any) -> dict[str, object]:
        self.read_kwargs[checkpoint_dir] = dict(kwargs, mapping=mapping)
        self.events.append(f"read:{checkpoint_dir}")
        return self.weights

    def get_initialized_weight_mapper(self, model: object, config: object) -> object:
        del model, config
        self.events.append("mapper_init")
        return self.target_weight_mapper


def _record_weight_copies(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> MagicMock:
    """Replace the copy of read weights into the model with a "materialize" event."""
    call_load_weights = MagicMock(side_effect=lambda *_args: events.append("materialize"))
    monkeypatch.setattr(model_hooks, "call_load_weights", call_load_weights)
    return call_load_weights


def test_sessionless_loader_times_its_read_as_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaders without an HF weight session read once, with nothing to finalize."""
    events = []
    metrics = {}
    weights = {"model.weight": object()}
    checkpoint_loader = SimpleNamespace(
        load_weights=MagicMock(
            side_effect=lambda *_args, **_kwargs: events.append("read") or weights
        )
    )

    @contextmanager
    def record_timing(metric_name: str, metric_values: dict[str, float]) -> Iterator[None]:
        events.append(f"{metric_name}_start")
        try:
            yield
        finally:
            metric_values[metric_name] = 1.0
            events.append(f"{metric_name}_end")

    monkeypatch.setattr(model_loader_module, "timing_metric", record_timing)
    _record_weight_copies(monkeypatch, events)
    model = SimpleNamespace(load_weights=object())
    mapping = object()

    _load_checkpoint_into_model(
        checkpoint_loader,
        "/checkpoint",
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=object(),
        extra_load_weights_kwargs={"source_identity": None},
        metrics=metrics,
    )

    checkpoint_loader.load_weights.assert_called_once_with(
        "/checkpoint", mapping=mapping, model=model, source_identity=None
    )
    assert events == [
        "checkpoint_preparation_seconds_start",
        "read",
        "checkpoint_preparation_seconds_end",
        "weight_population_seconds_start",
        "materialize",
        "weight_population_seconds_end",
    ]
    assert metrics == {
        "checkpoint_preparation_seconds": 1.0,
        "weight_population_seconds": 1.0,
        "checkpoint_finalization_seconds": 0.0,
    }


def test_target_mapper_is_initialized_before_checkpoint_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    checkpoint_loader = _RecordingCheckpointLoader(events)
    _record_weight_copies(monkeypatch, events)
    model = SimpleNamespace(load_weights=object())
    mapping, source_identity = object(), object()
    metrics = {}

    weight_mapper, loaded_weights_preloaded = load_model_from_auto_load_format(
        checkpoint_loader,
        "/checkpoint",
        model_config=object(),
        model=model,
        mapping=mapping,
        llm_args=SimpleNamespace(checkpoint_io_policy=CheckpointIOPolicy.NATIVE),
        spec_config=None,
        source_identity=source_identity,
        qualification=None,
        metrics=metrics,
        startup_metadata={},
    )

    assert weight_mapper is checkpoint_loader.target_weight_mapper
    # Only an MX P2P load writes the weights into the model itself.
    assert loaded_weights_preloaded is False
    assert events == ["mapper_init", "read:/checkpoint", "materialize"]
    # The target read gets the model, so a context-only worker reads lazily.
    assert checkpoint_loader.read_kwargs == {
        "/checkpoint": {
            "mapping": mapping,
            "model": model,
            "source_identity": source_identity,
        }
    }
    # The read is timed as preparation, with nothing to finalize.
    assert "checkpoint_preparation_seconds" in metrics
    assert metrics["checkpoint_finalization_seconds"] == 0.0
    assert "weight_population_seconds" in metrics


def test_read_ahead_copies_inside_weight_session_and_reports_its_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    checkpoint_loader = HfCheckpointLoader(
        weight_loader=HfWeightLoader(checkpoint_io_policy="rank_striped_read_ahead")
    )
    weight_loader = checkpoint_loader.weight_loader
    weights = {"model.weight": object()}
    session_kwargs = {}

    @contextmanager
    def open_weight_session(checkpoint_dir: str, **kwargs: Any) -> Iterator[dict[str, object]]:
        session_kwargs[checkpoint_dir] = kwargs
        events.append("session_enter")
        weight_loader.record_session_timings(preparation_seconds=2.0)
        try:
            yield weights
        finally:
            weight_loader.record_session_timings(preparation_seconds=2.0, finalization_seconds=3.0)
            events.append("session_exit")

    monkeypatch.setattr(weight_loader, "open_weight_session", open_weight_session)
    checkpoint_loader.load_weights = MagicMock(
        side_effect=AssertionError("read-ahead must read in the weight session")
    )
    call_load_weights = _record_weight_copies(monkeypatch, events)
    model = SimpleNamespace(load_weights=object())
    mapping = object()
    metrics = {}

    _load_checkpoint_into_model(
        checkpoint_loader,
        "/checkpoint",
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=None,
        extra_load_weights_kwargs={"source_identity": None},
        metrics=metrics,
    )

    # The weights are copied before the session finishes read-ahead.
    assert events == ["session_enter", "materialize", "session_exit"]
    assert call_load_weights.call_args.args[1] is weights
    assert session_kwargs == {
        "/checkpoint": {"mapping": mapping, "model": model, "source_identity": None}
    }
    # The model loader reports the timings the session recorded.
    assert metrics["checkpoint_preparation_seconds"] == 2.0
    assert metrics["checkpoint_finalization_seconds"] == 3.0


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
    checkpoint_loader = _RecordingCheckpointLoader(events)
    checkpoint_loader.build_checkpoint_catalog = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("catalog") or catalog
    )
    weight_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(side_effect=lambda _catalog: events.append("plan") or plan)
    )
    call_load_weights = _record_weight_copies(monkeypatch, events)

    _load_checkpoint_into_model(
        checkpoint_loader,
        "/checkpoint",
        model=SimpleNamespace(load_weights=object()),
        is_draft_model=False,
        mapping=object(),
        weight_mapper=weight_mapper,
        extra_load_weights_kwargs=None,
        metrics={},
    )

    # The plan is inspected after the copy.
    assert events == [
        "read:/checkpoint",
        "materialize",
        "catalog",
        "plan",
    ]
    call_load_weights.assert_called_once()


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


def test_draft_checkpoint_loads_after_target_with_its_own_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "1")
    events = []
    checkpoint_loader = _RecordingCheckpointLoader(events)
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
        side_effect=lambda checkpoint_dir, **_kwargs: (
            events.append(f"catalog:{checkpoint_dir}") or catalog
        )
    )
    _record_weight_copies(monkeypatch, events)
    draft_mapper = MagicMock()
    draft_mapper.init_model_and_config.side_effect = lambda *_args: events.append(
        "draft_mapper_init"
    )
    draft_mapper.build_weight_load_plan.side_effect = lambda _catalog: events.append("plan") or plan
    get_mapper = MagicMock(return_value=draft_mapper)
    monkeypatch.setattr(base_checkpoint_loader.AutoCheckpointMapper, "get", get_mapper)
    model = SimpleNamespace(
        load_weights=object(),
        load_draft_weights=object(),
        draft_model=object(),
        draft_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(architectures=["DraftForCausalLM"])
        ),
    )
    mapping = object()
    metrics = {}

    load_model_from_auto_load_format(
        checkpoint_loader,
        "/checkpoint",
        model_config=object(),
        model=model,
        mapping=mapping,
        llm_args=SimpleNamespace(checkpoint_io_policy=CheckpointIOPolicy.NATIVE),
        spec_config=SimpleNamespace(needs_separate_draft_weights=True, speculative_model="/draft"),
        source_identity=None,
        qualification=None,
        metrics=metrics,
        startup_metadata={},
    )

    # The target mapper has no plan builder, so only the draft read is inspected.
    assert events == [
        "mapper_init",
        "read:/checkpoint",
        "materialize",
        "draft_mapper_init",
        "read:/draft",
        "materialize",
        "catalog:/draft",
        "plan",
    ]
    get_mapper.assert_called_once_with("HF", "DraftForCausalLM")
    draft_mapper.init_model_and_config.assert_called_once_with(
        model.draft_model, model.draft_config
    )
    # The draft read does not get the model.
    assert checkpoint_loader.read_kwargs["/draft"] == {"mapping": mapping}
    assert "draft_checkpoint_preparation_seconds" in metrics
    assert "draft_weight_population_seconds" in metrics
    assert metrics["draft_checkpoint_finalization_seconds"] == 0.0


@pytest.mark.parametrize("target_preloaded", [False, True])
def test_mtp_draft_reuses_target_mapper(
    monkeypatch: pytest.MonkeyPatch, target_preloaded: bool
) -> None:
    monkeypatch.setenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", "1")
    events = []
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("draft.weight"),))
    plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.OPAQUE,
        demands=(WeightDemand("all", ("draft.weight",), (0,)),),
    )
    # The draft read reuses the target mapper, so that mapper also builds the
    # draft's shadow plan.
    target_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(side_effect=lambda _catalog: events.append("plan") or plan)
    )
    # An MX P2P target read writes the weights into the model and returns none
    # to copy. The draft read is still copied, even when it reads nothing, so
    # the model's draft load can reject an empty draft checkpoint.
    checkpoint_loader = _RecordingCheckpointLoader(
        events, weights={} if target_preloaded else None, target_weight_mapper=target_mapper
    )
    # Only an MX loader reports preloaded weights; stand in for one.
    monkeypatch.setattr(
        mx_loading, "is_mx_weights_preloaded", lambda _checkpoint_loader: target_preloaded
    )
    checkpoint_loader.build_checkpoint_catalog = MagicMock(
        side_effect=lambda checkpoint_dir, **_kwargs: (
            events.append(f"catalog:{checkpoint_dir}") or catalog
        )
    )
    call_load_weights = _record_weight_copies(monkeypatch, events)
    model = SimpleNamespace(load_weights=object(), load_draft_weights=object(), draft_config=None)

    weight_mapper, _ = load_model_from_auto_load_format(
        checkpoint_loader,
        "/checkpoint",
        model_config=object(),
        model=model,
        mapping=object(),
        llm_args=SimpleNamespace(checkpoint_io_policy=CheckpointIOPolicy.NATIVE),
        spec_config=SimpleNamespace(needs_separate_draft_weights=True, speculative_model="/draft"),
        source_identity=None,
        qualification=None,
        metrics={},
        startup_metadata={},
    )

    assert weight_mapper is target_mapper
    draft_copy = call_load_weights.call_args
    assert draft_copy.args[0] is model.load_draft_weights
    assert draft_copy.args[2] is target_mapper
    expected_events = ["mapper_init", "read:/checkpoint"]
    if not target_preloaded:
        expected_events.append("materialize")
    expected_events += [
        "catalog:/checkpoint",
        "plan",
        "read:/draft",
        "materialize",
        "catalog:/draft",
        "plan",
    ]
    assert events == expected_events
