# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.distributed import initialization as distributed_initialization
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
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
    _checkpoint_startup_metadata,
    _resolve_checkpoint_io_policy,
    construct_checkpoint_loader,
)
from tensorrt_llm._torch.pyexecutor.weight_load_strategies import (
    CheckpointLoadStrategy,
    ModelStreamerLoadStrategy,
    WeightLoadContext,
    model_hooks,
)
from tensorrt_llm._torch.pyexecutor.weight_load_strategies import (
    checkpoint as checkpoint_strategy_mod,
)
from tensorrt_llm._torch.pyexecutor.weight_load_strategies import (
    model_streamer as model_streamer_mod,
)
from tensorrt_llm._torch.pyexecutor.weight_load_strategies.checkpoint import (
    _open_checkpoint_weight_session,
    _timed_checkpoint_weight_session,
)
from tensorrt_llm._torch.pyexecutor.weight_load_strategies.model_streamer import (
    ModelStreamerMetricNames,
    ModelStreamerPlan,
    gather_tensor_names_from_rank_to_node,
    run_model_streamer,
    stream_batched_tensors,
)
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy, LoadFormat
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_checkpoint_startup_metadata_describes_actual_loader() -> None:
    checkpoint_loader = HfCheckpointLoader()

    metadata = _checkpoint_startup_metadata(
        checkpoint_loader, LoadFormat.AUTO, _checkpoint_load_strategy(checkpoint_loader)
    )

    assert metadata == {
        "checkpoint_loader_kind": "HfCheckpointLoader",
        "checkpoint_weight_loader_kind": "HfWeightLoader",
        "checkpoint_source_kind": "HF",
        "load_format": "auto",
        "weight_load_strategy_kind": "CheckpointLoadStrategy",
    }


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


@pytest.mark.parametrize("policy", ["native", "rank_striped_read_ahead"])
@pytest.mark.parametrize("session", [False, True])
def test_context_only_loading_keeps_unused_checkpoint_tensors_lazy(
    monkeypatch: pytest.MonkeyPatch, policy: str, session: bool
) -> None:
    loader = HfCheckpointLoader(weight_loader=HfWeightLoader(checkpoint_io_policy=policy))
    lazy_weights = {"layers.0.weight": object(), "layers.39.weight": object()}
    lazy_load = MagicMock(return_value=lazy_weights)
    eager_load = MagicMock(side_effect=AssertionError("must not read unused decoder tensors"))
    reader_session = MagicMock(side_effect=AssertionError("must not prefetch decoder tensors"))
    monkeypatch.setattr(loader.weight_loader, "_load_lazy_safetensors", lazy_load)
    monkeypatch.setattr(loader.weight_loader, "_load_weights_native", eager_load)
    monkeypatch.setattr(loader.weight_loader, "open_weight_session", reader_session)
    model = SimpleNamespace(model=SimpleNamespace(disagg_context_only=True))
    options = dict(mapping=Mapping(), model=model, use_consolidated=True)

    if session:
        with loader.open_weight_session("/checkpoint", **options) as weights:
            assert weights is lazy_weights
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
    model = SimpleNamespace(model=body)
    mapping = Mapping()

    with loader.open_weight_session("/checkpoint", mapping=mapping, model=model) as weights:
        assert weights is expected

    eager_load.assert_called_once_with("/checkpoint", mapping, False, model=model)
    lazy_load.assert_not_called()


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
    model = SimpleNamespace(model=SimpleNamespace(disagg_context_only=True))
    mapping = Mapping()

    with loader.open_weight_session("/checkpoint", mapping=mapping, model=model) as weights:
        assert weights is expected

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

    def __init__(self, modules: tuple[Any, ...] = ()) -> None:
        self._modules = modules
        self.planned_keys: list[set[str]] = []

    def modules(self) -> tuple[Any, ...]:
        return self._modules

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(())

    def plan_model_streamer_load(self, checkpoint_keys: set[str]) -> ModelStreamerPlan:
        self.planned_keys.append(checkpoint_keys)
        return ModelStreamerPlan(groups=[({"weight"}, lambda tensors: None)])


def _model_streamer_strategy(mapping: Mapping) -> ModelStreamerLoadStrategy:
    """Return a Model Streamer strategy for ``/checkpoint`` with fresh metrics."""
    return ModelStreamerLoadStrategy(
        WeightLoadContext(
            checkpoint_dir="/checkpoint",
            checkpoint_loader=MagicMock(name="checkpoint_loader"),
            llm_args=SimpleNamespace(
                load_format=LoadFormat.AUTO,
                checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER,
            ),
            mapping=mapping,
            spec_config=None,
            metrics={},
        )
    )


def test_model_streamer_strategy_streams_into_model(monkeypatch: pytest.MonkeyPatch) -> None:
    weight_map = {"weight": "/checkpoint/model.safetensors"}
    mapped_dirs = []
    streamed = []

    def get_weight_source_map(checkpoint_dir: str) -> dict[str, str]:
        mapped_dirs.append(checkpoint_dir)
        return weight_map

    def fake_run_model_streamer(plan, weights, *, rank, world_size):
        streamed.append((plan, weights, rank, world_size))
        return {ModelStreamerMetricNames.MODEL_STREAMER_SECONDS: 2.0}

    monkeypatch.setattr(model_streamer_mod, "get_weight_source_map", get_weight_source_map)
    monkeypatch.setattr(model_streamer_mod, "run_model_streamer", fake_run_model_streamer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    logger = MagicMock(name="logger")
    monkeypatch.setattr(model_streamer_mod, "logger", logger)
    weight_load_strategy = _model_streamer_strategy(Mapping(world_size=4, tp_size=4, rank=3))
    model = _StreamingModel()
    config = object()

    weight_load_strategy.prepare(model, config)
    assert weight_load_strategy.load_weights(model, config) is True

    # The model plans against the checkpoint's names, and the strategy streams the plan.
    assert mapped_dirs == ["/checkpoint"]
    assert model.planned_keys == [{"weight"}]
    [(plan, weights, rank, world_size)] = streamed
    assert plan.groups[0][0] == {"weight"}
    assert weights is weight_map
    assert (rank, world_size) == (3, 4)
    checkpoint_loader = weight_load_strategy.context.checkpoint_loader
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once_with(model, config)
    assert (
        weight_load_strategy.weight_mapper
        is checkpoint_loader.get_initialized_weight_mapper.return_value
    )
    # The Model Streamer metrics are logged; only the load time goes to the startup metrics.
    metrics = weight_load_strategy.context.metrics
    assert ModelStreamerMetricNames.MODEL_STREAMER_SECONDS not in metrics
    assert "weight_population_seconds" in metrics
    logger.info.assert_called_once_with(
        "Model Streamer metrics for rank 3: model_streamer_seconds=2.0000"
    )
    checkpoint_loader.load_weights.assert_not_called()

    # A separate draft checkpoint, e.g. a DSpark drafter, is read with the
    # regular checkpoint reader, using the target's weight mapper.
    draft_loads = []
    monkeypatch.setattr(
        CheckpointLoadStrategy,
        "load_draft_weights",
        lambda draft_strategy, draft_model: draft_loads.append((draft_strategy, draft_model)),
    )
    weight_load_strategy.load_draft_weights(model)
    [(draft_strategy, draft_model)] = draft_loads
    assert draft_model is model
    assert draft_strategy.context is weight_load_strategy.context
    assert draft_strategy.weight_mapper is weight_load_strategy.weight_mapper


def test_model_streamer_strategy_rejects_models_that_cannot_stream() -> None:
    with pytest.raises(ValueError, match="object does not support .* plan_model_streamer_load"):
        _model_streamer_strategy(Mapping()).prepare(object(), object())


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
    load runs `run_model_streamer()` for real. The returned list records the names
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
    monkeypatch.setattr(model_loader_module, "MetaInitMode", nullcontext)
    monkeypatch.setattr(
        model_loader_module, "maybe_create_moe_load_balancer", lambda *_: nullcontext()
    )
    monkeypatch.setattr(model_loader_module.AutoModelForCausalLM, "from_config", lambda _: model)
    monkeypatch.setattr(model_loader_module, "get_rank_model_storage", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_streamer_mod,
        "get_weight_source_map",
        lambda _: {"weight": "/checkpoint/model.safetensors"},
    )
    streamed = []

    def stream(weight_map, batches, consume, *, distributed):
        for batch in batches:
            for name in sorted(batch):
                streamed.append(name)
                consume(name, torch.ones(1))
        return {ModelStreamerMetricNames.MODEL_STREAMER_SECONDS: 1.0}

    monkeypatch.setattr(model_streamer_mod, "stream_batched_tensors", stream)
    checkpoint_loader = construct_checkpoint_loader(
        None, "HF", checkpoint_io_policy=CheckpointIOPolicy.MODEL_STREAMER
    )
    checkpoint_loader.get_initialized_weight_mapper = MagicMock()
    return loader, checkpoint_loader, model, streamed


def test_model_loader_streams_through_model_streamer_strategy(
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
    assert loader.startup_metadata["weight_load_strategy_kind"] == "ModelStreamerLoadStrategy"
    assert loader.startup_metadata["load_format"] == "auto"


def test_model_loader_rejects_loads_model_streamer_cannot_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader, checkpoint_loader, model, streamed = _model_streamer_load(
        monkeypatch, modules=(SimpleNamespace(_requires_standard_hf_loading=True),)
    )

    with pytest.raises(ValueError, match="Host-resident"):
        loader.load("/checkpoint", checkpoint_loader)

    model.plan_model_streamer_load.assert_not_called()
    assert streamed == []


@pytest.mark.parametrize("use_mpi", [True, False])
def test_gather_tensor_names_covers_different_expert_layers(monkeypatch, use_mpi):
    import torch.distributed as dist

    from tensorrt_llm import _utils

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(_utils, "mpi_disabled", lambda: not use_mpi)

    def peers(local):
        return [local, (local[0], ("layer1.expert3",))]

    class Comm:
        def allgather(self, local):
            assert use_mpi, "Ray workers have no MPI communicator"
            return peers(local)

    def gather(entries, local):
        # Under MPI, a default-group collective would keep NCCL buffers alive.
        assert not use_mpi, "MPI launches must not gather on the default process group"
        entries[:] = peers(local)

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    monkeypatch.setattr(dist, "all_gather_object", gather)
    selected, ranks = gather_tensor_names_from_rank_to_node(
        {"layer0.expert0"}, rank=0, world_size=2
    )
    assert selected == {"layer0.expert0", "layer1.expert3"}
    assert ranks == 2


def test_gather_tensor_names_bootstraps_torch_group_from_mpi(monkeypatch):
    import torch.distributed as dist

    from tensorrt_llm import _utils

    # Launcher values may describe an outer job; they are neither used nor changed.
    monkeypatch.setenv("MASTER_ADDR", "10.9.9.9")
    monkeypatch.setenv("MASTER_PORT", "1")
    monkeypatch.setattr(_utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(_utils, "mpi_rank", lambda: 1)
    monkeypatch.setattr(_utils, "mpi_world_size", lambda: 2)

    class Comm:
        def bcast(self, value, root):
            assert value is None and root == 0
            return "10.0.0.1", "29500"

        def allgather(self, local):
            return [local, local]

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    initialized = []
    monkeypatch.setattr(
        dist, "init_process_group", lambda *args, **kwargs: initialized.append((args, kwargs))
    )

    selected, ranks = gather_tensor_names_from_rank_to_node(
        {"layer0.expert0"}, rank=1, world_size=2
    )
    assert selected == {"layer0.expert0"} and ranks == 2
    assert initialized == [
        (("nccl",), {"init_method": "tcp://10.0.0.1:29500", "rank": 1, "world_size": 2})
    ]
    assert os.environ["MASTER_ADDR"] == "10.9.9.9"
    assert os.environ["MASTER_PORT"] == "1"


@pytest.mark.parametrize(
    ("hostnames", "host_address", "error"),
    [
        # Ranks on two nodes reach rank 0 through its routable address.
        (["node0", "node1"], "10.0.0.7", None),
        # Ranks on one node can use a loopback address.
        (["node0", "node0"], "127.0.1.1", None),
        # Ranks on other nodes cannot reach a loopback address.
        (["node0", "node1"], "127.0.1.1", "loopback address 127.0.1.1"),
        # A failed hostname lookup on rank 0 is broadcast, so every rank raises.
        (["node0", "node1"], socket.gaierror("no such host"), "no such host"),
    ],
)
def test_rank_zero_selects_fresh_rendezvous_endpoint(
    monkeypatch, hostnames: list[str], host_address: str | Exception, error: str | None
):
    import torch.distributed as dist

    from tensorrt_llm import _utils

    monkeypatch.setenv("MASTER_ADDR", "10.9.9.9")
    monkeypatch.setenv("MASTER_PORT", "1")
    monkeypatch.setattr(_utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(_utils, "mpi_rank", lambda: 0)
    monkeypatch.setattr(_utils, "mpi_world_size", lambda: 2)
    monkeypatch.setattr(socket, "gethostname", lambda: "node0")

    def gethostbyname(name):
        if isinstance(host_address, Exception):
            raise host_address
        return host_address

    monkeypatch.setattr(socket, "gethostbyname", gethostbyname)
    broadcast = []

    class Comm:
        def bcast(self, value, root):
            broadcast.append(value)
            return value

        def allgather(self, local):
            # The hostname gather tells if the ranks span nodes. The tensor-name
            # gather keeps both ranks on this node.
            return hostnames if isinstance(local, str) else [local, local]

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    initialized = []
    monkeypatch.setattr(
        dist, "init_process_group", lambda *args, **kwargs: initialized.append(kwargs)
    )

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            gather_tensor_names_from_rank_to_node({"layer0.expert0"}, rank=0, world_size=2)
        # Rank 0 broadcast the reason instead of an endpoint.
        (reason,) = broadcast
        assert isinstance(reason, str) and initialized == []
        return

    gather_tensor_names_from_rank_to_node({"layer0.expert0"}, rank=0, world_size=2)
    ((address, port),) = broadcast
    assert address == host_address
    assert port != "1" and int(port) > 0
    assert initialized == [
        {"init_method": f"tcp://{host_address}:{port}", "rank": 0, "world_size": 2}
    ]
    assert os.environ["MASTER_ADDR"] == "10.9.9.9"
    assert os.environ["MASTER_PORT"] == "1"


def test_gather_tensor_names_initializes_process_group_first(monkeypatch):
    from tensorrt_llm import _utils

    initialize = MagicMock()
    monkeypatch.setattr(
        distributed_initialization, "ensure_model_streamer_torch_distributed", initialize
    )
    monkeypatch.setattr(_utils, "mpi_disabled", lambda: False)

    class Comm:
        def allgather(self, local):
            initialize.assert_called_once_with(1, 2)
            return [local, local]

    monkeypatch.setattr(_utils, "mpi_comm", lambda: Comm())
    selected, ranks = gather_tensor_names_from_rank_to_node(
        {"layer0.expert0"}, rank=1, world_size=2
    )
    assert selected == {"layer0.expert0"} and ranks == 2


def test_stream_batches_share_one_session(monkeypatch):
    sessions = []
    calls = []

    class Streamer:
        def __enter__(self):
            sessions.append(self)
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, paths, *, device, is_distributed, tensor_names):
            calls.append((paths, device, is_distributed, set(tensor_names)))
            self.names = sorted(tensor_names)

        def get_tensors(self):
            for name in self.names:
                yield name, torch.zeros(2)

    monkeypatch.setitem(
        sys.modules, "runai_model_streamer", SimpleNamespace(SafetensorsStreamer=Streamer)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    weight_map = {
        "a": "/ckpt/1.safetensors",
        "b": "/ckpt/2.safetensors",
        "c": "/ckpt/2.safetensors",
    }
    consumed = []
    metrics = stream_batched_tensors(
        weight_map,
        [{"a"}, set(), {"b", "c"}],
        lambda name, tensor: consumed.append(name),
        distributed=True,
    )
    assert len(sessions) == 1
    # Each batch reads only the shards that hold its tensors; empty batches are skipped.
    assert calls == [
        (["/ckpt/1.safetensors"], "cuda", True, {"a"}),
        (["/ckpt/2.safetensors"], "cuda", True, {"b", "c"}),
    ]
    assert consumed == ["a", "b", "c"]
    assert metrics[ModelStreamerMetricNames.MODEL_STREAMER_BATCHES] == 2
    assert metrics[ModelStreamerMetricNames.MODEL_STREAMER_NODE_TENSORS] == 3


def test_run_model_streamer_loads_each_group_once_its_tensors_arrive(monkeypatch):
    values = {"a": 1.0, "b": 2.0, "c": 3.0, "other": 4.0}

    def gather(names, *, rank, world_size):
        assert (rank, world_size) == (1, 2)
        # The other rank on this node also needs "other".
        return set(names) | {"other"}, 2

    streamed_batches = []

    def stream(weight_map, batches, consume, *, distributed):
        assert distributed
        streamed_batches.extend(set(batch) for batch in batches)
        for name in ("b", "other", "c", "a"):
            tensor = torch.full((1,), values[name])
            consume(name, tensor)
            # The streamer reuses its buffer, so a group must get a copy.
            tensor.zero_()
        return {ModelStreamerMetricNames.MODEL_STREAMER_SECONDS: 1.0}

    monkeypatch.setattr(model_streamer_mod, "gather_tensor_names_from_rank_to_node", gather)
    monkeypatch.setattr(model_streamer_mod, "stream_batched_tensors", stream)
    events = []

    def load(label):
        return lambda tensors: events.append(
            (label, {name: tensor.item() for name, tensor in tensors.items()})
        )

    plan = ModelStreamerPlan(
        groups=[({"a", "b"}, load("ab")), ({"b"}, load("b")), ({"c"}, load("c"))],
        # "c" and "other" (which only the other rank needs) go in the first batch.
        batch_index=lambda name: 0 if name in ("c", "other") else 5,
        finish=lambda: events.append(("finish", {})),
    )
    weight_map = dict.fromkeys(("a", "b", "c", "other"), "/ckpt/model.safetensors")
    metrics = run_model_streamer(plan, weight_map, rank=1, world_size=2)

    assert streamed_batches == [{"c", "other"}, {"a", "b"}]
    # "b" is in two groups; "ab" loads only when "a" arrives last.
    assert events == [
        ("b", {"b": 2.0}),
        ("c", {"c": 3.0}),
        ("ab", {"a": 1.0, "b": 2.0}),
        ("finish", {}),
    ]
    assert metrics == {
        ModelStreamerMetricNames.MODEL_STREAMER_SECONDS: 1.0,
        ModelStreamerMetricNames.MODEL_STREAMER_RANK_TENSORS: 3.0,
    }


def test_run_model_streamer_rejects_missing_and_incomplete_tensors(monkeypatch):
    gather = MagicMock()
    monkeypatch.setattr(model_streamer_mod, "gather_tensor_names_from_rank_to_node", gather)
    plan = ModelStreamerPlan(groups=[({"a", "absent"}, MagicMock())])
    with pytest.raises(KeyError, match="absent"):
        run_model_streamer(plan, {"a": "/ckpt/model.safetensors"}, rank=0, world_size=2)
    # A missing tensor fails before any call between ranks.
    gather.assert_not_called()

    monkeypatch.setattr(
        model_streamer_mod,
        "gather_tensor_names_from_rank_to_node",
        lambda names, **_: (set(names), 1),
    )
    monkeypatch.setattr(model_streamer_mod, "stream_batched_tensors", lambda *_, **__: {})
    load, finish = MagicMock(), MagicMock()
    plan = ModelStreamerPlan(groups=[({"a"}, load)], finish=finish)
    with pytest.raises(RuntimeError, match="did not complete"):
        run_model_streamer(plan, {"a": "/ckpt/model.safetensors"}, rank=0, world_size=1)
    load.assert_not_called()
    finish.assert_not_called()


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
    with loader.open_weight_session("/checkpoint", mapping=mapping) as weights:
        assert list(weights) == ["custom.weight"]
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

    def get_initialized_weight_mapper(self, model: object, config: object) -> object:
        del model, config
        self.events.append("mapper_init")
        return object()


def _checkpoint_load_strategy(
    checkpoint_loader: object, *, spec_config: object = None
) -> CheckpointLoadStrategy:
    """Return a checkpoint load strategy for ``/checkpoint`` with fresh metrics."""
    return CheckpointLoadStrategy(
        WeightLoadContext(
            checkpoint_dir="/checkpoint",
            checkpoint_loader=checkpoint_loader,
            llm_args=SimpleNamespace(load_format=LoadFormat.AUTO),
            mapping=object(),
            spec_config=spec_config,
            metrics={},
        )
    )


def _model_without_llm_checkpoint_dir() -> MagicMock:
    """Return a mock model that reads its weights from the main checkpoint."""
    model = MagicMock()
    del model.llm_checkpoint_dir
    return model


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

    monkeypatch.setattr(checkpoint_strategy_mod, "timing_metric", record_timing)

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


def test_model_loader_session_spans_mapper_and_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    weight_load_strategy = _checkpoint_load_strategy(_SessionCheckpointLoader(events))
    monkeypatch.setattr(
        model_hooks,
        "call_load_weights",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("materialize")),
    )
    weights_preloaded = weight_load_strategy.load_weights(
        _model_without_llm_checkpoint_dir(), object()
    )

    assert not weights_preloaded
    assert weight_load_strategy.loaded_checkpoint_tensors
    assert events == [
        "session_enter",
        "mapper_init",
        "materialize",
        "session_exit",
    ]
    metrics = weight_load_strategy.context.metrics
    assert "checkpoint_preparation_seconds" in metrics
    assert "weight_population_seconds" in metrics
    assert "checkpoint_finalization_seconds" in metrics


def test_shadow_plan_disabled_does_not_call_inspection_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTLLM_SHADOW_WEIGHT_LOAD_PLAN", raising=False)
    checkpoint_loader = SimpleNamespace(build_checkpoint_catalog=MagicMock())
    weight_mapper = SimpleNamespace(build_weight_load_plan=MagicMock())

    result = checkpoint_strategy_mod._inspect_shadow_weight_load_plan(
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
        side_effect=lambda *_args, **_kwargs: events.append("catalog") or catalog
    )
    weight_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(side_effect=lambda _catalog: events.append("plan") or plan)
    )
    checkpoint_loader.get_initialized_weight_mapper = MagicMock(
        side_effect=lambda *_args: events.append("mapper_init") or weight_mapper
    )
    monkeypatch.setattr(
        model_hooks,
        "call_load_weights",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("materialize")),
    )

    _checkpoint_load_strategy(checkpoint_loader).load_weights(
        _model_without_llm_checkpoint_dir(), object()
    )

    assert events == [
        "session_enter",
        "mapper_init",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    model_hooks.call_load_weights.assert_called_once()


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

    result = checkpoint_strategy_mod._inspect_shadow_weight_load_plan(
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
        side_effect=lambda *_args, **_kwargs: events.append("catalog") or catalog
    )
    model = MagicMock()
    model.draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["DraftForCausalLM"])
    )
    weight_load_strategy = _checkpoint_load_strategy(
        checkpoint_loader, spec_config=SimpleNamespace(speculative_model="/draft")
    )
    monkeypatch.setattr(
        model_hooks,
        "call_load_weights",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("materialize")),
    )
    draft_mapper = MagicMock()
    draft_mapper.init_model_and_config.side_effect = lambda *_args: events.append("mapper_init")
    draft_mapper.build_weight_load_plan.side_effect = lambda _catalog: events.append("plan") or plan
    monkeypatch.setattr(
        checkpoint_strategy_mod.AutoCheckpointMapper, "get", MagicMock(return_value=draft_mapper)
    )

    weight_load_strategy.load_draft_weights(model)

    assert events == [
        "session_enter",
        "mapper_init",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    metrics = weight_load_strategy.context.metrics
    assert "draft_checkpoint_preparation_seconds" in metrics
    assert "draft_weight_population_seconds" in metrics
    assert "draft_checkpoint_finalization_seconds" in metrics


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
        side_effect=lambda *_args, **_kwargs: events.append("catalog") or catalog
    )
    model = MagicMock()
    model.draft_config = None
    weight_load_strategy = _checkpoint_load_strategy(
        checkpoint_loader, spec_config=SimpleNamespace(speculative_model="/draft")
    )
    weight_load_strategy.weight_mapper = SimpleNamespace(
        build_weight_load_plan=MagicMock(side_effect=lambda _catalog: events.append("plan") or plan)
    )
    monkeypatch.setattr(
        model_hooks,
        "call_load_weights",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("materialize")),
    )

    weight_load_strategy.load_draft_weights(model)

    assert model_hooks.call_load_weights.call_args.args[2] is weight_load_strategy.weight_mapper
    assert events == [
        "session_enter",
        "catalog",
        "plan",
        "materialize",
        "session_exit",
    ]
    metrics = weight_load_strategy.context.metrics
    assert "draft_checkpoint_preparation_seconds" in metrics
    assert "draft_weight_population_seconds" in metrics
    assert "draft_checkpoint_finalization_seconds" in metrics
