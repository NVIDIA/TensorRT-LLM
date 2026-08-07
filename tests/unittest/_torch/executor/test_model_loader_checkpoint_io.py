# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader, _construct_checkpoint_loader
from tensorrt_llm.llmapi.llm_args import LoadFormat

pytestmark = pytest.mark.cpu_only


class _PolymorphicCheckpointLoader(BaseCheckpointLoader):
    def __init__(self):
        self.calls = []

    def get_default_weight_loader(self):
        return None

    def get_default_config_loader(self):
        return None

    def cleanup(self):
        pass

    @property
    def weight_loader(self):
        return None

    @property
    def weight_mapper(self):
        return None

    @property
    def config_loader(self):
        return None

    @property
    def checkpoint_format(self):
        return "probe"

    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        self.calls.append((checkpoint_dir, mapping, kwargs))
        return {"probe.weight": object()}


class _CustomHfWeightLoader(HfWeightLoader):
    def __init__(self):
        self.calls = []

    def load_weights(self, checkpoint_dir, mapping, **kwargs):
        self.calls.append((checkpoint_dir, mapping, kwargs))
        return {"custom.weight": object()}


class _TinyModel(nn.Module):
    def _apply(self, fn):
        del fn
        return self

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def load_weights(self, weights, mapper):
        del weights, mapper


class _SessionCheckpointLoader:
    checkpoint_format = "HF"

    def __init__(self, events):
        self.events = events

    def coordinate_checkpoint_io_request(self, mapping):
        del mapping
        self.events.append("coordinate")

    @contextmanager
    def open_weight_session(self, checkpoint_dir, **kwargs):
        assert checkpoint_dir == "/checkpoint"
        assert "mapping" in kwargs
        self.events.append("session_enter")
        try:
            yield {"model.weight": object()}
        finally:
            self.events.append("session_exit")

    def is_weights_preloaded(self):
        return False

    def get_initialized_weight_mapper(self, model, config):
        del model, config
        self.events.append("mapper_init")
        return object()

    def post_load_apply(self, model, *, weights_preloaded=False):
        del model, weights_preloaded


def test_base_weight_session_calls_polymorphic_load_weights():
    loader = _PolymorphicCheckpointLoader()
    mapping = object()

    with loader.open_weight_session("/checkpoint", mapping=mapping, model="model") as weights:
        assert list(weights) == ["probe.weight"]

    assert loader.calls == [("/checkpoint", mapping, {"model": "model"})]


def test_custom_hf_weight_loader_is_not_bypassed_by_optimized_session():
    weight_loader = _CustomHfWeightLoader()
    checkpoint_loader = HfCheckpointLoader(weight_loader=weight_loader)
    mapping = object()

    with checkpoint_loader.open_weight_session(
        "/checkpoint", mapping=mapping, model="model"
    ) as weights:
        assert list(weights) == ["custom.weight"]

    assert weight_loader.calls == [("/checkpoint", mapping, {"model": "model"})]


@pytest.mark.parametrize("loader_cls", [MistralCheckpointLoader, MXCheckpointLoader])
def test_hf_checkpoint_subclass_load_weights_is_not_bypassed(loader_cls):
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


def test_construct_checkpoint_loader_configures_only_builtin_hf():
    native_loader = _construct_checkpoint_loader(
        "pytorch", None, "HF", checkpoint_io_policy="native"
    )
    striped_loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
    )
    mx_loader = _construct_checkpoint_loader(
        "pytorch",
        None,
        "MX",
        checkpoint_io_policy="rank_striped_read_ahead",
    )

    assert native_loader.weight_loader.checkpoint_io_policy == "native"
    assert striped_loader.weight_loader.checkpoint_io_policy == "rank_striped_read_ahead"
    assert mx_loader.checkpoint_format == "MX"
    assert mx_loader.weight_loader.checkpoint_io_policy == "native"

    custom_loader = object()
    assert (
        _construct_checkpoint_loader(
            "pytorch",
            custom_loader,
            "HF",
            checkpoint_io_policy="rank_striped_read_ahead",
        )
        is custom_loader
    )


def test_builtin_hf_coordinates_policy_at_orchestration_boundary():
    checkpoint_loader = HfCheckpointLoader()
    checkpoint_loader.weight_loader.coordinate_checkpoint_io_request = MagicMock()
    mapping = object()

    checkpoint_loader.coordinate_checkpoint_io_request(mapping)

    checkpoint_loader.weight_loader.coordinate_checkpoint_io_request.assert_called_once_with(
        mapping
    )


def test_model_loader_session_spans_mapper_and_materialization(monkeypatch):
    events = []
    model = _TinyModel()
    checkpoint_loader = _SessionCheckpointLoader(events)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            architectures=["TinyForCausalLM"],
            model_type="tiny",
        )
    )
    loader = ModelLoader(
        llm_args=SimpleNamespace(load_format=LoadFormat.AUTO),
        mapping=object(),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=16,
        max_seq_len=16,
    )
    loader._load_and_validate_config = MagicMock(return_value=config)
    loader._qualify_post_transform_profile = MagicMock(
        return_value=SimpleNamespace(
            qualified=False,
            profile=None,
            transform_abi_id=None,
        )
    )
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )
    loader._walk_full_post_load = MagicMock()
    loader._post_load_publish = MagicMock()

    monkeypatch.setattr(model_loader_module, "timing", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        model_loader_module,
        "maybe_create_moe_load_balancer",
        lambda *_args, **_kwargs: nullcontext(None),
    )
    monkeypatch.setattr(model_loader_module, "MetaInitMode", nullcontext)
    monkeypatch.setattr(
        model_loader_module.AutoModelForCausalLM, "from_config", MagicMock(return_value=model)
    )
    monkeypatch.setattr(
        model_loader_module.PostTransformConfigIdentity,
        "from_model_config",
        classmethod(lambda _cls, _config: object()),
    )
    monkeypatch.setattr(
        model_loader_module.torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(model_loader_module.torch.cuda, "empty_cache", lambda: None)

    loaded_model, moe_load_balancer = loader.load("/checkpoint", checkpoint_loader)

    assert loaded_model is model
    assert moe_load_balancer is None
    assert events == [
        "coordinate",
        "session_enter",
        "mapper_init",
        "materialize",
        "session_exit",
    ]
