# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader, _construct_checkpoint_loader
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("loader_cls", [MistralCheckpointLoader, MXCheckpointLoader])
def test_hf_subclasses_keep_polymorphic_load_weights(loader_cls):
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
    striped = _construct_checkpoint_loader(
        "pytorch",
        None,
        "HF",
        checkpoint_io_policy="rank_striped_read_ahead",
        partial_model_loading=True,
    )
    mx = _construct_checkpoint_loader(
        "pytorch",
        None,
        "MX",
        checkpoint_io_policy="rank_striped_read_ahead",
        partial_model_loading=True,
    )

    assert isinstance(striped.weight_loader, HfWeightLoader)
    assert striped.weight_loader.checkpoint_io_policy == "rank_striped_read_ahead"
    assert striped.weight_loader._partial_model_loading
    assert mx.checkpoint_format == "MX"
    assert mx.weight_loader.checkpoint_io_policy == "native"


def test_construct_checkpoint_loader_preserves_custom_hf_weight_loader(monkeypatch):
    class _CustomWeightLoader:
        def __init__(self):
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

    with pytest.raises(ValueError, match="built-in HfWeightLoader"):
        _construct_checkpoint_loader(
            "pytorch",
            None,
            "HF",
            checkpoint_io_policy="rank_striped_read_ahead",
        )


class _SessionCheckpointLoader:
    checkpoint_format = "HF"

    def __init__(self, events, checkpoint_dir="/checkpoint"):
        self.events = events
        self.checkpoint_dir = checkpoint_dir

    @contextmanager
    def open_weight_session(self, checkpoint_dir, **kwargs):
        assert checkpoint_dir == self.checkpoint_dir
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


def test_model_loader_session_spans_mapper_and_materialization():
    events = []
    model = MagicMock()
    checkpoint_loader = _SessionCheckpointLoader(events)
    loader = ModelLoader.__new__(ModelLoader)
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


def test_draft_session_spans_mapper_and_materialization(monkeypatch):
    events = []
    checkpoint_loader = _SessionCheckpointLoader(events, "/draft")
    model = MagicMock()
    model.draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["DraftForCausalLM"])
    )
    loader = ModelLoader.__new__(ModelLoader)
    loader.spec_config = SimpleNamespace(speculative_model="/draft")
    loader.mapping = object()
    loader._call_load_weights = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("materialize")
    )
    draft_mapper = MagicMock()
    draft_mapper.init_model_and_config.side_effect = lambda *_args: events.append("mapper_init")
    monkeypatch.setattr(
        model_loader_module.AutoCheckpointMapper, "get", MagicMock(return_value=draft_mapper)
    )

    loader._materialize_draft_checkpoint_weights(checkpoint_loader, model)

    assert events == [
        "session_enter",
        "mapper_init",
        "materialize",
        "session_exit",
    ]
