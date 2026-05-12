# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MX-specific branches in ``ModelLoader``."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm.llmapi.llm_args import LoadFormat


class _LinearStub(nn.Module):
    def post_load_weights(self):
        pass


class _DraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = _LinearStub()


class _TinyModel(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.linear = _LinearStub()
        self.draft_model = _DraftModel()
        self.draft_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(architectures=["DraftArch"])
        )
        self._events = events

    def _apply(self, fn):
        # The test is about ModelLoader's MX branching, not CUDA allocation.
        return self

    def to(self, *args, **kwargs):
        return self

    def load_weights(self, weights, mapper):
        self._events.append("load_weights")

    def load_draft_weights(self, weights, mapper):
        self._events.append("load_draft_weights")

    def post_load_weights(self):
        self._events.append("post_load_weights")


@contextmanager
def _moe_context(config, mapping):
    yield None


def _make_loader(monkeypatch, *, events, spec_config=None):
    llm_args = SimpleNamespace(load_format=LoadFormat.AUTO)
    loader = ModelLoader(
        llm_args=llm_args,
        mapping=MagicMock(name="mapping"),
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    loader._call_load_weights = MagicMock(
        side_effect=lambda fn, weights, mapper, **kwargs: fn(weights, mapper)
    )
    loader._load_and_validate_config = MagicMock(return_value=SimpleNamespace(name="config"))

    monkeypatch.setattr(model_loader_mod, "timing", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(model_loader_mod, "maybe_create_moe_load_balancer", _moe_context)
    monkeypatch.setattr(model_loader_mod, "MetaInitMode", lambda: nullcontext())
    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM,
        "from_config",
        MagicMock(return_value=_TinyModel(events)),
    )
    monkeypatch.setattr(model_loader_mod, "get_rank_model_storage", lambda _model: 0)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    return loader


def test_mx_success_initializes_mapper_skips_weight_mapping_and_reload_works(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    checkpoint_loader.load_weights.return_value = {}

    model, _ = loader.load("/ckpt", checkpoint_loader)

    checkpoint_loader.load_weights.assert_called_once()
    _args, kwargs = checkpoint_loader.load_weights.call_args
    assert kwargs["mapping"] is loader.mapping
    assert kwargs["model"] is model
    assert loader._call_load_weights.call_count == 0
    checkpoint_loader.get_initialized_weight_mapper.assert_called_once()
    assert loader.weight_mapper is checkpoint_loader.get_initialized_weight_mapper.return_value
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model, checkpoint_dir="/ckpt", weights_preloaded=True
    )

    # reload() uses self.weight_mapper unconditionally; MX success must
    # initialize it even though the initial load skipped _call_load_weights.
    loader.reload(model, {"reloaded": MagicMock()})
    assert loader._call_load_weights.call_count == 1


def test_mx_partial_fallback_merges_returned_weights(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = True
    fallback_weights = {"mismatched.weight": MagicMock()}
    checkpoint_loader.load_weights.return_value = fallback_weights

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    load_fn, weights, mapper = loader._call_load_weights.call_args.args
    assert load_fn == model.load_weights
    assert weights is fallback_weights
    assert mapper is loader.weight_mapper
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model, checkpoint_dir="/ckpt", weights_preloaded=True
    )


def test_mx_fallback_runs_standard_weight_mapping(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.is_weights_preloaded.return_value = False
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.get_initialized_weight_mapper.return_value = MagicMock()

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._call_load_weights.call_count == 1
    assert events[0] == "load_weights"
    assert "post_load_weights" in events
    checkpoint_loader.post_load_publish.assert_called_once_with(
        model, checkpoint_dir="/ckpt", weights_preloaded=False
    )
