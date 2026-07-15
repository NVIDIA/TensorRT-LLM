# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for semantic-layer checkpoint orchestration in ModelLoader."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm.llmapi.llm_args import LoadFormat


class _LayerwiseModel(nn.Module):
    def __init__(self, events, *, fail_on_bucket=None):
        super().__init__()
        self._events = events
        self._fail_on_bucket = fail_on_bucket
        self.llm_checkpoint_dir = "/model-checkpoint"

    def _apply(self, _fn):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def load_weights(self, weights, weight_mapper, *, initial_bucket_loading=False):
        bucket_name = next(iter(weights))
        self._events.append(("load", bucket_name, weight_mapper, initial_bucket_loading))
        if bucket_name == self._fail_on_bucket:
            raise RuntimeError("bucket load failed")

    def post_load_weights(self):
        self._events.append(("post_load",))


class _UnsupportedModel(_LayerwiseModel):
    def load_weights(self, weights, weight_mapper):
        self._events.append(("load", next(iter(weights)), weight_mapper))


@contextmanager
def _moe_context(_config, _mapping):
    yield None


def _make_loader(monkeypatch, model, *, spec_config=None):
    loader = ModelLoader(
        llm_args=SimpleNamespace(load_format=LoadFormat.AUTO),
        mapping=MagicMock(name="mapping"),
        spec_config=spec_config,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_seq_len=128,
    )
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(name="config", mapping=SimpleNamespace())
    )
    monkeypatch.setattr(model_loader_mod, "timing", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(model_loader_mod, "maybe_create_moe_load_balancer", _moe_context)
    monkeypatch.setattr(model_loader_mod, "MetaInitMode", lambda: nullcontext())
    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM, "from_config", MagicMock(return_value=model)
    )
    monkeypatch.setattr(model_loader_mod, "get_rank_model_storage", lambda _model: 0)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return loader


def _make_checkpoint_loader(events, buckets):
    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.is_layerwise_loading_enabled.return_value = True
    mapper = MagicMock(name="weight_mapper")

    def initialize_mapper(_model, _config):
        events.append(("mapper", mapper))
        return mapper

    def iter_buckets(checkpoint_dir, **kwargs):
        events.append(("iterate", checkpoint_dir, kwargs))
        try:
            yield from buckets
        finally:
            events.append(("iterator_closed",))

    checkpoint_loader.get_initialized_weight_mapper.side_effect = initialize_mapper
    checkpoint_loader.iter_layer_weight_buckets.side_effect = iter_buckets
    return checkpoint_loader, mapper


def test_layerwise_model_loader_orders_mapper_buckets_sync_and_post_load(monkeypatch):
    events = []
    model = _LayerwiseModel(events)
    loader = _make_loader(monkeypatch, model)
    checkpoint_loader, mapper = _make_checkpoint_loader(
        events,
        [{"top.weight": object()}, {"layers.0.weight": object()}],
    )
    synchronize = MagicMock(side_effect=lambda: events.append(("synchronize",)))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    loaded_model, _ = loader.load("/checkpoint", checkpoint_loader)

    assert loaded_model is model
    assert events[:7] == [
        ("mapper", mapper),
        (
            "iterate",
            "/model-checkpoint",
            {
                "mapping": loader.mapping,
                "model": model,
                "source_identity": None,
            },
        ),
        ("load", "top.weight", mapper, True),
        ("synchronize",),
        ("load", "layers.0.weight", mapper, True),
        ("synchronize",),
        ("iterator_closed",),
    ]
    assert events[-1] == ("post_load",)
    assert synchronize.call_count == 2
    checkpoint_loader.load_weights.assert_not_called()


def test_layerwise_model_loader_rejects_unsupported_model_before_iteration(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, _UnsupportedModel(events))
    checkpoint_loader, _mapper = _make_checkpoint_loader(events, [{"top.weight": object()}])

    with pytest.raises(RuntimeError, match="does not support initial_bucket_loading"):
        loader.load("/checkpoint", checkpoint_loader)

    checkpoint_loader.get_initialized_weight_mapper.assert_not_called()
    checkpoint_loader.iter_layer_weight_buckets.assert_not_called()
    assert events == []


def test_layerwise_model_loader_rejects_draft_checkpoint_before_iteration(monkeypatch):
    events = []
    spec_config = MagicMock()
    spec_config.spec_dec_mode.need_load_draft_weights.return_value = True
    loader = _make_loader(monkeypatch, _LayerwiseModel(events), spec_config=spec_config)
    checkpoint_loader, _mapper = _make_checkpoint_loader(events, [{"top.weight": object()}])

    with pytest.raises(RuntimeError, match="separate speculative draft checkpoint"):
        loader.load("/checkpoint", checkpoint_loader)

    checkpoint_loader.get_initialized_weight_mapper.assert_not_called()
    checkpoint_loader.iter_layer_weight_buckets.assert_not_called()
    assert events == []


def test_layerwise_model_loader_closes_iterator_and_stops_after_consumer_error(monkeypatch):
    events = []
    model = _LayerwiseModel(events, fail_on_bucket="layers.1.weight")
    loader = _make_loader(monkeypatch, model)
    checkpoint_loader, mapper = _make_checkpoint_loader(
        events,
        [
            {"layers.0.weight": object()},
            {"layers.1.weight": object()},
            {"layers.2.weight": object()},
        ],
    )
    synchronize = MagicMock(side_effect=lambda: events.append(("synchronize",)))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    with pytest.raises(RuntimeError, match="bucket load failed"):
        loader.load("/checkpoint", checkpoint_loader)

    assert ("load", "layers.0.weight", mapper, True) in events
    assert ("load", "layers.1.weight", mapper, True) in events
    assert not any(event[:2] == ("load", "layers.2.weight") for event in events)
    assert events[-1] == ("iterator_closed",)
    assert synchronize.call_count == 1
    checkpoint_loader.post_load_apply.assert_not_called()
    checkpoint_loader.post_load_publish.assert_not_called()
