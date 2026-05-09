# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GMS-specific branches in ``ModelLoader``."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch import memory as memory_mod
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm.llmapi.llm_args import LoadFormat


class _TinyModel(nn.Module):
    def __init__(self, events, *, include_draft=False):
        super().__init__()
        self._events = events
        if include_draft:
            self.draft_model = nn.Module()

    def _apply(self, fn):
        self._events.append("_apply")
        return self

    def to(self, *args, **kwargs):
        self._events.append("to")
        return self

    def load_weights(self, weights, mapper):
        self._events.append("load_weights")

    def post_load_weights(self):
        self._events.append("post_load_weights")


@contextmanager
def _moe_context(config, mapping):
    yield None


@contextmanager
def _pool_scope(events):
    events.append("pool_enter")
    yield
    events.append("pool_exit")


def _make_loader(monkeypatch, *, events, spec_config=None):
    llm_args = SimpleNamespace(
        load_format=LoadFormat.GMS,
        gms_config=SimpleNamespace(socket_path="/tmp/gms.sock", mode="auto", tag="weights"),
    )
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
        MagicMock(return_value=_TinyModel(events, include_draft=spec_config is not None)),
    )
    monkeypatch.setattr(model_loader_mod, "get_rank_model_storage", lambda _model: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    return loader


def _build_gms_backend(*, is_rw, events):
    backend = MagicMock()
    backend.connect.return_value = True
    backend.is_rw = is_rw
    if is_rw:
        backend.mem_pool_scope.side_effect = lambda _device: _pool_scope(events)
    else:

        def _materialize(_model):
            events.append("materialize")

        backend.materialize_module.side_effect = _materialize
    return backend


def _install_gms_backend(monkeypatch, backend):
    monkeypatch.setattr(memory_mod, "GMSBackend", MagicMock(return_value=backend))


def _spec_config_needing_draft_weights():
    return SimpleNamespace(
        spec_dec_mode=SimpleNamespace(need_load_draft_weights=lambda: True),
        speculative_model="/draft",
    )


@pytest.mark.parametrize(
    "is_rw, expected_events",
    [
        pytest.param(
            True,
            ["pool_enter", "load_weights", "pool_exit", "post_load_weights"],
            id="rw",
        ),
        pytest.param(
            False,
            ["post_load_weights", "materialize"],
            id="ro",
        ),
    ],
)
def test_gms_load_branch(monkeypatch, is_rw, expected_events):
    """Verify ``ModelLoader.load`` dispatches correctly per GMS lock mode.

    Cases:
        rw: the writer loads weights under the GMS memory pool, runs them
            through the standard mapping pipeline, then commits the
            populated pool for read-only consumers.
        ro: the reader runs ``post_load_weights`` to wire module aliases
            first, then GMS materializes weights via zero-copy mapping.
    """
    events = []
    loader = _make_loader(monkeypatch, events=events)
    backend = _build_gms_backend(is_rw=is_rw, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    if is_rw:
        # RW path: the checkpoint loader's returned weights flow into the
        # standard model.load_weights mapping pipeline.
        checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert loader._gms_backend is backend
    assert events == expected_events
    if is_rw:
        # RW: load weights, run them through the mapper, then commit
        # everything that landed in the GMS pool for RO consumers.
        checkpoint_loader.load_weights.assert_called_once_with("/ckpt", mapping=loader.mapping)
        loader._call_load_weights.assert_called_once()
        backend.move_untracked_params.assert_called_once_with(model)
        backend.finalize_write.assert_called_once_with(model)
    else:
        # RO: post_load_weights() must run before the GMS materialize
        # step so module aliases are wired up before zero-copy mapping.
        checkpoint_loader.load_weights.assert_not_called()
        loader._call_load_weights.assert_not_called()
        backend.materialize_module.assert_called_once_with(model)


def test_gms_rw_loader_preload_skips_mapping_pipeline(monkeypatch):
    """RW + empty ``weights`` + ``is_weights_preloaded()=True`` is legitimate.

    Mirrors the MX P2P preload scenario, where the checkpoint loader
    writes weights directly into model parameters and signals the
    preloaded state. ``ModelLoader`` must skip the standard mapping
    pipeline and still commit the populated pool for RO peers.
    """
    events = []
    loader = _make_loader(monkeypatch, events=events)
    backend = _build_gms_backend(is_rw=True, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "MX"
    checkpoint_loader.load_weights.return_value = {}
    checkpoint_loader.is_weights_preloaded.return_value = True

    model, _ = loader.load("/ckpt", checkpoint_loader)

    loader._call_load_weights.assert_not_called()
    backend.move_untracked_params.assert_called_once_with(model)
    backend.finalize_write.assert_called_once_with(model)


def test_gms_rw_no_load_and_no_preload_raises(monkeypatch):
    """RW + empty ``weights`` + ``is_weights_preloaded()=False`` is a bug.

    Without weights and without a loader-driven preload, the model is
    not populated. Committing it to GMS would expose an uninitialized
    layout to RO peers, so ``ModelLoader`` must raise.
    """
    events = []
    loader = _make_loader(monkeypatch, events=events)
    backend = _build_gms_backend(is_rw=True, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.load_weights.return_value = {}
    checkpoint_loader.is_weights_preloaded.return_value = False

    with pytest.raises(RuntimeError, match="Refusing to commit an unpopulated model"):
        loader.load("/ckpt", checkpoint_loader)

    # Pool was opened, but nothing must be committed.
    backend.finalize_write.assert_not_called()
    backend.cleanup.assert_called_once()
    assert loader._gms_backend is None


def test_gms_rw_exception_during_weight_mapping_cleans_up(monkeypatch):
    """A writer error after pool population must release the GMS session."""
    events = []
    loader = _make_loader(monkeypatch, events=events)
    loader._call_load_weights.side_effect = RuntimeError("mapping failed")
    backend = _build_gms_backend(is_rw=True, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}

    with pytest.raises(RuntimeError, match="mapping failed"):
        loader.load("/ckpt", checkpoint_loader)

    backend.cleanup.assert_called_once()
    backend.finalize_write.assert_not_called()
    assert loader._gms_backend is None


def test_gms_ro_with_spec_config_materializes_full_model_tree(monkeypatch):
    """RO materialization receives the root model, including draft_model."""
    events = []
    loader = _make_loader(
        monkeypatch,
        events=events,
        spec_config=_spec_config_needing_draft_weights(),
    )
    backend = _build_gms_backend(is_rw=False, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"

    model, _ = loader.load("/ckpt", checkpoint_loader)

    assert hasattr(model, "draft_model")
    backend.materialize_module.assert_called_once_with(model)
    checkpoint_loader.load_weights.assert_not_called()


def test_gms_connect_failure_raises(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)

    backend = MagicMock()
    backend.connect.return_value = False
    monkeypatch.setattr(memory_mod, "GMSBackend", MagicMock(return_value=backend))

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"

    with pytest.raises(RuntimeError, match="Failed to connect to GMS"):
        loader.load("/ckpt", checkpoint_loader)


def test_gms_unexpected_lock_state_raises(monkeypatch):
    """``is_rw`` must be True or False after a successful connect.

    A None value indicates an adapter bug or protocol violation; the
    loader should fail loudly rather than silently take the RO path.
    """
    events = []
    loader = _make_loader(monkeypatch, events=events)

    backend = MagicMock()
    backend.connect.return_value = True
    backend.is_rw = None  # Adapter bug: connected but lock state unset.
    monkeypatch.setattr(memory_mod, "GMSBackend", MagicMock(return_value=backend))

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"

    with pytest.raises(RuntimeError, match="lock state is unset"):
        loader.load("/ckpt", checkpoint_loader)

    # Neither branch should have run; nothing was written to the model.
    backend.mem_pool_scope.assert_not_called()
    backend.finalize_write.assert_not_called()
    backend.materialize_module.assert_not_called()
    checkpoint_loader.load_weights.assert_not_called()
