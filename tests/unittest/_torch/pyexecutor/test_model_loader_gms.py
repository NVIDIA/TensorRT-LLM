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

_SOURCE_IDENTITY = model_loader_mod.SourceIdentity(
    format_version=1,
    model_fingerprint="model",
    quant_fingerprint="quant",
    backend_fingerprint="backend",
    parallel_fingerprint="parallel",
    rank=0,
    shard_fingerprint="shard",
)


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

    def setup_aliases(self) -> None:
        self._events.append("setup_aliases")

    def cache_derived_state(self) -> None:
        self._events.append("cache_derived_state")

    def post_load_weights(self) -> None:
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
    loader._load_and_validate_config = MagicMock(
        return_value=SimpleNamespace(name="config", mapping=SimpleNamespace())
    )

    monkeypatch.setattr(model_loader_mod, "timing", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(model_loader_mod, "maybe_create_moe_load_balancer", _moe_context)
    monkeypatch.setattr(model_loader_mod, "MetaInitMode", lambda: nullcontext())
    # These tests stub ModelConfig, while SourceIdentity has dedicated
    # coverage. Keep this file focused on ModelLoader GMS branch behavior.
    monkeypatch.setattr(
        model_loader_mod.SourceIdentity,
        "from_model_config",
        classmethod(lambda cls, *_args, **_kwargs: _SOURCE_IDENTITY),
    )
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
        backend.get_source_identity.return_value = _SOURCE_IDENTITY

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
            [
                "pool_enter",
                "_apply",
                "to",
                "load_weights",
                "post_load_weights",
                "pool_exit",
            ],
            id="rw",
        ),
        pytest.param(
            False,
            ["setup_aliases", "materialize", "cache_derived_state"],
            id="ro",
        ),
    ],
)
def test_gms_load_branch(monkeypatch, is_rw, expected_events):
    """Verify ``ModelLoader.load`` dispatches correctly per GMS lock mode.

    Cases:
        rw: the writer opens the GMS pool BEFORE meta-tensor materialization
            and ``model.to('cuda')``, runs the entire model bring-up
            (``_apply`` for meta materialization, ``to('cuda')``, weight
            load, ``post_load_weights``) inside the pool, then commits via
            ``finalize_write`` once the scope exits.
        ro: the reader runs ``setup_aliases`` to wire module aliases, checks
            identity compatibility, materializes weights via zero-copy mapping,
            then refreshes derived state from real tensors.
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
        # ``model=model`` is passed for symmetry with the LoadFormat.AUTO
        # path (see model_loader.py); HF ignores it, MX uses it for direct
        # P2P writes when MX+GMS composition eventually lands.
        # ``source_identity`` is included so format-specific loaders can
        # publish the same compatibility fingerprint the RO path validates.
        checkpoint_loader.load_weights.assert_called_once_with(
            "/ckpt",
            mapping=loader.mapping,
            model=model,
            source_identity=loader._source_identity,
        )
        loader._call_load_weights.assert_called_once()
        backend.move_untracked_params.assert_called_once_with(model)
        backend.finalize_write.assert_called_once_with(model)
    else:
        # RO: setup_aliases() must run before the GMS materialize step so
        # module aliases are wired up before zero-copy mapping.
        checkpoint_loader.load_weights.assert_not_called()
        loader._call_load_weights.assert_not_called()
        backend.materialize_module.assert_called_once_with(model)


def test_gms_ro_materializes_between_alias_setup_and_cache_state(monkeypatch):
    events = []
    loader = _make_loader(monkeypatch, events=events)
    backend = _build_gms_backend(is_rw=False, events=events)
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"

    def record(event):
        def _append(*_args, **_kwargs):
            events.append(event)

        return _append

    checkpoint_loader.post_load_apply.side_effect = record("post_load_apply")
    checkpoint_loader.post_load_publish.side_effect = record("post_load_publish")

    # The STRICT pre-materialize identity gate runs between alias setup and
    # materialization; record it to pin the ordering without exercising the
    # comparison logic, which is covered in test_source_identity.py.
    monkeypatch.setattr(
        model_loader_mod,
        "check_weight_sharing_compatibility",
        lambda *_args, **_kwargs: events.append("check_source_identity"),
    )

    loader.load("/ckpt", checkpoint_loader)

    assert events == [
        "post_load_apply",
        "setup_aliases",
        "check_source_identity",
        "materialize",
        "cache_derived_state",
        "post_load_publish",
    ]
    assert "post_load_weights" not in events
    checkpoint_loader.load_weights.assert_not_called()
    backend.materialize_module.assert_called_once()


def test_gms_rw_post_load_runs_inside_pool_before_finalize(monkeypatch):
    """Every step that may allocate or rebind tensors must run inside the GMS pool.

    The widened ``mem_pool_scope`` covers meta-tensor materialization,
    ``model.to('cuda')``, weight load, and the post_load_* hooks. Only
    ``finalize_write`` runs after the pool closes, so the committed
    layout reflects the post-post_load model. Asserts the exact ordering
    so a future refactor cannot silently re-narrow the scope.
    """
    events = []
    loader = _make_loader(monkeypatch, events=events)
    backend = _build_gms_backend(is_rw=True, events=events)

    backend.move_untracked_params.side_effect = lambda _model: events.append(
        "move_untracked_params"
    )
    backend.finalize_write.side_effect = lambda _model: events.append("finalize_write")
    _install_gms_backend(monkeypatch, backend)

    checkpoint_loader = MagicMock(name="checkpoint_loader")
    checkpoint_loader.checkpoint_format = "HF"
    checkpoint_loader.load_weights.return_value = {"weight": MagicMock()}
    checkpoint_loader.post_load_apply.side_effect = lambda *_a, **_kw: events.append(
        "post_load_apply"
    )
    checkpoint_loader.post_load_publish.side_effect = lambda *_a, **_kw: events.append(
        "post_load_publish"
    )

    loader.load("/ckpt", checkpoint_loader)

    assert events == [
        "pool_enter",
        "_apply",
        "to",
        "load_weights",
        "post_load_apply",
        "post_load_publish",
        "post_load_weights",
        "move_untracked_params",
        "pool_exit",
        "finalize_write",
    ]

    # Belt-and-braces: inside the events list, every "interesting" step
    # must sit between pool_enter and pool_exit, with finalize_write the
    # only thing strictly after.
    enter_idx = events.index("pool_enter")
    exit_idx = events.index("pool_exit")
    finalize_idx = events.index("finalize_write")
    assert finalize_idx > exit_idx, "finalize_write must follow pool exit"
    for inside in (
        "_apply",
        "to",
        "load_weights",
        "post_load_apply",
        "post_load_publish",
        "post_load_weights",
        "move_untracked_params",
    ):
        idx = events.index(inside)
        assert enter_idx < idx < exit_idx, (
            f"{inside!r} (idx={idx}) must run inside the pool (enter={enter_idx}, exit={exit_idx})"
        )


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
