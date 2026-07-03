# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the KV-cache compression manager framework
(``BaseKVCacheCompressionManager`` in ``resource_manager.py``) — the
``BaseResourceManager``-based single-manager design.

Covers:
- :class:`BaseKVCacheCompressionManager` contract: the four lifecycle hooks
  default to no-op, zero resource counts, and it inherits
  :class:`BaseResourceManager` (so PyExecutor auto-drives it once registered).
- The resource-manager API -> lifecycle-hook translation, gated on PyExecutor's
  own signals: ``prepare_resources`` fires ``on_request_init`` on each
  request's first prefill chunk (``is_first_context_chunk``);
  ``update_resources`` fires ``on_context_step_end`` for each request in
  ``context_requests_last_chunk`` + one ``on_generation_step_end`` per
  iteration; ``free_resources`` fires ``on_request_finish``.
- :func:`create_kv_cache_compression_manager` factory.

The base class lives in ``resource_manager.py`` (it is a resource manager, not a
sparse-attention backend); the ``create_kv_cache_compression_manager`` factory
lives in ``_util.py`` next to ``_create_kv_cache_manager``.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseKVCacheCompressionManager,
    BaseResourceManager,
)

# ---------------------------------------------------------------------- #
# Mock infra: in-memory managers / requests (avoid touching V2 / model).  #
# ---------------------------------------------------------------------- #


class _RecordingMixin:
    """Mixin that records every hook invocation, to assert the RM-API -> hook
    translation without real algorithm side-effects."""

    def __init__(self, kv_cache_manager, record_list, name="m"):
        super().__init__(kv_cache_manager)
        self._record_list = record_list
        self._name = name

    def _record(self, hook_name: str):
        self._record_list.append(f"{self._name}:{hook_name}")


class _MockCompressionManager(_RecordingMixin, BaseKVCacheCompressionManager):
    """Mock manager that records the four lifecycle hooks."""

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_context_step_end(self, request, metadata):
        self._record("on_context_step_end")

    def on_generation_step_end(self, scheduled_batch, attn_metadata):
        self._record("on_generation_step_end")

    def on_request_finish(self, request):
        self._record("on_request_finish")


@pytest.fixture
def fake_kv_cache_manager():
    """A stand-in KVCacheManagerV2. The framework reads enable_block_reuse off
    it in __init__; default it to False, like a normal run with reuse off."""
    m = MagicMock(name="fake_KVCacheManagerV2")
    m.enable_block_reuse = False
    return m


def _req(rid, first_chunk=True):
    r = MagicMock(name=f"req{rid}")
    r.py_request_id = rid
    r.is_first_context_chunk = first_chunk
    return r


def _batch(context=(), generation=(), last_chunk=()):
    b = MagicMock(name="ScheduledRequests")
    b.context_requests = list(context)
    b.generation_requests = list(generation)
    b.context_requests_last_chunk = list(last_chunk)
    return b


# ---------------------------------------------------------------------- #
# 1. BaseKVCacheCompressionManager contract                               #
# ---------------------------------------------------------------------- #


class TestBaseABC:
    def test_inherits_base_resource_manager(self):
        # So PyExecutor's main loop auto-invokes prepare/update/free_resources.
        assert issubclass(BaseKVCacheCompressionManager, BaseResourceManager)

    def test_four_hooks_default_noop(self, fake_kv_cache_manager):
        m = BaseKVCacheCompressionManager(fake_kv_cache_manager)
        meta = MagicMock()
        assert m.on_request_init(MagicMock()) is None
        assert m.on_context_step_end(MagicMock(), meta) is None
        assert m.on_generation_step_end(MagicMock(), meta) is None
        assert m.on_request_finish(MagicMock()) is None

    def test_hooks_accept_extra_kwargs(self, fake_kv_cache_manager):
        # **kwargs lets the framework pass new args later without breaking
        # existing overrides.
        m = BaseKVCacheCompressionManager(fake_kv_cache_manager)
        assert m.on_request_init(MagicMock(), future_arg=1) is None
        assert m.on_generation_step_end(MagicMock(), MagicMock(), future_arg=1) is None

    def test_resource_counts_are_zero(self, fake_kv_cache_manager):
        m = BaseKVCacheCompressionManager(fake_kv_cache_manager)
        # The manager owns no physical resources (the V2 cache manager does),
        # so it must not gate the scheduler.
        assert m.get_max_resource_count() == 0
        assert m.get_needed_resource_to_completion(MagicMock()) == 0


# ---------------------------------------------------------------------- #
# 2. Resource-manager API -> lifecycle-hook translation                   #
#    (gated on PyExecutor signals, no manager-side bookkeeping)            #
# ---------------------------------------------------------------------- #


class TestResourceManagerAPI:
    def test_prepare_fires_init_on_first_chunk_only(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        # First prefill chunk -> init fires.
        m.prepare_resources(_batch(context=[_req(1, first_chunk=True)]))
        # A later (non-first) chunk of the same request -> no re-init.
        m.prepare_resources(_batch(context=[_req(1, first_chunk=False)]))
        assert rec == ["s:on_request_init"]

    def test_update_fires_context_end_on_last_chunk(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        req = _req(1)
        # Request's final prefill chunk this iteration -> context_step_end fires.
        m.update_resources(_batch(generation=[req], last_chunk=[req]), attn_metadata=MagicMock())
        assert "s:on_context_step_end" in rec
        assert rec[-1] == "s:on_generation_step_end"
        # Subsequent decode iteration (not in last_chunk) -> no context_step_end.
        rec.clear()
        m.update_resources(_batch(generation=[req]))
        assert rec == ["s:on_generation_step_end"]

    def test_step_end_fires_once_per_iteration(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        m.update_resources(_batch(generation=[_req(1), _req(2)]))
        assert rec.count("s:on_generation_step_end") == 1

    def test_free_fires_finish(self, fake_kv_cache_manager):
        rec = []
        m = _MockCompressionManager(fake_kv_cache_manager, rec, "s")
        m.free_resources(_req(1))
        assert rec == ["s:on_request_finish"]


# ---------------------------------------------------------------------- #
# 3. Factory                                                              #
# ---------------------------------------------------------------------- #


class TestFactory:
    def test_returns_none_when_no_algorithm_registered(self, fake_kv_cache_manager):
        # Framework-only: no concrete algorithm ships, so any config -> None.
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        assert create_kv_cache_compression_manager(cfg, fake_kv_cache_manager) is None

    def test_warns_for_unregistered_algorithm(self, fake_kv_cache_manager):
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        with patch.object(util_mod, "logger") as mock_logger:
            create_kv_cache_compression_manager(cfg, fake_kv_cache_manager)
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------- #
# 4. Canonical names live in resource_manager, not in the sparse module   #
# ---------------------------------------------------------------------- #


class TestCanonicalImports:
    def test_names_importable_from_canonical_modules(self):
        from tensorrt_llm._torch.pyexecutor import _util, resource_manager

        # Base class stays in resource_manager (it IS a resource manager); the
        # factory lives in _util next to _create_kv_cache_manager.
        assert hasattr(resource_manager, "BaseKVCacheCompressionManager")
        assert hasattr(_util, "create_kv_cache_compression_manager")

    def test_names_not_in_sparse_module(self):
        # The framework moved out of attention_backend/sparse/ (it is not a
        # sparse-attention backend); the sparse package no longer exports it.
        from tensorrt_llm._torch.attention_backend import sparse

        assert not hasattr(sparse, "BaseKVCacheCompressionManager")
        assert not hasattr(sparse, "create_kv_cache_compression_manager")


# ---------------------------------------------------------------------- #
# 5. Block-reuse guard                                                    #
# ---------------------------------------------------------------------- #


class TestBlockReuseGuard:
    """__init__ refuses block reuse for a method that changes the stored keys
    and values, the same check RocketKVCacheManager makes."""

    def _mgr(self, enable_block_reuse):
        m = MagicMock(name="KVCacheManagerV2")
        m.enable_block_reuse = enable_block_reuse
        return m

    def test_raises_when_reuse_on(self):
        with pytest.raises(ValueError, match="block reuse"):
            BaseKVCacheCompressionManager(self._mgr(enable_block_reuse=True))

    def test_ok_when_reuse_off(self):
        BaseKVCacheCompressionManager(self._mgr(enable_block_reuse=False))  # no raise
