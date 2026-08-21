# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the KV-cache compression manager lifecycle and factory."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheCompressionManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm.llmapi.llm_args import KvCacheCompressionConfig

# ---------------------------------------------------------------------- #
# Mock infra: in-memory managers / requests (avoid touching V2 / model).  #
# ---------------------------------------------------------------------- #


class _RecordingMixin:
    """Mixin that records every hook invocation, to assert the RM-API -> hook
    translation without real algorithm side-effects."""

    def __init__(self, kv_cache_manager, record_list, name="m"):
        super().__init__(_compression_config())
        self.bind_kv_cache_managers(kv_cache_manager)
        self._record_list = record_list
        self._name = name

    def _record(self, hook_name: str):
        self._record_list.append(f"{self._name}:{hook_name}")


class _MockCompressionManager(_RecordingMixin, KVCacheCompressionManager):
    """Mock manager that records iteration lifecycle hooks."""

    def on_request_init(self, request):
        self._record("on_request_init")

    def on_context_step_end(self, requests):
        self._record(f"on_context_step_end[{len(requests)}]")

    def on_generation_step_end(self, scheduled_batch):
        self._record("on_generation_step_end")

    def on_request_finish(self, request):
        self._record("on_request_finish")


class _PhysicalLengthChangingConfig(KvCacheCompressionConfig):
    changes_physical_kv_length: ClassVar[bool] = True


class _BlockReuseCompatibleConfig(KvCacheCompressionConfig):
    def supports_block_reuse(self) -> bool:
        return True


def _compression_config() -> KvCacheCompressionConfig:
    return KvCacheCompressionConfig(algorithm="test")


def _v2_manager(*, is_draft: bool):
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = False
    manager.kv_compression_manages_history = False
    manager.is_draft = is_draft
    return manager


@pytest.fixture
def fake_kv_cache_manager():
    """A stand-in KVCacheManagerV2 for compression-manager unit tests."""
    return _v2_manager(is_draft=False)


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
# 1. KVCacheCompressionManager contract                                   #
# ---------------------------------------------------------------------- #


class TestBaseABC:
    def test_inherits_base_resource_manager(self):
        # So PyExecutor's main loop auto-invokes prepare/update/free_resources.
        assert issubclass(KVCacheCompressionManager, BaseResourceManager)

    def test_lifecycle_hooks_default_noop(self, fake_kv_cache_manager):
        m = KVCacheCompressionManager(_compression_config())
        m.bind_kv_cache_managers(fake_kv_cache_manager)
        assert m.on_request_init(MagicMock()) is None
        assert m.on_context_step_end([MagicMock()]) is None
        assert m.on_generation_step_begin(MagicMock()) is None
        assert m.on_generation_step_end(MagicMock()) is None
        assert m.on_request_finish(MagicMock()) is None

    def test_hooks_accept_extra_kwargs(self, fake_kv_cache_manager):
        # **kwargs lets the framework pass new args later without breaking
        # existing overrides.
        m = KVCacheCompressionManager(_compression_config())
        m.bind_kv_cache_managers(fake_kv_cache_manager)
        assert m.on_request_init(MagicMock(), future_arg=1) is None
        assert m.on_generation_step_end(MagicMock(), future_arg=1) is None

    def test_resource_counts_are_zero(self, fake_kv_cache_manager):
        m = KVCacheCompressionManager(_compression_config())
        m.bind_kv_cache_managers(fake_kv_cache_manager)
        # The manager owns no physical resources (the V2 cache manager does),
        # so it must not gate the scheduler.
        assert m.get_max_resource_count() == 0
        assert m.get_needed_resource_to_completion(MagicMock()) == 0

    def test_physical_length_change_marks_target_and_draft_v2(self):
        # The draft cache is compacted together with the target, so both
        # managers diverge from the logical length in the same way.
        target = _v2_manager(is_draft=False)
        draft = _v2_manager(is_draft=True)

        config = _PhysicalLengthChangingConfig(algorithm="test")
        manager = KVCacheCompressionManager(config)
        manager.bind_kv_cache_managers(target, draft)

        assert manager.kv_cache_manager is target
        assert manager.draft_kv_cache_manager is draft
        assert manager.has_independent_draft_kv_cache
        assert target.kv_compression_manages_history is True
        assert draft.kv_compression_manages_history is True

    def test_rejects_non_v2_ownership(self):
        config = _compression_config()
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            KVCacheCompressionManager(config).bind_kv_cache_managers(MagicMock())
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            KVCacheCompressionManager(config).bind_kv_cache_managers(
                _v2_manager(is_draft=False), MagicMock()
            )

    def test_request_field_defaults_to_zero(self):
        """LlmRequest carries the compression count (the manager's only
        channel to the runtime); a fresh request must default to 0 so runs
        without a compression manager are unchanged."""
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
        from tensorrt_llm.bindings import SamplingConfig

        request = LlmRequest(
            request_id=1,
            max_new_tokens=8,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert request.py_num_compressed_tokens == 0


# ---------------------------------------------------------------------- #
# 2. Resource-manager API -> lifecycle-hook translation                   #
#    (gated on PyExecutor signals, no manager-side bookkeeping)            #
# ---------------------------------------------------------------------- #


class TestResourceManagerAPI:
    def test_target_update_receives_metadata_before_final_compression(self):
        calls = []
        metadata = MagicMock(name="attention_metadata")
        draft = MagicMock(name="draft_kv_cache_manager")
        target = MagicMock(name="target_kv_cache_manager")
        compression = MagicMock(name="compression_manager")
        draft.update_resources.side_effect = lambda *args: calls.append(("draft", args))
        target.update_resources.side_effect = lambda *args: calls.append(("target", args))
        compression.update_resources.side_effect = lambda *args: calls.append(("compression", args))
        manager = ResourceManager(
            {
                ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft,
                ResourceManagerType.KV_CACHE_MANAGER: target,
                ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression,
            }
        )
        batch = _batch(generation=[_req(1)])

        manager.update_resources(batch, metadata, 2.0)

        assert calls == [
            ("draft", (batch,)),
            ("target", (batch, metadata, 2.0)),
            ("compression", (batch,)),
        ]

    def test_real_v2_target_receives_relocation_metadata(self):
        from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kv_cache_v2_module

        target = _v2_manager(is_draft=False)
        target.kv_cache_map = {}
        batch = _batch(generation=[_req(1)])
        metadata = MagicMock(name="attention_metadata")
        manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: target})

        with patch.object(kv_cache_v2_module, "_update_kv_cache_draft_token_location") as relocate:
            manager.update_resources(batch, metadata, 2.0)

        relocate.assert_called_once_with(target, batch, metadata, 2.0)

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
        # Final prefill chunks this iteration -> one batched context_step_end.
        req2 = _req(2)
        m.update_resources(_batch(generation=[req], last_chunk=[req, req2]))
        assert "s:on_context_step_end[2]" in rec
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
    def test_returns_none_when_no_algorithm_registered(self):
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        assert create_kv_cache_compression_manager(cfg) is None

    def test_warns_for_unregistered_algorithm(self):
        cfg = MagicMock()
        cfg.algorithm = "made_up_method"
        with patch.object(util_mod, "logger") as mock_logger:
            create_kv_cache_compression_manager(cfg)
            mock_logger.warning.assert_called_once()

    def test_triattention_requires_sm100_family(self):
        cfg = MagicMock()
        cfg.algorithm = "triattention"
        with (
            patch.object(util_mod, "is_sm_100f", return_value=False),
            pytest.raises(RuntimeError, match="SM100-family"),
        ):
            create_kv_cache_compression_manager(cfg)

    def test_capabilities_default_false(self):
        config = KvCacheCompressionConfig(algorithm="offload")
        target = _v2_manager(is_draft=False)
        assert config.changes_physical_kv_length is False
        assert config.supports_block_reuse() is False
        assert config.supports_speculative_decoding() is False
        m = KVCacheCompressionManager(config)
        m.bind_kv_cache_managers(target)
        assert target.kv_compression_manages_history is False
        assert not hasattr(m, "spec_config")

    def test_spec_gate_uses_config_capability(self):
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_compatibility
        from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

        config = KvCacheCompressionConfig(algorithm="offload")
        kv_cache_config = SimpleNamespace(enable_block_reuse=False)
        spec_config = SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.DFLASH)
        with pytest.raises(ValueError, match="speculative decoding"):
            validate_kv_cache_compression_compatibility(
                config,
                kv_cache_config,
                spec_config,
            )
        validate_kv_cache_compression_compatibility(
            config,
            kv_cache_config,
            None,
        )


class TestKvCacheCreatorLifecycle:
    def test_estimation_still_creates_triattention_manager(self):
        config = SimpleNamespace(algorithm="triattention")
        pretrained_config = object()
        expected_manager = SimpleNamespace(provides_cold_page_codec=False)
        creator = object.__new__(util_mod.KvCacheCreator)
        creator._skip_est = False
        creator._max_seq_len = 1024
        creator._kv_cache_config = SimpleNamespace()
        creator._llm_args = SimpleNamespace(kv_cache_compression_config=config)
        creator._model_engine = SimpleNamespace(
            model=SimpleNamespace(model_config=SimpleNamespace(pretrained_config=pretrained_config))
        )
        creator._draft_model_engine = None
        creator._is_encoder_decoder = MagicMock(return_value=False)
        creator._should_create_separate_draft_kv_cache = MagicMock(return_value=False)
        target_manager = object()
        creator._create_kv_cache_manager = MagicMock(return_value=target_manager)

        with patch.object(
            util_mod,
            "create_kv_cache_compression_manager",
            return_value=expected_manager,
        ) as factory:
            resources = {}
            creator.build_managers(resources, estimating_kv_cache=True)

        factory.assert_called_once_with(
            config,
            pretrained_config=pretrained_config,
        )
        assert resources[ResourceManagerType.KV_CACHE_MANAGER] is target_manager
        assert resources[ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER] is expected_manager

    def test_teardown_pops_and_shuts_down_compression_manager(self):
        creator = object.__new__(util_mod.KvCacheCreator)
        compression_manager = MagicMock()
        resources = {
            ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER: compression_manager,
            ResourceManagerType.KV_CACHE_MANAGER: MagicMock(),
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: None,
        }

        creator.teardown_managers(resources)

        compression_manager.shutdown.assert_called_once_with()
        assert ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in resources


@pytest.mark.parametrize(
    "provides_cold_page_codec",
    (True, False),
    ids=("cold-codec", "iteration-manager"),
)
def test_build_routes_compression_manager_by_capabilities(provides_cold_page_codec):
    creator = object.__new__(util_mod.KvCacheCreator)
    creator._skip_est = False
    creator._max_seq_len = 1024
    creator._kv_cache_config = SimpleNamespace()
    compression_config = SimpleNamespace(algorithm="triattention")
    pretrained_config = object()
    creator._llm_args = SimpleNamespace(kv_cache_compression_config=compression_config)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(pretrained_config=pretrained_config))
    )
    creator._draft_model_engine = None
    creator._kv_connector_manager = None
    creator._is_kv_cache_manager_v2 = True
    creator._fp8_ctx_mla_kv_len_cap = None
    creator._is_encoder_decoder = MagicMock(return_value=False)
    creator._should_create_separate_draft_kv_cache = MagicMock(return_value=True)
    creator._needs_gpu_kv_cache_budget_split = MagicMock(return_value=False)

    build_order = []
    compression_manager = SimpleNamespace(
        provides_cold_page_codec=provides_cold_page_codec,
    )
    target_config = object()
    draft_config = object()
    target_manager = SimpleNamespace()
    draft_manager = object()
    creator._split_kv_cache_budget_for_draft = MagicMock(
        side_effect=[(target_config, draft_config), (target_config, draft_config)]
    )
    creator._create_kv_cache_manager = MagicMock(
        side_effect=lambda *_args, **_kwargs: build_order.append("target") or target_manager
    )
    creator._create_one_model_draft_kv_cache_manager = MagicMock(
        side_effect=lambda *_args, **_kwargs: build_order.append("draft") or draft_manager
    )

    resources = {}
    with patch.object(
        util_mod,
        "create_kv_cache_compression_manager",
        side_effect=lambda *_args, **_kwargs: build_order.append("factory")
        or compression_manager,
    ) as factory:
        creator.build_managers(resources)

    expected_codec_provider = compression_manager if provides_cold_page_codec else None
    factory.assert_called_once_with(
        compression_config,
        pretrained_config=pretrained_config,
    )
    assert (
        creator._create_kv_cache_manager.call_args.kwargs["cold_page_codec_provider"]
        is expected_codec_provider
    )
    assert (
        creator._create_one_model_draft_kv_cache_manager.call_args.kwargs[
            "cold_page_codec_provider"
        ]
        is expected_codec_provider
    )
    assert resources[ResourceManagerType.KV_CACHE_MANAGER] is target_manager
    assert resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] is draft_manager
    assert resources[ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER] is compression_manager
    assert build_order == ["factory", "target", "draft"]


# ---------------------------------------------------------------------- #
# 4. Canonical names live in resource_manager, not in the sparse module   #
# ---------------------------------------------------------------------- #


class TestCanonicalImports:
    def test_names_importable_from_canonical_modules(self):
        from tensorrt_llm._torch.pyexecutor import _util, resource_manager

        # Base class stays in resource_manager (it IS a resource manager); the
        # factory lives in _util next to _create_kv_cache_manager.
        assert hasattr(resource_manager, "KVCacheCompressionManager")
        assert hasattr(_util, "create_kv_cache_compression_manager")

    def test_names_not_in_sparse_module(self):
        # The framework moved out of attention_backend/sparse/ (it is not a
        # sparse-attention backend); the sparse package no longer exports it.
        from tensorrt_llm._torch.attention_backend import sparse

        assert not hasattr(sparse, "KVCacheCompressionManager")
        assert not hasattr(sparse, "create_kv_cache_compression_manager")


# ---------------------------------------------------------------------- #
# 5. Compression compatibility gate                                      #
# ---------------------------------------------------------------------- #


class TestCompressionCompatibility:
    def test_raises_when_reuse_on(self):
        config = _compression_config()
        with pytest.raises(ValueError, match="block reuse"):
            util_mod.validate_kv_cache_compression_compatibility(
                config,
                SimpleNamespace(enable_block_reuse=True),
                None,
            )

    def test_ok_when_reuse_off(self):
        util_mod.validate_kv_cache_compression_compatibility(
            _compression_config(),
            SimpleNamespace(enable_block_reuse=False),
            None,
        )

    def test_block_reuse_capability_allows_reuse(self):
        config = _BlockReuseCompatibleConfig(algorithm="test")
        util_mod.validate_kv_cache_compression_compatibility(
            config,
            SimpleNamespace(enable_block_reuse=True),
            None,
        )
