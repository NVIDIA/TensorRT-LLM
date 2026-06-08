"""Unit tests for the RocketKV pipeline.

Covers:

- :class:`RocketKVSparseAttentionConfig` Pydantic discriminator dispatch via
  the ``SparseAttentionConfig`` Annotated Union (``algorithm="rocketkv"``).
- Coexistence with the legacy :class:`RocketSparseAttentionConfig`
  (``algorithm="rocket"``) which routes to the ``RocketKVCacheManager``
  cache-manager subclass — both must remain selectable.
- :attr:`BaseSparseAttentionConfig.is_behavior_layer_method` property
  semantics: ``True`` for ``rocketkv``, ``False`` for legacy ``rocket``.
- :func:`create_sparse_attention_manager` factory dispatch returning a
  :class:`RocketKV` instance for ``rocketkv``, ``None`` for legacy
  ``rocket`` (which goes through ``get_sparse_attn_kv_cache_manager``).
- :class:`RocketKV` hook contract and capability ClassVars.
- :class:`KVCacheManagerV2` isinstance assertion at factory level.

This is a pipeline-level wire test covering config dispatch and factory
adaptation, not the KT-summary / HSA-mask numerics.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from tensorrt_llm._torch.attention_backend.sparse import (
    SparseAttentionExecutor,
    create_sparse_attention_manager,
)
from tensorrt_llm._torch.attention_backend.sparse.rocketkv import RocketKV
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    RocketKVSparseAttentionConfig,
    RocketSparseAttentionConfig,
    SkipSoftmaxAttentionConfig,
    SparseAttentionConfig,
)

# ---------------------------------------------------------------------------
# 1. Pydantic discriminator: ``algorithm="rocketkv"`` resolves to the new
#    config class (NOT the legacy ``RocketSparseAttentionConfig``).


class TestPydanticDiscriminator:
    def test_rocketkv_string_resolves_to_rocketkv_config(self):
        adapter = TypeAdapter(SparseAttentionConfig)
        cfg = adapter.validate_python({"algorithm": "rocketkv"})
        assert isinstance(cfg, RocketKVSparseAttentionConfig)
        assert cfg.algorithm == "rocketkv"

    def test_legacy_rocket_string_still_resolves_to_legacy_rocket_config(self):
        """Coexistence check: legacy ``algorithm="rocket"`` MUST keep routing
        to the ``RocketSparseAttentionConfig`` after the rocketkv addition."""
        adapter = TypeAdapter(SparseAttentionConfig)
        cfg = adapter.validate_python({"algorithm": "rocket"})
        assert isinstance(cfg, RocketSparseAttentionConfig)
        assert cfg.algorithm == "rocket"

    def test_unknown_algorithm_rejected(self):
        adapter = TypeAdapter(SparseAttentionConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({"algorithm": "rocketkv-mistyped"})

    def test_default_field_values(self):
        cfg = RocketKVSparseAttentionConfig()
        assert cfg.algorithm == "rocketkv"
        assert cfg.page_size == 16
        assert cfg.prompt_budget == 2048
        assert cfg.kt_cache_dtype == "bfloat16"
        assert cfg.kt_tokens_per_block is None

    def test_custom_field_values(self):
        cfg = RocketKVSparseAttentionConfig(
            page_size=8,
            prompt_budget=4096,
            kt_cache_dtype="float8_e5m2",
            kt_tokens_per_block=4,
        )
        assert cfg.page_size == 8
        assert cfg.prompt_budget == 4096
        assert cfg.kt_cache_dtype == "float8_e5m2"
        assert cfg.kt_tokens_per_block == 4


# ---------------------------------------------------------------------------
# 2. is_behavior_layer_method property semantics


class TestBehaviorLayerMethodProperty:
    def test_rocketkv_is_NOT_behavior_layer(self):
        # RocketKV is a MEMORY-layer method: it owns a cache-manager
        # subclass (RocketKVCacheManagerV2 registers the KT pool) plus an
        # attention shim, so is_behavior_layer_method is False (dispatch
        # is gated on coordinator presence, not this flag).
        cfg = RocketKVSparseAttentionConfig()
        assert cfg.is_behavior_layer_method is False

    def test_legacy_rocket_is_NOT_behavior_layer(self):
        """Coexistence: legacy ``rocket`` config keeps routing to the
        memory-layer path (cache-manager subclass)."""
        cfg = RocketSparseAttentionConfig()
        assert cfg.is_behavior_layer_method is False

    def test_dsa_and_skipsoftmax_are_NOT_behavior_layer(self):
        for cfg in (DeepSeekSparseAttentionConfig(), SkipSoftmaxAttentionConfig()):
            assert cfg.is_behavior_layer_method is False


# ---------------------------------------------------------------------------
# 3. create_sparse_attention_manager factory dispatch


class TestFactoryDispatch:
    def _fake_v2(self):
        """Build a MagicMock that passes ``isinstance(_, KVCacheManagerV2)``."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

        m = MagicMock(spec=KVCacheManagerV2)
        return m

    def test_rocketkv_returns_rocketkv_instance_with_v2(self):
        cfg = RocketKVSparseAttentionConfig(page_size=16, prompt_budget=2048)
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert isinstance(mgr, RocketKV)
        assert isinstance(mgr, SparseAttentionExecutor)
        # Constructor params forwarded
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048

    def test_rocketkv_custom_values_forwarded(self):
        cfg = RocketKVSparseAttentionConfig(
            page_size=8,
            prompt_budget=4096,
            kt_cache_dtype="float8_e5m2",
            kt_tokens_per_block=4,
        )
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert isinstance(mgr, RocketKV)
        assert mgr.page_size == 8
        assert mgr.prompt_budget == 4096
        # The executor stores the torch dtype, not the string from the
        # Pydantic config. Config field stays string.
        import torch

        assert mgr.kt_cache_dtype is torch.float8_e5m2
        assert mgr.kt_tokens_per_block == 4

    def test_legacy_rocket_returns_none(self):
        """Coexistence: legacy ``rocket`` config returns None from this
        factory (it goes through ``get_sparse_attn_kv_cache_manager`` which
        returns the cache-manager subclass instead)."""
        cfg = RocketSparseAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_dsa_returns_none(self):
        cfg = DeepSeekSparseAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_skip_softmax_returns_none(self):
        cfg = SkipSoftmaxAttentionConfig()
        mgr = create_sparse_attention_manager(cfg, self._fake_v2())
        assert mgr is None

    def test_rocketkv_raises_type_error_for_non_v2_cache_mgr(self):
        cfg = RocketKVSparseAttentionConfig()
        # Plain MagicMock without spec → fails isinstance(KVCacheManagerV2)
        not_v2 = MagicMock(name="not_v2_manager")
        with pytest.raises(TypeError, match="KVCacheManagerV2"):
            create_sparse_attention_manager(cfg, not_v2)


# ---------------------------------------------------------------------------
# 4. RocketKV class — hook + capability contract


class TestRocketKVClass:
    def test_subclass_chain(self):
        from tensorrt_llm._torch.attention_backend.sparse import BaseKVCacheCompressionExecutor

        assert issubclass(RocketKV, SparseAttentionExecutor)
        assert issubclass(RocketKV, BaseKVCacheCompressionExecutor)

    def test_axis_classvar(self):
        assert RocketKV.axis == "sparse"

    def test_capability_declarations(self):
        # RocketKV is a 2-stage hybrid: Stage I-b at prefill end PHYSICALLY
        # evicts (SnapKV top-pB keep), then Stage II does sparse mask over
        # the shrunk cache. So physically_evicts_kv MUST be True.
        assert RocketKV.physically_evicts_kv is True
        # KT cache + Stage I-b keep-set are both request-specific.
        assert RocketKV.supports_kv_cache_reuse is False
        # Pattern 2: uses default plain KVCacheManagerV2 with KT_CACHE BufferConfig
        # added via multi-pool extension (no Pattern 3 subclass).
        assert RocketKV.kv_cache_manager_class is None

    def test_construct_with_minimal_args(self):
        import torch

        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048
        # Executor stores torch dtype.
        assert mgr.kt_cache_dtype is torch.bfloat16
        # kt_tokens_per_block defaults to None at config-level but the
        # executor computes it from page_size + tokens_per_block when
        # the cache manager exposes those. With MagicMock cache manager,
        # the safe path returns kt_tokens_per_block based on default
        # page_size (16). Just assert it's an int.
        assert isinstance(mgr.kt_tokens_per_block, int)

    def test_stage_i_returns_none_without_kt_pool(self):
        """Stage I (on_context_attention) returns None when no KT pool is
        wired (MagicMock manager), so the kernel falls back to dense
        attention."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        result = mgr.on_context_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_stage_ii_returns_none_without_kt_pool(self):
        """Stage II (on_generation_attention) returns None when no KT pool is
        wired (MagicMock manager), so the kernel falls back to dense
        attention."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        result = mgr.on_generation_attention(0, None, None, None, MagicMock())
        assert result is None

    def test_other_hooks_default_noop(self):
        """The 3 hooks RocketKV does NOT override inherit base no-op
        (request_init / generation_step_end / request_finish).

        The 3 hooks it DOES override are tested separately:
        - on_context_attention (Stage I-a) — sparse-kv prediction
        - on_context_end       (Stage I-b) — SnapKV physical evict
        - on_generation_attention (Stage II) — HSA mask
        """
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.on_request_init(MagicMock()) is None
        assert mgr.on_generation_step_end(MagicMock(), MagicMock()) is None
        assert mgr.on_request_finish(MagicMock()) is None

    def test_stage_i_b_overrides_context_end(self):
        """Stage I-b (physical evict at prefill end) lives in
        on_context_end. RocketKV MUST override it (not inherit the base
        no-op) — that is how the framework dispatches the evict. The body
        does real work, so check the override is present rather than
        invoking it with bare mocks."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert mgr.implements("on_context_end") is True

    def test_attention_shims_defined_in_same_module(self):
        """RocketKV ships its own attention shim classes alongside the
        executor, so the executor pipeline routes to the correct KV manager
        AND the correct attention class."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention,
            RocketKVTrtllmAttentionMetadata,
            RocketKVVanillaAttention,
            RocketKVVanillaAttentionMetadata,
        )

        assert RocketKVTrtllmAttention.Metadata is RocketKVTrtllmAttentionMetadata
        assert RocketKVVanillaAttention.Metadata is RocketKVVanillaAttentionMetadata

    def test_attention_factory_routes_rocketkv_to_its_shim(self):
        """``get_trtllm_sparse_attn_attention_backend`` must return the
        RocketKV-specific attention class for algorithm="rocketkv", NOT
        the default ``TrtllmAttention`` short-circuit used by other
        behavior-layer methods."""
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention,
            RocketKVVanillaAttention,
        )
        from tensorrt_llm._torch.attention_backend.sparse.utils import (
            get_trtllm_sparse_attn_attention_backend,
            get_vanilla_sparse_attn_attention_backend,
        )

        cfg = RocketKVSparseAttentionConfig()
        assert get_trtllm_sparse_attn_attention_backend(cfg) is RocketKVTrtllmAttention
        assert get_vanilla_sparse_attn_attention_backend(cfg) is RocketKVVanillaAttention


# ---------------------------------------------------------------------------
# 5. Coexistence — both configs can be selected without conflict


class TestCoexistence:
    """Critical: both algorithm="rocket" (legacy) and algorithm="rocketkv"
    must remain selectable. Verifies no Pydantic discriminator collision and
    no factory cross-talk."""

    def test_both_configs_distinct_classes(self):
        cfg_rocket = RocketSparseAttentionConfig()
        cfg_rocketkv = RocketKVSparseAttentionConfig()
        assert type(cfg_rocket) is not type(cfg_rocketkv)
        assert cfg_rocket.algorithm != cfg_rocketkv.algorithm

    def test_legacy_rocket_does_not_get_factory_routed_to_rocketkv(self):
        """Legacy ``algorithm="rocket"`` must NOT return a RocketKV
        executor — must route through cache-manager subclass instead."""
        cfg = RocketSparseAttentionConfig()
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2

        fake_v2 = MagicMock(spec=KVCacheManagerV2)
        mgr = create_sparse_attention_manager(cfg, fake_v2)
        assert mgr is None  # not routed here; legacy path picks up via
        # ``get_sparse_attn_kv_cache_manager`` → ``RocketKVCacheManager``

    def test_rocketkv_does_not_collide_with_rocket_in_discriminator(self):
        """Discriminator must round-trip both algorithm values cleanly."""
        adapter = TypeAdapter(SparseAttentionConfig)
        # legacy rocket
        rocket = adapter.validate_python({"algorithm": "rocket"})
        assert rocket.algorithm == "rocket"
        # rocketkv
        rocketkv = adapter.validate_python({"algorithm": "rocketkv"})
        assert rocketkv.algorithm == "rocketkv"
        # Cross-check: classes are distinct
        assert type(rocket).__name__ == "RocketSparseAttentionConfig"
        assert type(rocketkv).__name__ == "RocketKVSparseAttentionConfig"


# ---------------------------------------------------------------------------
# 6. RocketKV class structure — hook overrides, kernel imports, helper
#    methods, KT pool ownership.
# ---------------------------------------------------------------------------


class TestRocketKVAlgorithmBodyPort:
    """RocketKV algorithm-body structure: expected kernel calls, metadata
    field names, and hook fire timing."""

    def test_executor_overrides_all_three_algorithm_hooks(self):
        """RocketKV must override HOOK 2 (Stage I-a / KT build +
        SnapKV mask) + HOOK 3 (Stage I-b physical evict) + HOOK 4
        (Stage II HSA mask)."""
        from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
            BaseKVCacheCompressionExecutor,
        )

        mgr = RocketKV(kv_cache_manager=MagicMock())
        for hook_name in ("on_context_attention", "on_context_end", "on_generation_attention"):
            mgr_method = getattr(type(mgr), hook_name)
            base_method = getattr(BaseKVCacheCompressionExecutor, hook_name)
            assert mgr_method is not base_method, (
                f"RocketKV must override {hook_name} (algorithm body). "
                f"Inherited default no-op means the body is missing."
            )

    def test_executor_overrides_request_lifecycle_hooks(self):
        """Request lifecycle (prepare/free resources) lives in HOOK 1 /
        HOOK 6 on the executor."""
        from tensorrt_llm._torch.attention_backend.sparse.kv_cache_compression_executor import (
            BaseKVCacheCompressionExecutor,
        )

        mgr = RocketKV(kv_cache_manager=MagicMock())
        for hook_name in ("on_request_init", "on_request_finish"):
            assert getattr(type(mgr), hook_name) is not getattr(
                BaseKVCacheCompressionExecutor, hook_name
            ), f"RocketKV must override {hook_name} (cache-manager lifecycle)."

    def test_kt_pool_ownership_helpers_exist(self):
        """get_kt_buffers + copy_kt_block_offsets are executor instance
        methods (Pattern 2: the KT_CACHE BufferConfig owns the pool; the
        executor delegates via a thin shim)."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        assert hasattr(mgr, "get_kt_buffers"), (
            "RocketKV must expose get_kt_buffers(layer_idx) — delegates "
            "to V2 KT_CACHE BufferConfig pool (Pattern 2)."
        )
        assert hasattr(mgr, "copy_kt_block_offsets"), (
            "RocketKV must expose copy_kt_block_offsets(...) — delegates "
            "to V2 KEY-role block offsets (shared block IDs)."
        )

    def test_kt_pool_carries_expected_params(self):
        """The executor stores prompt_budget / page_size /
        kt_tokens_per_block + window_size / kernel_size / topk / topr for
        algorithm-body access."""
        mgr = RocketKV(
            kv_cache_manager=MagicMock(),
            page_size=16,
            prompt_budget=2048,
            window_size=32,
            kernel_size=5,
            topk=256,
            topr=32,
        )
        assert mgr.page_size == 16
        assert mgr.prompt_budget == 2048
        assert mgr.window_size == 32
        assert mgr.kernel_size == 5
        assert mgr.topk == 256
        assert mgr.topr == 32

    def test_triton_kernel_imports(self):
        """The algorithm body imports a specific set of triton kernels
        from sparse/kernel.py. We verify by checking the kernel symbols
        appear in the module's source."""
        import inspect

        from tensorrt_llm._torch.attention_backend.sparse import rocketkv

        src = inspect.getsource(rocketkv)
        # The 8 triton kernel names the algorithm body imports.
        for kernel_name in (
            "triton_bmm",
            "triton_flatten_to_batch",
            "triton_rocket_batch_to_flatten",
            "triton_rocket_paged_kt_cache_bmm",
            "triton_rocket_qk_split",
            "triton_rocket_reduce_scores",
            "triton_rocket_update_kt_cache_ctx",
            "triton_rocket_update_kt_cache_gen",
            "triton_softmax",
            "triton_topk",
        ):
            assert kernel_name in src, (
                f"Expected triton kernel ``{kernel_name}`` to be referenced in rocketkv.py."
            )

    def test_helpers_dont_crash_with_mocked_manager(self):
        """The helper methods (get_kt_buffers / copy_kt_block_offsets) must
        not crash when called with a mocked cache manager that has no
        actual KT pool — they should return safely (None / passthrough)."""
        mgr = RocketKV(kv_cache_manager=MagicMock())
        # No actual GPU pool allocated for mocked manager → helpers degrade
        assert mgr.get_kt_buffers(0) is None
        # copy_kt_block_offsets with no kt_cache_manager returns input
        # passthrough
        fake_buf = MagicMock()
        result = mgr.copy_kt_block_offsets([], fake_buf)
        assert result is fake_buf


class TestRocketKVAttentionShimsCarryMetadata:
    """The attention shim classes (RocketKVTrtllmAttention /
    RocketKVVanillaAttention) carry the RocketKV-specific Metadata class
    so backend dispatch picks them up."""

    def test_trtllm_shim_carries_correct_metadata(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVTrtllmAttention,
            RocketKVTrtllmAttentionMetadata,
        )

        assert RocketKVTrtllmAttention.Metadata is RocketKVTrtllmAttentionMetadata

    def test_vanilla_shim_carries_correct_metadata(self):
        from tensorrt_llm._torch.attention_backend.sparse.rocketkv import (
            RocketKVVanillaAttention,
            RocketKVVanillaAttentionMetadata,
        )

        assert RocketKVVanillaAttention.Metadata is RocketKVVanillaAttentionMetadata

    def test_trtllm_metadata_has_expected_field_names(self):
        """The metadata carries the expected field names (prompt_budget /
        window_size / page_size / topk / kt_cache_block_offsets / etc.).
        We verify by inspecting the source for these field references."""
        import inspect

        from tensorrt_llm._torch.attention_backend.sparse import rocketkv

        src = inspect.getsource(rocketkv.RocketKVTrtllmAttentionMetadata)
        for field_name in (
            "prompt_budget",
            "window_size",
            "page_size",
            "topk",
            "kt_cache_block_offsets",
            "host_kt_cache_block_offsets",
            "context_cumsum_cuda",
            "k_context_lens_cuda",
            "k_context_start_cuda",
            "valid_seq_indices_cuda",
            "q_cu_seqlens_cuda",
            "k_cu_seqlens_cuda",
            "sparse_offsets_ctx_cuda",
            "sparse_offsets_gen_cuda",
            "cum_kt_lens_cuda",
            "num_kt_tokens",
            "max_kt_tokens",
        ):
            assert field_name in src, (
                f"RocketKVTrtllmAttentionMetadata must carry field ``{field_name}``."
            )


class TestRocketKVAlgorithmBodyKernelCalls:
    """Source-level assertion that the HOOK 2/4 callback bodies invoke the
    expected triton kernels in the expected order. Locks the algorithm body
    against accidental drift."""

    def _get_hook_source(self, hook_name):
        import inspect

        return inspect.getsource(getattr(RocketKV, hook_name))

    def test_on_context_attention_kernel_sequence(self):
        src = self._get_hook_source("on_context_attention")
        # The 6 triton kernels of Stage I-a sparse-kv prediction.
        for kernel_name in (
            "triton_rocket_qk_split",
            "triton_bmm",
            "triton_softmax",
            "triton_flatten_to_batch",
            "triton_rocket_batch_to_flatten",
            "triton_rocket_update_kt_cache_ctx",
        ):
            assert kernel_name in src, (
                f"on_context_attention must call ``{kernel_name}`` "
                f"(Stage I-a sparse-kv prediction sequence)."
            )

    def test_on_generation_attention_kernel_sequence(self):
        src = self._get_hook_source("on_generation_attention")
        # The triton kernels of Stage II HSA.
        for kernel_name in (
            "triton_rocket_update_kt_cache_gen",
            "triton_rocket_paged_kt_cache_bmm",
            "triton_softmax",
            "triton_rocket_reduce_scores",
            "triton_topk",
        ):
            assert kernel_name in src, (
                f"on_generation_attention must call ``{kernel_name}`` (Stage II HSA sequence)."
            )

    def test_on_context_end_does_physical_evict(self):
        """HOOK 3 performs the Stage I-b rewind — must invoke
        rewind_kv_cache.

        Note: Pattern 2 has no separate kt_cache_manager.rewind_cache
        call — KT slots live inside the same V2 logical blocks as KEY/VALUE
        (shared block IDs via multi-pool BufferConfig), so V2's
        rewind_kv_cache frees them in one shot."""
        src = self._get_hook_source("on_context_end")
        assert "rewind_kv_cache" in src, (
            "on_context_end must call rewind_kv_cache (Stage I-b SnapKV physical evict)."
        )
        assert "kt_cache_manager.rewind_cache" not in src, (
            "Pattern 2 must NOT call kt_cache_manager.rewind_cache — "
            "KT shares block IDs with KEY/VALUE via V2 multi-pool "
            "BufferConfig, so rewind_kv_cache frees KT slots "
            "automatically."
        )
