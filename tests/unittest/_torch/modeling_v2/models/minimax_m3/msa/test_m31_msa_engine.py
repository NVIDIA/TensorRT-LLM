# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family MSA classes and how the M3.1 targets put them in front of the engine (``msa.engine``).

Every family class derives from main's MiniMax-M3 MSA class of the same role, the backend tries the family FMHA
libraries first, the targets select the family classes only on an engine with ``RuntimeHooks.sparse_attention`` (and
not under the kill switch), and the NVFP4 index-K cache's sizing replaces the BF16 index row main prices. CPU.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import runtime_hooks_compat
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import (
    cache_manager,
    engine,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_backend import (
    MiniMaxM31MsaSparseAttention,
    MiniMaxM31MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_decode import (
    M31MsaDecodeFmha,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_prefill import (
    M31MsaPrefillFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.msa_decode import MsaDecodeFmha
from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import MsaPrefillFmha
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttention,
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def test_family_classes_derive_from_main():
    assert issubclass(MiniMaxM31MsaSparseAttention, MiniMaxM3MsaSparseAttention)
    assert issubclass(MiniMaxM31MsaSparseAttentionMetadata, MiniMaxM3MsaSparseAttentionMetadata)
    assert MiniMaxM31MsaSparseAttention.Metadata is MiniMaxM31MsaSparseAttentionMetadata
    assert issubclass(M31MsaPrefillFmha, MsaPrefillFmha)
    assert issubclass(M31MsaDecodeFmha, MsaDecodeFmha)
    assert issubclass(cache_manager.MiniMaxM31KVCacheManagerV2, MiniMaxM3KVCacheManagerV2)


def test_the_backend_tries_the_family_fmha_libraries_first():
    assert MiniMaxM31MsaSparseAttention.prepended_fmha_lib_classes == (
        M31MsaDecodeFmha,
        M31MsaPrefillFmha,
    )


def test_the_kv_cache_manager_resolves_index_k_views_through_the_shim():
    mro = cache_manager.MiniMaxM31KVCacheManagerV2.__mro__
    assert mro.index(MiniMaxM3KVCacheManagerV2) < mro.index(cache_manager._IndexKViewShim)
    assert mro.index(cache_manager._IndexKViewShim) < mro.index(KVCacheManagerV2)


def test_the_kill_switch_keeps_main_classes(monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_FAMILY_MSA", "1")
    assert not engine.family_msa_enabled()
    assert engine.sparse_attention_classes() is None
    assert not engine.use_family_backend(SimpleNamespace(attn=object()))


def test_an_engine_without_the_hook_keeps_main_classes(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_FAMILY_MSA", raising=False)
    monkeypatch.setattr(runtime_hooks_compat, "has_hook", lambda name: False)
    assert not engine.family_msa_enabled()
    assert engine.sparse_attention_classes() is None


def test_an_engine_with_the_hook_takes_the_family_classes(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_FAMILY_MSA", raising=False)
    if not runtime_hooks_compat.has_hook("sparse_attention"):
        pytest.skip("this engine has no RuntimeHooks.sparse_attention")
    classes = engine.sparse_attention_classes()
    assert classes.backend is MiniMaxM31MsaSparseAttention
    assert classes.kv_cache_manager is cache_manager.MiniMaxM31KVCacheManagerV2
    assert classes.warmup_scope is engine.warmup_scope
    assert classes.indexer_prewarm is engine.indexer_prewarm


def _keep_stock_lookups(monkeypatch):
    monkeypatch.setattr(engine, "_stock_routed", False)
    modules = engine.stock_msa_lookups()
    for module, attr in modules:
        monkeypatch.setattr(module, attr, getattr(module, attr))
    return modules


def test_main_classes_resolve_the_vendored_fmha_sm100(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import msa_prefill
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
        msa_utils,
    )

    monkeypatch.delenv("TRTLLM_M31_DISABLE_FAMILY_MSA", raising=False)
    modules = _keep_stock_lookups(monkeypatch)
    vendored = {
        "require_msa_module": msa_utils.require_msa_module,
        "_msa_sparse_api": msa_prefill._msa_sparse_api,
    }
    assert engine.route_stock_msa_to_vendored()
    for module, attr in modules:
        assert getattr(module, attr) is vendored[attr], (module.__name__, attr)
    assert engine.route_stock_msa_to_vendored()  # idempotent


def test_the_kill_switch_keeps_the_packaged_fmha_sm100(monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_FAMILY_MSA", "1")
    modules = _keep_stock_lookups(monkeypatch)
    before = [getattr(module, attr) for module, attr in modules]
    assert not engine.route_stock_msa_to_vendored()
    assert [getattr(module, attr) for module, attr in modules] == before


def test_the_hook_field_is_classified_for_engines_without_it():
    assert "sparse_attention" in runtime_hooks_compat.PERF_ONLY


def test_use_family_backend_leaves_a_non_msa_backend(monkeypatch):
    monkeypatch.setattr(engine, "family_msa_enabled", lambda: True)
    module = SimpleNamespace(attn=object())
    assert not engine.use_family_backend(module)


def _size_inputs(indexer_kv_dtype, num_layers=8):
    pretrained = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=1024,
        num_attention_heads=8,
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained,
        sparse_attention_config=SimpleNamespace(
            indexer_kv_dtype=indexer_kv_dtype, sparse_index_dim=128
        ),
        quant_config=None,
    )
    mapping = SimpleNamespace(pp_size=1, enable_attention_dp=False, tp_size=1)
    return model_config, mapping, {"sparse_layer_ids": list(range(3, num_layers))}


def test_nvfp4_index_k_prices_its_packed_row_and_the_shared_sink():
    cls = cache_manager.MiniMaxM31KVCacheManagerV2
    bf16 = cls._extra_bytes_per_token(*_size_inputs("bf16")[:2], None, _size_inputs("bf16")[2])
    nvfp4 = cls._extra_bytes_per_token(*_size_inputs("nvfp4")[:2], None, _size_inputs("nvfp4")[2])
    num_sparse = 5
    assert nvfp4 == (
        bf16
        - num_sparse * 128 * 2
        + num_sparse * cache_manager.NVFP4_INDEX_K_BYTES_PER_TOKEN
        + cache_manager.NVFP4_INDEX_K_DIM
    )
    assert cache_manager.NVFP4_INDEX_K_BYTES_PER_TOKEN == 72


def test_disable_index_value_follows_the_checkpoint():
    get = cache_manager.get_sparse_disable_index_value_layer_ids
    assert get(SimpleNamespace()) is None
    flags = [False, False, False, True, True]
    assert get(SimpleNamespace(sparse_attention_config={"disable_index_value": flags})) == [3, 4]
