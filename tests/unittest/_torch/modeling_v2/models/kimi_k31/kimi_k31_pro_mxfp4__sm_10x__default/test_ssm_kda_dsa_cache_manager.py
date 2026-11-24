# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi K3.1 KDA + DSA hybrid cache manager: routing and indexer K-cache views."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_cache import (
    is_dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    is_dsa_cache_manager as is_engine_dsa_cache_manager,
)
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MambaStateConfig,
)
from tensorrt_llm.mapping import Mapping

_INDEX_HEAD_DIM = 128
# FP8 keys plus one fp32 scale per 128-wide block.
_INDEXER_BYTES_PER_TOKEN = _INDEX_HEAD_DIM + 4
_TOKENS_PER_BLOCK = 32
# Layers 0 and 2 are KDA, layers 1 and 3 are DSA MLA (1-indexed 2 and 4).
_KDA_MASK = [True, False, True, False]
_MLA_MASK = [False, True, False, True]


def _sparse_config(indexer_k_dtype: str = "fp8") -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_n_heads=4,
        index_head_dim=_INDEX_HEAD_DIM,
        index_topk=64,
        indexer_k_dtype=indexer_k_dtype,
    )


def _kimi_dsa_text_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _kimi_dsa_model_config() -> SimpleNamespace:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        KimiK31ForConditionalGeneration,
    )

    return SimpleNamespace(
        pretrained_config=_kimi_dsa_text_config(),
        mapping=Mapping(),
        quant_config=None,
        sparse_attention_config=_sparse_config(),
        spec_config=None,
        get_num_mamba_layers=lambda: 2,
        # get_kv_cache_manager_cls asks the routed modeling_v2 target (a subclass of
        # KimiK31ForConditionalGeneration) for its manager first.
        modeling_v2_target=KimiK31ForConditionalGeneration,
    )


def test_kimi_dsa_routes_to_dsa_hybrid_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_dsa_model_config()
    cls = get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))
    assert cls is KimiK3DSAHybridCacheManager


def test_kimi_k31_requires_kv_cache_manager_v2() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        KimiK31Config,
        KimiK31TextConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31ForCausalLM,
        K31VLWrapper,
        KimiK31ForConditionalGeneration,
    )
    from tensorrt_llm._torch.configs import KimiK3Config, KimiLinearConfig
    from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto

    text = {"model_type": "kimi_linear", "hidden_size": 64, "num_attention_heads": 4}
    dsa_config = {"dsa_layers": [2, 4], "index_k_norm_type": "layernorm"}
    k3_text = KimiLinearConfig(**text)
    k31_text = KimiK31TextConfig(**text, dsa_config=dsa_config)
    assert not K31ForCausalLM.requires_kv_cache_manager_v2(k3_text)
    assert K31ForCausalLM.requires_kv_cache_manager_v2(k31_text)
    assert not K31VLWrapper.requires_kv_cache_manager_v2(KimiK3Config(text_config=dict(text)))
    k31 = KimiK31Config(text_config=dict(text, dsa_config=dsa_config))
    assert KimiK31ForConditionalGeneration.requires_kv_cache_manager_v2(k31)

    # "auto" never demotes K3.1 to V1, and an explicit V1 request fails early.
    def llm_args(setting):
        return SimpleNamespace(
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=setting),
            cache_transceiver_config=None,
        )

    assert _resolve_kv_cache_manager_v2_auto(llm_args("auto"), K31ForCausalLM, k31_text)
    with pytest.raises(ValueError, match="requires KV cache manager v2"):
        _resolve_kv_cache_manager_v2_auto(llm_args(False), K31ForCausalLM, k31_text)


def test_kimi_dsa_rejects_v1_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    with pytest.raises(ValueError, match="use_kv_cache_manager_v2=True"):
        get_kv_cache_manager_cls(
            _kimi_dsa_model_config(), KvCacheConfig(use_kv_cache_manager_v2=False)
        )


_MLA_HEAD_DIM = 64


def _make_manager(
    indexer_k_dtype: str = "fp8",
    dtype: DataType = DataType.BF16,
    head_dim: int = _MLA_HEAD_DIM,
    enable_block_reuse: bool = False,
    kv_cache_dtype: str = "auto",
    **draft_kwargs,
) -> KimiK3DSAHybridCacheManager:
    return KimiK3DSAHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
            dtype=kv_cache_dtype,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        layer_mask=_MLA_MASK,
        sparse_attention_config=_sparse_config(indexer_k_dtype),
        pretrained_config=_kimi_dsa_text_config(),
        **draft_kwargs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "dtype,view_dtype,elements_per_page",
    [
        (DataType.BF16, torch.bfloat16, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM),
        (DataType.FP8, torch.float8_e4m3fn, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM),
    ],
)
def test_kimi_dsa_pool_view_dtype(
    dtype: DataType, view_dtype: torch.dtype, elements_per_page: int
) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_manager(dtype=dtype)
    try:
        pool = manager.get_unique_primary_pool()
        assert pool.dtype == view_dtype
        assert pool.shape[1:] == (2, 1, elements_per_page)
        # The view steps one K page per MLA layer, as the pool lays them out.
        page_stride = manager.impl.get_page_stride(manager._attention_local_layers[0], Role.KEY)
        assert int(page_stride) == elements_per_page * pool.element_size()
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("indexer_k_dtype, bytes_per_token", [("fp8", _INDEXER_BYTES_PER_TOKEN)])
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8])
def test_kimi_dsa_hybrid_manager_indexer_views(
    indexer_k_dtype: str, bytes_per_token: int, dtype: DataType
) -> None:
    manager = _make_manager(indexer_k_dtype, dtype)
    try:
        assert is_dsa_cache_manager(manager)
        assert is_engine_dsa_cache_manager(manager)
        assert manager.indexer_k_dtype == indexer_k_dtype
        assert manager.indexer_k_cache_local_layer_mask == _MLA_MASK
        assert manager.get_layer_bytes_per_token(1, _index_key_role()) == bytes_per_token
        assert manager.get_layer_bytes_per_token(0, _index_key_role()) == 0

        # The two MLA layers share one coalesced K pool, in MLA-layer order.
        scale1, offset1 = manager.get_primary_pool_page_index_params(1)
        scale3, offset3 = manager.get_primary_pool_page_index_params(3)
        assert scale1 == scale3 == 2
        assert (offset1, offset3) == (0, 1)

        for layer_idx in (1, 3):
            buffers = manager.get_indexer_k_cache_buffers(layer_idx)
            assert buffers.dtype == torch.uint8
            assert buffers.shape[1:] == (_TOKENS_PER_BLOCK, 1, bytes_per_token)
        with pytest.raises(AssertionError):
            manager.get_indexer_k_cache_buffers(0)

        pool = manager.get_unique_primary_pool()
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kimi_dsa_pool_block_indices_cover_batch_capacity() -> None:
    # The per-step gather reads only the columns that can hold a page of the batch; every
    # column it skips holds no page (the safe block 0).
    manager = _make_manager()
    try:
        request_ids = [0, 1]
        manager.add_dummy_requests(request_ids, token_nums=[40, 100], is_gen=True)
        blocks = manager.get_pool_block_indices(len(request_ids), request_ids=request_ids)
        capacity = max(int(manager.kv_cache_map[r].capacity) for r in request_ids)
        num_cols = -(-capacity // _TOKENS_PER_BLOCK)
        width = manager.host_kv_cache_block_offsets.shape[-1]
        assert blocks.shape == (len(request_ids), num_cols) and num_cols < width
        copy_idx = manager.index_mapper.get_copy_index(request_ids, 0, 1).to(torch.long)
        full = manager.host_kv_cache_block_offsets[manager._attention_pool_id, copy_idx, 0, :]
        full = full.clamp(min=0, max=manager._num_attention_slots - 1).to(torch.int32)
        assert torch.equal(blocks, full[:, :num_cols])
        assert bool((full[:, num_cols:] == 0).all())
        # A request the manager does not know falls back to the whole table.
        assert manager._batch_block_columns([0, 7]) == width
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kimi_dsa_allocated_block_counts() -> None:
    # Pages each request holds, in batch order; their maximum is the gathered column count.
    manager = _make_manager()
    try:
        request_ids = [1, 0]
        manager.add_dummy_requests([0, 1], token_nums=[40, 100], is_gen=True)
        counts = manager.get_allocated_block_counts(request_ids)
        expected = [
            -(-int(manager.kv_cache_map[r].capacity) // _TOKENS_PER_BLOCK) for r in request_ids
        ]
        assert counts.dtype == torch.int32 and counts.tolist() == expected
        assert int(counts.max()) == manager._batch_block_columns(request_ids)
        width = manager.host_kv_cache_block_offsets.shape[-1]
        assert manager.get_allocated_block_counts([0, 7]).tolist() == [expected[1], width]
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "backend, runtime, accepted",
    [
        ("NIXL", "PYTHON", True),
        ("NIXL", "CPP", False),
        ("UCX", "PYTHON", False),
    ],
)
def test_kimi_dsa_disagg_requires_python_nixl(
    monkeypatch: pytest.MonkeyPatch, backend: str, runtime: str, accepted: bool
) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    kwargs = dict(
        is_disagg=True,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, transceiver_runtime=runtime
        ),
    )
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    if accepted:
        cls = get_kv_cache_manager_cls(_kimi_dsa_model_config(), kv_cache_config, **kwargs)
        assert cls is KimiK3DSAHybridCacheManager
    else:
        with pytest.raises(ValueError, match="transceiver_runtime='PYTHON'"):
            get_kv_cache_manager_cls(_kimi_dsa_model_config(), kv_cache_config, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8])
def test_kimi_dsa_hybrid_manager_disagg_page_table(dtype: DataType) -> None:
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind

    manager = _make_manager(dtype=dtype)
    try:
        index_key = str(_index_key_role())
        index_views = []
        key_views = []
        for group in build_page_table_from_manager(manager).layer_groups:
            for view in group.pool_views:
                if index_key in view.pool_role:
                    index_views.append(view)
                elif "key" in view.pool_role:
                    key_views.append(view)
        # The indexer K-cache travels as its own replicated view, one buffer
        # per MLA layer; the latent KV keeps the head-indexed mapper.
        assert index_views and all(v.mapper_kind is MapperKind.REPLICATED for v in index_views)
        index_layers = sorted(
            int(layer) for v in index_views for layer in v.buffer_entries["local_layer_id"]
        )
        assert index_layers == [1, 3]
        assert all(
            v.bytes_per_layer == _TOKENS_PER_BLOCK * _INDEXER_BYTES_PER_TOKEN for v in index_views
        )
        assert key_views and all(v.mapper_kind is MapperKind.INDEXED for v in key_views)
    finally:
        manager.shutdown()


def _index_key_role():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    return Role.INDEX_KEY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_standalone_draft_buffers_do_not_change_target_indexer_views() -> None:
    """A unified draft cache retains the target's original DSA buffer layout."""
    manager = _make_manager(
        draft_num_layers=2,
        draft_num_kv_heads=1,
        draft_head_dim=576,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_scratch_tokens=8,
    )
    try:
        assert manager.get_unique_primary_pool().shape[1:] == (
            2,
            1,
            _TOKENS_PER_BLOCK * _MLA_HEAD_DIM,
        )
        for layer in (1, 3):
            assert manager.get_indexer_k_cache_buffers(layer).shape[1:] == (
                _TOKENS_PER_BLOCK,
                1,
                _INDEXER_BYTES_PER_TOKEN,
            )
        for layer in range(2):
            pool = manager.get_draft_buffers(layer)
            assert pool.dtype == torch.bfloat16
            assert pool.shape[1:] == (1, 1, _TOKENS_PER_BLOCK, 576)
            assert pool.numel() > 0
    finally:
        manager.shutdown()
