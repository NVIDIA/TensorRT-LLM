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
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import is_dsa_cache_manager
from tensorrt_llm._torch.modules.kimi_kda.dsa_cache_manager import KimiK3DSAHybridCacheManager
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MambaStateConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

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
    return SimpleNamespace(
        pretrained_config=_kimi_dsa_text_config(),
        mapping=Mapping(),
        quant_config=None,
        sparse_attention_config=_sparse_config(),
        spec_config=None,
        get_num_mamba_layers=lambda: 2,
    )


def test_kimi_dsa_routes_to_dsa_hybrid_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_USE_PY_MAMBA", raising=False)
    monkeypatch.delenv("TLLM_MAMBA_MANAGER_PREFERENCE", raising=False)
    model_config = _kimi_dsa_model_config()
    cls = get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))
    assert cls is KimiK3DSAHybridCacheManager


def test_kimi_k31_requires_kv_cache_manager_v2() -> None:
    from tensorrt_llm._torch.configs import KimiK3Config, KimiK31Config, KimiLinearConfig
    from tensorrt_llm._torch.models.modeling_kimi_k3_vl import (
        KimiK3ForConditionalGeneration,
        KimiK31ForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
    from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto

    text = {"model_type": "kimi_linear", "hidden_size": 64, "num_attention_heads": 4}
    dsa_config = {"dsa_layers": [2, 4], "index_k_norm_type": "layernorm"}
    k3_text = KimiLinearConfig(**text)
    k31_text = KimiLinearConfig(**text, dsa_config=dsa_config)
    assert not KimiLinearForCausalLM.requires_kv_cache_manager_v2(k3_text)
    assert KimiLinearForCausalLM.requires_kv_cache_manager_v2(k31_text)
    assert not KimiK3ForConditionalGeneration.requires_kv_cache_manager_v2(
        KimiK3Config(text_config=dict(text))
    )
    k31 = KimiK31Config(text_config=dict(text, dsa_config=dsa_config))
    assert KimiK31ForConditionalGeneration.requires_kv_cache_manager_v2(k31)

    # "auto" never demotes K3.1 to V1, and an explicit V1 request fails early.
    def llm_args(setting):
        return SimpleNamespace(
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=setting),
            cache_transceiver_config=None,
        )

    assert _resolve_kv_cache_manager_v2_auto(llm_args("auto"), KimiLinearForCausalLM, k31_text)
    with pytest.raises(ValueError, match="requires KV cache manager v2"):
        _resolve_kv_cache_manager_v2_auto(llm_args(False), KimiLinearForCausalLM, k31_text)


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
def test_kimi_dsa_nvfp4_ds_mla_pool_holds_records() -> None:
    # kv_cache_config.dtype "nvfp4_ds_mla": one 352-byte record per token and MLA layer in the FP8 container.
    from tensorrt_llm._torch.attention.backends.sparse.nvfp4_ds_mla_kv import TOKEN_BYTES
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_manager(dtype=DataType.FP8, head_dim=576, kv_cache_dtype="nvfp4_ds_mla")
    try:
        assert manager.use_nvfp4_ds_mla
        pool = manager.get_unique_primary_pool()
        assert pool.dtype == torch.float8_e4m3fn
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * TOKEN_BYTES)
        page_stride = manager.impl.get_page_stride(manager._attention_local_layers[0], Role.KEY)
        assert int(page_stride) == _TOKENS_PER_BLOCK * TOKEN_BYTES
        assert manager._attention_cache_bytes_per_token() == sum(_MLA_MASK) * (
            TOKEN_BYTES + _INDEXER_BYTES_PER_TOKEN
        )
    finally:
        manager.shutdown()
    with pytest.raises(ValueError, match="nvfp4_ds_mla requires"):
        _make_manager(dtype=DataType.FP8, kv_cache_dtype="nvfp4_ds_mla")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kimi_dsa_hybrid_manager_with_standalone_draft_layers() -> None:
    # A unified standalone MLA drafter (DSpark) stores one BF16 latent per token per layer and is
    # registered as model layers; the DSA views must still cover the two target MLA layers only.
    num_draft_layers = 3
    manager = _make_manager(
        draft_num_layers=num_draft_layers,
        draft_num_kv_heads=1,
        draft_head_dim=_MLA_HEAD_DIM,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_scratch_tokens=8,
    )
    try:
        assert len(manager.draft_layer_ids) == num_draft_layers
        draft_local = {manager.layer_offsets[layer] for layer in manager.draft_layer_ids}
        assert draft_local.isdisjoint(manager._attention_local_layers)
        assert manager._attention_local_layers == [1, 3]
        pool = manager.get_unique_primary_pool()
        assert pool.shape[1] == sum(_MLA_MASK)
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM)
        assert manager.indexer_k_cache_local_layer_mask == _MLA_MASK + [False] * num_draft_layers

        # Per token: a BF16 latent and the index K for each MLA layer, a BF16 latent per draft layer.
        mla_bytes = sum(_MLA_MASK) * (2 * _MLA_HEAD_DIM + _INDEXER_BYTES_PER_TOKEN)
        draft_bytes = num_draft_layers * 2 * _MLA_HEAD_DIM
        assert manager._attention_cache_bytes_per_token() == mla_bytes + draft_bytes
        for layer_idx in (1, 3):
            assert manager.get_indexer_k_cache_buffers(layer_idx).shape[1:] == (
                _TOKENS_PER_BLOCK,
                1,
                _INDEXER_BYTES_PER_TOKEN,
            )
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
# BF16 keys take two bytes per value and keep the 4-byte scale slot.
@pytest.mark.parametrize(
    "indexer_k_dtype, bytes_per_token",
    [("fp8", _INDEXER_BYTES_PER_TOKEN), ("bf16", 2 * _INDEX_HEAD_DIM + 4)],
)
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
def test_kimi_dsa_hybrid_manager_indexer_views(
    indexer_k_dtype: str, bytes_per_token: int, dtype: DataType
) -> None:
    manager = _make_manager(indexer_k_dtype, dtype)
    try:
        assert is_dsa_cache_manager(manager)
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
        elements_per_token = _MLA_HEAD_DIM // (2 if dtype == DataType.NVFP4 else 1)
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * elements_per_token)
        if dtype == DataType.NVFP4:
            assert pool.dtype == torch.int8
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
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.NVFP4])
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
@skip_pre_blackwell
@pytest.mark.parametrize("prefix_len", [15, 17, 31, 33, 63, 65])
@pytest.mark.parametrize("reuse_prefix", [False, True])
def test_kimi_dsa_nvfp4_append_across_token_and_page_boundaries(
    prefix_len: int, reuse_prefix: bool
) -> None:
    """Appending a chunk preserves cached rows and the hybrid side buffers."""
    head_dim = 576
    total_len = prefix_len + 3
    manager = _make_manager(
        dtype=DataType.NVFP4, head_dim=head_dim, enable_block_reuse=reuse_prefix
    )
    try:
        if not reuse_prefix:
            manager.add_dummy_requests([42], token_nums=[total_len], is_gen=False)
        offsets = torch.empty(
            (manager.num_attention_op_pools, 1, 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        rope = torch.zeros((256, 128), dtype=torch.float32, device="cuda")
        rope[:, ::2] = 1
        quant_mode = int(QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4).layer_quant_mode)
        dequant_scale = torch.tensor([100.0 / (448 * 6)], device="cuda")
        quant_scale = dequant_scale.reciprocal()
        state = manager.get_ssm_states(0)
        state.fill_(7)
        saved_prefixes = {}
        for start, end in ((0, prefix_len), (prefix_len, total_len)):
            request_id = 42 + int(start > 0) if reuse_prefix else 42
            if reuse_prefix:
                request = LlmRequest(
                    request_id=request_id,
                    input_tokens=list(range(end)),
                    max_new_tokens=1,
                    sampling_config=SamplingConfig(),
                    is_streaming=False,
                )
                manager.prepare_expect_snapshot_points([request])
                assert manager.prepare_context(request)
                assert request.prepopulated_prompt_len == start
                assert manager.resize_context(request, num_tokens=request.context_remaining_length)
                batch = ScheduledRequests()
                batch.append_context_request(request)
                manager.prepare_resources(batch)
                state_index = manager.get_state_indices([request_id], [False])[0]
                for buffer in (*manager.all_ssm_states, *manager.all_conv_states):
                    if start == 0:
                        buffer[state_index].fill_(7)
                    else:
                        assert torch.all(buffer[state_index] == 7)
            manager.copy_batch_block_offsets(offsets, [request_id], 1, 1, 1)
            raw_pages = manager.get_pool_block_indices(1, request_ids=[request_id], num_contexts=1)[
                0
            ]
            for layer_idx, value in ((1, 1.0), (3, -2.0)):
                indexer = manager.get_indexer_k_cache_buffers(layer_idx)
                if start == 0:
                    indexer.fill_(63)
                q = torch.zeros((end - start, 8 * 192), dtype=torch.bfloat16, device="cuda")
                latent = torch.full(
                    (end - start, head_dim), value, dtype=torch.bfloat16, device="cuda"
                )
                torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
                    q,
                    latent,
                    1,
                    torch.tensor([0, start], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, end], dtype=torch.int64, device="cuda"),
                    end - start,
                    rope,
                    8,
                    128,
                    64,
                    512,
                    offsets,
                    manager.kv_cache_pool_pointers,
                    manager.kv_cache_pool_mapping,
                    quant_scale,
                    0,
                    layer_idx,
                    manager.tokens_per_block,
                    manager.max_seq_len,
                    1,
                    quant_mode,
                )
                positions = torch.arange(end, device="cuda")
                scale, layer_offset = manager.get_primary_pool_page_index_params(layer_idx)
                indices = (
                    (
                        raw_pages.to("cuda")[positions // manager.tokens_per_block] * scale
                        + layer_offset
                    )
                    * manager.tokens_per_block
                    + positions % manager.tokens_per_block
                ).to(torch.int32)[None]
                output = torch.empty((1, end, head_dim), dtype=torch.float8_e4m3fn, device="cuda")
                compact_indices = torch.empty_like(indices)
                pool = manager.get_unique_primary_pool()
                torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                    manager.kv_cache_pool_pointers,
                    manager.kv_cache_pool_mapping,
                    indices,
                    output,
                    compact_indices,
                    dequant_scale,
                    layer_idx,
                    0,
                    pool.shape[0] * pool.shape[1] * manager.tokens_per_block,
                )
                torch.testing.assert_close(
                    output.float(), torch.full_like(output.float(), value), rtol=0.125, atol=0.01
                )
                cached_positions = torch.arange(prefix_len, device="cuda")
                assert torch.all(
                    indexer[
                        raw_pages.to("cuda").long()[cached_positions // manager.tokens_per_block],
                        cached_positions % manager.tokens_per_block,
                    ]
                    == 63
                )
                if start == 0:
                    saved_prefixes[layer_idx] = output.clone()
                else:
                    torch.testing.assert_close(
                        output[:, :prefix_len].float(),
                        saved_prefixes[layer_idx].float(),
                        rtol=0,
                        atol=0,
                    )
            if reuse_prefix:
                request.context_current_position = request.prompt_len
                manager.update_context_resources(batch)
                manager.free_resources(request)
        if not reuse_prefix:
            assert torch.all(state == 7)
    finally:
        manager.shutdown()
