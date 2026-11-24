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

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.cache_manager import (  # noqa: E501
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.cache_policy import (  # noqa: E501
    KIMI_K3_FP8_KV_LAYERS_ENV,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.cache_utils import (  # noqa: E501
    is_dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.cache_utils import (  # noqa: E501
    is_dsa_cache_manager as is_target_dsa_cache_manager,
)
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


def _kimi_dsa_text_config(num_mla_layers: int = 2) -> SimpleNamespace:
    full_attn_layers = list(range(2, 2 * num_mla_layers + 1, 2))
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=2 * num_mla_layers,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": list(range(1, 2 * num_mla_layers, 2)),
            "full_attn_layers": full_attn_layers,
        },
        dsa_config={"dsa_layers": full_attn_layers, "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _kimi_dsa_model_config() -> SimpleNamespace:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling import (  # noqa: E501
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
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.modeling import (  # noqa: E501
        KimiK3ForConditionalGeneration,
        KimiK31ForConditionalGeneration,
        KimiLinearForCausalLM,
    )
    from tensorrt_llm._torch.configs import KimiK3Config, KimiLinearConfig
    from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto

    text = {"model_type": "kimi_linear", "hidden_size": 64, "num_attention_heads": 4}
    dsa_config = {"dsa_layers": [2, 4], "index_k_norm_type": "layernorm"}
    k3_text = KimiLinearConfig(**text)
    k31_text = KimiK31TextConfig(**text, dsa_config=dsa_config)
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
    num_mla_layers: int = 2,
    state_dtype: torch.dtype = torch.float32,
    host_cache_size: int | None = None,
    **draft_kwargs,
) -> KimiK3DSAHybridCacheManager:
    return KimiK3DSAHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=num_mla_layers,
        mamba_layer_mask=[True, False] * num_mla_layers,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=state_dtype,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            host_cache_size=host_cache_size,
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
            dtype=kv_cache_dtype,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_mla_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        layer_mask=[False, True] * num_mla_layers,
        sparse_attention_config=_sparse_config(indexer_k_dtype),
        pretrained_config=_kimi_dsa_text_config(num_mla_layers),
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
@pytest.mark.parametrize(
    "indexer_k_dtype, bytes_per_token", [("fp8", _INDEXER_BYTES_PER_TOKEN), ("fp4", 68)]
)
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8])
def test_kimi_dsa_hybrid_manager_indexer_views(
    indexer_k_dtype: str, bytes_per_token: int, dtype: DataType
) -> None:
    manager = _make_manager(indexer_k_dtype, dtype)
    try:
        assert is_dsa_cache_manager(manager)
        assert is_target_dsa_cache_manager(manager)
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
@pytest.mark.parametrize("main", ["fp8", "nvfp4", "mixed"])
@pytest.mark.parametrize("indexer", ["fp8", "fp4"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_independent_target_cache_formats(monkeypatch, main, indexer, state_dtype):
    """Admission and cache views preserve each requested storage format."""
    if main == "mixed":
        monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    dtype = DataType.FP8 if main == "fp8" else DataType.NVFP4
    manager = _make_manager(
        dtype=dtype,
        head_dim=576,
        indexer_k_dtype=indexer,
        state_dtype=state_dtype,
        num_mla_layers=4,
        kv_cache_dtype="fp8" if main == "fp8" else "nvfp4",
        draft_num_layers=2,
        draft_num_kv_heads=1,
        draft_head_dim=576,
        draft_dtype=DataType.BF16,
        draft_kv_factor=1,
        draft_scratch_tokens=8,
    )
    try:
        manager.add_dummy_requests([42], token_nums=[65], is_gen=False)
        for layer in (1, 3, 5, 7):
            expected = (
                DataType.FP8
                if main == "fp8" or (main == "mixed" and layer in (5, 7))
                else DataType.NVFP4
            )
            assert manager.get_kv_cache_dtype(layer) == expected
            pool = manager.get_primary_pool(layer)
            assert pool.dtype == (torch.float8_e4m3fn if expected == DataType.FP8 else torch.int8)
            keys = manager.get_indexer_k_cache_buffers(layer)
            assert keys.shape[-1] == (132 if indexer == "fp8" else 68)
            keys.fill_(layer)
            assert bool((keys == layer).all())
        for state in manager.all_ssm_states:
            assert state.dtype == state_dtype
            state.fill_(0.5)
            assert bool((state == 0.5).all())
        for layer in range(2):
            assert manager.get_draft_buffers(layer).dtype == torch.bfloat16
    finally:
        manager.shutdown()


def _offload_request_and_clear_device_frames(manager, request, committed_tokens):
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CACHE_LEVEL1, _introspection

    stream = torch.cuda.current_stream().cuda_stream
    cache = manager.kv_cache_map[request.py_request_id]
    if cache.num_committed_tokens < committed_tokens:
        cache.commit(list(range(committed_tokens)))
    manager.suspend_request(request)
    pressure = []
    attention_slots = max(
        pool.num_slots
        for pool in manager.impl.pool_group_descs
        if all(
            variant.layer_group_id != manager.ssm_layer_group_id
            for variant in pool.slot_desc.variants
        )
    )
    pages_per_request = committed_tokens // manager.tokens_per_block
    attention_requests = (attention_slots + pages_per_request - 1) // pages_per_request + 1
    try:
        # Fill every physical pool with newer caches, including the attention pages and SSM slots.
        for index in range(max(pool.num_slots for pool in manager.impl.pool_group_descs) + 4):
            other = manager.impl.create_kv_cache()
            pressure.append(other)
            assert other.resume(stream)
            if index < attention_requests:
                assert other.resize(committed_tokens), (
                    f"Attention pressure {index}/{attention_requests}; "
                    f"pool slots: {[pool.num_slots for pool in manager.impl.pool_group_descs]}"
                )
                # Full committed pages can move to host; an uncommitted tail stays pinned on GPU.
                other.commit([1000 + index] * committed_tokens)
            other.suspend()
    finally:
        for other in pressure:
            other.close()
    assert max(_introspection.storage_utilization(manager.impl, CACHE_LEVEL1)) > 0
    torch.cuda.synchronize()
    # Clear the released GPU frames so a missing onboarding copy cannot pass by reading old bytes.
    cleared = set()
    for layer in (1, 3, 5, 7):
        pool = manager.get_primary_pool(layer)
        if pool.data_ptr() not in cleared:
            pool.zero_()
            cleared.add(pool.data_ptr())
            if manager.get_kv_cache_dtype(layer) == DataType.NVFP4:
                pool_id = int(manager.kv_cache_pool_mapping[layer, 0])
                scale_address = int(manager.kv_cache_pool_pointers[pool_id, 0, 1])
                pool_tokens = pool.shape[0] * pool.shape[1] * manager.tokens_per_block
                scales = convert_to_torch_tensor(
                    TensorWrapper(
                        scale_address, torch.float8_e4m3fn, [pool_tokens, manager.head_dim // 16]
                    )
                )
                scales.zero_()
        manager.get_indexer_k_cache_buffers(layer).zero_()
    for buffer in (*manager.all_ssm_states, *manager.all_conv_states):
        buffer.zero_()
    assert manager.resume_request(request)
    torch.cuda.synchronize()
    slot = cache.get_ssm_block_base_index(manager.ssm_layer_group_id)
    for buffer in (*manager.all_ssm_states, *manager.all_conv_states):
        assert torch.all(buffer[slot] == 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "prefix_len,reuse_prefix",
    [(length, False) for length in (15, 17, 31, 33, 63, 65)]
    + [(length, True) for length in (32, 64, 96)],
)
@pytest.mark.parametrize(
    "main_dtype,fp8_layers",
    [(DataType.FP8, None), (DataType.NVFP4, ""), (DataType.NVFP4, "3"), (DataType.NVFP4, None)],
    ids=["fp8", "nvfp4", "explicit-fp8", "last-two-fp8"],
)
@pytest.mark.parametrize("indexer_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_kimi_dsa_nvfp4_append_across_token_and_page_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    prefix_len: int,
    reuse_prefix: bool,
    main_dtype: DataType,
    fp8_layers: str | None,
    indexer_dtype: str,
    state_dtype: torch.dtype,
    host_offload: bool = False,
) -> None:
    """Preserve cached rows on append and on the supported full-block prefix reuse."""
    if fp8_layers is None:
        monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, fp8_layers)
    head_dim = 576
    total_len = prefix_len + 3
    manager = _make_manager(
        dtype=main_dtype,
        head_dim=head_dim,
        enable_block_reuse=reuse_prefix or host_offload,
        num_mla_layers=4,
        indexer_k_dtype=indexer_dtype,
        state_dtype=state_dtype,
        host_cache_size=64 << 20 if host_offload else None,
    )
    try:
        expected_fp8_layers = (
            {1, 3, 5, 7}
            if main_dtype == DataType.FP8
            else ({5, 7} if fp8_layers is None else ({3} if fp8_layers else set()))
        )
        assert {
            layer for layer in (1, 3, 5, 7) if manager.get_kv_cache_dtype(layer) == DataType.FP8
        } == expected_fp8_layers
        if expected_fp8_layers and main_dtype == DataType.NVFP4:
            with pytest.raises(ValueError, match="layer-specific primary pool"):
                manager.get_unique_primary_pool()
        if not reuse_prefix and not host_offload:
            manager.add_dummy_requests([42], token_nums=[total_len], is_gen=False)
        offsets = torch.empty(
            (manager.num_attention_op_pools, 1, 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        rope = torch.zeros((256, 128), dtype=torch.float32, device="cuda")
        rope[:, ::2] = 1
        dequant_scale = torch.tensor([100.0 / (448 * 6)], device="cuda")
        quant_scale = dequant_scale.reciprocal()
        state = manager.get_ssm_states(0)
        state.fill_(7)
        saved_prefixes = {}
        for start, end in ((0, prefix_len), (prefix_len, total_len)):
            request_id = 42 + int(start > 0) if reuse_prefix else 42
            if reuse_prefix or (host_offload and start == 0):
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
            elif host_offload:
                assert manager.kv_cache_map[request_id].resize(end)
            manager.copy_batch_block_offsets(offsets, [request_id], 1, 1, 1)
            raw_pages = manager.get_pool_block_indices(1, request_ids=[request_id], num_contexts=1)[
                0
            ]
            for layer_idx, value in ((1, 1.0), (3, -2.0), (5, 0.5), (7, -1.0)):
                is_fp8 = manager.get_kv_cache_dtype(layer_idx) == DataType.FP8
                algo = QuantAlgo.FP8 if is_fp8 else QuantAlgo.NVFP4
                quant_mode = int(QuantConfig(kv_cache_quant_algo=algo).layer_quant_mode)
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
                    None if is_fp8 else quant_scale,
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
                pool = manager.get_primary_pool(manager.layer_offsets[layer_idx])
                if is_fp8:
                    output.copy_(
                        pool.view(torch.uint8)
                        .view(-1, head_dim)[indices.long()]
                        .view(torch.float8_e4m3fn)
                    )
                else:
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
                        raw_pages.to("cuda").long()[cached_positions // manager.tokens_per_block]
                        * manager.indexer_k_cache_page_scale,
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
            if reuse_prefix or (host_offload and start == 0):
                request.context_current_position = request.prompt_len
                manager.update_context_resources(batch)
            if host_offload and start == 0:
                _offload_request_and_clear_device_frames(manager, request, prefix_len)
            if reuse_prefix:
                manager.free_resources(request)
        if not reuse_prefix and not host_offload:
            assert torch.all(state == 7)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@skip_pre_blackwell
@pytest.mark.parametrize("reuse_prefix", [False, True])
@pytest.mark.parametrize(
    "main_dtype,fp8_layers",
    [(DataType.FP8, None), (DataType.NVFP4, ""), (DataType.NVFP4, "3"), (DataType.NVFP4, None)],
    ids=["fp8", "nvfp4", "explicit-fp8", "last-two-fp8"],
)
@pytest.mark.parametrize("indexer_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_independent_cache_formats_survive_host_offload(
    monkeypatch, reuse_prefix, main_dtype, fp8_layers, indexer_dtype, state_dtype
):
    """Resume main/indexer pages and recurrent state, then append or reuse the cached prefix."""
    test_kimi_dsa_nvfp4_append_across_token_and_page_boundaries(
        monkeypatch,
        64,
        reuse_prefix,
        main_dtype,
        fp8_layers,
        indexer_dtype,
        state_dtype,
        host_offload=True,
    )


@skip_pre_blackwell
@pytest.mark.parametrize("heads", [12, 96])
@pytest.mark.parametrize("decode_tokens", [1, 4])
@pytest.mark.parametrize("override", [None, "3"], ids=["last-two", "one-fp8-layer"])
@pytest.mark.parametrize("indexer_dtype", ["fp8", "fp4"])
def test_mixed_kv_mla_context_chunk_and_decode(
    monkeypatch, heads, decode_tokens, override, indexer_dtype
):
    """Compare MLA absorption across alternating physical pools and both indexer formats."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        KimiK31TextConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.graph_ops import (  # noqa: E501
        forward_context_sparse_attn,
        forward_generation_sparse_attn,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.ops.attention.nvfp4_mla_gather import (  # noqa: E501
        nvfp4_mla_kv_cache_gather,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_10x__default.weights import (  # noqa: E501
        declare_mla,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    if override is None:
        monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, override)
    fp8_layers = {5, 7} if override is None else {3}
    topk_count = 2048
    sparse_config = _sparse_config(indexer_dtype).model_copy(update={"index_topk": topk_count})
    torch.manual_seed(91)
    cfg = KimiK31TextConfig(
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=8,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=256,
        mla_use_nope=True,
        mla_use_output_gate=False,
        torch_dtype=torch.bfloat16,
        linear_attn_config=dict(
            full_attn_layers=[2, 4, 6, 8], kda_layers=[1, 3, 5, 7], num_heads=32
        ),
        dsa_config=dict(
            dsa_layers=[2, 4, 6, 8],
            index_k_norm_type="layernorm",
            index_rope_dim=0,
            index_num_heads=4,
            index_head_dim=128,
            index_topk=topk_count,
            index_topk_freq=1,
        ),
    )
    config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
        sparse_attention_config=sparse_config,
    )
    # Alternate precision to force the same step's metadata to change physical pools.
    layer_order = [1, 5, 3, 7]
    modules = {}
    with torch.device("cuda"):
        for layer in layer_order:
            bank = torch.nn.ParameterDict()
            mla = declare_mla(bank, cfg, layer, config, None)
            mla.mqa.indexer = None
            mla.k_b_proj_trans = torch.nn.Parameter(
                torch.randn(heads, 512, 128, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            mla.v_b_proj = torch.nn.Parameter(
                torch.randn(heads, 128, 512, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            assert mla.mqa.has_fp8_kv_cache == (layer in fp8_layers)
            assert mla.mqa.has_fp4_kv_cache == (layer not in fp8_layers)
            modules[layer] = mla
    manager = _make_manager(
        dtype=DataType.NVFP4, head_dim=576, num_mla_layers=4, indexer_k_dtype=indexer_dtype
    )
    try:
        manager.add_dummy_requests([42], token_nums=[33 + decode_tokens], is_gen=False)
        for start, count, generation in (
            (0, 17, False),
            (17, 16, False),
            (33, decode_tokens, True),
        ):
            metadata = modules[layer_order[0]].mqa.Metadata(
                seq_lens=torch.tensor([count], dtype=torch.int32),
                request_ids=[42],
                max_num_requests=1,
                num_contexts=0 if generation else 1,
                prompt_lens=[33 if generation else count],
                max_num_tokens=max(17, count),
                kv_cache_manager=manager,
                mapping=Mapping(),
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[start]),
                sparse_attention_config=sparse_config,
            )
            metadata.update_spec_dec_param(
                batch_size=1,
                is_spec_decoding_enabled=decode_tokens > 1,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=decode_tokens - 1,
                max_total_draft_tokens=decode_tokens - 1,
                num_contexts=0 if generation else 1,
            )
            metadata.prepare()
            topk = (
                torch.arange(topk_count, dtype=torch.int32, device="cuda").expand(count, -1).clone()
            )
            topk[topk > (start + torch.arange(count, device="cuda"))[:, None]] = -1
            metadata.shared_topk_indices = topk
            for layer in layer_order:
                mla = modules[layer]
                mla.mqa.predicted_tokens_per_seq = count if generation else 1
                q = torch.randn(count, heads * 192, device="cuda", dtype=torch.bfloat16) * 0.2
                latent = torch.randn(count, 576, device="cuda", dtype=torch.bfloat16) * 0.5
                output = torch.empty(count, heads * 128, device="cuda", dtype=torch.bfloat16)
                run = forward_generation_sparse_attn if generation else forward_context_sparse_attn
                run(mla, q, latent[:, :512], latent[:, 512:], metadata, output, latent_cache=latent)
                scale, offset = manager.get_primary_pool_page_index_params(layer)
                pages = manager.get_pool_block_indices(
                    1, request_ids=[42], num_contexts=int(not generation)
                )[0].to("cuda")
                positions = torch.arange(start + count, device="cuda")
                indices = (
                    (pages[positions // manager.tokens_per_block] * scale + offset)
                    * manager.tokens_per_block
                    + positions % manager.tokens_per_block
                ).to(torch.int32)
                pool = manager.get_primary_pool(layer)
                pool_tokens = pool.shape[0] * pool.shape[1] * manager.tokens_per_block
                if layer in fp8_layers:
                    kv = (
                        pool.view(torch.uint8)
                        .view(-1, 576)[indices.long()]
                        .view(torch.float8_e4m3fn)
                        .float()
                    )
                else:
                    pool_id = int(manager.kv_cache_pool_mapping[layer, 0])
                    address = int(manager.kv_cache_pool_pointers[pool_id, 0, 1])
                    sf = convert_to_torch_tensor(
                        TensorWrapper(address, torch.float8_e4m3fn, [pool_tokens, 36])
                    )
                    kv = nvfp4_mla_kv_cache_gather.reference(
                        pool.view(-1, 288), sf, indices, mla.mqa._nvfp4_mla_kv_scale_quant_orig
                    ).float()
                q3 = q.view(count, heads, 192)
                q_latent = torch.bmm(
                    q3[:, :, :128].transpose(0, 1), mla.k_b_proj_trans.transpose(1, 2)
                ).transpose(0, 1)
                fused = torch.cat((q_latent, q3[:, :, 128:]), dim=-1).float()
                # Sparse MLA uses E4M3 Q in both context and generation.
                fused = fused.to(torch.float8_e4m3fn).float()
                scores = torch.einsum("qhd,kd->qhk", fused, kv) / (192**0.5)
                scores.masked_fill_(
                    positions[None, None, :]
                    > (start + torch.arange(count, device="cuda"))[:, None, None],
                    float("-inf"),
                )
                latent_out = torch.einsum("qhk,kd->qhd", scores.softmax(-1), kv[:, :512]).to(
                    torch.bfloat16
                )
                expected = (
                    torch.bmm(latent_out.transpose(0, 1), mla.v_b_proj.transpose(1, 2))
                    .transpose(0, 1)
                    .reshape_as(output)
                )
                # FP8 attention rounds softmax intermediates. Normalize the
                # maximum error so cancellation near zero does not dominate.
                error = output.float() - expected.float()
                relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
                    expected.float()
                )
                phase = (layer, start, count, generation)
                assert relative_l2 < 0.03, (phase, relative_l2.item())
                max_relative_error = error.abs().max() / expected.abs().max()
                assert max_relative_error < 0.05, (phase, max_relative_error.item())
    finally:
        manager.shutdown()


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
