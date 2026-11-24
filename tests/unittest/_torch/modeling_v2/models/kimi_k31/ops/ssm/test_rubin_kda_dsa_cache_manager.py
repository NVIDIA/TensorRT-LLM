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
"""The family KDA + DSA hybrid cache manager of the Kimi K3.1 Rubin targets: indexer K-cache
views, page tables, FP8 KV layers, draft layers and replay seeding."""

import importlib
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import is_dsa_cache_manager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MambaStateConfig,
    MTPDecodingConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# sm_107 only: l0_gb300 collects ops/ as a directory.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7),
    reason="Kimi K3.1 Rubin: sm_107",
)

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


_MLA_HEAD_DIM = 64


def _make_manager(
    indexer_k_dtype: str = "fp8",
    dtype: DataType = DataType.BF16,
    head_dim: int = _MLA_HEAD_DIM,
    enable_block_reuse: bool = False,
    num_mla_layers: int = 2,
    **draft_kwargs,
) -> RubinKdaDsaHybridCacheManager:
    """``num_mla_layers`` KDA + DSA layer pairs (KDA first), with partial reuse and a snapshot at
    the end of each prompt."""
    return _make_dsa_manager(
        indexer_k_dtype,
        dtype,
        mla_layers=tuple(range(2, 2 * num_mla_layers + 1, 2)),
        num_hidden_layers=2 * num_mla_layers,
        head_dim=head_dim,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=enable_block_reuse,
            enable_partial_reuse=True,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
        ),
        **draft_kwargs,
    )


@pytest.mark.parametrize(
    "dtype,view_dtype,elements_per_page",
    [
        (DataType.BF16, torch.bfloat16, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM),
        (DataType.FP8, torch.float8_e4m3fn, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM),
        # Two E2M1 values per byte: the NVFP4 view is int8 over the packed latents.
        (DataType.NVFP4, torch.int8, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM // 2),
    ],
)
def test_kimi_dsa_pool_view_dtype(
    dtype: DataType,
    view_dtype: torch.dtype,
    elements_per_page: int,
) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_manager(dtype=dtype, fp8_kv_layers=())
    try:
        pool = manager.get_unique_primary_pool()
        assert pool.dtype == view_dtype
        assert pool.shape[1:] == (2, 1, elements_per_page)
        # The view steps one K page per MLA layer, as the pool lays them out.
        page_stride = manager.impl.get_page_stride(manager._attention_local_layers[0], Role.KEY)
        assert int(page_stride) == elements_per_page * pool.element_size()
    finally:
        manager.shutdown()


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


@pytest.mark.parametrize(
    "indexer_k_dtype, bytes_per_token",
    [("fp8", _INDEXER_BYTES_PER_TOKEN)],
)
def test_kimi_dsa_hybrid_manager_indexer_views(indexer_k_dtype: str, bytes_per_token: int) -> None:
    manager = _make_manager(indexer_k_dtype)
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
        assert pool.shape[1:] == (2, 1, _TOKENS_PER_BLOCK * _MLA_HEAD_DIM)
    finally:
        manager.shutdown()


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


def test_kimi_dsa_hybrid_manager_disagg_page_table() -> None:
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind

    manager = _make_manager()
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


_KDA_HEADS = 8
_KDA_HEAD_DIM = 128
_NUM_SPEC = 2


def _make_replay_manager() -> RubinKdaDsaHybridCacheManager:
    """A generation server's manager: with drafts the KDA layers keep replay caches in V2 pools."""
    return RubinKdaDsaHybridCacheManager(
        mamba_d_state=_KDA_HEAD_DIM,
        mamba_d_conv=4,
        mamba_num_heads=_KDA_HEADS,
        mamba_n_groups=_KDA_HEADS,
        mamba_head_dim=_KDA_HEAD_DIM,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=_MLA_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        spec_config=MTPDecodingConfig(max_draft_len=_NUM_SPEC),
        kda_replay_num_spec=_NUM_SPEC,
        layer_mask=_MLA_MASK,
        sparse_attention_config=_sparse_config(),
        pretrained_config=_dsa_text_config((2, 4), 4),
    )


def test_kimi_dsa_manager_keeps_the_k3_package_out(monkeypatch) -> None:
    """On the real V2 pools: the K3.1 replay state plans when the pools are bound to seed
    transferred slots in all layers at once from stacked views of them, and with block reuse the
    snapshot probe asks this package's KDA prefill; neither loads the K3 package or patches the
    CuTe DSL compile pipeline."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
        rubin_kda_dsa_cache_manager,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
        RubinKdaReplayState,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_metadata import (
        KDA_PREFILL_CHUNK_SIZE,
    )
    from tensorrt_llm._torch.modules.kimi_kda import cache_manager as shared_cache_manager
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    k3_package = shared_cache_manager._K3_KDA_PACKAGE.rsplit(".", 1)[0]
    requested = []
    import_module = importlib.import_module

    def spy(name, package=None):
        if name == k3_package or name.startswith(k3_package + "."):
            requested.append(name)
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    monkeypatch.setattr(shared_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    try:
        from cutlass.cutlass_dsl.cutlass import CuTeDSL
    except ImportError:
        CuTeDSL = None
    pipeline = None if CuTeDSL is None else CuTeDSL._get_pipeline

    manager = _make_replay_manager()
    try:
        replay = manager._kda_replay
        assert type(replay) is RubinKdaReplayState

        # A context server's manager has block reuse on.
        manager.enable_block_reuse = True
        snapshots = (
            hasattr(_KVCache, "acquire_ssm_snapshot_slot")
            and rubin_kda_dsa_cache_manager._kda_prefill_stores_snapshots()
        )
        alignment = manager._get_kda_midchunk_snapshot_alignment()
        assert alignment == (KDA_PREFILL_CHUNK_SIZE if snapshots else None)
        assert alignment == KDA_PREFILL_CHUNK_SIZE

        slots = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        window = replay._committed_window
        offsets = [0]
        for section_dim in replay._conv_section_dims:
            offsets.append(offsets[-1] + section_dim)
        replay.prev_num_accepted_tokens.index_fill_(0, slots, 1)
        for layer, conv_state in enumerate(replay._conv_states):
            fresh = torch.randn(conv_state[slots].shape, device="cuda").to(conv_state.dtype)
            conv_state.index_copy_(0, slots, fresh)
            for name, buffer in replay._layer_cache_fields(layer).items():
                if name != "prev_num_accepted_tokens" and buffer is not None:
                    buffer.index_fill_(0, slots, 1)

        # The pools space each kind's per-layer views evenly, so the plan built at binding stacks.
        assert replay._batched_seed_plan
        assert replay._get_batched_seed_plan() is not None
        replay.seed_transferred_slots(slots)
        torch.cuda.synchronize()

        assert replay.prev_num_accepted_tokens[slots].tolist() == [0, 0]
        for layer, conv_state in enumerate(replay._conv_states):
            fields = replay._layer_cache_fields(layer)
            for section, name in enumerate(("kda_conv_q", "kda_conv_k", "kda_conv_v")):
                buffer = fields[name]
                got = buffer.float()[slots]
                committed = conv_state[slots, offsets[section] : offsets[section + 1]]
                assert torch.equal(got[:, :, :window], committed.to(buffer.dtype).float())
                assert not got[:, :, window:].any()
            for name in ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache"):
                if fields.get(name) is not None:
                    assert not fields[name].float()[slots].any()
    finally:
        manager.shutdown()
    # The plan's stacked views go with the pool views they were built from.
    assert replay._batched_seed_plan is None
    assert requested == []
    assert CuTeDSL is None or CuTeDSL._get_pipeline is pipeline


def _index_key_role():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    return Role.INDEX_KEY


@pytest.mark.parametrize(
    "prefix_len,reuse_prefix",
    [(length, False) for length in (15, 17, 31, 33, 63, 65)]
    + [(length, True) for length in (32, 64, 96)],
)
@pytest.mark.parametrize(
    "fp8_layers", [(), (3,), None], ids=["nvfp4", "explicit-fp8", "last-two-fp8"]
)
def test_kimi_dsa_nvfp4_append_across_token_and_page_boundaries(
    prefix_len: int, reuse_prefix: bool, fp8_layers: tuple[int, ...] | None
) -> None:
    """Preserve cached rows on append and on the supported full-block prefix reuse."""
    head_dim = 576
    total_len = prefix_len + 3
    manager = _make_manager(
        dtype=DataType.NVFP4,
        head_dim=head_dim,
        enable_block_reuse=reuse_prefix,
        num_mla_layers=4,
        fp8_kv_layers=fp8_layers,
    )
    try:
        expected_fp8_layers = {5, 7} if fp8_layers is None else set(fp8_layers)
        assert {
            layer for layer in (1, 3, 5, 7) if manager.get_kv_cache_dtype(layer) == DataType.FP8
        } == expected_fp8_layers
        if expected_fp8_layers:
            with pytest.raises(ValueError, match="layer-specific primary pool"):
                manager.get_unique_primary_pool()
        if not reuse_prefix:
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


def test_kimi_dsa_fp8_tail_requires_nvfp4() -> None:
    """Explicit FP8 KV layers need NVFP4 storage; the default rule selects none without it."""
    with pytest.raises(ValueError, match="fp8_kv_layers requires the NVFP4 KV cache storage"):
        _make_manager(dtype=DataType.BF16, fp8_kv_layers=(3,))
    manager = _make_manager(dtype=DataType.BF16, fp8_kv_layers=None)
    try:
        assert not manager._fp8_mla_global_layer_ids
        assert all(manager.get_kv_cache_dtype(layer) == DataType.BF16 for layer in (1, 3))
    finally:
        manager.shutdown()


# FP8 KV layers beside NVFP4 ones: one page table per lifecycle plus an FP8 plane

# Eight layers: the even ones (0, 2, 4, 6) are KDA and the odd ones (1-indexed 2, 4, 6, 8) are DSA
# MLA layers. By default the last two MLA layers, zero-based 5 and 7, keep FP8 latent pages beside
# the NVFP4 pages of 1 and 3.
_MIXED_MLA_LAYERS = (2, 4, 6, 8)
_MIXED_NUM_LAYERS = 8
_MIXED_DSA_LAYERS = (1, 3, 5, 7)
_MIXED_NVFP4_LAYERS = (1, 3)
_MIXED_FP8_LAYERS = (5, 7)
# Per-token latent bytes: NVFP4 packs two E2M1 values per byte and one E4M3 scale per 16 values.
_NVFP4_LATENT_BYTES = _MLA_HEAD_DIM // 2
_NVFP4_SCALE_BYTES = _MLA_HEAD_DIM // 16
_FP8_LATENT_BYTES = _MLA_HEAD_DIM
# The production DSA geometry: 93 layers whose DSA (= MLA) layers are 1-indexed 4, 8, ..., 88, 92
# and 93, with 576-wide latents in 64-token pages and an FP4 indexer K-cache (68 B per token).
_PROD_NUM_LAYERS = 93
_PROD_MLA_LAYERS = (*range(4, 89, 4), 92, 93)
_PROD_DSA_LAYERS = tuple(layer - 1 for layer in _PROD_MLA_LAYERS)
_PROD_FP8_LAYERS = (91, 92)
_PROD_HEAD_DIM = 576
_PROD_TOKENS_PER_BLOCK = 64
_PROD_INDEXER_BYTES = 68
# A unified standalone drafter registered as model layers (small, for the eight-layer schedule).
_DRAFT_KWARGS = dict(
    draft_num_layers=3,
    draft_num_kv_heads=1,
    draft_head_dim=_MLA_HEAD_DIM,
    draft_dtype=DataType.BF16,
    draft_kv_factor=1,
    draft_scratch_tokens=8,
)
# The production DSpark drafter: five layers of one 576-wide FP8 latent per token.
_PROD_DRAFT_KWARGS = dict(
    draft_num_layers=5,
    draft_num_kv_heads=1,
    draft_head_dim=_PROD_HEAD_DIM,
    draft_dtype=DataType.FP8,
    draft_kv_factor=1,
    draft_scratch_tokens=8,
)


def _dsa_text_config(
    mla_layers: tuple,
    num_hidden_layers: int,
    kv_lora_rank: int = 32,
    qk_rope_head_dim: int = 32,
) -> SimpleNamespace:
    """Text config whose MLA layers (1-indexed) are all DSA layers; every other layer is KDA."""
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=num_hidden_layers,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [
                layer for layer in range(1, num_hidden_layers + 1) if layer not in mla_layers
            ],
            "full_attn_layers": list(mla_layers),
        },
        dsa_config={"dsa_layers": list(mla_layers), "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _make_dsa_manager(
    indexer_k_dtype: str = "fp8",
    dtype: DataType = DataType.NVFP4,
    *,
    mla_layers: tuple = _MIXED_MLA_LAYERS,
    num_hidden_layers: int = _MIXED_NUM_LAYERS,
    head_dim: int = _MLA_HEAD_DIM,
    kv_lora_rank: int = 32,
    qk_rope_head_dim: int = 32,
    tokens_per_block: int = _TOKENS_PER_BLOCK,
    max_tokens: int = 512,
    max_seq_len: int = 256,
    max_batch_size: int = 2,
    host_cache_size: int | None = None,
    kv_cache_config: KvCacheConfig | None = None,
    mamba_ssm_cache_dtype: torch.dtype = torch.float32,
    **draft_kwargs,
) -> RubinKdaDsaHybridCacheManager:
    # The MLA layers (1-indexed) are the DSA layers; every other layer is KDA.
    mla_mask = [layer + 1 in mla_layers for layer in range(num_hidden_layers)]
    kda_mask = [not is_mla for is_mla in mla_mask]
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(
            max_tokens=max_tokens, enable_block_reuse=False, host_cache_size=host_cache_size
        )
    return RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=sum(kda_mask),
        mamba_layer_mask=kda_mask,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=sum(mla_mask),
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        layer_mask=mla_mask,
        sparse_attention_config=_sparse_config(indexer_k_dtype),
        pretrained_config=_dsa_text_config(
            mla_layers, num_hidden_layers, kv_lora_rank, qk_rope_head_dim
        ),
        **draft_kwargs,
    )


def _make_production_manager(
    dtype: DataType = DataType.NVFP4, indexer_k_dtype: str = "fp4", **kwargs
) -> RubinKdaDsaHybridCacheManager:
    """A manager over the production DSA geometry, NVFP4 with FP4 index keys unless ``dtype`` and
    ``indexer_k_dtype`` say otherwise, with tiny KDA states."""
    kwargs.setdefault("max_tokens", 8192)
    kwargs.setdefault("max_seq_len", 1024)
    return _make_dsa_manager(
        indexer_k_dtype,
        dtype,
        mla_layers=_PROD_MLA_LAYERS,
        num_hidden_layers=_PROD_NUM_LAYERS,
        head_dim=_PROD_HEAD_DIM,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_block=_PROD_TOKENS_PER_BLOCK,
        **kwargs,
    )


def _page_index_params(manager, local_layer_idx: int, role) -> tuple:
    """(scale, expansion, layer offset) of a buffer's V2 page-index converter."""
    converter = manager.impl.get_page_index_converter(local_layer_idx, role)
    return int(converter.scale), int(converter.expansion), int(converter.layer_offset)


def _slot_sizes_of_group(manager, layer_group: int) -> list:
    """Slot bytes of each physical pool of the pool group that holds ``layer_group``."""
    sizes = [
        [int(size) for size in pool_group.slot_desc.slot_size_list]
        for pool_group in manager.impl.pool_group_descs
        if any(
            int(variant.layer_group_id) == layer_group for variant in pool_group.slot_desc.variants
        )
    ]
    assert len(sizes) == 1, sizes
    return sizes[0]


def _copy_block_offsets(manager, request_ids: list, num_contexts: int = 0) -> torch.Tensor:
    """The attention ops' [pool, sequence, K/V, block] page table of a batch, on the host."""
    planes = torch.full(
        (manager.num_attention_op_pools, len(request_ids), 2, manager.max_blocks_per_seq),
        -7,
        dtype=torch.int32,
        device="cuda",
    )
    manager.copy_batch_block_offsets(planes, request_ids, 1, num_contexts, len(request_ids))
    torch.cuda.synchronize()
    return planes.cpu()


def _assert_planes_scale_the_host_table(
    manager, planes: torch.Tensor, request_ids: list, num_contexts: int = 0
) -> None:
    """Each lifecycle plane holds its host slots times its page scale (0 for an unmapped block);
    the FP8 plane holds the DSA lifecycle's slots times the FP8 pool's V2 page scale."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

    copy_idx = manager.index_mapper.get_copy_index(request_ids, num_contexts, 1)
    copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
    host = manager.host_kv_cache_block_offsets[:, copy_idx, 0, :].to(torch.int64)

    def assert_plane(plane: int, slots: torch.Tensor, scale: int, kv_offset: int) -> None:
        unmapped = slots == int(BAD_PAGE_INDEX)
        want_k = torch.where(unmapped, 0, slots * scale)
        want_v = torch.where(unmapped, 0, slots * scale + kv_offset)
        assert torch.equal(planes[plane, :, 0].to(torch.int64), want_k), f"plane {plane}, K"
        assert torch.equal(planes[plane, :, 1].to(torch.int64), want_v), f"plane {plane}, V"

    assert planes.shape[0] == manager.num_attention_op_pools
    for pool_id in range(manager.num_pools):
        assert_plane(
            pool_id,
            host[pool_id],
            int(manager.index_scales[pool_id]),
            int(manager.kv_offset[pool_id]),
        )
    if manager._fp8_latent_op_pool is not None:
        # The scale V2 gives the FP8 pages, not the manager's own conversion table.
        fp8_layer = manager._fp8_latent_local_layers[0]
        fp8_scale = int(manager.impl.get_page_index_scale(fp8_layer, Role.KEY))
        assert fp8_scale == len(manager._fp8_latent_local_layers)
        assert_plane(
            manager._fp8_latent_op_pool, host[manager._fp8_latent_source_pool], fp8_scale, 0
        )


def test_fp8_kv_layer_layout() -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

    manager = _make_dsa_manager(fp8_kv_layers=None)
    try:
        assert manager._fp8_mla_global_layer_ids == set(_MIXED_FP8_LAYERS)
        assert manager._attention_local_layers == list(_MIXED_DSA_LAYERS)
        assert manager._nvfp4_latent_local_layers == list(_MIXED_NVFP4_LAYERS)
        assert manager._fp8_latent_local_layers == list(_MIXED_FP8_LAYERS)
        # One page table per lifecycle: no per-layer planes.
        assert not manager._use_per_layer_page_tables

        # The selected layers report FP8 storage, to the buffer views too; the manager stays NVFP4.
        assert manager.dtype == DataType.NVFP4
        assert manager.get_kv_cache_dtype() == DataType.NVFP4
        for layer in _MIXED_DSA_LAYERS:
            want = DataType.FP8 if layer in _MIXED_FP8_LAYERS else DataType.NVFP4
            assert manager.get_kv_cache_dtype(layer) == want, layer
            assert manager._get_layer_dtype_by_local_id(layer) == want, layer

        # Per-token bytes of each role: FP8 pages carry no block scales.
        index_bytes = _INDEXER_BYTES_PER_TOKEN
        for layer in _MIXED_NVFP4_LAYERS:
            assert manager.get_layer_bytes_per_token(layer, Role.KEY) == _NVFP4_LATENT_BYTES
            assert (
                manager.get_layer_bytes_per_token(layer, Role.KEY_BLOCK_SCALE) == _NVFP4_SCALE_BYTES
            )
            assert manager.get_layer_bytes_per_token(layer, Role.INDEX_KEY) == index_bytes
        for layer in _MIXED_FP8_LAYERS:
            assert manager.get_layer_bytes_per_token(layer, Role.KEY) == _FP8_LATENT_BYTES
            assert manager.get_layer_bytes_per_token(layer, Role.KEY_BLOCK_SCALE) == 0
            assert manager.get_layer_bytes_per_token(layer, Role.INDEX_KEY) == index_bytes

        # One latent pool per format in the DSA layer group, each holding one page per layer of
        # its format in layer order; the indexer K-cache keeps one page per DSA layer.
        def params(layers, role):
            return [_page_index_params(manager, layer, role) for layer in layers]

        assert params(_MIXED_NVFP4_LAYERS, Role.KEY) == [(2, 1, 0), (2, 1, 1)]
        assert params(_MIXED_NVFP4_LAYERS, Role.KEY_BLOCK_SCALE) == [(2, 1, 0), (2, 1, 1)]
        assert params(_MIXED_FP8_LAYERS, Role.KEY) == [(2, 1, 0), (2, 1, 1)]
        assert params(_MIXED_DSA_LAYERS, Role.INDEX_KEY) == [(4, 1, k) for k in range(4)]
        tpb = _TOKENS_PER_BLOCK
        dsa = manager._attention_pool_id
        assert sorted(_slot_sizes_of_group(manager, dsa)) == sorted(
            [
                2 * tpb * _NVFP4_LATENT_BYTES,
                2 * tpb * _NVFP4_SCALE_BYTES,
                2 * tpb * _FP8_LATENT_BYTES,
                4 * tpb * index_bytes,
            ]
        )
        num_slots = manager._num_attention_slots
        assert num_slots > 0
        assert manager._count_attention_slots(1) == manager._count_attention_slots(5) == num_slots

        # The DSA lifecycle's pointer row addresses the NVFP4 pool; one virtual attention-op pool
        # after every lifecycle pool addresses the FP8 pool from the same slots.
        num_pools = manager.num_pools
        assert manager.num_attention_op_pools == num_pools + 1
        assert manager._fp8_latent_op_pool == num_pools
        assert manager._fp8_latent_source_pool == dsa
        assert tuple(manager.kv_cache_pool_pointers.shape) == (num_pools + 1, 2, 2)
        pointers = manager.kv_cache_pool_pointers.tolist()

        def base(layer, role, mode):
            return int(manager.impl.get_mem_pool_base_address(layer, role, mode))

        assert pointers[num_pools] == [[base(5, Role.KEY, PageIndexMode.PER_LAYER), 0], [0, 0]]
        assert pointers[dsa] == [
            [
                base(1, Role.KEY, PageIndexMode.SHARED),
                base(1, Role.KEY_BLOCK_SCALE, PageIndexMode.SHARED),
            ],
            [0, 0],
        ]
        mapping = manager.kv_cache_pool_mapping.tolist()
        assert [mapping[layer] for layer in _MIXED_DSA_LAYERS] == [
            [dsa, 0],
            [dsa, 1],
            [num_pools, 0],
            [num_pools, 1],
        ]
        for layer in range(_MIXED_NUM_LAYERS):
            if layer not in _MIXED_DSA_LAYERS:
                assert mapping[layer][0] == manager.layer_to_pool_mapping_dict[layer], layer
        assert manager.index_scales.numel() == manager.kv_offset.numel() == num_pools
        assert manager.host_kv_cache_block_offsets.shape[0] == num_pools
        assert int(manager.index_scales[dsa]) == manager._index_scale_ints[dsa] == 2
        assert manager._fp8_latent_index_scales.tolist() == [2]
        assert manager._fp8_latent_kv_offsets.tolist() == [0]
        for layer, want in zip(_MIXED_DSA_LAYERS, [(2, 0), (2, 1), (2, 0), (2, 1)]):
            assert manager.get_primary_pool_page_index_params(layer) == want, layer

        # Zero-copy views of the two latent pools, one per format.
        nvfp4_pool = manager.get_primary_pool(1)
        assert manager.get_primary_pool(3) is nvfp4_pool
        assert nvfp4_pool.dtype == torch.int8
        assert tuple(nvfp4_pool.shape) == (num_slots, 2, 1, tpb * _NVFP4_LATENT_BYTES)
        assert nvfp4_pool.data_ptr() == pointers[dsa][0][0]
        fp8_pool = manager.get_primary_pool(5)
        assert manager.get_primary_pool(7) is fp8_pool
        assert fp8_pool.dtype == torch.float8_e4m3fn
        assert tuple(fp8_pool.shape) == (num_slots, 2, 1, tpb * _FP8_LATENT_BYTES)
        assert fp8_pool.data_ptr() == pointers[num_pools][0][0]
        with pytest.raises(ValueError, match="layer-specific primary pool"):
            manager.get_unique_primary_pool()
        # The FP8 layers' buffer views see E4M3 pages.
        assert manager.get_buffers(5, "HND").dtype == torch.float8_e4m3fn
    finally:
        manager.shutdown()


def test_fp8_kv_layers_off_keeps_the_nvfp4_layout() -> None:
    # With none selected, every DSA layer stays NVFP4 with one page table per lifecycle (an
    # all-NVFP4 cache).
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_dsa_manager(fp8_kv_layers=())
    try:
        assert manager._fp8_mla_global_layer_ids == frozenset()
        assert manager._fp8_latent_local_layers == []
        assert manager._nvfp4_latent_local_layers == list(_MIXED_DSA_LAYERS)
        assert manager._fp8_latent_op_pool is None
        assert manager._fp8_latent_materializer is None
        for ordinal, layer in enumerate(_MIXED_DSA_LAYERS):
            assert manager.get_kv_cache_dtype(layer) == DataType.NVFP4
            assert manager._get_layer_dtype_by_local_id(layer) == DataType.NVFP4
            assert _page_index_params(manager, layer, Role.KEY) == (4, 1, ordinal)
            assert _page_index_params(manager, layer, Role.KEY_BLOCK_SCALE) == (4, 1, ordinal)
        tpb = _TOKENS_PER_BLOCK
        dsa = manager._attention_pool_id
        assert sorted(_slot_sizes_of_group(manager, dsa)) == sorted(
            [
                4 * tpb * _NVFP4_LATENT_BYTES,
                4 * tpb * _NVFP4_SCALE_BYTES,
                4 * tpb * _INDEXER_BYTES_PER_TOKEN,
            ]
        )
        num_pools = manager.num_pools
        assert manager.num_attention_op_pools == num_pools
        assert tuple(manager.kv_cache_pool_pointers.shape) == (num_pools, 2, 2)
        mapping = manager.kv_cache_pool_mapping.tolist()
        assert [mapping[layer] for layer in _MIXED_DSA_LAYERS] == [[dsa, k] for k in range(4)]
        assert manager._index_scale_ints[dsa] == 4
        pool = manager.get_unique_primary_pool()
        assert tuple(pool.shape[1:]) == (4, 1, tpb * _NVFP4_LATENT_BYTES)

        # The base copy fills one plane per lifecycle.
        request_ids = [0, 1]
        manager.add_dummy_requests(request_ids, token_nums=[40, 100], is_gen=True)
        planes = _copy_block_offsets(manager, request_ids)
        _assert_planes_scale_the_host_table(manager, planes, request_ids)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "value, slot_sizes, num_fp8_layers, dsa_bytes_per_token",
    [
        # 22 NVFP4 layers x (288 + 36 + 68) B and 2 FP8 layers x (576 + 68) B per token.
        (None, [405504, 104448, 73728, 50688], 2, 9912),
        ((91, 92), [405504, 104448, 73728, 50688], 2, 9912),
        # 24 NVFP4 layers x (288 + 36 + 68) B per token: the all-NVFP4 layout.
        ((), [442368, 104448, 55296], 0, 9408),
    ],
    ids=["default", "explicit_91_92", "none"],
)
def test_fp8_kv_layers_production_geometry(
    value, slot_sizes, num_fp8_layers, dsa_bytes_per_token
) -> None:
    # The production geometry with the production DSpark drafter (five FP8 layers).
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_production_manager(fp8_kv_layers=value, **_PROD_DRAFT_KWARGS)
    try:
        fp8_layers = list(_PROD_FP8_LAYERS[:num_fp8_layers])
        nvfp4_layers = [layer for layer in _PROD_DSA_LAYERS if layer not in fp8_layers]
        num_nvfp4 = len(nvfp4_layers)
        assert manager._fp8_latent_local_layers == fp8_layers
        assert manager._nvfp4_latent_local_layers == nvfp4_layers
        assert not manager._use_per_layer_page_tables
        dsa = manager._attention_pool_id
        assert _slot_sizes_of_group(manager, dsa) == slot_sizes
        assert (
            sum(manager.get_layer_bytes_per_token(layer, Role.ALL) for layer in _PROD_DSA_LAYERS)
            == dsa_bytes_per_token
        )
        for ordinal, layer in enumerate(nvfp4_layers):
            assert _page_index_params(manager, layer, Role.KEY) == (num_nvfp4, 1, ordinal)
            assert manager.get_primary_pool_page_index_params(layer) == (num_nvfp4, ordinal)
        for ordinal, layer in enumerate(fp8_layers):
            assert _page_index_params(manager, layer, Role.KEY) == (2, 1, ordinal)
            assert manager.get_primary_pool_page_index_params(layer) == (2, ordinal)
        assert [
            _page_index_params(manager, layer, Role.INDEX_KEY) for layer in _PROD_DSA_LAYERS
        ] == [(24, 1, k) for k in range(24)]

        # The drafter: five 576 B FP8 latents per token in a layer group of its own, decoded by
        # DSpark from that group's plane, which the FP8 plane does not move.
        draft_locals = [manager.layer_offsets[layer] for layer in manager.draft_layer_ids]
        assert len(draft_locals) == 5
        draft_group = manager.layer_to_pool_mapping_dict[draft_locals[0]]
        assert draft_group != dsa and draft_group < manager.num_pools
        assert _slot_sizes_of_group(manager, draft_group) == [5 * _PROD_HEAD_DIM * 64]
        mapping = manager.kv_cache_pool_mapping.tolist()
        assert [mapping[layer][0] for layer in draft_locals] == [draft_group] * 5
        assert manager._index_scale_ints[draft_group] == manager.get_layer_page_index_scale(
            manager.draft_layer_ids[0]
        )

        tpb = _PROD_TOKENS_PER_BLOCK
        num_slots = manager._num_attention_slots
        pool = manager.get_primary_pool(nvfp4_layers[0])
        assert pool.dtype == torch.int8
        assert tuple(pool.shape) == (num_slots, num_nvfp4, 1, tpb * _PROD_HEAD_DIM // 2)
        if fp8_layers:
            fp8_pool = manager.get_primary_pool(fp8_layers[0])
            assert tuple(fp8_pool.shape) == (num_slots, 2, 1, tpb * _PROD_HEAD_DIM)
            # One FP8 plane after the lifecycle planes (KDA, DSA, drafter), not one per layer.
            assert manager.num_attention_op_pools == manager.num_pools + 1
            assert manager._index_scale_ints[dsa] == 22
        else:
            assert manager.num_attention_op_pools == manager.num_pools
            assert manager._index_scale_ints[dsa] == 24
    finally:
        manager.shutdown()


# The schedules of the page-table test, as (manager factory, kwargs, FP8 env). On the eight-layer
# schedule both latent pools have page scale 2. On the production one the NVFP4 pool has page
# scale 22 and the FP8 pool 2, so a plane filled with the other pool's scale lands on other pages.
_PAGE_TABLE_SCHEDULES = {
    "mixed8": (_make_dsa_manager, _DRAFT_KWARGS),
    "production": (_make_production_manager, _PROD_DRAFT_KWARGS),
}


@pytest.mark.parametrize("schedule", sorted(_PAGE_TABLE_SCHEDULES))
@pytest.mark.parametrize("copy_mode", ["host_table", "triton", "device_expansion"])
def test_fp8_kv_page_tables_address_the_per_layer_pages(
    monkeypatch: pytest.MonkeyPatch, copy_mode: str, schedule: str
) -> None:
    # For every MLA page of a batch, the address the attention ops derive from the pool pointers,
    # the materialized plane and the layer mapping is the one V2's per-layer addressing gives.
    from tensorrt_llm._torch.pyexecutor import step_upload_kernels
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, PageIndexMode

    make_manager, draft_kwargs = _PAGE_TABLE_SCHEDULES[schedule]
    monkeypatch.setenv(
        "TLLM_KVCM_V2_DEVICE_PAGE_TABLE", "1" if copy_mode == "device_expansion" else "0"
    )
    # TRTLLM_KV_BLOCK_OFFSETS_TRITON (helix8 GEN sets 2), read once at import.
    monkeypatch.setattr(step_upload_kernels, "_BLOCK_OFFSETS_TRITON", copy_mode == "triton")
    reference = make_manager(fp8_kv_layers=(), **draft_kwargs)
    manager = make_manager(fp8_kv_layers=None, **draft_kwargs)
    try:
        device_expansion = copy_mode == "device_expansion"
        assert manager._page_table_materializer.uses_device_expansion == device_expansion
        assert (manager._fp8_latent_materializer is not None) == device_expansion
        num_pools, dsa = manager.num_pools, manager._attention_pool_id
        pointers = manager.kv_cache_pool_pointers.tolist()
        mapping = manager.kv_cache_pool_mapping.tolist()
        nvfp4_layers = manager._nvfp4_latent_local_layers
        fp8_layers = manager._fp8_latent_local_layers
        fp8_scale = int(manager.impl.get_page_index_scale(fp8_layers[0], Role.KEY))
        assert (manager._index_scale_ints[dsa], fp8_scale) == (len(nvfp4_layers), len(fp8_layers))
        if schedule == "production":
            assert (len(nvfp4_layers), fp8_scale) == (22, 2)

        # DSpark decodes its draft table from its layer group's plane with the page scale it
        # compares at start-up; both are those of the all-NVFP4 cache.
        draft_layers = [manager.layer_offsets[layer] for layer in manager.draft_layer_ids]
        draft_plane = manager.layer_to_pool_mapping_dict[draft_layers[0]]
        assert draft_plane != dsa and draft_plane < num_pools
        assert [mapping[layer][0] for layer in draft_layers] == [draft_plane] * len(draft_layers)
        assert manager.index_scales.numel() == manager.kv_offset.numel() == num_pools
        draft_scale = manager.get_layer_page_index_scale(manager.draft_layer_ids[0])
        assert manager._index_scale_ints[draft_plane] == draft_scale
        assert manager.layer_to_pool_mapping_dict == reference.layer_to_pool_mapping_dict
        reference_mapping = reference.kv_cache_pool_mapping.tolist()
        assert [mapping[layer] for layer in draft_layers] == [
            reference_mapping[layer] for layer in draft_layers
        ]
        assert manager._index_scale_ints[draft_plane] == reference._index_scale_ints[draft_plane]

        request_ids = [0, 1]
        manager.add_dummy_requests(request_ids, token_nums=[40, 100], is_gen=True)
        planes = _copy_block_offsets(manager, request_ids)
        _assert_planes_scale_the_host_table(manager, planes, request_ids)
        copy_idx = manager.index_mapper.get_copy_index(request_ids, 0, 1)
        copy_idx = copy_idx.to(device="cpu", dtype=torch.long)
        host_slots = manager.host_kv_cache_block_offsets[dsa, copy_idx, 0, :].to(torch.int64)
        mapped = host_slots != int(BAD_PAGE_INDEX)
        assert bool(mapped.any())
        for plane, scale in ((dsa, len(nvfp4_layers)), (num_pools, fp8_scale)):
            plane_slots = planes[plane, :, 0].to(torch.int64)
            assert torch.equal(plane_slots[mapped], host_slots[mapped] * scale), plane

        tpb = manager.tokens_per_block
        checked = 0
        for seq, request_id in enumerate(request_ids):
            kv_cache = manager.kv_cache_map[request_id]
            slots = [
                int(slot) for slot in kv_cache.get_base_page_indices(dsa)[: kv_cache.num_blocks]
            ]
            assert slots and int(BAD_PAGE_INDEX) not in slots
            for layer in manager._attention_local_layers:
                pool_id, layer_in_pool = mapping[layer]
                # E4M3 pages hold one byte per latent value; NVFP4 pages two values per byte and
                # one E4M3 block scale per 16 values.
                elements = tpb * manager.head_dim_per_layer[layer]
                if layer in fp8_layers:
                    lanes = [(Role.KEY, 0, elements)]
                else:
                    lanes = [
                        (Role.KEY, 0, elements // 2),
                        (Role.KEY_BLOCK_SCALE, 1, elements // 16),
                    ]
                for role, lane, page_bytes in lanes:
                    converter = manager.impl.get_page_index_converter(layer, role)
                    pages = converter(slots, PageIndexMode.PER_LAYER)
                    pool_base = int(
                        manager.impl.get_mem_pool_base_address(layer, role, PageIndexMode.PER_LAYER)
                    )
                    assert int(manager.impl.get_page_stride(layer, role)) == page_bytes
                    for block, page in enumerate(pages):
                        plane = int(planes[pool_id, seq, 0, block])
                        address = pointers[pool_id][0][lane] + (plane + layer_in_pool) * page_bytes
                        want = pool_base + int(page) * page_bytes
                        assert address == want, (request_id, layer, str(role), block)
                        checked += 1
        assert checked > 0

        # A batch table without the FP8 plane is refused.
        short = torch.zeros(
            (num_pools, len(request_ids), 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        with pytest.raises(ValueError, match="attention-op pools"):
            manager.copy_batch_block_offsets(short, request_ids, 1, 0, len(request_ids))
    finally:
        manager.shutdown()
        reference.shutdown()


def test_the_fp8_plane_keeps_the_host_page_table_at_production_widths() -> None:
    """At a DEP16 context server's batch and block-table width (a 1M-token pool, so max_seq_len
    bounds the width), the FP8 plane adds one attention-op plane and leaves the host page table
    the all-NVFP4 cache's."""
    import gc

    shapes = {}
    for arm, fp8_kv_layers in (("all_nvfp4", ()), ("fp8_plane", None)):
        manager = _make_production_manager(
            max_batch_size=32,
            max_seq_len=997696,
            max_tokens=1 << 20,
            fp8_kv_layers=fp8_kv_layers,
            **_PROD_DRAFT_KWARGS,
        )
        try:
            shapes[arm] = (
                manager.num_pools,
                manager.num_attention_op_pools,
                tuple(manager.host_kv_cache_block_offsets.shape),
            )
        finally:
            manager.shutdown()
            manager = None
            gc.collect()
    num_pools, op_pools, host_shape = shapes["fp8_plane"]
    assert op_pools == num_pools + 1
    assert shapes["all_nvfp4"] == (num_pools, num_pools, host_shape)


def test_fp8_kv_disagg_page_table() -> None:
    # The transfer views of the production geometry with FP8 layers 91 and 92.
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
    from tensorrt_llm._torch.disaggregation.resource.utils import get_pool_view_global_layer_ids

    manager = _make_production_manager(fp8_kv_layers=None)
    try:
        page_table = build_page_table_from_manager(manager)
        group = page_table.layer_groups[manager._attention_pool_id]
        views = {}
        for view in group.pool_views:
            views.setdefault("/".join(sorted(view.pool_role)), []).append(
                (
                    int(view.bytes_per_layer),
                    view.mapper_kind,
                    [int(layer) for layer in get_pool_view_global_layer_ids(view, group)],
                )
            )
        tpb = _PROD_TOKENS_PER_BLOCK
        nvfp4_layers = [layer for layer in _PROD_DSA_LAYERS if layer not in _PROD_FP8_LAYERS]
        assert sorted(views) == ["index_key", "key", "key_block_scale"], views.keys()
        # Two latent views: 22 NVFP4 pages of 64 x 288 B and 2 FP8 pages of 64 x 576 B per slot.
        assert sorted(views["key"], key=lambda view: view[0]) == [
            (tpb * _PROD_HEAD_DIM // 2, MapperKind.INDEXED, nvfp4_layers),
            (tpb * _PROD_HEAD_DIM, MapperKind.INDEXED, list(_PROD_FP8_LAYERS)),
        ]
        assert views["index_key"] == [
            (tpb * _PROD_INDEXER_BYTES, MapperKind.REPLICATED, list(_PROD_DSA_LAYERS))
        ]
        assert [(size, layers) for size, _, layers in views["key_block_scale"]] == [
            (tpb * _PROD_HEAD_DIM // 16, nvfp4_layers)
        ]
    finally:
        manager.shutdown()


def _dsa_group_pools(manager: RubinKdaDsaHybridCacheManager) -> dict:
    """Every physical pool of the DSA layer group as [slots, pages per slot, page bytes]."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
    from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

    def pool(local_layer_idx: int, role) -> torch.Tensor:
        converter = manager.impl.get_page_index_converter(local_layer_idx, role)
        scale = int(converter.scale)
        num_pages = int(manager.impl.get_page_index_upper_bound(local_layer_idx, role))
        num_pages = num_pages // int(converter.expansion) + int(converter.layer_offset)
        assert num_pages % scale == 0
        stride = int(manager.impl.get_page_stride(local_layer_idx, role))
        base = int(
            manager.impl.get_mem_pool_base_address(local_layer_idx, role, PageIndexMode.PER_LAYER)
        )
        return convert_to_torch_tensor(
            TensorWrapper(base, torch.uint8, [num_pages // scale, scale, stride])
        )

    first_nvfp4 = manager._nvfp4_latent_local_layers[0]
    pools = {
        "nvfp4_latent": pool(first_nvfp4, Role.KEY),
        "nvfp4_block_scale": pool(first_nvfp4, Role.KEY_BLOCK_SCALE),
        "index_key": pool(manager._attention_local_layers[0], Role.INDEX_KEY),
    }
    if manager._fp8_latent_local_layers:
        pools["fp8_latent"] = pool(manager._fp8_latent_local_layers[0], Role.KEY)
    return pools


def _dsa_slots(manager: RubinKdaDsaHybridCacheManager, request_id: int) -> torch.Tensor:
    kv_cache = manager.kv_cache_map[request_id]
    slots = kv_cache.get_base_page_indices(manager._attention_pool_id)[: kv_cache.num_blocks]
    return torch.tensor([int(slot) for slot in slots], dtype=torch.long, device="cuda")


@pytest.mark.parametrize("value", [None, ()], ids=["fp8_91_92", "none"])
def test_fp8_kv_pages_round_trip_through_the_host_tier(value) -> None:
    # A suspended request's pages move to the host tier when another request takes their GPU
    # slots, and come back, mostly into other slots, when it resumes. Every page of every pool of
    # the DSA layer group (the NVFP4 latent pages and block scales, the FP8 latent pages of layers
    # 91 and 92, the indexer K) must come back byte for byte.
    manager = _make_production_manager(
        max_tokens=2048,
        max_seq_len=1024,
        max_batch_size=4,
        host_cache_size=256 << 20,
        fp8_kv_layers=value,
    )
    stream = torch.cuda.current_stream().cuda_stream
    pressure = None
    try:
        from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

        pools = _dsa_group_pools(manager)
        expected_pools = ["index_key", "nvfp4_block_scale", "nvfp4_latent"]
        if value is None:
            expected_pools.append("fp8_latent")
            assert tuple(pools["fp8_latent"].shape[1:]) == (2, 64 * _PROD_HEAD_DIM)
        assert sorted(pools) == sorted(expected_pools)
        request_ids = [0, 1]
        manager.add_dummy_requests(request_ids, token_nums=[300, 700])
        generator = torch.Generator(device="cuda").manual_seed(3)
        written = {}
        for request_id in request_ids:
            slots = _dsa_slots(manager, request_id)
            contents = {}
            for name, pool in pools.items():
                contents[name] = torch.randint(
                    0,
                    256,
                    (len(slots), *pool.shape[1:]),
                    dtype=torch.uint8,
                    device="cuda",
                    generator=generator,
                )
                pool[slots] = contents[name]
            written[request_id] = (slots, contents)
        torch.cuda.synchronize()

        caches = [manager.kv_cache_map[request_id] for request_id in request_ids]
        for cache in caches:
            cache.suspend()
        # Another request takes every GPU slot of the DSA pool group that it can get, which needs
        # the suspended requests' slots: their pages go to the host tier (level 1).
        pressure = manager.impl.create_kv_cache()
        assert pressure.resume(stream)
        num_slots = manager._num_attention_slots
        for blocks in range(num_slots, 0, -1):
            if pressure.resize(blocks * manager.tokens_per_block):
                break
        torch.cuda.synchronize()
        for request_id, cache in zip(request_ids, caches):
            counts, _ = _introspection.active_page_stats(cache)
            assert len(counts) > 1 and counts[1] >= len(written[request_id][0]), (
                f"request {request_id}: pages by cache level {counts}, pressure {blocks} of "
                f"{num_slots} DSA slots"
            )
        pressure.close()
        pressure = None

        for cache in caches:
            assert cache.resume(stream)
        torch.cuda.synchronize()
        for request_id, cache in zip(request_ids, caches):
            counts, _ = _introspection.active_page_stats(cache)
            assert counts[0] == sum(counts), counts
            old_slots, contents = written[request_id]
            slots = _dsa_slots(manager, request_id)
            assert len(slots) == len(old_slots)
            for name, pool in pools.items():
                assert torch.equal(pool[slots], contents[name]), (request_id, name)
    finally:
        if pressure is not None:
            pressure.close()
        manager.shutdown()


def _pool_group_desc(manager, layer_group: int):
    """The pool group descriptor that holds ``layer_group``."""
    (desc,) = [
        pool_group
        for pool_group in manager.impl.pool_group_descs
        if any(
            int(variant.layer_group_id) == layer_group for variant in pool_group.slot_desc.variants
        )
    ]
    return desc


@pytest.mark.parametrize(
    "indexer_k_dtype", ["fp4", "fp8"], ids=["fp4_index_keys", "fp8_index_keys"]
)
def test_fp8_latent_cache_logs_its_pools(indexer_k_dtype) -> None:
    """The production geometry with the production DSpark drafter. The FP8 latent cache that serves
    logs the GPU slots of each pool group on one line: a slot of the DSA pool group holds the FP8
    latents (24 x 576 B) and index keys (24 x 68 B, or 24 x 132 B with FP8 index keys) of 64
    tokens, one of the drafter's pool group five 576 B latents per token."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
        rubin_kda_dsa_cache_manager as dsa_cache_manager,
    )

    with patch.object(dsa_cache_manager.logger, "info") as info:
        manager = _make_production_manager(DataType.FP8, indexer_k_dtype, **_PROD_DRAFT_KWARGS)
    try:
        lines = [
            call.args[0]
            for call in info.call_args_list
            if str(call.args[0]).startswith("Kimi K3.1 KV cache pools")
        ]
        assert manager.dtype == DataType.FP8 and not manager._fp8_latent_local_layers
        (line,) = lines
        segments = re.findall(
            r"pool_group_id=(\d+), num_slots=(\d+), slot_size=\[([\d, ]*)\], bytes=(\d+)", line
        )
        logged_groups = {
            int(group): (int(slots), [int(size) for size in sizes.split(", ")], int(total))
            for group, slots, sizes, total in segments
        }
        assert sorted(logged_groups) == list(range(len(segments)))
        for slots, sizes, total in logged_groups.values():
            assert total == slots * sum(sizes)
        tpb = _PROD_TOKENS_PER_BLOCK
        index_bytes = {"fp4": _PROD_INDEXER_BYTES, "fp8": 132}[indexer_k_dtype]
        draft_group = manager.layer_to_pool_mapping_dict[
            manager.layer_offsets[manager.draft_layer_ids[0]]
        ]
        for layer_group, sizes in (
            (manager._attention_pool_id, [tpb * 24 * _PROD_HEAD_DIM, tpb * 24 * index_bytes]),
            (draft_group, [tpb * 5 * _PROD_HEAD_DIM]),
        ):
            desc = _pool_group_desc(manager, layer_group)
            assert logged_groups[int(desc.pool_group_index)][:2] == (int(desc.num_slots), sizes)
    finally:
        manager.shutdown()


#: (state, conv window) bytes of a slot of ``_make_production_manager``'s KDA pool group: 69 KDA
#: layers, each with a [4, 8, 8] recurrent state and a [96, 3] BF16 conv window.
_PROD_KDA_SLOT_BYTES = {
    torch.float32: (69 * 4 * 8 * 8 * 4, 69 * 96 * 3 * 2),
    torch.bfloat16: (69 * 4 * 8 * 8 * 2, 69 * 96 * 3 * 2),
}


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_kda_state_slot_follows_the_state_dtype(state_dtype) -> None:
    """``kv_cache_config.mamba_ssm_cache_dtype``, with the production DSA layer layout and drafter
    (the FP8 KV cache, FP8 index keys, the production DSpark drafter) and tiny KDA states
    ([4, 8, 8], conv [96, 3]). A slot of the KDA pool group holds each local KDA layer's recurrent
    state at that dtype and its BF16 conv window: an FP32 state takes twice the bytes of a BF16
    one, the conv windows the same. The state views have that dtype, and the FP8 pool line logs the
    KDA pool group's slot."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
        rubin_kda_dsa_cache_manager as dsa_cache_manager,
    )

    with patch.object(dsa_cache_manager.logger, "info") as info:
        manager = _make_production_manager(
            DataType.FP8, "fp8", mamba_ssm_cache_dtype=state_dtype, **_PROD_DRAFT_KWARGS
        )
    try:
        assert manager.ssm_state_dtype == state_dtype
        assert {state.dtype for state in manager.all_ssm_states} == {state_dtype}
        assert {conv.dtype for conv in manager.all_conv_states} == {torch.bfloat16}
        num_kda_layers = manager.local_num_mamba_layers
        heads, head_dim, d_state = manager.ssm_state_shape
        state_bytes = num_kda_layers * heads * head_dim * d_state * state_dtype.itemsize
        conv_bytes = num_kda_layers * manager.conv_bytes
        assert (state_bytes, conv_bytes) == _PROD_KDA_SLOT_BYTES[state_dtype]
        assert num_kda_layers * manager.ssm_bytes == state_bytes
        # A pool group orders its pools by slot bytes; compare them as a set.
        sizes = _slot_sizes_of_group(manager, manager.ssm_layer_group_id)
        assert sorted(sizes) == sorted([state_bytes, conv_bytes])
        (line,) = [
            call.args[0]
            for call in info.call_args_list
            if str(call.args[0]).startswith("Kimi K3.1 KV cache pools")
        ]
        logged = {
            int(group): [int(size) for size in group_sizes.split(", ")]
            for group, _, group_sizes, _ in re.findall(
                r"pool_group_id=(\d+), num_slots=(\d+), slot_size=\[([\d, ]*)\], bytes=(\d+)", line
            )
        }
        desc = _pool_group_desc(manager, manager.ssm_layer_group_id)
        assert logged[int(desc.pool_group_index)] == sizes
        # The FP32 state names itself at startup, with the same state bytes; a BF16 one does not.
        state_lines = [
            call.args[0]
            for call in info.call_args_list
            if str(call.args[0]).startswith("Kimi K3.1 KDA recurrent state")
        ]
        if state_dtype == torch.float32:
            assert state_lines == [
                f"Kimi K3.1 KDA recurrent state: float32, {num_kda_layers} local KDA layers of "
                f"{[heads, head_dim, d_state]}, {state_bytes} B of state per slot"
            ]
        else:
            assert state_lines == []
    finally:
        manager.shutdown()
