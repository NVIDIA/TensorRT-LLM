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
"""A standalone DSpark drafter's layers as explicit draft layer specs of the MiniMax-M3 manager."""

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.dflash_attention import DFLASH_FOLDED_DECODE_LOWERING
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import QuantMode

NUM_TARGET = 4  # M3 convention without a checkpoint: layers 0..2 dense, 3 sparse
NUM_DRAFT = 2
DRAFT_WINDOW = 256
HEADS = 2
HEAD_DIM = 128


def _specs(draft_window: Optional[int] = DRAFT_WINDOW) -> tuple[KVCacheLayerSpec, ...]:
    """Target layers in the NVFP4 cache dtype, then a BF16 drafter (windowed by default)."""
    scale_roles = (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
    target = [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.NVFP4,
            num_kv_heads=HEADS,
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE, *scale_roles),
        )
        for layer in range(NUM_TARGET)
    ]
    draft = [
        KVCacheLayerSpec(
            global_layer_id=NUM_TARGET + layer,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=DataType.BF16,
            num_kv_heads=HEADS,
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=draft_window,
        )
        for layer in range(NUM_DRAFT)
    ]
    return tuple(target + draft)


def _manager(
    layer_specs: Optional[tuple[KVCacheLayerSpec, ...]],
    dtype: DataType = DataType.NVFP4,
    implementation: str = "msa",
    is_disagg: bool = False,
    max_gpu_total_bytes: int = 64 << 20,
    max_seq_len: int = 1024,
    max_batch_size: int = 2,
    enable_block_reuse: bool = False,
) -> MiniMaxM3KVCacheManagerV2:
    """A MiniMax-M3 manager (hybrid NVFP4 MSA by default), with or without draft layer specs."""
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=max_gpu_total_bytes,
            host_cache_size=0,
            enable_block_reuse=enable_block_reuse,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation=implementation,
            indexer_kv_dtype="fp8" if implementation == "msa" else "bf16",
        ),
        layer_specs=layer_specs,
        is_disagg=is_disagg,
        enable_stats=enable_block_reuse,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_draft_layer_specs_join_the_hybrid_nvfp4_manager() -> None:
    """Draft layers join the hybrid manager without changing the target's pages."""
    torch.cuda.init()
    plain = _manager(None)
    unified = _manager(_specs())
    try:
        assert unified.target_layer_ids == tuple(range(NUM_TARGET))
        assert unified.draft_layer_ids == tuple(range(NUM_TARGET, NUM_TARGET + NUM_DRAFT))
        # Dense target layers keep the FP8 half of the hybrid layout; draft layers their own dtype.
        dtypes = [unified.get_layer_dtype(layer) for layer in unified.pp_layers]
        assert dtypes == [DataType.FP8] * 3 + [DataType.NVFP4] + [DataType.BF16] * NUM_DRAFT
        for layer in range(NUM_TARGET):
            for role in (Role.KEY, Role.VALUE):
                assert unified.impl.get_page_stride(
                    unified.layer_offsets[layer], role
                ) == plain.impl.get_page_stride(plain.layer_offsets[layer], role)
        assert unified.get_index_k_buffer(3).shape[1:] == plain.get_index_k_buffer(3).shape[1:]
        # Index-K is the manager's own buffer, not a layer-spec role.
        index_k_bytes = [
            unified.get_layer_bytes_per_token(unified.layer_offsets[layer], Role.INDEX_KEY)
            for layer in unified.pp_layers
        ]
        assert index_k_bytes == [0, 0, 0, 128] + [0] * NUM_DRAFT
        # The cache-created event divides each tier quota by a total that includes index-K.
        assert unified.get_cache_bytes_per_token() == sum(
            spec.bytes_per_token() for spec in unified.layer_specs
        ) + sum(index_k_bytes)

        # Draft layers: their own lifecycle, the base view in their dtype.
        groups = unified.layer_to_pool_mapping_dict
        draft_groups = {groups[unified.layer_offsets[layer]] for layer in unified.draft_layer_ids}
        target_groups = {groups[unified.layer_offsets[layer]] for layer in unified.target_layer_ids}
        assert len(draft_groups) == 1 and not draft_groups & target_groups
        view = unified.get_buffers(NUM_TARGET, kv_layout="HND")
        assert view.dtype == torch.bfloat16
        assert tuple(view.shape[1:]) == (2, HEADS, 128, HEAD_DIM)
        assert unified.get_draft_kv_cache_view() is unified
        assert plain.get_draft_kv_cache_view() is None

        # The unified drafter reads its layers' block offsets, so they keep attention-op pools.
        assert plain.num_attention_op_pools == 0
        assert unified.num_attention_op_pools > 0
        # CUDA-graph padding passes the draft view (this manager) back in.
        (request,) = unified.add_dummy_requests(
            [7], token_nums=[200], draft_kv_cache_manager=unified.get_draft_kv_cache_view()
        )
        # Target layers get raw slot ids, draft layers rows of their base view.
        for layer in unified.pp_layers:
            raw = unified.get_batch_cache_indices([7], layer_idx=layer, raw_indices=True)[0]
            indices = unified.get_batch_cache_indices([7], layer_idx=layer)[0]
            if layer in unified.draft_layer_ids:
                scale = unified.get_layer_page_index_scale(layer)
                expected = [s * scale // 2 if s >= 0 else s for s in raw]
            else:
                expected = raw
            assert indices[: len(raw)] == expected, layer
        offsets = torch.zeros(
            unified.num_attention_op_pools,
            1,
            2,
            unified.max_blocks_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
        unified.copy_batch_block_offsets(offsets, [7], 1, 1, 1)
        torch.cuda.synchronize()
        pool = int(unified.kv_cache_pool_mapping[unified.layer_offsets[NUM_TARGET], 0])
        slots = [
            s
            for s in unified.get_batch_cache_indices([7], layer_idx=NUM_TARGET, raw_indices=True)[0]
            if s >= 0
        ]
        assert len(slots) == 2
        # The row the drafter reads holds the draft layer's K page of each slot, V right after it.
        scale = unified.get_layer_page_index_scale(NUM_TARGET)
        pages = [[slot * scale + plane for slot in slots] for plane in (0, 1)]
        assert offsets[pool, 0, :, : len(slots)].tolist() == pages
        # Draft-layer indices address the base view's contiguous K/V rows, like the flat table.
        rows = [page // 2 for page in pages[0]]
        assert unified.get_batch_cache_indices([7], layer_idx=NUM_TARGET)[0][: len(slots)] == rows
        flat = unified.get_batch_cache_indices_flat([7], [len(slots)], layer_idx=NUM_TARGET)
        assert flat.tolist() == rows
        # Target layers keep MiniMax's raw slot ids.
        sparse_slots = unified.get_batch_cache_indices([7], layer_idx=3)[0]
        assert (
            sparse_slots == unified.get_batch_cache_indices([7], layer_idx=3, raw_indices=True)[0]
        )
        unified.free_resources(request)

        # The draft lifecycle is sized by its window, not by its share of the bytes per token.
        target_ratio, draft_ratio = unified.kv_cache_manager_py_config.initial_pool_ratio
        assert target_ratio + draft_ratio == pytest.approx(1.0)
        assert (
            0
            < draft_ratio
            < 2 * HEADS * HEAD_DIM * 2 * NUM_DRAFT / (unified.get_cache_bytes_per_token())
        )
    finally:
        unified.shutdown()
        plain.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_only_the_draft_layers_block_offset_rows_are_built() -> None:
    """Target layers address pages by slot id, so each step rebuilds only the draft layers' rows."""
    torch.cuda.init()
    unified = _manager(_specs())
    try:
        (request,) = unified.add_dummy_requests([7], token_nums=[200])
        sentinel = -7
        offsets = torch.full(
            (unified.num_attention_op_pools, 1, 2, unified.max_blocks_per_seq),
            sentinel,
            dtype=torch.int32,
            device="cuda",
        )
        unified.copy_batch_block_offsets(offsets, [7], 1, 1, 1)
        torch.cuda.synchronize()
        pool_of = {
            layer: int(unified.kv_cache_pool_mapping[unified.layer_offsets[layer], 0])
            for layer in unified.pp_layers
        }
        draft_rows = {pool_of[layer] for layer in unified.draft_layer_ids}
        target_rows = {pool_of[layer] for layer in unified.target_layer_ids} - draft_rows
        assert target_rows, "the target layers need attention-op rows of their own"
        for row in sorted(target_rows):
            assert (offsets[row] == sentinel).all(), row
        for row in sorted(draft_rows):
            assert (offsets[row, 0, :, :2] >= 0).all(), row
        unified.free_resources(request)
    finally:
        unified.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_draft_layers_keep_the_dense_target_subpage_factor() -> None:
    """Explicit draft layers do not change the dense target layers' sub-page factor."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (  # noqa: E501
        uniform_subpages_per_slot,
    )

    torch.cuda.init()
    plain = None
    unified = None
    try:
        plain = _manager(None)
        unified = _manager(_specs())
        assert uniform_subpages_per_slot(unified) == uniform_subpages_per_slot(plain) > 0
    finally:
        for manager in (unified, plain):
            if manager is not None:
                manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_warmup_scrub_clears_every_slot() -> None:
    """The post-warmup NaN/Inf scrub zeroes every slot of target and draft layers alike."""
    torch.cuda.init()
    manager = _manager(_specs())
    try:
        for layer in manager.pp_layers:
            view = manager.get_buffers(layer)
            if view.dtype.is_floating_point:
                view.fill_(float("nan"))
        assert manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
        for layer in manager.pp_layers:
            view = manager.get_buffers(layer)
            if view.dtype.is_floating_point:
                assert not view.float().isnan().any(), layer
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_nvfp4_draft_layers_are_rejected() -> None:
    """Block scales belong to the sparse target layers, so an NVFP4 drafter cannot join."""
    specs = _specs()
    nvfp4_draft = replace(
        specs[NUM_TARGET],
        dtype=DataType.NVFP4,
        buffer_roles=(Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE),
    )
    with pytest.raises(ValueError, match="does not host NVFP4 draft layers"):
        _manager((*specs[:NUM_TARGET], nvfp4_draft, *specs[NUM_TARGET + 1 :]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize(
    "sparse_attention_freq,num_specs,expected",
    [
        (None, NUM_TARGET + NUM_DRAFT, [3]),
        ([0, 1, 0, 1], NUM_TARGET + NUM_DRAFT, [1, 3]),
        ([0, 1, 0, 1], NUM_TARGET, [1, 3]),
    ],
    ids=["default", "config", "config-target-only"],
)
def test_specs_decide_the_sparse_target_layers(
    sparse_attention_freq: Optional[list[int]], num_specs: int, expected: list[int]
) -> None:
    """A per-layer head list spanning the drafter keeps the target's sparse split."""
    torch.cuda.init()
    pretrained_config = None
    if sparse_attention_freq is not None:
        pretrained_config = SimpleNamespace(
            num_hidden_layers=NUM_TARGET,
            sparse_attention_config={"sparse_attention_freq": sparse_attention_freq},
        )
    manager = MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=[HEADS] * (NUM_TARGET + NUM_DRAFT),
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
        layer_specs=_specs()[:num_specs],
        pretrained_config=pretrained_config,
    )
    try:
        assert manager.sparse_layer_ids == expected
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_draft_layers_need_their_own_lifecycle() -> None:
    """Unwindowed draft layers would share the target's lifecycle and pool offsets."""
    torch.cuda.init()
    with pytest.raises(ValueError, match="only in their own lifecycle"):
        _manager(_specs(draft_window=None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_disagg_draft_layers_need_the_hnd_layout() -> None:
    """A Triton FP8 cache transfers token-major K/V, but drafters write head-major K/V."""
    specs = tuple(
        replace(spec, dtype=DataType.FP8, buffer_roles=(Role.KEY, Role.VALUE))
        if spec.owner is KVCacheLayerOwner.TARGET
        else spec
        for spec in _specs()
    )
    with pytest.raises(ValueError, match="only with the HND K/V layout"):
        _manager(specs, dtype=DataType.FP8, implementation="triton", is_disagg=True)


class _HostingKVCacheManagerV2(KVCacheManagerV2):
    """A plain V2 manager that sizes its hosted draft lifecycle as a hosting manager does."""

    hosts_unified_draft_layers = True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_window_sized_draft_pool_admits_a_full_context_chunk() -> None:
    """Allocation rounds the window-sized draft share, but one full chunk still fits."""
    torch.cuda.init()
    max_num_tokens = 8320
    specs = tuple(
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=owner,
            dtype=DataType.BF16,
            num_kv_heads=heads,
            head_dim=HEAD_DIM,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=window,
        )
        for layer, owner, heads, window in (
            (0, KVCacheLayerOwner.TARGET, 2 * HEADS, None),
            (1, KVCacheLayerOwner.DRAFT, HEADS, DRAFT_WINDOW),
        )
    )
    manager = _HostingKVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=1 << 30,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=[2 * HEADS, HEADS],
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=16384,
        max_batch_size=1,
        max_num_tokens=max_num_tokens,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        vocab_size=16,
        layer_specs=specs,
    )
    try:
        assert manager.kv_cache_manager_py_config.initial_pool_ratio is not None
        # Different page sizes keep the draft lifecycle in its own physical pool.
        assert int(manager.kv_cache_pool_mapping[0, 0]) != int(manager.kv_cache_pool_mapping[1, 0])
        requests = manager.add_dummy_requests([7], token_nums=[max_num_tokens])
        assert requests is not None
        manager.free_resources(requests[0])
    finally:
        manager.shutdown()


def _prefill_once(manager: MiniMaxM3KVCacheManagerV2, request: LlmRequest) -> int:
    """Claim, allocate and run one request's whole context like the scheduler; return its reuse."""
    assert manager.prepare_context(request)
    reused_blocks = int(
        manager.kv_cache_map[request.py_request_id].commit_pending_stats().reused_blocks
    )
    assert manager.resize_context(request, request.context_remaining_length)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    return reused_blocks


def _finish_context(manager: MiniMaxM3KVCacheManagerV2, request: LlmRequest) -> None:
    """Advance the request past its context and commit its blocks for reuse."""
    batch = ScheduledRequests()
    batch.append_context_request(request)
    request.context_chunk_size = request.context_remaining_length
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_a_prefix_hit_hands_back_the_draft_layers_pages() -> None:
    """A prefix hit reuses the draft lifecycle's live-window pages with their contents.

    The second prompt shares the first one's first five pages and then diverges, after the first
    request ran three more pages past them. The target claims the same prefix as a target-only
    manager, and every draft page the drafter's window reads at the claimed end is the first
    request's page, still holding what the first request stored.
    """
    torch.cuda.init()
    shared = list(range(5000, 5000 + 5 * 128))
    first_prompt = shared + list(range(20000, 20000 + 3 * 128 + 17))
    second_prompt = shared + list(range(30000, 30000 + 3 * 128 + 17))

    def request(request_id: int, tokens: list[int]) -> LlmRequest:
        return LlmRequest(
            request_id=request_id,
            max_new_tokens=4,
            input_tokens=tokens,
            sampling_config=SamplingConfig(1),
            is_streaming=False,
            return_perf_metrics=True,
        )

    claims = {}
    for arm, specs in (("target_only", _specs()[:NUM_TARGET]), ("unified", _specs())):
        manager = _manager(specs, enable_block_reuse=True, max_seq_len=4096)
        try:
            first = request(1, first_prompt)
            _prefill_once(manager, first)
            stored = {}
            for layer in manager.draft_layer_ids:
                view = manager.get_buffers(layer, kv_layout="HND")
                rows = manager.get_batch_cache_indices([1], layer_idx=layer)[0]
                for ordinal, row in enumerate(rows):
                    if row >= 0:
                        view[row].fill_(10 + ordinal)
                stored[layer] = rows
            torch.cuda.synchronize()
            _finish_context(manager, first)
            manager.free_resources(first)

            second = request(2, second_prompt)
            reused_blocks = _prefill_once(manager, second)
            claims[arm] = (second.context_current_position, reused_blocks)
            claimed_pages = second.context_current_position // 128
            for layer in manager.draft_layer_ids:
                view = manager.get_buffers(layer, kv_layout="HND")
                rows = manager.get_batch_cache_indices([2], layer_idx=layer)[0]
                window_pages = range(claimed_pages - DRAFT_WINDOW // 128, claimed_pages)
                for ordinal in window_pages:
                    assert rows[ordinal] == stored[layer][ordinal], (layer, ordinal)
                    assert float(view[rows[ordinal]].float().mean()) == 10 + ordinal
            _finish_context(manager, second)
            manager.free_resources(second)
        finally:
            manager.shutdown()
    assert claims["unified"][0] == claims["target_only"][0] == len(shared)
    # Per-request reused blocks sum over attention lifecycles: the draft lifecycle adds the pages
    # it holds at the claimed end (sink-free window), so hosting the drafter raises the count.
    target_pages = claims["target_only"][1]
    assert claims["unified"][1] == target_pages + DRAFT_WINDOW // 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_unified_target_capacity_matches_a_separate_draft_pool() -> None:
    """Hosting the drafter costs the target only the draft lifecycle's bytes.

    The separate-pool arm is the target manager alone on the quota a draft pool of the same size
    leaves. A bytes-per-token split would give the BF16 drafter about half of the quota instead.
    """
    torch.cuda.init()
    quota, max_seq_len, max_batch_size = 256 << 20, 1 << 18, 8
    sizes = dict(max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    unified = _manager(_specs(), max_gpu_total_bytes=quota, **sizes)
    try:
        _, draft_ratio = unified.kv_cache_manager_py_config.initial_pool_ratio
        draft_bytes = int(draft_ratio * quota)
        unified_tokens = unified.get_num_available_tokens(token_num_upper_bound=max_seq_len)
        # Every sequence holds a window and one context chunk fits beside them.
        requests = unified.add_dummy_requests(
            list(range(max_batch_size)),
            token_nums=[DRAFT_WINDOW] * (max_batch_size - 1) + [unified.max_num_tokens],
        )
        assert requests is not None and len(requests) == max_batch_size
        for request in requests:
            unified.free_resources(request)
    finally:
        unified.shutdown()
    capacities = []
    for target_quota in (quota, quota - draft_bytes):
        target_only = _manager(_specs()[:NUM_TARGET], max_gpu_total_bytes=target_quota, **sizes)
        try:
            capacities.append(
                target_only.get_num_available_tokens(token_num_upper_bound=max_seq_len)
            )
        finally:
            target_only.shutdown()
    whole_quota_tokens, separate_pool_tokens = capacities
    assert whole_quota_tokens < max_seq_len  # The quota, not max_seq_len, bounds the capacity.
    # Within the allocation grain of the draft lifecycle's pools at this small quota.
    assert unified_tokens >= 0.98 * separate_pool_tokens
    assert unified_tokens >= 0.95 * whole_quota_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_max_tokens_quota_pays_for_the_index_k_pages() -> None:
    """The profiling pass's max_tokens buys the warmup's longest request with hosted draft specs.

    Specs carry K/V and scales only; the sparse layer's index-K shares its pages, so a quota that
    left it out would cut the longest CUDA-graph warmup request short of max_seq_len.
    """
    torch.cuda.init()
    tokens_per_block, max_seq_len, max_batch_size = 128, 1 << 16, 2
    # _get_token_num_for_estimation: one max_seq_len request plus one block per other request.
    blocks = -(-(max_seq_len + 1) // tokens_per_block) + max_batch_size - 1
    manager = MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=blocks * tokens_per_block,
            max_gpu_total_bytes=1 << 30,
            avg_seq_len=max_seq_len,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
        layer_specs=_specs(),
        is_disagg=True,
    )
    try:
        assert manager.sparse_layer_ids  # Index-K pages exist.
        assert (
            manager.get_num_available_tokens(
                token_num_upper_bound=max_seq_len, batch_size=max_batch_size
            )
            == max_seq_len
        )
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_only_a_hosting_manager_sizes_the_draft_lifecycle() -> None:
    """A plain V2 manager with a windowed draft lifecycle keeps the backend's pool split."""
    torch.cuda.init()
    specs = tuple(
        replace(spec, dtype=DataType.BF16, buffer_roles=(Role.KEY, Role.VALUE)) for spec in _specs()
    )
    plain = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=NUM_TARGET,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        vocab_size=16,
        layer_specs=specs,
    )
    try:
        assert plain.kv_cache_manager_py_config.initial_pool_ratio is None
    finally:
        plain.shutdown()
    unified = _manager(_specs())
    try:
        assert unified.kv_cache_manager_py_config.initial_pool_ratio is not None
    finally:
        unified.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize("pp_size", [1, 2])
def test_window_sized_draft_ratio(pp_size: int) -> None:
    """A windowed draft lifecycle gets its windows plus one chunk, below a bytes-per-token share."""

    def _configs(specs: tuple[KVCacheLayerSpec, ...], target_sinks: Optional[int] = None) -> list:
        return [
            SimpleNamespace(
                sliding_window_size=spec.sliding_window_size,
                num_sink_tokens=target_sinks if spec.owner is KVCacheLayerOwner.TARGET else None,
                reuse_policy=spec.reuse_policy,
                buffers=[SimpleNamespace(size=spec.bytes_per_token() * 128)],
            )
            for spec in specs
        ]

    manager = object.__new__(KVCacheManagerV2)
    manager.layer_specs = _specs()[NUM_TARGET - 1 :]
    manager.max_batch_size = 2
    manager.max_num_tokens = 256
    manager.mapping = Mapping(world_size=pp_size, rank=0, pp_size=pp_size)
    manager.is_disagg = False
    manager._disable_overlap_scheduler = False
    manager._generation_kv_capacity_headroom = 1
    manager.num_extra_kv_tokens = 0
    manager.gen_preemption_headroom = 0
    configs = _configs(manager.layer_specs)
    draft_bytes_per_token = 2 * HEADS * HEAD_DIM * 2 * NUM_DRAFT
    token_share = draft_bytes_per_token / sum(
        spec.bytes_per_token() for spec in manager.layer_specs
    )
    quota = 1 << 30
    target_ratio, draft_ratio = manager._windowed_draft_pool_ratio(configs, quota, 128)
    assert target_ratio + draft_ratio == pytest.approx(1.0)
    # Every PP stage's batch holds a window, and one context chunk fits beside them.
    windows_and_chunk = (manager.max_batch_size * pp_size * DRAFT_WINDOW + 256) * (
        draft_bytes_per_token
    )
    assert windows_and_chunk <= draft_ratio * quota < token_share * quota
    # Requests paused in place by GEN slot preemption keep their windows.
    manager.gen_preemption_headroom = 2
    paused_ratio = manager._windowed_draft_pool_ratio(configs, quota, 128)[1]
    paused = windows_and_chunk + 2 * DRAFT_WINDOW * draft_bytes_per_token
    assert paused <= paused_ratio * quota < token_share * quota
    assert paused_ratio > draft_ratio
    manager.gen_preemption_headroom = 0
    # Without the overlap scheduler a single-stage batch's retiring request frees its KV at once,
    # so half as many sequences hold a window.
    manager._disable_overlap_scheduler = True
    no_overlap_ratio = manager._windowed_draft_pool_ratio(configs, quota, 128)[1]
    assert (no_overlap_ratio < draft_ratio) if pp_size == 1 else (no_overlap_ratio == draft_ratio)
    manager._disable_overlap_scheduler = False
    # Resume refuses a pool above its watermark, so a lower watermark needs a larger share.
    assert manager._windowed_draft_pool_ratio(configs, quota, 128, 0.5)[1] > draft_ratio
    # A quota too small for every window keeps the bytes-per-token split.
    manager.max_batch_size = 64
    assert manager._windowed_draft_pool_ratio(configs, 8 << 20, 128)[1] == pytest.approx(
        token_share
    )
    manager.max_batch_size = 2
    # A quota without room for one context chunk per lifecycle keeps the backend's split.
    assert manager._windowed_draft_pool_ratio(configs, 1 << 20, 128) is None
    # Without a context-chunk bound the backend keeps its own split.
    manager.max_num_tokens = None
    assert manager._windowed_draft_pool_ratio(configs, quota, 128) is None
    manager.max_num_tokens = 256
    # Without a windowed draft lifecycle the backend keeps its own split.
    manager.layer_specs = _specs(draft_window=None)[NUM_TARGET - 1 :]
    assert manager._windowed_draft_pool_ratio(_configs(manager.layer_specs), quota, 128) is None
    # A target sharing the drafter's window shares its lifecycle too: sink tokens that round to
    # no sink block do not make a second one.
    manager.layer_specs = tuple(
        replace(spec, sliding_window_size=DRAFT_WINDOW) for spec in _specs()[NUM_TARGET - 1 :]
    )
    assert manager._windowed_draft_pool_ratio(_configs(manager.layer_specs, 0), quota, 128) is None


def _dspark_creator(*, is_disagg: bool, context_kernel: bool = False) -> KvCacheCreator:
    """A creator for an NVFP4 MiniMax-M3 target with a BF16 standalone DSpark drafter."""

    def model_config(kv_quant_algo: Optional[QuantAlgo] = None) -> SimpleNamespace:
        return SimpleNamespace(
            pretrained_config=SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=HEAD_DIM,
                hidden_size=16 * HEAD_DIM,
                torch_dtype=torch.bfloat16,
            ),
            quant_config=SimpleNamespace(
                quant_mode=QuantMode.from_quant_algo(kv_cache_quant_algo=kv_quant_algo)
            ),
        )

    target, draft = model_config(QuantAlgo.NVFP4), model_config()
    # DSpark's checkpoint declares its sliding window.
    draft.pretrained_config.sliding_window = 4096
    # A drafter that cannot report its lowerings is assumed to need the context kernel.
    drafter = SimpleNamespace(dflash_attention_backend="TRTLLM")
    if not context_kernel:
        drafter.validate_block_attention_windows = (
            lambda: ["causal decode + future-block merge"] * 2
        )
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: False, is_dspark=lambda: True),
        attention_backend="AUTO",
        draft_is_embedded_in_target=False,
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
    )
    creator._draft_config = draft
    creator._mapping = SimpleNamespace(
        pp_size=1, cp_size=1, tp_size=2, enable_attention_dp=False, pp_layers=range
    )
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=target,
            config=target.pretrained_config,
            draft_model=drafter,
        ),
    )
    creator._kv_cache_manager_cls = MiniMaxM3KVCacheManagerV2
    creator._sparse_attention_config = SimpleNamespace(implementation="msa")
    creator._kv_cache_config = SimpleNamespace(max_attention_window=None)
    creator._tokens_per_block = 128
    creator._unification_max_seq_len = 8192
    creator._max_seq_len = 8192
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    creator._get_draft_max_attention_window = lambda *_: [4096]
    return creator


def _gate_creator(
    *,
    is_disagg: bool,
    tokens_per_block: int = 128,
    context_kernel: bool = False,
    nvfp4_draft: bool = False,
    draft_window: Optional[int] = 4096,
    pool_ratio: Optional[list[float]] = None,
    implementation: str = "msa",
    tp_size: int = 2,
    draft_kv_heads: int = 4,
    plain_v2: bool = False,
    nvfp4_target: bool = True,
    speculative: bool = True,
    pp_size: int = 1,
    full_attention_drafter: bool = False,
    omit_draft_window_metadata: bool = False,
    folded_decode: bool = False,
) -> KvCacheCreator:
    """``_dspark_creator`` with one setting changed, for the unify-gate table."""
    creator = _dspark_creator(is_disagg=is_disagg, context_kernel=context_kernel)
    if folded_decode:
        # A full non-causal draft layer lowers to the folded decode.
        creator._model_engine.model.draft_model.validate_block_attention_windows = lambda: [
            "causal decode + future-block merge",
            DFLASH_FOLDED_DECODE_LOWERING,
        ]
    creator._tokens_per_block = tokens_per_block
    draft = creator._get_draft_kv_model_config()
    if nvfp4_draft:
        draft.quant_config = SimpleNamespace(
            quant_mode=QuantMode.from_quant_algo(kv_cache_quant_algo=QuantAlgo.NVFP4)
        )
    draft.pretrained_config.num_key_value_heads = draft_kv_heads
    if full_attention_drafter:
        draft.pretrained_config.use_sliding_window = False
        draft_window = None
    if omit_draft_window_metadata:
        # A checkpoint without window fields is full attention too.
        del draft.pretrained_config.sliding_window
        draft_window = None
    window = creator._unification_max_seq_len if draft_window is None else draft_window
    creator._get_draft_max_attention_window = lambda *_: [window]
    if pool_ratio is not None:
        creator._pool_ratio_in = pool_ratio
    creator._sparse_attention_config = SimpleNamespace(implementation=implementation)
    creator._mapping.tp_size = tp_size
    creator._mapping.pp_size = pp_size
    if plain_v2:
        creator._kv_cache_manager_cls = KVCacheManagerV2
    if not nvfp4_target:
        creator._model_engine.model.model_config.quant_config = SimpleNamespace(
            quant_mode=QuantMode.from_quant_algo()
        )
    if not speculative:
        creator._speculative_config = None
    return creator


_BOTH = (False, True)
# A disaggregated worker raises at startup rather than decide apart from its peer.
_RAISES = "raises"
# (id, workers, TRTLLM_DISAGG_SHARED_DRAFT_KV, creator settings, unifies, blocker substring or None)
_GATE_CASES = [
    ("hosting-manager", _BOTH, "1", {}, True, None),
    # Unset, the switch shares a hosting manager's draft KV, whose separate pool is not
    # transferred; a plain V2 manager stays opt-in.
    ("unset-env", _BOTH, None, {}, True, None),
    ("opt-out", (True,), "0", {}, False, None),
    ("context-kernel-64", _BOTH, "1", dict(tokens_per_block=64, context_kernel=True), True, None),
    ("context-kernel-128", _BOTH, "1", dict(context_kernel=True), False, "128-token pages"),
    ("page-96", _BOTH, "1", dict(tokens_per_block=96), False, "96-token pages"),
    ("page-256", _BOTH, "1", dict(tokens_per_block=256), True, None),
    # The folded decode keeps the fixed page sizes: it is not validated on dynamic pages.
    ("folded-decode-64", _BOTH, "1", dict(tokens_per_block=64, folded_decode=True), True, None),
    ("folded-decode-128", _BOTH, "1", dict(folded_decode=True), False, "128-token pages"),
    ("nvfp4-draft", _BOTH, "1", dict(nvfp4_draft=True), False, "NVFP4 draft layers"),
    # Settings each disaggregated worker sets on its own: aggregated serving and an explicit
    # opt-out fall back to the separate draft KV cache; otherwise the worker refuses to start.
    (
        "shared-lifecycle",
        (False,),
        "1",
        dict(draft_window=None),
        False,
        "share a KV cache lifecycle",
    ),
    (
        "shared-lifecycle",
        (True,),
        "1",
        dict(draft_window=None),
        _RAISES,
        "share a KV cache lifecycle",
    ),
    ("shared-lifecycle-unset-env", (True,), None, dict(draft_window=None), _RAISES, "lifecycle"),
    ("shared-lifecycle-opt-out", (True,), "0", dict(draft_window=None), False, "lifecycle"),
    ("pool-ratio-arity", (False,), "1", dict(pool_ratio=[1.0]), False, "pool_ratio has 1 entries"),
    ("pool-ratio-arity", (True,), "1", dict(pool_ratio=[1.0]), _RAISES, "pool_ratio has 1 entries"),
    ("pool-ratio-arity-unset-env", (True,), None, dict(pool_ratio=[1.0]), _RAISES, "pool_ratio"),
    ("pool-ratio-arity-opt-out", (True,), "0", dict(pool_ratio=[1.0]), False, "pool_ratio"),
    ("pp2", (False,), None, dict(pp_size=2), False, "pipeline_parallel_size is 2"),
    ("pp2", (True,), None, dict(pp_size=2), _RAISES, "pipeline_parallel_size is 2"),
    ("pp2-opt-out", (True,), "0", dict(pp_size=2), False, "pipeline_parallel_size is 2"),
    # A drafter whose checkpoint is full attention falls back on every worker alike.
    (
        "full-attention-drafter",
        _BOTH,
        None,
        dict(full_attention_drafter=True),
        False,
        "drafter uses full attention",
    ),
    (
        "full-attention-drafter-without-window-fields",
        _BOTH,
        None,
        dict(omit_draft_window_metadata=True),
        False,
        "drafter uses full attention",
    ),
    # A setting every worker shares decides first, so a worker-local one alongside it falls back.
    ("pp2-nvfp4-draft", (True,), None, dict(pp_size=2, nvfp4_draft=True), False, "NVFP4 draft"),
    ("pool-ratio-per-lifecycle", _BOTH, "1", dict(pool_ratio=[0.9, 0.1]), True, None),
    ("triton-nhd", (False,), "1", dict(implementation="triton"), True, None),
    ("triton-nhd", (True,), "1", dict(implementation="triton"), False, "HND K/V layout"),
    # TP 4 sees one head per rank on both sides, TP 2 does not: disagg peers with different TP
    # would choose different layouts.
    ("unequal-global-heads", (False,), "1", dict(tp_size=4, draft_kv_heads=2), True, None),
    (
        "unequal-global-heads",
        (True,),
        "1",
        dict(tp_size=4, draft_kv_heads=2),
        False,
        "2 KV heads, the target 4",
    ),
    ("no-speculation", _BOTH, "1", dict(speculative=False), False, "not enabled"),
    (
        "plain-v2-64",
        _BOTH,
        "1",
        dict(plain_v2=True, tokens_per_block=64, nvfp4_target=False),
        True,
        None,
    ),
    (
        "plain-v2-unset-env",
        (False,),
        None,
        dict(plain_v2=True, tokens_per_block=64, nvfp4_target=False),
        True,
        None,
    ),
    (
        "plain-v2-unset-env",
        (True,),
        None,
        dict(plain_v2=True, tokens_per_block=64, nvfp4_target=False),
        False,
        None,
    ),
    (
        "plain-v2-128",
        _BOTH,
        "1",
        dict(plain_v2=True, nvfp4_target=False),
        False,
        "tokens_per_block is 128",
    ),
    # An NVFP4 target pairs with a BF16 draft in transfer only when the draft has pools of its own.
    (
        "plain-v2-nvfp4-target",
        (True,),
        "1",
        dict(plain_v2=True, tokens_per_block=64),
        False,
        "k_only",
    ),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize(
    ("draft_window", "full_attention_drafter"),
    [(256, False), (1024, False), (None, True)],
    ids=["windowed", "window-at-max-seq-len", "full-attention"],
)
def test_the_gate_refuses_what_the_manager_refuses(
    monkeypatch: pytest.MonkeyPatch, draft_window: Optional[int], full_attention_drafter: bool
) -> None:
    """Aggregated serving shares draft KV only in layouts the manager builds, else falls back."""
    torch.cuda.init()
    monkeypatch.delenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", raising=False)
    max_seq_len = 1024
    creator = _gate_creator(
        is_disagg=False, draft_window=draft_window, full_attention_drafter=full_attention_drafter
    )
    creator._unification_max_seq_len = creator._max_seq_len = max_seq_len
    unifies = creator._should_unify_dflash_kv_cache()
    # Specs carry a window reaching max_seq_len as full attention.
    spec_window = None if draft_window is None or draft_window >= max_seq_len else draft_window
    try:
        _manager(_specs(spec_window), max_seq_len=max_seq_len).shutdown()
        builds = True
    except ValueError as error:
        assert "own lifecycle" in str(error)
        builds = False
    assert unifies is builds


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("is_disagg", "env", "settings", "unifies", "blocker"),
    [
        pytest.param(
            is_disagg,
            env,
            settings,
            unifies,
            blocker,
            id=f"{name}-{'disagg' if is_disagg else 'agg'}",
        )
        for name, workers, env, settings, unifies, blocker in _GATE_CASES
        for is_disagg in workers
    ],
)
def test_unify_gate(
    monkeypatch: pytest.MonkeyPatch,
    is_disagg: bool,
    env: Optional[str],
    settings: dict,
    unifies: bool,
    blocker: Optional[str],
) -> None:
    """MiniMax-M3's manager hosts a DSpark drafter unless a layout it or transfer refuses applies.

    Disaggregated workers share a hosting manager's draft KV unless opted out, and a plain V2
    manager's only on request. Layouts refused for a reason every worker shares keep the separate
    draft KV cache; a disaggregated worker whose own settings are refused fails at startup.
    """
    if env is None:
        monkeypatch.delenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", env)
    creator = _gate_creator(is_disagg=is_disagg, **settings)
    reason = creator._dflash_unify_blocker()
    if blocker is None:
        assert reason is None
    else:
        assert reason is not None and blocker in reason
    if is_disagg and env == "1" and blocker is not None and creator._speculative_config is not None:
        # Required sharing that a blocker refuses stops the worker at startup.
        with pytest.raises(ValueError, match=re.escape(blocker)):
            creator._check_shared_draft_kv_request()
    if unifies == _RAISES:
        match = re.escape(blocker) + r".*TRTLLM_DISAGG_SHARED_DRAFT_KV=0 on every worker"
        with pytest.raises(ValueError, match=match):
            creator._should_unify_dflash_kv_cache()
    else:
        assert creator._should_unify_dflash_kv_cache() is unifies
