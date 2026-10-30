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

from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping

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
) -> MiniMaxM3KVCacheManagerV2:
    """A MiniMax-M3 manager (hybrid NVFP4 MSA by default), with or without draft layer specs."""
    return MiniMaxM3KVCacheManagerV2(
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
        dtype=dtype,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation=implementation,
            indexer_kv_dtype="fp8" if implementation == "msa" else "bf16",
        ),
        layer_specs=layer_specs,
        is_disagg=is_disagg,
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
def test_page_indices_follow_the_layer_owner() -> None:
    """Target layers get raw slot ids and draft layers base view rows."""
    torch.cuda.init()
    manager = _manager(_specs())
    try:
        (request,) = manager.add_dummy_requests([7], token_nums=[200])
        for layer in manager.pp_layers:
            raw = manager.get_batch_cache_indices([7], layer_idx=layer, raw_indices=True)[0]
            indices = manager.get_batch_cache_indices([7], layer_idx=layer)[0]
            if layer in manager.draft_layer_ids:
                scale = manager.get_layer_page_index_scale(layer)
                expected = [s * scale // 2 if s >= 0 else s for s in raw]
            else:
                expected = raw
            assert indices[: len(raw)] == expected, layer
        manager.free_resources(request)
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
def test_target_specs_need_the_cache_dtype() -> None:
    """Target views and block scales follow the manager dtype, so target specs must use it."""
    specs = _specs()
    fp8_target = replace(specs[3], dtype=DataType.FP8, buffer_roles=(Role.KEY, Role.VALUE))
    with pytest.raises(ValueError, match="target layer specs in the cache dtype"):
        _manager((*specs[:3], fp8_target, *specs[NUM_TARGET:]))


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
    manager = KVCacheManagerV2(
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("is_disagg", [False, True])
def test_window_sized_draft_ratio(is_disagg: bool, pp_size: int) -> None:
    """A windowed draft lifecycle gets its windows plus one chunk, not a bytes-per-token share."""

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
    manager.is_disagg = is_disagg
    manager._generation_kv_capacity_headroom = 1
    manager.num_extra_kv_tokens = 0
    configs = _configs(manager.layer_specs)
    draft_bytes_per_token = 2 * HEADS * HEAD_DIM * 2 * NUM_DRAFT
    # Admissible sequences (batch x PP stages, doubled for the overlap scheduler's retiring
    # request unless an aggregated PP pipeline) hold three pages of window each, plus one chunk.
    num_sequences = 2 * pp_size * (2 if is_disagg or pp_size == 1 else 1)
    draft_bytes = (num_sequences * 3 * 128 + 256) * draft_bytes_per_token
    assert manager._windowed_draft_pool_ratio(configs, 1 << 30, 128) == pytest.approx(
        [1 - draft_bytes / (1 << 30), draft_bytes / (1 << 30)]
    )
    # Resume refuses a pool above its watermark, so the share covers the windows below it.
    assert manager._windowed_draft_pool_ratio(configs, 1 << 30, 128, 0.5)[1] == pytest.approx(
        2 * draft_bytes / (1 << 30)
    )
    # A quota too small for the windows keeps the bytes-per-token split.
    token_share = draft_bytes_per_token / sum(
        spec.bytes_per_token() for spec in manager.layer_specs
    )
    assert manager._windowed_draft_pool_ratio(configs, 1 << 20, 128)[1] == pytest.approx(
        token_share
    )
    # The quota must hold the context-chunk floor: two blocks per lifecycle, over the resume
    # watermark; otherwise the backend keeps its own split.
    floor_bytes = 3 * sum(spec.bytes_per_token() * 128 for spec in manager.layer_specs)
    assert manager._windowed_draft_pool_ratio(configs, floor_bytes - 1, 128, 0.95) is None
    assert manager._windowed_draft_pool_ratio(configs, floor_bytes, 128, 0.95) is not None
    # Without a context-chunk bound the backend keeps its own split.
    manager.max_num_tokens = None
    assert manager._windowed_draft_pool_ratio(configs, 1 << 30, 128) is None
    manager.max_num_tokens = 256
    # Without a windowed draft lifecycle the backend keeps its own split.
    manager.layer_specs = _specs(draft_window=None)[NUM_TARGET - 1 :]
    assert manager._windowed_draft_pool_ratio(_configs(manager.layer_specs), 1 << 30, 128) is None
    # A target sharing the drafter's window shares its lifecycle too: sink tokens that round to
    # no sink block do not make a second one.
    manager.layer_specs = tuple(
        replace(spec, sliding_window_size=DRAFT_WINDOW) for spec in _specs()[NUM_TARGET - 1 :]
    )
    assert (
        manager._windowed_draft_pool_ratio(_configs(manager.layer_specs, 0), 1 << 30, 128) is None
    )
