# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The QSA hybrid cache manager's V2 page layout, on real managers.

The attention op addresses a layer's K/V page as ``pool + (scale * slot + 2 *
offset) * page`` and its NVFP4 block-scale page with the same block index in
the scale pool. These tests build the managers a Qwen3.8-Flash-Next engine
builds -- the 48-layer trunk (12 QSA layers among 36 recurrent ones) and the
one-layer MTP draft manager -- and check that every view the manager serves is
where that formula lands, for bf16, fp8 and NVFP4 pools.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.qsa import QSAMambaHybridCacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.qsa.cache_manager import (
    QSA_INDEX_POSITION,
    QSA_SIDE_PLANES,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, QSASparseAttentionConfig
from tensorrt_llm.mapping import Mapping

_CHECKPOINT = SimpleNamespace(
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)
_TRUNK_LAYERS = 48
_QSA_LAYERS = list(range(3, _TRUNK_LAYERS, 4))
_DRAFT_LAYER = _TRUNK_LAYERS
_KV_HEADS, _HEAD_DIM, _TPB = 2, 256, 32
_INDEX_PAGE = 1 * 128 * 2 * _TPB
_POSITION_PAGE = 3 * 4 * _TPB


def _manager(
    dtype: DataType, *, draft: bool, tp_size: int = 1, attention_dp: bool = False
) -> QSAMambaHybridCacheManagerV2:
    """The trunk or the separate MTP draft manager, as the engine builds them (rank 0 of
    `tp_size`, with attention data parallelism when `attention_dp`)."""
    if draft:
        attention_mask = [False] * _TRUNK_LAYERS + [True]
        mamba_mask = [False] * (_TRUNK_LAYERS + 1)
    else:
        attention_mask = [layer in _QSA_LAYERS for layer in range(_TRUNK_LAYERS)]
        mamba_mask = [not attention for attention in attention_mask]
    return QSAMambaHybridCacheManagerV2(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=sum(mamba_mask),
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=4096, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=sum(attention_mask),
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TPB,
        max_seq_len=1024,
        max_batch_size=4,
        mapping=Mapping(
            world_size=tp_size,
            rank=0,
            tp_size=tp_size,
            pp_size=1,
            enable_attention_dp=attention_dp,
        ),
        dtype=dtype,
        layer_mask=attention_mask,
        is_draft=draft,
        vocab_size=1024,
        conv_state_layout="q_k_v",
        sparse_attention_config=QSASparseAttentionConfig(),
        pretrained_config=_CHECKPOINT,
    )


def _layers(draft: bool) -> list[int]:
    return [_DRAFT_LAYER] if draft else _QSA_LAYERS


def _kv_page_bytes(dtype: DataType) -> int:
    element_bits = {DataType.BF16: 16, DataType.FP8: 8, DataType.NVFP4: 4}[dtype]
    return _KV_HEADS * _TPB * _HEAD_DIM * element_bits // 8


def _thop_addresses(cm, layer: int, slot: int, page_bytes: int, plane: int) -> tuple[int, int]:
    """Byte addresses of this layer's K and V page for `slot`, as the attention
    op forms them from the manager's pool pointers, mapping and page-index
    scale (`plane` 0 = data, 1 = NVFP4 block scales)."""
    local = cm.layer_offsets[layer]
    pool, offset = cm.kv_cache_pool_mapping[local].tolist()
    pointers = cm.kv_cache_pool_pointers
    base = int(pointers[pool, 0, plane]) if pointers.dim() == 3 else int(pointers[pool, 0])
    key_entry = int(cm.index_scales[pool]) * slot
    value_entry = key_entry + int(cm.kv_offset[pool])
    shift = offset * cm.kv_factor * page_bytes
    return base + key_entry * page_bytes + shift, base + value_entry * page_bytes + shift


def _slots_to_check(view: torch.Tensor) -> list[int]:
    return [0, 1, view.shape[0] // 2, view.shape[0] - 1]


@pytest.fixture(scope="module", autouse=True)
def _require_cuda():
    assert torch.cuda.is_available(), "the managers allocate their pools on the GPU"


@pytest.mark.parametrize("draft", [False, True], ids=["trunk", "draft"])
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
def test_views_are_where_the_attention_op_addresses(dtype: DataType, draft: bool) -> None:
    """Every K/V (and NVFP4 block-scale) view of every QSA layer starts, at
    every checked slot, at the address the attention op computes for it."""
    cm = _manager(dtype, draft=draft)
    try:
        page = _kv_page_bytes(dtype)
        for layer in _layers(draft):
            kv = cm.get_buffers(layer, kv_layout="HND")
            assert kv.shape[1:] == (
                2,
                _KV_HEADS,
                _TPB,
                _HEAD_DIM // (2 if dtype == DataType.NVFP4 else 1),
            )
            for slot in _slots_to_check(kv):
                key, value = _thop_addresses(cm, layer, slot, page, plane=0)
                assert kv[slot, 0].data_ptr() == key, (layer, slot)
                assert kv[slot, 1].data_ptr() == value, (layer, slot)
            if dtype != DataType.NVFP4:
                continue
            scales = cm.get_block_scale_buffers(layer, kv_layout="HND")
            assert scales.dtype == torch.uint8
            assert scales.shape == (kv.shape[0], 2, _KV_HEADS, _TPB, _HEAD_DIM // 16)
            for slot in _slots_to_check(scales):
                key, value = _thop_addresses(cm, layer, slot, page // 8, plane=1)
                assert scales[slot, 0].data_ptr() == key, (layer, slot)
                assert scales[slot, 1].data_ptr() == value, (layer, slot)
    finally:
        cm.shutdown()


@pytest.mark.parametrize("draft", [False, True], ids=["trunk", "draft"])
@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
def test_side_planes_pack_when_a_page_could_collide(dtype: DataType, draft: bool) -> None:
    """An index page could be a K/V page here -- FP8 at one KV head a rank (1 x 256 x 32
    tokens), NVFP4 at both -- so the index and position planes are packed into a buffer
    of their own instead of landing in the K/V pool, and under NVFP4 the K/V and
    block-scale pools hold the same number of pages per slot."""
    cm = _manager(dtype, draft=draft)
    try:
        layers = _layers(draft)
        assert cm.qsa_side_planes_size == len(layers) * _INDEX_PAGE + _POSITION_PAGE
        for ordinal, layer in enumerate(layers):
            local = cm.layer_offsets[layer]
            data_scale = cm.impl.get_page_index_scale(local, Role.KEY)
            if dtype == DataType.NVFP4:
                assert data_scale == cm.impl.get_page_index_scale(local, Role.KEY_BLOCK_SCALE)
            assert data_scale == 2 * len(layers)
            assert cm.kv_cache_pool_mapping[local, 1] == ordinal
        planes_owner = cm.layer_offsets[layers[0]]
        assert cm.impl.get_page_index_scale(planes_owner, QSA_SIDE_PLANES) == 1
    finally:
        cm.shutdown()


@pytest.mark.parametrize("draft", [False, True], ids=["trunk", "draft"])
def test_bf16_keeps_per_layer_side_roles(draft: bool) -> None:
    """A bf16 K/V page (at least 1 x 256 x 2 bytes x 32 tokens) can never be an index or
    position page, so the side planes stay the per-layer INDEX_KEY role plus one position
    role, served by the same views the generic accessor builds."""
    cm = _manager(DataType.BF16, draft=draft)
    try:
        layers = _layers(draft)
        assert cm.qsa_side_planes_size is None
        for layer in layers:
            local = cm.layer_offsets[layer]
            assert cm.impl.get_page_index_scale(local, Role.INDEX_KEY) == len(layers)
            assert cm.impl.get_page_index_scale(local, Role.KEY) == 2 * len(layers)
            generic = MambaHybridCacheManagerV2.get_index_k_buffer(
                cm, layer, num_heads=1, head_dim=128, dtype=torch.bfloat16
            )
            served = cm.get_index_k_buffer(layer)
            assert (served.data_ptr(), served.shape, served.stride()) == (
                generic.data_ptr(),
                generic.shape,
                generic.stride(),
            )
        owner = cm.layer_offsets[layers[0]]
        assert cm.impl.get_page_index_scale(owner, QSA_INDEX_POSITION) == 1
        assert cm.get_qsa_position_buffer().stride(0) == _TPB * 3
    finally:
        cm.shutdown()


@pytest.mark.parametrize("draft", [False, True], ids=["trunk", "draft"])
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
def test_side_and_main_views_do_not_alias(dtype: DataType, draft: bool) -> None:
    """One slot written through every view reads back unchanged through each:
    no K/V, scale, index or position page overlaps another."""
    cm = _manager(dtype, draft=draft)
    try:
        views = []
        for layer in _layers(draft):
            kv = cm.get_buffers(layer, kv_layout="HND")
            views += [kv[:, 0], kv[:, 1]]
            if dtype == DataType.NVFP4:
                scales = cm.get_block_scale_buffers(layer, kv_layout="HND")
                views += [scales[:, 0], scales[:, 1]]
            views.append(cm.get_index_k_buffer(layer))
        views.append(cm.get_qsa_position_buffer())
        slots = {view.shape[0] for view in views}
        assert len(slots) == 1, f"views disagree on the slot count: {sorted(slots)}"
        slot = slots.pop() - 1
        as_bytes = [view[slot].view(torch.uint8) for view in views]
        for i, raw in enumerate(as_bytes):
            raw.fill_(i + 1)
        torch.cuda.synchronize()
        for i, raw in enumerate(as_bytes):
            assert bool((raw == i + 1).all()), f"view {i} was overwritten by another view"
    finally:
        cm.shutdown()


def _logical_page(view: torch.Tensor) -> tuple:
    """What a peer copies of one side view per slot: the page's shape, dtype and bytes, and
    whether they are contiguous. Where the page sits in its pool (the slot stride, other
    buffers coalesced beside it) is physical and may differ between layouts."""
    page = view[0]
    return tuple(page.shape), page.dtype, page.numel() * page.element_size(), page.is_contiguous()


#: (tp_size, attention_dp): the layouts a disaggregated pair can mix over one global
#: geometry (2 KV heads): TP1 and ADP2 hold both heads per rank, TP2 one.
_LAYOUTS = {"tp1": (1, False), "tp2": (2, False), "adp2": (2, True)}


@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
def test_transfer_views_do_not_depend_on_the_kv_head_split(dtype: DataType) -> None:
    """Disaggregated peers compare the replicated side roles each layer registers and copy
    their pages byte for byte, so roles, their tensors and every side view's page (shape,
    dtype, bytes) must not depend on how many KV heads a rank holds; where pages sit in
    their pools may. At TP2 an FP8 K/V page (1 head x 256 x 32 tokens) is the size of an
    index page; under NVFP4 the same holds at TP1 and ADP2 (2 heads x 128 bytes x 32
    tokens). Both pack on every layout; bf16, which can never collide, on none."""
    seen = {}
    for name, (tp_size, attention_dp) in _LAYOUTS.items():
        cm = _manager(dtype, draft=False, tp_size=tp_size, attention_dp=attention_dp)
        try:
            buffers = cm._extra_buffers_per_layer(tokens_per_block=_TPB)
            roles = {
                layer: sorted(
                    (str(b.role), b.size) for b in buffers.get(cm.layer_offsets[layer], [])
                )
                for layer in _QSA_LAYERS
            }
            views = {layer: cm.get_index_k_buffer(layer) for layer in _QSA_LAYERS}
            views["position"] = cm.get_qsa_position_buffer()
            logical = {key: _logical_page(view) for key, view in views.items()}
            # How each side buffer spreads over ranks, as the page table declares it.
            tensors = {
                (layer, role): cm._page_buffer_tensors(
                    cm.layer_offsets[layer], role, size, _TPB, False
                )
                for layer, layer_roles in roles.items()
                for role, size in layer_roles
            }
            seen[name] = (roles, logical, tensors)
        finally:
            cm.shutdown()
    assert seen["tp1"] == seen["tp2"] == seen["adp2"]
    roles = {role for layer in seen["tp1"][0].values() for role, _ in layer}
    if dtype == DataType.BF16:
        assert roles == {str(Role.INDEX_KEY), str(QSA_INDEX_POSITION)}
    else:
        assert roles == {str(QSA_SIDE_PLANES)}
