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
"""MiniMax-M3 shared Eagle3 draft layouts and real heterogeneous page tables.

Draft views use an attention-op pool rooted at the draft K page inside the slot.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.utils import get_kv_page_offset
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    cache_manager as m3_cache_manager,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3DraftKVCacheView,
    MiniMaxM3KVCacheManagerV2,
    derive_shared_draft_layout,
    extend_attention_op_pools_for_shared_draft_layers,
    shared_draft_layer_count,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
    uniform_subpages_per_slot,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

DRAFT_LOCAL_LAYER = 60
SCALE = 179  # sub-pages per M3 mega-slot: 3 dense x 2 + 57 sparse x 3 + draft x 2
DRAFT_K_ADDR = 0x7000_0000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
def test_nvfp4_shared_draft_constructs_and_converts_heterogeneous_page_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real pool allocation and both target/draft page-table consumers."""
    torch.cuda.init()
    manager = MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=4,
        num_kv_heads=[2, 2, 2, 2, 1],
        head_dim=128,
        tokens_per_block=128,
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=256,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        vocab_size=16,
        spec_config=Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused"),
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa", indexer_kv_dtype="fp8"
        ),
    )
    try:
        assert manager._use_per_layer_page_tables
        assert manager._shared_draft_layer_ids == [4]
        # The shared Eagle layer keeps native P128 pages.
        assert manager.impl.get_page_index_converter(4, Role.KEY).expansion == 1
        # Fill two logical request rows; no context scratch is active.
        for row, slot in enumerate((2, 5)):
            manager.host_kv_cache_block_offsets[:, row].fill_(slot)
        output = torch.empty(
            (manager.num_attention_op_pools, 2, 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        manager._copy_batch_block_offsets_per_layer(
            output, [101, 102], torch.tensor([0, 1], dtype=torch.long), 0, 2
        )
        actual = output.cpu()
        for layer_id in range(manager.num_local_layers):
            for role_idx, role in enumerate((Role.KEY, Role.VALUE)):
                converter = manager.impl.get_page_index_converter(layer_id, role)
                for row, slot in enumerate((2, 5)):
                    expected = converter(
                        [slot] * manager.max_blocks_per_seq, PageIndexMode.PER_LAYER
                    )
                    assert actual[layer_id, row, role_idx].tolist() == expected

        # The source pool's scale and K/V offset describe its representative
        # layer; poison them so only the view's own values produce the pages below.
        manager.index_scales.zero_()
        manager.kv_offset.zero_()

        # The draft view is an FP8/P128 cache rooted at the draft layer's K page.
        draft = manager.get_draft_kv_cache_view()
        assert isinstance(draft, MiniMaxM3DraftKVCacheView)
        assert draft.dtype == DataType.FP8
        assert draft.trtllm_gen_extra_tokens_per_block == frozenset({128})
        assert draft.kv_cache_pool_pointers.tolist() == [
            [manager.impl.get_mem_pool_base_address(4, Role.KEY, PageIndexMode.SHARED), 0]
        ]
        assert draft.kv_cache_pool_mapping[4].tolist() == [0, 0]
        source_pool = manager.impl.get_layer_group_id(4)
        assert (
            draft.host_kv_cache_block_offsets.data_ptr()
            == manager.host_kv_cache_block_offsets[source_pool].data_ptr()
        )

        # One device copy converts the draft layer's slot ids to P128 pages.
        copy_idx = torch.tensor([0, 1], dtype=torch.int32, pin_memory=True)
        monkeypatch.setattr(
            manager,
            "index_mapper",
            SimpleNamespace(get_copy_index=lambda ids, num_contexts, beam_width: copy_idx),
        )
        device_copy = Mock(wraps=m3_cache_manager.copy_batch_block_offsets_to_device)
        monkeypatch.setattr(m3_cache_manager, "copy_batch_block_offsets_to_device", device_copy)
        draft_output = torch.full(
            (1, 2, 2, draft.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda"
        )
        draft.copy_batch_block_offsets(draft_output, [101, 102], 1, 0, 2)
        torch.cuda.synchronize()
        device_copy.assert_called_once()
        converter = manager.impl.get_page_index_converter(4, Role.KEY)
        for row, slot in enumerate((2, 5)):
            expected_k = converter([slot] * draft.max_blocks_per_seq, PageIndexMode.SHARED)
            assert draft_output[0, row, 0].tolist() == expected_k
            assert draft_output[0, row, 1].tolist() == [page + 1 for page in expected_k]

        # FMHA must use the view's extent and K/V displacement rather than
        # rejecting the view or delegating to its differently rooted owner.
        attn = TrtllmAttention.__new__(TrtllmAttention)
        attn.local_layer_idx = 4
        fmha = FlashInferTrtllmGenFmha.__new__(FlashInferTrtllmGenFmha)
        fmha._attn_ref = lambda: attn
        metadata = TrtllmAttentionMetadata(
            max_num_requests=2, max_num_tokens=256, kv_cache_manager=draft, mapping=manager.mapping
        )
        assert fmha._get_total_num_blocks(metadata) == manager.get_kv_subpage_pool(4)[0].shape[0]
        assert get_kv_page_offset(attn, metadata, 0) == 1
        with pytest.raises(ValueError, match="only addresses its shared draft layer"):
            draft.get_attention_op_num_blocks(0)

        # Dense decode stages one sub-page factor for the dense target layers; the
        # NVFP4 layer and the view-read draft layer may sit in other pools.
        assert uniform_subpages_per_slot(manager) == manager.get_kv_subpage_pool(0)[1]
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
def test_block_scale_role_uses_global_layer_ids(dtype: DataType) -> None:
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = dtype
    manager.pp_layers = [60, 7, 2]
    manager.sparse_layer_ids = {7}
    for local_layer, expected_nvfp4 in enumerate((None, Role.KEY_BLOCK_SCALE, None)):
        expected = expected_nvfp4 if dtype == DataType.NVFP4 else None
        assert manager._get_block_scale_role(Role.KEY, local_layer) == expected
        assert manager._get_block_scale_role(Role.VALUE, local_layer) is None
    generic = KVCacheManagerV2.__new__(KVCacheManagerV2)
    generic.dtype = dtype
    expected = Role.KEY_BLOCK_SCALE if dtype == DataType.NVFP4 else None
    assert generic._get_block_scale_role(Role.KEY, 0) == expected


def test_virtual_pool_is_rooted_at_the_draft_k_page():
    pool_pointers = torch.tensor([[0x6000_0000, 0]], dtype=torch.int64)
    pool_mapping = torch.tensor([[0, i] for i in range(61)], dtype=torch.int32)

    pointers, mapping, index_scales, kv_offsets, op_pools = (
        extend_attention_op_pools_for_shared_draft_layers(
            pool_pointers, pool_mapping, 1, [(DRAFT_LOCAL_LAYER, DRAFT_K_ADDR, SCALE)]
        )
    )

    assert pointers.tolist() == [[0x6000_0000, 0], [DRAFT_K_ADDR, 0]]
    # The draft layer moves to the new pool at offset 0; target rows are untouched.
    assert mapping[DRAFT_LOCAL_LAYER].tolist() == [1, 0]
    assert mapping[:DRAFT_LOCAL_LAYER].tolist() == pool_mapping[:DRAFT_LOCAL_LAYER].tolist()
    # Slot s -> page s * SCALE for K and s * SCALE + 1 for V.
    assert index_scales.tolist() == [SCALE]
    assert kv_offsets.tolist() == [1]
    assert op_pools == [(1, 0)]


def test_block_offset_copy_fills_the_virtual_pool_from_the_source_pool(monkeypatch):
    """The base copy fills the storage pools; the override then fills the virtual pool
    from the source pool's slot ids, and does nothing extra without draft layers.
    """
    calls = []

    def fake_base_copy(
        self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None
    ):
        calls.append(("base", request_ids, num_seqs, max_blocks))

    def fake_device_copy(host, dst, copy_idx, index_scales, kv_offsets, stream):
        calls.append(
            ("virtual", host, dst, copy_idx, index_scales.tolist(), kv_offsets.tolist(), stream)
        )

    monkeypatch.setattr(KVCacheManagerV2, "copy_batch_block_offsets", fake_base_copy)
    monkeypatch.setattr(m3_cache_manager, "copy_batch_block_offsets_to_device", fake_device_copy)

    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.FP8
    manager._draft_op_pools = ((1, 0),)
    manager._draft_index_scales = torch.tensor([SCALE], dtype=torch.int32)
    manager._draft_kv_offsets = torch.tensor([1], dtype=torch.int32)
    manager.host_kv_cache_block_offsets = torch.zeros((1, 4, 2, 8), dtype=torch.int32)
    copy_idx = torch.tensor([2, 0], dtype=torch.int32)
    manager.index_mapper = SimpleNamespace(get_copy_index=lambda ids, nc, bw: copy_idx)
    manager._stream = SimpleNamespace(cuda_stream=1234)
    dst = torch.zeros((2, 4, 2, 8), dtype=torch.int32)

    manager.copy_batch_block_offsets(dst, [7, 9], 1, 0, 2, max_blocks=5)

    assert calls[0] == ("base", [7, 9], 2, 5)
    kind, host, dst_slice, idx, scales, offsets, stream = calls[1]
    assert kind == "virtual"
    assert (
        host.shape == (1, 4, 2, 8)
        and host.data_ptr() == manager.host_kv_cache_block_offsets.data_ptr()
    )
    assert dst_slice.shape == (1, 4, 2, 8) and dst_slice.data_ptr() == dst[1].data_ptr()
    assert idx is copy_idx
    assert scales == [SCALE] and offsets == [1] and stream == 1234

    # Non-speculative MiniMax-M3: the base copy is all that runs.
    calls.clear()
    plain = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    plain.copy_batch_block_offsets(dst, [7], 1, 0, 1)
    assert calls == [("base", [7], 1, None)]


@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("swa_scratch_reuse", [False, True])
def test_per_layer_page_tables_get_no_virtual_pools(monkeypatch, dtype, swa_scratch_reuse):
    """With per-layer page tables every layer already has its own pool."""
    pointers = torch.tensor([[0x6000_0000 + i, 0] for i in range(61)])
    mapping = torch.tensor([[i, 0] for i in range(61)], dtype=torch.int32)

    def fake_base_prepare(self, index_mapper_capacity):
        self._use_per_layer_page_tables = True
        self.kv_cache_pool_pointers = pointers.clone()
        self.kv_cache_pool_mapping = mapping.clone()
        self.num_attention_op_pools = 61

    monkeypatch.setattr(KVCacheManagerV2, "_prepare_page_table_tensor", fake_base_prepare)
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = dtype
    manager._shared_draft_layer_ids = [DRAFT_LOCAL_LAYER]
    manager.layer_offsets = {i: i for i in range(61)}
    manager.is_draft = False
    manager.enable_swa_scratch_reuse = swa_scratch_reuse
    manager.tokens_per_block = 128

    if swa_scratch_reuse:
        with pytest.raises(NotImplementedError, match="SWA scratch reuse"):
            manager._prepare_page_table_tensor(8)
        return

    manager._prepare_page_table_tensor(8)

    assert torch.equal(manager.kv_cache_pool_pointers, pointers)
    assert torch.equal(manager.kv_cache_pool_mapping, mapping)
    assert manager.num_attention_op_pools == 61
    assert manager._draft_op_pools == ()
    expected_extra_pages = {128} if dtype == DataType.FP8 else set()
    assert manager.trtllm_gen_extra_tokens_per_block == frozenset(expected_extra_pages)


def test_update_resources_refuses_tree_relocation(monkeypatch):
    """Linear acceptance rewinds through the base; tree acceptance is refused."""
    calls = []
    monkeypatch.setattr(KVCacheManagerV2, "update_resources", lambda self, *a: calls.append(a))
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.FP8
    linear = SimpleNamespace(
        py_num_accepted_draft_tokens=2, py_num_accepted_draft_tokens_indices=[]
    )
    tree = SimpleNamespace(
        py_num_accepted_draft_tokens=2, py_num_accepted_draft_tokens_indices=[0, 2]
    )

    manager.update_resources(SimpleNamespace(generation_requests=[linear]), None, 2)
    assert len(calls) == 1

    with pytest.raises(NotImplementedError, match="relocate"):
        manager.update_resources(SimpleNamespace(generation_requests=[linear, tree]), None, 2)


def test_draft_layout_locates_the_appended_tail():
    # num_layers is the target count; a per-layer head list already includes the
    # draft tail.
    assert derive_shared_draft_layout(60, 4, 1) == ([60], 60)
    assert derive_shared_draft_layout(60, [4] * 60 + [64], 1) == ([60], 60)
    assert derive_shared_draft_layout(60, 4, 0) == ([], 60)
    assert derive_shared_draft_layout(None, 4, 1) == ([], None)


def test_shared_draft_layer_count_follows_the_base_manager():
    # Same rule as get_pp_layers: a spec config and no layer_mask.
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=lambda: False,
        is_mtp_vanilla=lambda: False,
        is_eagle3_one_model=lambda: True,
    )
    eagle3 = SimpleNamespace(spec_dec_mode=mode, _num_draft_hidden_layers=None)
    assert shared_draft_layer_count(eagle3, None) == 1
    assert shared_draft_layer_count(eagle3, [True] * 60) == 0
    assert shared_draft_layer_count(None, None) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("local_draft_layers", [[], [60], [61]])
@pytest.mark.parametrize("swa_scratch_reuse", [False, True])
def test_draft_kv_cache_view_accessor_respects_local_layers(
    monkeypatch: pytest.MonkeyPatch, local_draft_layers: list[int], swa_scratch_reuse: bool
) -> None:
    """Only local draft layers may create a view; scratch reuse is unsupported on every rank."""
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.NVFP4
    manager.is_draft = False
    manager._shared_draft_layer_ids = [60, 61]
    manager.layer_offsets = {layer: local for local, layer in enumerate([30, *local_draft_layers])}
    manager.enable_swa_scratch_reuse = swa_scratch_reuse
    manager._draft_kv_cache_view = None
    view = SimpleNamespace(blocks_in_primary_pool=1024)
    create_view = Mock(return_value=view)
    monkeypatch.setattr(m3_cache_manager, "MiniMaxM3DraftKVCacheView", create_view)

    if swa_scratch_reuse:
        with pytest.raises(NotImplementedError, match="SWA scratch reuse"):
            manager.get_draft_kv_cache_view()
        create_view.assert_not_called()
        assert manager._draft_kv_cache_view is None
    elif not local_draft_layers:
        assert manager.get_draft_kv_cache_view() is None
        create_view.assert_not_called()
        assert manager._draft_kv_cache_view is None
    else:
        assert manager.get_draft_kv_cache_view() is view
        assert manager.get_draft_kv_cache_view() is view
        create_view.assert_called_once_with(manager, local_draft_layers)
    assert manager._shared_draft_layer_ids == [60, 61]


@pytest.mark.cpu_only
@pytest.mark.parametrize("via_accessor", [False, True])
def test_draft_kv_cache_view_rejects_multiple_local_layers(via_accessor: bool) -> None:
    """The single-pool view must reject a second layer before publishing wrong pointers."""
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.NVFP4
    manager.is_draft = False
    manager._shared_draft_layer_ids = [60, 61]
    manager.layer_offsets = {60: 0, 61: 1}
    manager.enable_swa_scratch_reuse = False
    manager._draft_kv_cache_view = None

    with pytest.raises(NotImplementedError, match="exactly one local shared draft layer"):
        if via_accessor:
            manager.get_draft_kv_cache_view()
        else:
            MiniMaxM3DraftKVCacheView(manager, [60, 61])
    assert manager._draft_kv_cache_view is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("manager_cls", [KVCacheManagerV2, MiniMaxM3KVCacheManagerV2])
@pytest.mark.parametrize("dtype", [None, DataType.HALF, DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("spec_mode", ["none", "linear", "dynamic"])
def test_speculative_validation_precedes_cache_setup(
    monkeypatch: pytest.MonkeyPatch,
    manager_cls: type[KVCacheManagerV2],
    dtype: DataType | None,
    spec_mode: str,
) -> None:
    class CacheSetupReached(Exception):
        pass

    setup = Mock(spec_set=kv_cache_manager_v2.get_pp_layers, side_effect=CacheSetupReached)
    monkeypatch.setattr(kv_cache_manager_v2, "get_pp_layers", setup)
    spec_config = (
        None
        if spec_mode == "none"
        else Eagle3DecodingConfig(
            max_draft_len=3,
            speculative_model="unused",
            use_dynamic_tree=spec_mode == "dynamic",
            dynamic_tree_max_topK=2 if spec_mode == "dynamic" else None,
        )
    )
    manager = manager_cls.__new__(manager_cls)
    dtype_kwargs = {} if dtype is None else {"dtype": dtype}
    reject = (
        manager_cls is MiniMaxM3KVCacheManagerV2
        and dtype == DataType.NVFP4
        and spec_mode == "dynamic"
    )
    with pytest.raises(
        NotImplementedError if reject else CacheSetupReached,
        match="does not yet move NVFP4 K/V block scales" if reject else None,
    ):
        manager.__init__(
            KvCacheConfig(max_tokens=256, enable_block_reuse=False),
            CacheType.SELF,
            num_layers=4,
            num_kv_heads=2,
            head_dim=128,
            tokens_per_block=128,
            max_seq_len=256,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            spec_config=spec_config,
            **dtype_kwargs,
        )
    assert manager.dtype == (DataType.HALF if dtype is None else dtype)
    if reject:
        setup.assert_not_called()
    else:
        setup.assert_called_once()
