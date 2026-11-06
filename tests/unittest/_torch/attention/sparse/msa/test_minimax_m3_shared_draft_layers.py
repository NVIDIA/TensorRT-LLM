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
"""MiniMax-M3 dense and shared Eagle3 layouts and heterogeneous page tables.

Native layers use an attention-op pool rooted at their K page inside the slot.
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
    default_sparse_layer_ids,
    derive_shared_draft_layout,
    extend_attention_op_pools_for_shared_draft_layers,
    shared_draft_layer_count,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (
    uniform_subpages_per_slot,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _DEVICE_PAGE_TABLE_ENV,
    KVCacheManagerV2,
    Role,
    _BasePageTableMaterializer,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, PageIndexMode

DRAFT_LOCAL_LAYER = 60
_CACHE_MANAGER_MODULE = "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager"
SCALE = 179  # sub-pages per M3 mega-slot: 3 dense x 2 + 57 sparse x 3 + draft x 2
DRAFT_K_ADDR = 0x7000_0000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA cache pools")
@pytest.mark.parametrize("setting", ["0", "1"])
def test_nvfp4_shared_draft_constructs_and_converts_heterogeneous_page_tables(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    """Exercise real pool allocation, the draft page table and the resolved cache views."""
    torch.cuda.init()
    if setting == "0" and not prefer_pinned():
        pytest.skip("native host-reading kernel requires GPU-addressable host memory")
    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, setting)
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
        assert manager.num_attention_op_pools == 0
        assert not hasattr(manager, "_device_attention_op_block_offsets_staging")

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
        copy_idx = torch.tensor([0, 1], dtype=torch.int32, pin_memory=prefer_pinned())
        monkeypatch.setattr(
            manager,
            "index_mapper",
            SimpleNamespace(get_copy_index=lambda ids, num_contexts, beam_width: copy_idx),
        )
        device_copy = Mock(wraps=m3_cache_manager.copy_batch_block_offsets_to_device)
        monkeypatch.setattr(m3_cache_manager, "copy_batch_block_offsets_to_device", device_copy)
        monkeypatch.setattr(kv_cache_manager_v2, "copy_batch_block_offsets_to_device", device_copy)
        draft_output = torch.full(
            (1, 2, 2, draft.max_blocks_per_seq), -7, dtype=torch.int32, device="cuda"
        )
        draft.copy_batch_block_offsets(draft_output, [101, 102], 1, 0, 2)
        torch.cuda.synchronize()
        device_copy.assert_called_once()
        source, _, indices, scales, offsets, _ = device_copy.call_args.args
        assert all(t.is_cuda == (setting == "1") for t in (source, indices, scales, offsets))
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

        # Views are resolved at construction: the warmup scrub zeroes K/V outside inference mode.
        with torch.inference_mode():
            kv_view = manager.get_buffers(0, "HND")
            scale_view = manager.get_block_scale_buffers(3, "HND")
        assert kv_view is manager.get_buffers(0) and not kv_view.is_inference()
        assert scale_view is manager.get_block_scale_buffers(3) and not scale_view.is_inference()
        manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
def test_block_scale_role_uses_global_layer_ids(dtype: DataType) -> None:
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.layer_specs = None
    manager.dtype = dtype
    manager.pp_layers = [60, 7, 2]
    manager.sparse_layer_ids = {7}
    for local_layer, expected_nvfp4 in enumerate((None, Role.KEY_BLOCK_SCALE, None)):
        expected = expected_nvfp4 if dtype == DataType.NVFP4 else None
        assert manager._get_block_scale_role(Role.KEY, local_layer) == expected
        assert manager._get_block_scale_role(Role.VALUE, local_layer) is None
    generic = KVCacheManagerV2.__new__(KVCacheManagerV2)
    generic.layer_specs = None
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
    from the source pool's slot ids, does nothing extra without virtual pools, and
    nothing at all without attention-op pools.
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
    manager.num_attention_op_pools = 2
    manager._draft_op_pools = ((1, 0),)
    manager.num_pools = 1
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

    # Without native-layer virtual pools, only the base copy runs.
    calls.clear()
    plain = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    plain.num_attention_op_pools = 1
    plain.copy_batch_block_offsets(dst, [7], 1, 0, 1)
    assert calls == [("base", [7], 1, None)]

    calls.clear()
    hybrid = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    hybrid.num_attention_op_pools = 0
    hybrid.copy_batch_block_offsets(dst, [7], 1, 0, 1)
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("setting", ["0", "1"], ids=["native", "forced_cc"])
def test_shared_draft_page_tables_use_real_materialization(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    """Exercise base and virtual pool launches without allocating a model's KV cache."""
    if setting == "0" and not prefer_pinned():
        pytest.skip("native expansion requires GPU-addressable host memory")
    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, setting)
    pin = setting == "0"
    host = torch.arange(32, dtype=torch.int32).reshape(1, 4, 8)
    table = torch.full((1, 4, 2, 8), 777, dtype=torch.int32, pin_memory=pin)
    table[:, :, 0] = host
    table[0, 2, 0, -1] = BAD_PAGE_INDEX
    stream = torch.cuda.current_stream()

    def prepare_base(self, index_mapper_capacity: int) -> None:
        # Only storage setup is stubbed. Both initialization of the draft
        # materializers and all row gathers, H2D copies and kernels are real.
        self.num_pools = 1
        self._use_per_layer_page_tables = False
        self.host_kv_cache_block_offsets = table
        self._stream = stream
        self.index_scales = torch.tensor([3], dtype=torch.int32, pin_memory=pin)
        self.kv_offset = torch.tensor([1], dtype=torch.int32, pin_memory=pin)
        self._page_table_materializer = _BasePageTableMaterializer(
            table, stream, self.index_scales, self.kv_offset
        )
        self.kv_cache_pool_pointers = torch.tensor([[0, 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.int32)

    monkeypatch.setattr(KVCacheManagerV2, "_prepare_page_table_tensor", prepare_base)
    monkeypatch.setattr(
        MiniMaxM3KVCacheManagerV2,
        "_kv_slot_geometry",
        lambda self, layer: (0, None, 4, 5 + 2 * layer, None),
    )
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager._shared_draft_layer_ids = [1, 2]
    manager.dtype = DataType.FP8
    manager.layer_offsets = {0: 0, 1: 1, 2: 2}
    manager.is_draft = False
    manager.enable_swa_scratch_reuse = False
    manager.tokens_per_block = 128
    manager.max_copy_beam_width = 1
    manager.kv_cache_type = CacheType.SELF
    manager._prepare_page_table_tensor(4)
    assert manager._draft_op_pools == ((1, 0), (2, 0))
    assert len(manager._draft_page_table_materializers) == (2 if setting == "1" else 0)

    # Reorder and shrink the batch between launches; retained staging must
    # reread the live source instead of reusing the preceding batch's rows.
    for request_ids in ([2, 0], [1]):
        copy_idx = torch.tensor(request_ids, dtype=torch.int32, pin_memory=pin)
        manager.index_mapper = SimpleNamespace(
            get_copy_index=lambda ids, nc, bw, replicate_beam_zero=False: copy_idx
        )
        output = torch.full((3, 4, 2, 8), -12345, dtype=torch.int32, device="cuda")
        manager.copy_batch_block_offsets(output, request_ids, 1, 0, len(request_ids))
        stream.synchronize()
        actual = output.cpu()
        pages = table[0, request_ids, 0]
        for pool, scale in enumerate((3, 7, 9)):
            keys = torch.where(pages == BAD_PAGE_INDEX, 0, pages * scale)
            values = torch.where(pages == BAD_PAGE_INDEX, 0, pages * scale + 1)
            torch.testing.assert_close(actual[pool, : len(request_ids), 0], keys)
            torch.testing.assert_close(actual[pool, : len(request_ids), 1], values)
        assert torch.all(actual[:, len(request_ids) :] == -12345)
        table[:, :, 0, 0] += 10


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
    assert manager.num_attention_op_pools == (0 if dtype == DataType.NVFP4 else 61)
    assert manager._draft_op_pools == ()
    expected_extra_pages = {128} if dtype == DataType.FP8 else set()
    assert manager.trtllm_gen_extra_tokens_per_block == frozenset(expected_extra_pages)


@pytest.mark.parametrize(
    "dtype,tokens_per_block,expected",
    [
        (DataType.FP8, 128, {128}),
        (DataType.BF16, 128, {128}),
        (DataType.FP8, 32, set()),
        (DataType.NVFP4, 128, set()),
    ],
)
def test_separate_draft_manager_opts_into_p128(monkeypatch, dtype, tokens_per_block, expected):
    """A separate (is_draft) manager holds only native draft layers: at 128-token pages
    it lists 128 as a trtllm-gen extra page size (the only P128 route on SM107) and
    builds no virtual pools."""
    pointers = torch.tensor([[0x6000_0000, 0]])
    mapping = torch.tensor([[0, 0]], dtype=torch.int32)

    def fake_base_prepare(self, index_mapper_capacity):
        self._use_per_layer_page_tables = False
        self.kv_cache_pool_pointers = pointers.clone()
        self.kv_cache_pool_mapping = mapping.clone()
        self.num_attention_op_pools = 1

    monkeypatch.setattr(KVCacheManagerV2, "_prepare_page_table_tensor", fake_base_prepare)
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = dtype
    manager._shared_draft_layer_ids = []
    manager.layer_offsets = {0: 0}
    manager.is_draft = True
    manager.enable_swa_scratch_reuse = False
    manager.tokens_per_block = tokens_per_block

    manager._prepare_page_table_tensor(8)

    assert manager.trtllm_gen_extra_tokens_per_block == frozenset(expected)
    assert manager._draft_op_pools == ()
    assert torch.equal(manager.kv_cache_pool_pointers, pointers)
    assert manager.num_attention_op_pools == (0 if dtype == DataType.NVFP4 else 1)


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


def _text_config(sparse_attention_freq):
    return SimpleNamespace(
        num_hidden_layers=len(sparse_attention_freq),
        sparse_attention_config={"sparse_attention_freq": list(sparse_attention_freq)},
    )


def test_default_sparse_layers_follow_the_text_config():
    # The model reads its dense/sparse split from the config; so does the cache.
    assert default_sparse_layer_ids(_text_config([1, 0, 1, 1, 0]), 5) == [0, 2, 3]
    assert default_sparse_layer_ids(_text_config([0, 0, 0, 1, 1]), 5) == [3, 4]
    # Without a config spanning the target layers, layers 0..2 stay dense.
    assert default_sparse_layer_ids(None, 5) == [3, 4]
    assert default_sparse_layer_ids(_text_config([1, 0, 1, 1, 0]), 6) == [3, 4, 5]
    assert default_sparse_layer_ids(_text_config([1, 0, 1, 1, 0]), None) == []


def test_default_sparse_layers_log_a_fallback_that_ignores_the_config(monkeypatch):
    import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager as cache_manager

    log = Mock()
    monkeypatch.setattr(cache_manager, "logger", log)
    # A config whose split is not the default, on a manager with a different layer count.
    assert default_sparse_layer_ids(_text_config([1, 0, 1, 1, 0]), 6) == [3, 4, 5]
    assert log.info.call_count == 1
    # The default split, a matching layer count, or no config: nothing to report.
    default_sparse_layer_ids(_text_config([0, 0, 0, 1, 1]), 6)
    default_sparse_layer_ids(_text_config([1, 0, 1, 1, 0]), 5)
    default_sparse_layer_ids(None, 6)
    assert log.info.call_count == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("implementation", ["msa", "triton"])
def test_index_k_cache_follows_the_config_sparse_layers(
    monkeypatch: pytest.MonkeyPatch, implementation: str
) -> None:
    """Every layer the config makes sparse gets an index-K buffer and view, and only those."""
    resolved = []
    registered = {}

    def fake_base_init(self, *args, **kwargs):
        self.is_disagg = False
        self.dtype = DataType.BF16
        # Five target layers and the shared Eagle3 layer appended after them.
        self.layer_offsets = {layer: layer for layer in range(6)}
        # The base _build_base_config registers the extra pool buffers here.
        registered.update(self._extra_buffers_per_layer(tokens_per_block=32))

    def fake_get_index_k_buffer(self, layer_idx, **kwargs):
        resolved.append(layer_idx)
        return torch.empty(0)

    monkeypatch.setattr(KVCacheManagerV2, "__init__", fake_base_init)
    monkeypatch.setattr(KVCacheManagerV2, "get_index_k_buffer", fake_get_index_k_buffer)
    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "_compute_num_total_slots", lambda self: 0)
    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "get_buffers", lambda self, *args: None)
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=lambda: False,
        is_mtp_vanilla=lambda: False,
        is_eagle3_one_model=lambda: True,
    )

    manager = MiniMaxM3KVCacheManagerV2(
        num_layers=5,
        spec_config=SimpleNamespace(spec_dec_mode=mode, _num_draft_hidden_layers=None),
        sparse_attention_config=SimpleNamespace(
            implementation=implementation, indexer_kv_dtype="bf16"
        ),
        pretrained_config=_text_config([1, 0, 1, 1, 0]),
    )

    assert manager._shared_draft_layer_ids == [5]
    assert manager.sparse_layer_ids == [0, 2, 3]
    assert manager.disable_index_value_layer_ids == {0, 2, 3}
    assert sorted(registered) == [0, 2, 3]
    for buffers in registered.values():
        assert [buffer.role for buffer in buffers] == [Role.INDEX_KEY]
    assert resolved == [0, 2, 3]
    for layer in (0, 2, 3):
        assert manager.get_index_k_buffer(layer) is not None
    for layer in (1, 4, 5):
        assert manager.get_index_k_buffer(layer) is None
    # The Triton path runs the dense target layers on the native attention op.
    expected_dense_native = (1, 4) if implementation == "triton" else ()
    assert manager._dense_native_layer_ids == expected_dense_native


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
@pytest.mark.parametrize(
    "sm,fmha_libs,error",
    [
        (107, None, None),
        (107, "-flashinfer_trtllm_gen", "TLLM_FMHA_LIBS excludes it"),
        (107, "fallback", "TLLM_FMHA_LIBS excludes it"),
        (107, "flashinfer_trtllm_gen,fallback", None),
        # Elsewhere the fallback has P128 paged-context kernels, so nothing is required.
        (103, "-flashinfer_trtllm_gen", None),
    ],
)
@pytest.mark.parametrize("site", ["separate_draft_manager", "nvfp4_draft_view"])
def test_p128_opt_in_fails_fast_without_the_flashinfer_route(
    monkeypatch, sm, fmha_libs, error, site
):
    """On SM107 the P128 opt-in is checked when the KV manager is set up, not at the
    first prefill: a missing FlashInfer route raises an error that names the fix."""
    monkeypatch.setattr(f"{_CACHE_MANAGER_MODULE}.get_sm_version", lambda: sm)
    if fmha_libs is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", fmha_libs)

    if site == "separate_draft_manager":
        monkeypatch.setattr(
            KVCacheManagerV2,
            "_prepare_page_table_tensor",
            lambda self, capacity: setattr(self, "_use_per_layer_page_tables", False),
        )
        manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
        manager.dtype = DataType.FP8
        manager._shared_draft_layer_ids = []
        manager.layer_offsets = {0: 0}
        manager.is_draft = True
        manager.enable_swa_scratch_reuse = False
        manager.tokens_per_block = 128
        setup = lambda: manager._prepare_page_table_tensor(8)  # noqa: E731
    else:
        # Stop right after the check: the rest of the view needs real pools.
        manager = SimpleNamespace(tokens_per_block=128, layer_offsets={})
        setup = lambda: MiniMaxM3DraftKVCacheView(manager, [DRAFT_LOCAL_LAYER])  # noqa: E731

    if error is not None:
        with pytest.raises(RuntimeError, match=error) as excinfo:
            setup()
        assert "128-token KV pages" in str(excinfo.value)
    elif site == "separate_draft_manager":
        setup()
        assert manager.trtllm_gen_extra_tokens_per_block == frozenset({128})
    else:
        # The check passed; the view then fails on the stub's missing layer.
        with pytest.raises(KeyError, match=str(DRAFT_LOCAL_LAYER)):
            setup()


@pytest.mark.cpu_only
@pytest.mark.parametrize("shared_draft", [False, True])
def test_p128_target_fails_fast_only_with_a_shared_draft_layer(monkeypatch, shared_draft):
    """On SM107 a BF16 target's dense layers keep SDPA, so a missing FlashInfer route only
    fails setup when a shared draft layer needs it."""
    monkeypatch.setattr(f"{_CACHE_MANAGER_MODULE}.get_sm_version", lambda: 107)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "-flashinfer_trtllm_gen")
    monkeypatch.setattr(
        KVCacheManagerV2,
        "_prepare_page_table_tensor",
        lambda self, capacity: setattr(self, "_use_per_layer_page_tables", True),
    )
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager.dtype = DataType.HALF
    manager.is_draft = False
    manager.enable_swa_scratch_reuse = False
    manager.tokens_per_block = 128
    manager._dense_native_layer_ids = (0, 1, 2)
    manager._shared_draft_layer_ids = [60] if shared_draft else []
    manager.layer_offsets = {0: 0, 1: 1, 2: 2, **({60: 3} if shared_draft else {})}

    if shared_draft:
        with pytest.raises(RuntimeError, match="TLLM_FMHA_LIBS excludes it"):
            manager._prepare_page_table_tensor(8)
    else:
        manager._prepare_page_table_tensor(8)
        assert manager.trtllm_gen_extra_tokens_per_block == frozenset({128})


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


@pytest.mark.cpu_only
def test_dense_layers_get_virtual_pools_without_speculation(monkeypatch):
    """Dense target layers need the same stride correction as draft layers."""

    def fake_base_prepare(self, index_mapper_capacity):
        self._page_table_materializer = SimpleNamespace(uses_device_expansion=False)
        self._use_per_layer_page_tables = False
        self.kv_cache_pool_pointers = torch.tensor([[0x6000_0000, 0]], dtype=torch.int64)
        self.kv_cache_pool_mapping = torch.tensor([[0, i] for i in range(4)], dtype=torch.int32)
        self.num_pools = 1

    monkeypatch.setattr(KVCacheManagerV2, "_prepare_page_table_tensor", fake_base_prepare)
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager._shared_draft_layer_ids = []
    manager._dense_native_layer_ids = (0, 1, 2)
    manager.dtype = DataType.BF16
    manager.layer_offsets = {i: i for i in range(4)}
    manager.is_draft = False
    manager.enable_swa_scratch_reuse = False
    manager.tokens_per_block = 128
    manager._kv_slot_geometry = lambda i: (0x7000_0000 + i * 4096, None, 8, 9, None)
    manager._prepare_page_table_tensor(8)
    assert manager.kv_cache_pool_mapping.tolist() == [[1, 0], [2, 0], [3, 0], [0, 3]]
    assert manager.kv_cache_pool_pointers[1:, 0].tolist() == [
        0x7000_0000 + i * 4096 for i in range(3)
    ]
    assert manager._draft_index_scales.tolist() == [9, 9, 9]
    assert manager._draft_kv_offsets.tolist() == [1, 1, 1]
    assert manager.num_attention_op_pools == 4


@pytest.mark.cpu_only
def test_sparse_view_writes_native_hnd_storage_without_copy(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
        write_kv_slots,
    )

    # Multiple heads distinguish HND from NHD; include adjacent layer/index pages.
    heads = 2
    storage = (torch.arange(3 * 7 * heads * 8 * 16, dtype=torch.float32) + 1).reshape(
        3, 7, heads, 8, 16
    )
    before = storage.clone()
    manager = MiniMaxM3KVCacheManagerV2.__new__(MiniMaxM3KVCacheManagerV2)
    manager._kv_buffers = {}
    manager._kv_slot_geometry = lambda i, layout: (
        storage.data_ptr(),
        storage.dtype,
        3,
        7,
        [heads, 8, 16],
    )
    monkeypatch.setattr(m3_cache_manager, "convert_to_torch_tensor", lambda wrapper: storage)
    nhd = manager.get_buffers(0)
    hnd = manager.get_buffers(0, "HND")
    assert nhd.data_ptr() == hnd.data_ptr() == storage.data_ptr()
    slots = torch.tensor([0, 7, 8, 23])
    values = torch.arange(4 * heads * 16, dtype=storage.dtype).reshape(4, heads, 16)
    write_kv_slots(nhd[:, 0], slots, values, num_live_tokens=slots.numel())
    torch.testing.assert_close(hnd[slots // 8, 0, :, slots % 8], values)
    touched = torch.zeros_like(storage, dtype=torch.bool)
    touched[slots // 8, 0, :, slots % 8] = True
    assert torch.equal(storage[~touched], before[~touched])
    torch.testing.assert_close(nhd, hnd.transpose(2, 3))
    # Index-K has one head even when main K/V has multiple heads. Its
    # NHD/HND views must alias the untouched adjacent physical page.
    index_hnd = storage[:, 2, :1]
    manager._index_k_buffers = {0: index_hnd.transpose(1, 2)}
    index_nhd = manager.get_index_k_buffer(0)
    assert index_nhd.shape == (3, 8, 1, 16)
    assert index_nhd.data_ptr() == index_hnd.data_ptr()
    torch.testing.assert_close(manager.get_index_k_buffer(0, "HND"), index_hnd)
    torch.testing.assert_close(index_hnd, before[:, 2, :1])
