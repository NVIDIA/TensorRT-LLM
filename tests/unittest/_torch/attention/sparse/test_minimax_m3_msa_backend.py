# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate backend selection, decode scratch-buffer sizing, and the paged
HND view contract passed to the packaged MSA kernel. Numerical parity against
the Triton reference is covered by the SM100 integration accuracy test.
"""

import sys
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
    MiniMaxM3KVCacheManagerV2,
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import MsaDecodeSpan
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    msa_paged_kv,
)
from tensorrt_llm._torch.attention_backend.sparse.registry import _resolve_minimax_m3_backend_cls
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_msa_package_availability_installs_cutlass_46_compatibility_aliases(monkeypatch):
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        msa_package_available,
    )

    cute = ModuleType("cutlass.cute")
    cute.core = SimpleNamespace()
    cute.ThrMma = object()
    cute.make_rmem_tensor = object()
    cutlass = ModuleType("cutlass")
    cutlass.cute = cute
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    monkeypatch.setattr("importlib.util.find_spec", lambda unused_name: object())

    msa_package_available.cache_clear()
    try:
        assert msa_package_available()
        assert cute.core.ThrMma is cute.ThrMma
        assert cute.make_fragment is cute.make_rmem_tensor
    finally:
        msa_package_available.cache_clear()


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(implementation="msa").to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is MiniMaxM3MsaSparseAttention


def test_msa_requires_block_size_128():
    # The MSA implementation is fixed to a 128-token page size; a mismatched
    # sparse_block_size must fail loudly at config construction rather than being
    # silently overridden at runtime.
    with pytest.raises(ValueError, match=r"sparse_block_size == 128"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", sparse_block_size=64)

    # The Triton reference is unaffected by the constraint.
    cfg = MiniMaxM3SparseAttentionConfig(implementation="triton", sparse_block_size=64)
    assert cfg.sparse_block_size == 64


def test_msa_fp8_indexer_config_is_explicit_and_lowered() -> None:
    """FP8 is explicit, MSA-only, and incompatible with index values."""
    cfg = MiniMaxM3SparseAttentionConfig(implementation="msa", indexer_kv_dtype="fp8")
    assert cfg.to_sparse_params().indexer_kv_dtype == "fp8"

    with pytest.raises(ValueError, match=r"requires the 'msa' implementation"):
        MiniMaxM3SparseAttentionConfig(implementation="triton", indexer_kv_dtype="fp8")
    with pytest.raises(ValueError, match=r"sparse_disable_index_value=True"):
        MiniMaxM3SparseAttentionConfig(
            implementation="msa",
            indexer_kv_dtype="fp8",
            sparse_disable_index_value=False,
        )
    for num_index_heads in (0, -1):
        with pytest.raises(ValueError, match=r"greater than 0"):
            MiniMaxM3SparseAttentionConfig(sparse_num_index_heads=num_index_heads)
    for sparse_index_dim in (0, -1):
        with pytest.raises(ValueError, match=r"greater than 0"):
            MiniMaxM3SparseAttentionConfig(sparse_index_dim=sparse_index_dim)


@pytest.mark.parametrize(
    (
        "configured_sparse_index_dim",
        "expected_sparse_index_dim",
        "indexer_kv_dtype",
        "base_dtype",
        "expected_dtype",
    ),
    [
        (None, 128, "bf16", DataType.BF16, torch.bfloat16),
        (96, 96, "bf16", DataType.BF16, torch.bfloat16),
        (96, 96, "bf16", DataType.HALF, torch.bfloat16),
        (96, 96, "bf16", DataType.FLOAT, torch.bfloat16),
        (128, 128, "fp8", DataType.HALF, torch.float8_e4m3fn),
    ],
)
def test_cache_manager_honors_executor_sparse_attention_config(
    monkeypatch: pytest.MonkeyPatch,
    configured_sparse_index_dim: int | None,
    expected_sparse_index_dim: int,
    indexer_kv_dtype: str,
    base_dtype: DataType,
    expected_dtype: torch.dtype,
) -> None:
    """The production keyword controls both index width and storage dtype."""

    observed_index_buffer_args = {}

    def fake_base_init(self, *args, **kwargs) -> None:
        del args, kwargs
        self.is_disagg = False
        self.dtype = base_dtype
        self.layer_offsets = {}

    def fake_get_index_k_buffer(self, layer_idx, **kwargs):
        del self, layer_idx
        observed_index_buffer_args.update(kwargs)
        return None

    monkeypatch.setattr(KVCacheManagerV2, "__init__", fake_base_init)
    monkeypatch.setattr(KVCacheManagerV2, "get_index_k_buffer", fake_get_index_k_buffer)
    monkeypatch.setattr(MiniMaxM3KVCacheManagerV2, "_compute_num_total_slots", lambda self: 0)
    sparse_config = SimpleNamespace(
        sparse_index_dim=configured_sparse_index_dim,
        indexer_kv_dtype=indexer_kv_dtype,
    )

    manager = MiniMaxM3KVCacheManagerV2(
        num_layers=4,
        sparse_attention_config=sparse_config,
    )

    assert manager.sparse_index_dim == expected_sparse_index_dim
    assert manager.indexer_kv_dtype == indexer_kv_dtype
    assert manager._torch_dtype_for_index_cache() is expected_dtype
    assert manager.get_index_k_buffer(3) is None
    assert observed_index_buffer_args["head_dim"] == expected_sparse_index_dim
    assert observed_index_buffer_args["dtype"] is expected_dtype


@pytest.mark.parametrize("sparse_index_dim", [0, -1])
def test_cache_manager_rejects_non_positive_sparse_index_dim(sparse_index_dim: int) -> None:
    for kwargs in (
        {"sparse_index_dim": sparse_index_dim},
        {"sparse_attention_config": SimpleNamespace(sparse_index_dim=sparse_index_dim)},
    ):
        with pytest.raises(ValueError, match=r"sparse_index_dim must be greater than 0"):
            MiniMaxM3KVCacheManagerV2(num_layers=4, **kwargs)


def test_msa_metadata_rejects_undersized_max_score_buffer():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    # Flat backing store sized for 4 heads * 8 k-tiles * 2 batch = 64 elements,
    # too small for the plan's required 4 * 16 * 2 = 128.
    metadata.msa_max_score = torch.zeros(4 * 8 * 2)
    metadata.kv_cache_manager = None

    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata._ensure_msa_decode_scratch_buffers(
            num_index_heads=4,
            max_batch=2,
            capture_graph=False,
            required_max_k_tiles=16,
        )


MAX_NUM_SEQUENCES = 8
MAX_BLOCKS_PER_SEQ = 64


class _RecordingBuffers:
    """The graph buffer pool, recording what each buffer was reserved as."""

    def __init__(self):
        self.requested = {}

    def get_buffer(self, tensor_shape, dtype, cache_name, capture_graph):
        self.requested[cache_name] = (tuple(tensor_shape), dtype, capture_graph)
        return torch.zeros(tensor_shape, device="cuda", dtype=dtype)


def _buffer_metadata(**manager_fields):
    """Metadata ready for _create_msa_buffers, under capture."""
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.kv_cache_manager = SimpleNamespace(
        max_blocks_per_seq=MAX_BLOCKS_PER_SEQ,
        tokens_per_block=128,
        get_index_k_buffer=lambda layer_idx, kv_layout=None: None,
        **manager_fields,
    )
    metadata.is_cuda_graph = True
    metadata.cuda_graph_buffers = _RecordingBuffers()
    metadata.max_num_sequences = MAX_NUM_SEQUENCES
    metadata.max_num_tokens = 512
    # No sparse params, so the fmha_sm100 proxy scratch is skipped and this
    # exercises only the layer-invariant buffers.
    metadata._msa_params = None
    return metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_msa_buffers_include_graph_stable_block_table():
    """The 2-D page table and per-request length the decode kernels take must
    come from the graph buffer pool at the manager's worst-case geometry, so
    their addresses survive capture."""
    metadata = _buffer_metadata()

    metadata._create_msa_buffers()

    assert metadata._msa_buffers_ready
    assert metadata.msa_block_table.shape == (MAX_NUM_SEQUENCES, MAX_BLOCKS_PER_SEQ)
    assert metadata.msa_seq_lens_cuda.shape == (MAX_NUM_SEQUENCES,)
    requested = metadata.cuda_graph_buffers.requested
    assert requested["msa_block_table"] == (
        (MAX_NUM_SEQUENCES, MAX_BLOCKS_PER_SEQ),
        torch.int32,
        True,
    )
    assert requested["msa_seq_lens_cuda"] == ((MAX_NUM_SEQUENCES,), torch.int32, True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("factors", "expected"),
    [((9, 9, 9), 9), ((9, 4, 9), 0)],
    ids=["uniform-pool", "groups-disagree"],
)
def test_msa_buffers_stage_the_subpage_table_only_for_a_uniform_pool(factors, expected):
    """prepare() stages the sub-page expansion before any layer is named, which
    is sound only where every layer of the pool packs the same number of
    sub-pages per slot. Where they disagree, no table is staged and each dense
    layer expands its own.
    """
    metadata = _buffer_metadata(
        layer_offsets=dict.fromkeys(range(len(factors)), 0),
        get_kv_subpage_pool=lambda layer_idx, kv_layout="HND": (None, factors[layer_idx]),
    )

    metadata._create_msa_buffers()

    assert metadata._msa_subpages_per_slot == expected
    if expected == 0:
        assert metadata.msa_subpage_block_table is None
        assert "msa_subpage_block_table" not in metadata.cuda_graph_buffers.requested
    else:
        # One K row and one V row per slot, at the same worst-case geometry as
        # the slot table it expands.
        assert metadata.cuda_graph_buffers.requested["msa_subpage_block_table"] == (
            (MAX_NUM_SEQUENCES, 2, MAX_BLOCKS_PER_SEQ),
            torch.int32,
            True,
        )


def test_msa_subpage_rows_slice_the_generation_span():
    """A mixed step hands the dense kernel only the span's rows, so its sub-page
    table has to be sliced the same way the slot table is. The factor travels
    with it so the kernel can tell a stale staging from its own geometry."""
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.msa_subpage_block_table = torch.arange(4 * 2 * 3, dtype=torch.int32).reshape(4, 2, 3)
    metadata._msa_subpages_per_slot = 9

    table, factor = metadata.msa_subpage_rows(2, 4)
    assert factor == 9
    assert torch.equal(table, metadata.msa_subpage_block_table[2:4])

    # Nothing staged: the caller expands its own layer's table, and the 0
    # factor is what tells it to.
    metadata.msa_subpage_block_table = None
    assert metadata.msa_subpage_rows(2, 4) == (None, 0)


def test_msa_proxy_max_score_view_is_contiguous_over_stable_store():
    """The proxy view fed to fmha_sm100 must be contiguous in the exact
    [num_index_heads, plan_max_k_tiles, num_tokens] shape the kernel writes,
    backed by a stable store so its data_ptr survives CUDA graph replay.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    # Worst-case store: 4 heads * 16 k-tiles * 8 batch.
    num_index_heads, worst_k, max_batch = 4, 16, 8
    metadata.msa_max_score = torch.zeros(num_index_heads * worst_k * max_batch)
    store_ptr = metadata.msa_max_score.data_ptr()

    # A smaller live step still yields a contiguous view sized to that step,
    # which is what the kernel's stride-agnostic write requires.
    view = metadata.msa_proxy_max_score_view(num_index_heads, 5, 3)
    assert view.shape == (num_index_heads, 5, 3)
    assert view.is_contiguous()
    assert view.data_ptr() == store_ptr

    # Oversized requests are rejected rather than silently corrupting memory.
    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata.msa_proxy_max_score_view(num_index_heads, worst_k, max_batch + 1)

    # So is an empty one. Both writers address the view by block id, so a zero
    # extent is not a small view but writes past the end of one.
    with pytest.raises(ValueError, match=r"no block extent"):
        metadata.msa_proxy_max_score_view(num_index_heads, 0, max_batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_msa_paged_kv_preserves_tma_compatible_outer_stride() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")

    from fmha_sm100.cute import interface as sparse_interface

    pages, roles, heads, page_size, head_dim = 5, 2, 2, 128, 128
    pool = torch.empty(
        pages,
        roles,
        heads,
        page_size,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_cache_manager = Mock()
    kv_cache_manager.get_buffers.return_value = pool

    k_view, v_view = msa_paged_kv(kv_cache_manager, layer_idx=3)

    kv_cache_manager.get_buffers.assert_called_once_with(3, kv_layout="HND")
    for view in (k_view, v_view):
        assert not view.is_contiguous()
        prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)
        assert prepared.data_ptr() == view.data_ptr()
        assert prepared.stride() == view.stride()

    mismatched = sparse_interface._prepare_paged_hnd_input(k_view, page_size // 2)
    assert mismatched.data_ptr() == k_view.data_ptr()
    with pytest.raises(ValueError, match="page_size == blk_kv"):
        sparse_interface._prepare_paged_kv_for_tma(mismatched, mismatched, page_size // 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_msa_paged_hnd_input_materializes_unaligned_outer_stride() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")

    from fmha_sm100.cute import interface as sparse_interface

    pages, heads, page_size, head_dim = 5, 2, 128, 128
    outer_stride = heads * page_size * head_dim + 1
    storage = torch.empty(
        pages * outer_stride,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    view = storage.as_strided(
        (pages, heads, page_size, head_dim),
        (outer_stride, page_size * head_dim, head_dim, 1),
    )

    prepared = sparse_interface._prepare_paged_hnd_input(view, page_size)

    assert prepared.is_contiguous()
    assert prepared.data_ptr() != view.data_ptr()
    torch.testing.assert_close(prepared, view)


def test_msa_index_k_uses_hnd_cache_view_and_writer():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    num_pages, coalescing_scale, page_size, head_dim = 2, 7, 8, 16
    pool = torch.zeros(
        num_pages,
        coalescing_scale,
        1,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
    )
    hnd_cache = pool[:, 0]

    class FakeCacheManager:
        def __init__(self):
            self.calls = []

        def get_index_k_buffer(self, layer_idx, kv_layout="NHD"):
            self.calls.append((layer_idx, kv_layout))
            return hnd_cache

    manager = FakeCacheManager()
    metadata.kv_cache_manager = manager
    metadata.msa_out_cache_loc = torch.tensor([2, page_size + 5], dtype=torch.int32)
    values = torch.arange(2 * head_dim, dtype=torch.float32).reshape(2, 1, head_dim)

    returned = metadata.msa_idx_k_cache(3)
    metadata.msa_write_idx_k(3, values)

    assert returned.data_ptr() == hnd_cache.data_ptr()
    assert not returned.is_contiguous()
    assert manager.calls == [(3, "HND"), (3, "HND")]
    torch.testing.assert_close(hnd_cache[0, 0, 2], values[0, 0].to(torch.bfloat16))
    torch.testing.assert_close(hnd_cache[1, 0, 5], values[1, 0].to(torch.bfloat16))


def test_msa_indexer_preserves_strided_hnd_index_k(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer as indexer_module
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseConfig

    config = MiniMaxM3SparseConfig(
        num_q_heads=4,
        num_kv_heads=1,
        head_dim=128,
        num_index_heads=4,
        sparse_index_dim=128,
        block_size=128,
        topk=16,
    )
    indexer = indexer_module.MsaIndexer(config)
    pool = torch.randn(2, 7, 1, 128, 128, dtype=torch.bfloat16)
    idx_k_paged = pool[:, 0]
    captured = {}

    def fake_proxy_max_score(idx_q, passed_idx_k, **kwargs):
        del kwargs
        captured["idx_k"] = passed_idx_k
        return torch.zeros(4, 2, idx_q.shape[0])

    expected = torch.zeros(1, 1, 16, dtype=torch.int32)

    def fake_select_blocks_from_maxscore(*args, **kwargs):
        del args, kwargs
        return expected

    monkeypatch.setattr(indexer_module, "_proxy_max_score", fake_proxy_max_score)
    monkeypatch.setattr(
        indexer_module,
        "select_blocks_from_maxscore",
        fake_select_blocks_from_maxscore,
    )

    result = indexer.select_blocks(
        torch.zeros(1, 4, 128, dtype=torch.bfloat16),
        idx_k_paged,
        idx_sm_scale=128**-0.5,
        kv_indices=torch.arange(2, dtype=torch.int32),
        qo_lens_cpu=torch.tensor([1], dtype=torch.int32),
        kv_lens_cpu=torch.tensor([256], dtype=torch.int32),
        qo_offset_cpu=torch.tensor([255], dtype=torch.int32),
    )

    assert captured["idx_k"] is idx_k_paged
    assert captured["idx_k"].data_ptr() == idx_k_paged.data_ptr()
    assert not captured["idx_k"].is_contiguous()
    assert result is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_msa_indexer_enforces_real_fp8_and_bf16_handoff_states() -> None:
    """Only producer states reachable from the FP8 and BF16 model paths pass."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseConfig

    config = MiniMaxM3SparseConfig(
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=128,
        num_index_heads=4,
        sparse_index_dim=128,
        block_size=128,
        topk=16,
    )
    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.m3_config = config
    attention.layer_idx = 3
    attention.indexer_kv_dtype = "fp8"
    captured = {}

    class FakeIndexer:
        def select_blocks(self, idx_q, idx_k_cache, **kwargs):
            captured["idx_q"] = idx_q
            captured["index_k_cache"] = idx_k_cache
            captured["kwargs"] = kwargs
            return torch.zeros(2, 4, 16, dtype=torch.int32, device="cuda")

    attention.indexer = FakeIndexer()

    class FakeMetadata:
        # A chunked-prefill step, whose rows are all fmha_sm100's, so the
        # indexer takes the proxy path and needs no decode span.
        msa_decode_span = None
        msa_prefill_proxy_plan = (False, 0, 2, {}, None)
        msa_prefill_n_valid_blocks = torch.ones(2, dtype=torch.int32, device="cuda")
        msa_kv_indices = torch.arange(2, dtype=torch.int32, device="cuda")
        msa_qo_lens_cpu = torch.ones(2, dtype=torch.int32)
        msa_kv_lens_cpu = torch.full((2,), 128, dtype=torch.int32)
        msa_qo_offset_cpu = torch.full((2,), 127, dtype=torch.int32)
        num_contexts = 2
        num_generations = 0

        def __init__(self, dtype: torch.dtype) -> None:
            backing = torch.empty(2 * 7, 1, 128, 128, dtype=dtype, device="cuda")
            self.cache = backing[::7]
            self.msa_out_cache_loc = torch.tensor([0, 128], dtype=torch.int32, device="cuda")

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
                write_kv_slots,
            )

            captured["write"] = (layer_idx, idx_k)
            write_kv_slots(
                self.cache,
                self.msa_out_cache_loc,
                idx_k,
                layout="HND",
            )

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            captured["read_layer"] = layer_idx
            return self.cache

    idx_q = torch.randn(2, 4 * 128, dtype=torch.bfloat16, device="cuda")
    idx_k = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    fp8_metadata = FakeMetadata(torch.float8_e4m3fn)

    # The production FP8 path is fused. BF16 Q plus a live K is a test-only
    # state and must not be silently converted or written into an FP8 cache.
    with pytest.raises(ValueError, match=r"requires fused FP8 index-Q"):
        attention.run_indexer(idx_q, idx_k, fp8_metadata)

    # The fused producer has already inserted K and passes no live K tensor;
    # E4M3 Q flows to the scorer without a duplicate cache write.
    fused_q = idx_q.to(torch.float8_e4m3fn)
    result = attention.run_indexer(fused_q, None, fp8_metadata)
    assert result.shape == (2, 4, 16)
    assert captured["idx_q"].data_ptr() == fused_q.data_ptr()
    assert captured["index_k_cache"].dtype == torch.float8_e4m3fn
    assert captured["index_k_cache"].stride(0) == 7 * 128 * 128
    assert "write" not in captured

    # The default BF16 path keeps both live tensors and populates its cache.
    attention.indexer_kv_dtype = "bf16"
    bf16_metadata = FakeMetadata(torch.bfloat16)
    with pytest.raises(ValueError, match=r"does not match indexer_kv_dtype"):
        attention.run_indexer(idx_q, idx_k, fp8_metadata)
    with pytest.raises(ValueError, match=r"requires BF16 index-Q"):
        attention.run_indexer(fused_q, idx_k, bf16_metadata)
    with pytest.raises(ValueError, match=r"live BF16 index-K tensor"):
        attention.run_indexer(idx_q, None, bf16_metadata)
    for unsupported_dtype in (torch.float16, torch.float32):
        with pytest.raises(ValueError, match=r"requires BF16 index-Q"):
            attention.run_indexer(
                idx_q.to(unsupported_dtype),
                idx_k.to(unsupported_dtype),
                bf16_metadata,
            )
        with pytest.raises(ValueError, match=r"does not match indexer_kv_dtype"):
            attention.run_indexer(idx_q, idx_k, FakeMetadata(unsupported_dtype))
    result = attention.run_indexer(idx_q, idx_k, bf16_metadata)
    assert result.shape == (2, 4, 16)
    assert captured["idx_q"].data_ptr() == idx_q.data_ptr()
    assert captured["index_k_cache"].dtype == torch.bfloat16
    assert captured["write"][0] == 3
    assert captured["write"][1].data_ptr() == idx_k.data_ptr()
    torch.testing.assert_close(bf16_metadata.cache[0, 0, 0], idx_k[0])
    torch.testing.assert_close(bf16_metadata.cache[1, 0, 0], idx_k[1])


@pytest.mark.parametrize(
    (
        "num_contexts",
        "num_generations",
        "expected_gen_first",
        "expected_ctx_rows",
        "expected_query_len",
    ),
    [(2, 0, 0, 0, None), (1, 1, 1, 1, 1), (0, 2, 0, 0, 1)],
    ids=["prefill", "mixed", "decode"],
)
def test_run_indexer_hands_the_indexer_this_steps_generation_span(
    num_contexts: int,
    num_generations: int,
    expected_gen_first: int,
    expected_ctx_rows: int,
    expected_query_len: int | None,
) -> None:
    """The scorer split follows the batch, and both halves must agree on it.

    The CuTe DSL scorer takes the generation span and the fmha_sm100 proxy the
    context prefix ahead of it, so the first query token of the span is derived
    from the row count and the uniform query length exactly as PhasedFmha
    derives the attention phase's token offset.

    A pure-prefill step reports no split at all rather than one covering every
    row: it has no span, so the proxy plan scores the whole batch and there is
    no boundary for the two halves to disagree about.
    """
    num_tokens, num_index_heads, sparse_index_dim = num_contexts + num_generations, 4, 128
    captured = {}

    class FakeIndexer:
        def select_blocks(self, *args: object, **kwargs: object) -> torch.Tensor:
            del args
            captured.update(kwargs)
            return torch.zeros(num_tokens, 1, 16, dtype=torch.int32)

    class FakeMetadata:
        msa_prefill_n_valid_blocks = torch.ones(num_tokens, dtype=torch.int32)
        msa_n_valid_blocks = torch.ones(num_tokens, dtype=torch.int32)
        msa_worst_case_max_k_tiles = 8
        msa_kv_indices = torch.arange(num_tokens, dtype=torch.int32)
        msa_qo_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_kv_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_qo_offset_cpu = torch.tensor([0], dtype=torch.int32)

        def __init__(self) -> None:
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            self.num_seqs = num_contexts + num_generations
            # The suffix of single-token generation rows prepare() would
            # describe, which is empty for a pure-prefill step.
            self.msa_decode_span = (
                MsaDecodeSpan(row_first=num_contexts, query_len=1) if num_generations > 0 else None
            )
            # Planned for exactly the context rows, so a pure-decode step has
            # no plan at all.
            self.msa_prefill_proxy_plan = ("prefill",) if num_contexts > 0 else None
            self.msa_block_table = torch.zeros(self.num_seqs, 4, dtype=torch.int32)
            self.msa_seq_lens_cuda = torch.zeros(self.num_seqs, dtype=torch.int32)
            self.idx_k_cache = torch.empty(
                num_tokens,
                1,
                sparse_index_dim,
                dtype=torch.bfloat16,
            )

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            del layer_idx
            self.idx_k_cache = idx_k

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            del layer_idx
            return self.idx_k_cache

        def msa_proxy_max_score_view(
            self, num_index_heads: int, max_k_tiles: int, tokens: int
        ) -> torch.Tensor:
            return torch.zeros(num_index_heads, max_k_tiles, tokens)

    attention = SimpleNamespace(
        layer_idx=0,
        m3_config=SimpleNamespace(
            sparse_index_dim=sparse_index_dim,
            num_index_heads=num_index_heads,
            num_kv_heads=1,
        ),
        indexer=FakeIndexer(),
        indexer_kv_dtype="bf16",
    )
    metadata = FakeMetadata()

    result = MiniMaxM3MsaSparseAttention.run_indexer(
        attention,
        torch.zeros(num_tokens, num_index_heads * sparse_index_dim, dtype=torch.bfloat16),
        torch.zeros(num_tokens, sparse_index_dim, dtype=torch.bfloat16),
        metadata,
    )

    assert result.shape == (num_tokens, 1, 16)
    assert captured["gen_token_first"] == expected_gen_first
    assert captured["ctx_rows"] == expected_ctx_rows
    # Set together with the rest of the scorer's inputs, so it says whether the
    # step has a generation span for the scorer to take at all.
    assert captured["decode_query_len"] == expected_query_len


@pytest.mark.parametrize(
    "indexer_dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8_e4m3fn"],
)
def test_msa_proxy_max_score_strided_index_k_matches_packed(
    indexer_dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import _proxy_max_score
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")

    page_size = head_dim = 128
    num_index_heads = 4
    coalescing_scale = 57
    kv_lens_cpu = torch.tensor([1, 130, 257, 128, 511, 1024, 33, 900], dtype=torch.int32)
    pages_per_sequence = (kv_lens_cpu + page_size - 1) // page_size
    num_pages = int(pages_per_sequence.sum().item())

    generator = torch.Generator(device="cuda").manual_seed(0)
    index_k_pool = torch.randn(
        num_pages,
        coalescing_scale,
        1,
        page_size,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(indexer_dtype)
    index_k_strided = index_k_pool[:, 0]
    index_k_packed = index_k_strided.contiguous()
    index_q = torch.randn(
        kv_lens_cpu.numel(),
        num_index_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(indexer_dtype)
    kwargs = {
        "qo_lens_cpu": torch.ones_like(kv_lens_cpu),
        "kv_lens_cpu": kv_lens_cpu,
        "qo_offset_cpu": kv_lens_cpu - 1,
        "kv_indices": torch.arange(num_pages, device="cuda", dtype=torch.int32),
        "sm_scale": head_dim**-0.5,
        "causal": True,
    }

    strided_scores = _proxy_max_score(index_q, index_k_strided, **kwargs)
    packed_scores = _proxy_max_score(index_q, index_k_packed, **kwargs)
    torch.cuda.synchronize()

    assert not index_k_strided.is_contiguous()
    assert index_k_strided.stride(0) == coalescing_scale * page_size * head_dim
    assert torch.equal(strided_scores, packed_scores)


def _expand_slot_rows(block_ids: torch.Tensor, tokens_per_block: int) -> torch.Tensor:
    """req_to_token reference: block_id * tokens_per_block + offset_in_block."""
    within = torch.arange(tokens_per_block, dtype=torch.int64)
    grid = block_ids.to(torch.int64).unsqueeze(-1) * tokens_per_block + within
    return grid.reshape(block_ids.shape[0], -1).to(torch.int32)


def test_build_kv_page_indices_matches_first_slot_of_each_page():
    """The host page table must equal the page ids each request's req_to_token
    row holds at its page boundaries, since both use the manager's
    tokens_per_block as the page size. Rows are ragged (0-padded block ids,
    global and non-contiguous) and one request has no KV at all."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        build_kv_page_indices,
    )

    page_size = 8
    block_ids = torch.tensor(
        [[11, 4, 7, 0], [5, 9, 0, 0], [3, 0, 0, 0], [21, 13, 6, 2]],
        dtype=torch.int32,
    )
    # 3 pages (partial last), 2 pages (exact), no pages, 4 pages.
    kv_lens = torch.tensor([17, 16, 0, 32], dtype=torch.int32)
    req_to_token = _expand_slot_rows(block_ids, page_size)

    reference = torch.cat(
        [
            req_to_token[b, : int(kv_lens[b]) : page_size] // page_size
            for b in range(block_ids.shape[0])
        ]
    )
    page_indices = build_kv_page_indices(block_ids, kv_lens, page_size)

    assert page_indices.dtype == torch.int32
    assert page_indices.tolist() == [11, 4, 7, 5, 9, 21, 13, 6, 2]
    torch.testing.assert_close(page_indices, reference, rtol=0, atol=0)


def test_build_paged_kv_slot_mapping_out_cache_loc_matches_slot_grid():
    """out_cache_loc must name the same slots as indexing req_to_token per new
    token, for a mixed batch of one context request plus decode rows."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
        build_paged_kv_slot_mapping,
    )

    tokens_per_block = 4
    block_ids = torch.tensor([[6, 2, 9], [4, 0, 0], [7, 1, 0]], dtype=torch.int32)

    class FakeCacheManager:
        tokens_per_block = 4

        def get_block_ids_per_seq(self, request_ids):
            assert request_ids == [0, 1, 2]
            return block_ids

    # Request 0 prefills 6 tokens over a 3-token prefix; 1 and 2 decode.
    qo_lens_cpu = torch.tensor([6, 1, 1], dtype=torch.int32)
    kv_lens_cpu = torch.tensor([9, 3, 5], dtype=torch.int32)
    qo_offset_cpu = kv_lens_cpu - qo_lens_cpu

    mapping = build_paged_kv_slot_mapping(
        kv_cache_manager=FakeCacheManager(),
        request_ids=[0, 1, 2],
        qo_lens_cpu=qo_lens_cpu,
        qo_offset_cpu=qo_offset_cpu,
        device=torch.device("cpu"),
    )

    req_to_token = _expand_slot_rows(block_ids, tokens_per_block)
    reference = [
        int(req_to_token[b, int(qo_offset_cpu[b]) + offset])
        for b in range(3)
        for offset in range(int(qo_lens_cpu[b]))
    ]

    torch.testing.assert_close(mapping.req_to_token, req_to_token, rtol=0, atol=0)
    assert mapping.slot_ids.tolist() == [0, 1, 2]
    assert mapping.out_cache_loc.dtype == torch.int32
    assert mapping.out_cache_loc.tolist() == reference
    assert mapping.block_ids_cpu.tolist() == block_ids.tolist()

    # A zero-length CUDA-graph padding row offsets to -1. Its slot is a
    # placeholder no forward reads, so it only has to stay inside the row
    # rather than index off the table.
    padded = build_paged_kv_slot_mapping(
        kv_cache_manager=FakeCacheManager(),
        request_ids=[0, 1, 2],
        qo_lens_cpu=torch.tensor([1, 1, 1], dtype=torch.int32),
        qo_offset_cpu=torch.tensor([0, -1, -1], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    for b, slot in enumerate(padded.out_cache_loc.tolist()):
        assert slot in req_to_token[b].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lazily_allocated_scratch_publishes_the_bound_it_used(monkeypatch):
    """The scratch is normally sized in _create_msa_buffers, but a metadata
    built without sparse params allocates it here on first use. Either way the
    worst-case bound has to be published: msa_proxy_max_score_view shapes the
    view from it, including on a step that skips the proxy plan and so never
    computes a bound of its own.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import msa_backend

    # Both stand in for the fmha_sm100 submodule, which need not be built to
    # test what is done with the bound it returns.
    monkeypatch.setattr(msa_backend, "require_msa_module", lambda: None)
    monkeypatch.setattr(msa_backend, "_worst_case_proxy_max_k_tiles", lambda *a, **kw: 32)

    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.kv_cache_manager = SimpleNamespace()
    metadata.cuda_graph_buffers = None
    metadata.max_num_sequences = 2
    metadata.max_num_tokens = 8
    metadata.msa_max_score = None
    metadata._msa_worst_case_max_k_tiles = 0

    metadata._ensure_msa_decode_scratch_buffers(
        num_index_heads=4,
        max_batch=2,
        capture_graph=False,
        required_max_k_tiles=16,
    )

    assert metadata.msa_worst_case_max_k_tiles == 32
    # The store was sized against that bound, so a view shaped by it fits.
    assert metadata.msa_proxy_max_score_view(4, 32, 2).shape == (4, 32, 2)


def _span_metadata(*, num_contexts=0, qo_lens=(1, 1), kv_lens=(9, 11), is_cuda_graph=False):
    """Metadata with just enough state for _set_decode_span.

    The span is a description of the batch's rows, so seq_lens/kv_lens are
    enough to drive the real msa_*_cpu length properties; no cache pool is
    needed.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.mapping = None
    # Assigned behind the seq_lens property, whose setter would stage a device
    # copy nothing here reads; num_seqs derives from it. The num_contexts
    # setter then runs on_update() over both, as it does in a real step.
    metadata._seq_lens = torch.tensor(qo_lens, dtype=torch.int32)
    metadata.num_contexts = num_contexts
    metadata.kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    metadata.is_cuda_graph = is_cuda_graph
    # prepare() stages these before the span, and the span reads them.
    metadata._stage_host_lengths()
    return metadata


def test_the_decode_span_of_a_pure_decode_step_is_the_whole_batch():
    """A pure-decode step leaves fmha_sm100 nothing, which is what lets
    prepare() skip its plans and its page table entirely."""
    metadata = _span_metadata()

    metadata._set_decode_span()

    assert metadata.msa_decode_span == (0, 1)
    assert metadata.msa_decode_query_len == 1
    assert metadata.msa_max_kv_len == 11
    assert metadata._msa_runs_no_fmha() is True


def test_the_decode_span_of_a_mixed_step_is_its_generation_suffix():
    """A context request does not move the generation rows off their kernels.

    The generation requests are the batch's row suffix, so the decode kernels
    take that span and fmha_sm100 keeps the context prefix.
    """
    # Two context requests (7 and 5 query tokens, the first a chunk of a long
    # prompt) ahead of two decode rows.
    metadata = _span_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))

    metadata._set_decode_span()

    assert metadata.msa_decode_span == (2, 1)
    # fmha_sm100 still runs the context prefix, so its page table stays live.
    assert metadata._msa_runs_no_fmha() is False
    # The trtllm-gen scheduling bound must come from the span's own rows: the
    # 4096-token context row here would inflate a whole-batch maximum by 100x.
    assert metadata.msa_max_kv_len == 40


def test_a_pure_prefill_step_has_no_decode_span():
    """A step with no generation row has nothing for the decode kernels, and
    fmha_sm100 keeps every plan and the page table they read."""
    metadata = _span_metadata(num_contexts=2, qo_lens=(5, 7), kv_lens=(5, 7))

    metadata._set_decode_span()

    assert metadata.msa_decode_span is None
    assert metadata.msa_decode_query_len is None
    assert metadata._msa_runs_no_fmha() is False


def test_plan_rows_narrow_to_the_rows_fmha_sm100_still_runs():
    """The plans must cover exactly those rows.

    A plan is built from the host lengths of the requests it covers, so one that
    claimed the whole batch while only the context prefix ran would schedule the
    kernel over generation rows nothing dispatches to it. The attention plan
    covers the context rows because that is the only phase MsaPrefillFmha runs;
    the indexer's proxy plan covers whatever the CuTe DSL scorer did not take,
    which is the same range plus a pure-prefill step's generation-free batch.
    """
    mixed = _span_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))
    mixed._set_decode_span()
    # The span took the generation suffix, so what is left is the prefix.
    assert mixed._msa_proxy_plan_rows() == (0, 2)
    assert mixed._msa_attn_plan_rows() == (0, 2)

    decode = _span_metadata()
    decode._set_decode_span()
    # Nothing is left to plan on a pure-decode step the kernels fully own.
    assert decode._msa_proxy_plan_rows() is None
    assert decode._msa_attn_plan_rows() is None

    prefill = _span_metadata(num_contexts=2, qo_lens=(5, 7), kv_lens=(5, 7))
    prefill._set_decode_span()
    # No generation row, so the proxy scores every row.
    assert prefill._msa_proxy_plan_rows() == (0, 2)
    assert prefill._msa_attn_plan_rows() == (0, 2)


@pytest.mark.parametrize("head_major", [False, True])
def test_combined_topk_table_preserves_the_requested_backing(head_major):
    """Joining the two halves of a mixed step's table must not change its layout.

    The Triton sparse decode kernel reads the top-k table head-major, so a
    joined table has to permute to a contiguous [num_kv_heads, total_q, topk]
    exactly as the selector's own output does; a token-major join would silently
    hand the kernel a strided view where production hands it a dense one.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import (
        _combined_topk_table,
    )

    num_kv_heads, topk = 2, 16
    ctx = torch.arange(5 * num_kv_heads * topk, dtype=torch.int32).reshape(5, num_kv_heads, topk)
    gen = -ctx[:3] - 1

    combined = _combined_topk_table(ctx, gen, head_major=head_major)

    assert combined.shape == (8, num_kv_heads, topk)
    assert torch.equal(combined[:5], ctx)
    assert torch.equal(combined[5:], gen)
    assert combined.permute(1, 0, 2).is_contiguous() is head_major
    assert combined.is_contiguous() is not head_major


def _phase_libraries():
    """One instance of each library, sharing a stub layer.

    Built without __init__ because neither library's state is under test here;
    _attn_ref is what Fmha.attn reads.
    """
    from tensorrt_llm._torch.attention_backend.fmha.msa_decode import MsaDecodeFmha
    from tensorrt_llm._torch.attention_backend.fmha.msa_prefill import MsaPrefillFmha

    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 3
    attn_ref = weakref.ref(attention)
    libraries = []
    for cls in (MsaDecodeFmha, MsaPrefillFmha):
        library = cls.__new__(cls)
        library._attn_ref = attn_ref
        libraries.append(library)
    return attention, libraries


def _generation_params(attention, metadata, *, seq_offset, input_seq_length):
    """Phase params as PhasedFmha.forward builds them for a generation phase."""
    from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams

    params = FmhaParams(
        attn=attention,
        meta=metadata,
        fwd=SimpleNamespace(
            sparse_runtime_params=SimpleNamespace(sparse_attn_indices=None),
        ),
        workspace=torch.zeros(1),
    )
    params.seq_offset = seq_offset
    params.input_seq_length = input_seq_length
    return params


def test_decode_fmha_runs_the_phase_its_span_describes():
    """The agreeing case is the one production takes, so the check must let it
    through to the dispatch by layer type."""
    attention, (decode, _) = _phase_libraries()
    metadata = SimpleNamespace(
        msa_decode_span=(2, 1),
        num_generations=2,
        msa_block_table=torch.zeros(4, 4, dtype=torch.int32),
        msa_seq_lens_cuda=torch.zeros(4, dtype=torch.int32),
    )
    params = _generation_params(attention, metadata, seq_offset=2, input_seq_length=1)
    dispatched = []
    decode._run_dense = lambda params, block_table, seq_lens: dispatched.append(
        (block_table.shape[0], seq_lens.shape[0])
    )

    decode.run_generation(params)

    # The span's rows, not the whole batch: a dense layer attends the page
    # table of the generation requests alone.
    assert dispatched == [(2, 2)]


def _mixed_batch_sparse_gqa_case(*, page_size, head_dim, num_kv_heads, group, topk, seed):
    """A one-context-plus-three-decode batch for run_msa_prefill_gqa.

    Returns the attention stub, the metadata fields both runs share, q, the
    per-query top-k table, and the batch's context token count. Pages are
    shuffled so a kernel that ignored the block table and indexed the cache by
    logical block would not pass.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        build_kv_page_indices,
        per_token_valid_blocks,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_heads = num_kv_heads * group
    # Row 0 prefills a fresh 260-token prompt; rows 1-3 decode one token each.
    qo_lens_cpu = torch.tensor([260, 1, 1, 1], dtype=torch.int32)
    kv_lens_cpu = torch.tensor([260, 300, 1500, 129], dtype=torch.int32)
    qo_offset_cpu = kv_lens_cpu - qo_lens_cpu
    batch = int(qo_lens_cpu.shape[0])
    total_q = int(qo_lens_cpu.sum())
    max_blocks = int((kv_lens_cpu.max().item() + page_size - 1) // page_size)
    num_pages = batch * max_blocks

    block_table = (
        torch.randperm(num_pages, device="cuda", generator=generator)
        .to(torch.int32)
        .reshape(batch, max_blocks)
    )
    pool = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    q = torch.randn(
        total_q, num_heads * head_dim, device="cuda", generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    # Select each token's earliest valid blocks, ascending with a -1 tail, as
    # the indexer emits them. Deterministic, and valid for the context rows,
    # whose causal extent grows token by token.
    n_valid = per_token_valid_blocks(
        qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
    )
    table = torch.full((total_q, num_kv_heads, topk), -1, dtype=torch.int32)
    for token, valid in enumerate(n_valid.tolist()):
        real = min(topk, max(int(valid), 0))
        table[token, :, :real] = torch.arange(real, dtype=torch.int32)
    # Head-major backing, so the .permute(1, 0, 2) in run_msa_prefill_gqa is the
    # zero-copy view it is in production.
    head_major = table.permute(1, 0, 2).contiguous().cuda()

    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 0
    attention.head_dim = head_dim
    attention.num_heads = num_heads
    attention.q_scaling = 1.0

    fields = dict(
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=page_size,
            get_buffers=lambda layer_idx, kv_layout=None: pool,
        ),
        msa_block_table=block_table,
        msa_seq_lens_cuda=kv_lens_cpu.cuda(),
        msa_kv_indices=build_kv_page_indices(block_table.cpu(), kv_lens_cpu, page_size).cuda(),
        msa_qo_lens_cpu=qo_lens_cpu,
        msa_kv_lens_cpu=kv_lens_cpu,
        msa_qo_offset_cpu=qo_offset_cpu,
        msa_max_kv_len=int(kv_lens_cpu[1:].max()),
        max_num_requests=batch,
    )
    return attention, fields, q, head_major.permute(1, 0, 2), int(qo_lens_cpu[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_batch_generation_span_matches_the_whole_batch_msa_path():
    """Splitting a mixed batch by phase must not change any row's answer.

    The generation rows go to the Triton sparse decode kernel while the context
    rows stay on fmha_sm100 under a context-only plan, so the correctness gate
    is that both halves still agree with a whole-batch fmha_sm100 run.
    """
    from tensorrt_llm._torch.attention_backend.fmha.msa_prefill import run_msa_prefill_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
        msa_package_available,
        msa_paged_kv,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.triton_sparse_decode import (
        minimax_m3_sparse_attn_decode,
    )
    from tensorrt_llm._utils import get_sm_version

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA submodule) required")
    if get_sm_version() not in (100, 103):
        pytest.skip("fmha_sm100 requires SM100/SM103")

    page_size, head_dim = 128, 128
    attention, fields, q, kv_block_indexes, num_ctx_tokens = _mixed_batch_sparse_gqa_case(
        page_size=page_size,
        head_dim=head_dim,
        num_kv_heads=1,
        group=8,
        topk=MSA_REQUIRED_TOPK,
        seed=61,
    )
    total_q = int(q.shape[0])
    metadata = SimpleNamespace(**fields)
    num_heads = attention.num_heads
    sm_scale = head_dim**-0.5

    def run_gqa(output, *, token_first, token_last, row_first, num_rows):
        """One fmha_sm100 call, as MsaPrefillFmha.run_context makes it."""
        run_msa_prefill_gqa(
            attention,
            q[token_first:token_last],
            metadata,
            output[token_first:token_last],
            kv_block_indexes=kv_block_indexes[token_first:token_last],
            plan=None,
            row_first=row_first,
            num_rows=num_rows,
        )

    # Reference: one whole-batch fmha_sm100 call over every row, which is what
    # the split has to reproduce.
    reference = torch.zeros_like(q)
    run_gqa(reference, token_first=0, token_last=total_q, row_first=0, num_rows=4)

    # Split: the context prefix keeps fmha_sm100 under a one-row plan and the
    # generation suffix goes to the Triton kernel, as MsaDecodeFmha dispatches
    # it.
    split = torch.zeros_like(q)
    run_gqa(split, token_first=0, token_last=num_ctx_tokens, row_first=0, num_rows=1)
    k_paged, v_paged = msa_paged_kv(metadata.kv_cache_manager, attention.layer_idx)
    num_gen_tokens = total_q - num_ctx_tokens
    minimax_m3_sparse_attn_decode(
        q[num_ctx_tokens:].view(num_gen_tokens, num_heads, head_dim),
        k_paged,
        v_paged,
        kv_block_indexes[num_ctx_tokens:].permute(1, 0, 2),
        metadata.msa_block_table[1:4],
        metadata.msa_seq_lens_cuda[1:4],
        sm_scale=sm_scale,
        output=split[num_ctx_tokens:].view(num_gen_tokens, num_heads, head_dim),
        decode_query_len=1,
    )
    torch.cuda.synchronize()

    reference = reference.view(total_q, num_heads, head_dim).float()
    split = split.view(total_q, num_heads, head_dim).float()

    assert torch.isfinite(split).all()
    # The context prefix runs on fmha_sm100 either way, but under a 1-row plan
    # rather than a 4-row one, so its work partitioning differs.
    torch.testing.assert_close(
        split[:num_ctx_tokens], reference[:num_ctx_tokens], rtol=1e-2, atol=1e-2
    )
    # The generation rows change kernel outright, so they carry the wider
    # tolerance the Triton-vs-fmha_sm100 A/B uses elsewhere.
    torch.testing.assert_close(
        split[num_ctx_tokens:], reference[num_ctx_tokens:], rtol=6e-2, atol=6e-2
    )
