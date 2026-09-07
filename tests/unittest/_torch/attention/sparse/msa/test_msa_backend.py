# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 integration tests for its MSA sparse attention backend.

These validate MiniMax-M3 backend selection, indexer/cache integration, decode
scratch-buffer sizing, and the paged HND contract passed to the packaged MSA
kernel. Generic block-sparse MQA/GQA numerical coverage lives in the parent
``test_sparse_mqa_gqa.py`` module.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    MiniMaxM3KVCacheManagerV2,
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import msa_paged_kv
from tensorrt_llm._torch.attention.backends.sparse.registry import _resolve_minimax_m3_backend_cls
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_msa_package_availability_installs_cutlass_compatibility_aliases(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
        msa_package_available,
    )

    cute = ModuleType("cutlass.cute")
    cute.core = SimpleNamespace()
    cute.ThrCopy = object()
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
        assert cute.core.ThrCopy is cute.ThrCopy
        assert cute.core.ThrMma is cute.ThrMma
        assert cute.make_fragment is cute.make_rmem_tensor
    finally:
        msa_package_available.cache_clear()


def test_msa_import_preserves_cute_compile_option_selection() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")

    import cutlass.cute as cute

    original_compile = cute.compile
    from fmha_sm100.cute import interface as sparse_interface

    del sparse_interface
    assert cute.compile is original_compile
    assert callable(cute.compile)
    selected_compile = cute.compile[cute.FrontendNext]
    assert callable(selected_compile)


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_availability as avail

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
    # too small for the plan's required 4 * 16 * 2 = 128. One query token per
    # decode row, so the token bound equals the batch here.
    metadata.msa_max_score = torch.zeros(4 * 8 * 2)
    metadata.kv_cache_manager = None
    metadata.max_num_sequences = 2
    metadata.max_num_tokens = 2

    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata._ensure_msa_decode_scratch_buffers(
            num_index_heads=4,
            max_batch=2,
            capture_graph=False,
            required_max_k_tiles=16,
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_msa_paged_kv_preserves_tma_compatible_outer_stride() -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
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

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
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


def test_per_token_valid_blocks_multi_token_decode():
    """Spec-verify decode rows get one entry per query token, following the causal
    ladder inside the verify window.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
        per_token_valid_blocks,
    )

    # One request verifying 4 tokens against kv_len 10 (offset 6): token t
    # attends 7 + t positions; with 2-token blocks that is ceil((7+t)/2).
    qo = torch.tensor([4], dtype=torch.int32)
    kv = torch.tensor([10], dtype=torch.int32)
    off = torch.tensor([6], dtype=torch.int32)
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=2)
    assert n_valid.tolist() == [4, 4, 5, 5]

    # Mixed batch: an ordinary decode row (qo=1) alongside a verify row.
    qo = torch.tensor([1, 3], dtype=torch.int32)
    kv = torch.tensor([9, 6], dtype=torch.int32)
    off = kv - qo
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=4)
    # Row 0: 9 positions -> 3 blocks. Row 1 tokens attend 4, 5, 6 -> 1, 2, 2.
    assert n_valid.tolist() == [3, 1, 2, 2]


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
    import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer as indexer_module
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
        MiniMaxM3SparseConfig,
    )

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
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
        MiniMaxM3SparseConfig,
    )

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
        msa_decode_proxy_plan = None
        msa_eager_proxy_plan = (False, 0, 2, {}, None)
        msa_eager_all_blocks_empty = False
        msa_eager_n_valid_blocks = torch.ones(2, dtype=torch.int32, device="cuda")
        msa_kv_indices = torch.arange(2, dtype=torch.int32, device="cuda")
        msa_qo_lens_cpu = torch.ones(2, dtype=torch.int32)
        msa_kv_lens_cpu = torch.full((2,), 128, dtype=torch.int32)
        msa_qo_offset_cpu = torch.full((2,), 127, dtype=torch.int32)
        num_contexts = 0
        num_generations = 2

        def __init__(self, dtype: torch.dtype) -> None:
            backing = torch.empty(2 * 7, 1, 128, 128, dtype=dtype, device="cuda")
            self.cache = backing[::7]
            self.msa_out_cache_loc = torch.tensor([0, 128], dtype=torch.int32, device="cuda")

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
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
    ("num_contexts", "num_generations", "expected_head_major"),
    [(2, 0, True), (1, 1, False), (0, 2, False)],
)
def test_run_indexer_routes_head_major_output_by_batch_mode(
    num_contexts: int,
    num_generations: int,
    expected_head_major: bool,
) -> None:
    num_tokens, num_index_heads, sparse_index_dim = 3, 4, 128
    captured = {}

    class FakeIndexer:
        def select_blocks(self, *args: object, **kwargs: object) -> torch.Tensor:
            del args
            captured["head_major_output"] = kwargs["head_major_output"]
            return torch.zeros(num_tokens, 1, 16, dtype=torch.int32)

    class FakeMetadata:
        msa_decode_proxy_plan = None
        msa_eager_proxy_plan = ("eager",)
        msa_eager_n_valid_blocks = torch.ones(num_tokens, dtype=torch.int32)
        msa_kv_indices = torch.arange(num_tokens, dtype=torch.int32)
        msa_qo_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_kv_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_qo_offset_cpu = torch.tensor([0], dtype=torch.int32)

        def __init__(self) -> None:
            self.num_contexts = num_contexts
            self.num_generations = num_generations
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
    assert captured["head_major_output"] is expected_head_major


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

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
        _proxy_max_score,
    )
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
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
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
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
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
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


def test_msa_scratch_sizing_covers_spec_verify_tokens():
    """With Eagle3 a decode step has 1 + draft_len query tokens per request, so the
    proxy scratch must be sized by tokens, not by batch.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.kv_cache_manager = None
    # 2 sequences, 4 tokens each (draft_len=3): 8 decode tokens per step.
    metadata.max_num_sequences = 2
    metadata.max_num_tokens = 8
    # Store sized for batch-only sizing (4 heads * 16 k-tiles * 2), which is
    # too small once tokens are accounted for (4 * 16 * 8).
    metadata.msa_max_score = torch.zeros(4 * 16 * 2)

    with pytest.raises(ValueError, match=r"msa_max_score backing store"):
        metadata._ensure_msa_decode_scratch_buffers(
            num_index_heads=4,
            max_batch=2,
            capture_graph=False,
            required_max_k_tiles=16,
        )


def _kv_lens_update_metadata(monkeypatch, *, qo_lens, kv_staged, kv_corrected, page_size):
    """Metadata with just enough staged state for on_update_kv_lens: two requests
    with several query tokens each, optimistic kv_lens from prepare(), then the
    corrected kv_lens. Everything lives on the host.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend

    # The base hook only touches MLA caches this metadata never builds.
    monkeypatch.setattr(msa_backend.TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: None)
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    qo = torch.tensor(qo_lens, dtype=torch.int32)
    batch = int(qo.shape[0])
    total_q = int(qo.sum())
    metadata._seq_lens = qo
    metadata._seq_lens_cuda = qo.clone()
    metadata._num_tokens = total_q
    metadata.kv_lens_cuda = torch.tensor(kv_corrected, dtype=torch.int32)
    metadata.msa_kv_lens_staged = torch.tensor(kv_staged, dtype=torch.int32)
    metadata.kv_cache_manager = SimpleNamespace(tokens_per_block=page_size)
    # Slot table: request b holds slot 100 * b + position.
    width = 16
    metadata.msa_req_to_token = torch.arange(width, dtype=torch.int32).unsqueeze(
        0
    ) + 100 * torch.arange(batch, dtype=torch.int32).unsqueeze(1)
    qo_long = qo.to(torch.long)
    metadata.msa_q_batch_row = torch.repeat_interleave(
        torch.arange(batch, dtype=torch.int32), qo_long
    )
    starts = torch.cumsum(qo_long, 0) - qo_long
    metadata.msa_q_intra = (
        torch.arange(total_q, dtype=torch.int64) - torch.repeat_interleave(starts, qo_long)
    ).to(torch.int32)
    metadata._msa_q_token_starts = (0, *torch.cumsum(qo_long, 0).tolist())
    metadata.msa_out_cache_loc = torch.full((total_q + 3,), -1, dtype=torch.int32)
    metadata.msa_n_valid_blocks = torch.zeros(total_q + 3, dtype=torch.int32)
    metadata._msa_eager_n_valid_blocks = None
    metadata._msa_proxy_plan = None
    metadata._msa_gqa_plan = None
    metadata._msa_dense_plan = None
    metadata._msa_eager_proxy_plan = None
    metadata._msa_eager_gqa_plan = None
    metadata._msa_eager_dense_plan = None
    metadata._msa_fields_ready = True
    metadata._msa_kv_lens_dynamic = True
    return metadata


def test_on_update_kv_lens_rederives_slots_counts_and_plan_lengths(monkeypatch):
    """Request 0 loses one rejected draft token (staged 9 -> corrected 8), request 1
    is unchanged. Slots, valid-block counts and the plans' length rows must
    follow, per request or per query token depending on the row count.
    """
    metadata = _kv_lens_update_metadata(
        monkeypatch, qo_lens=(2, 3), kv_staged=(9, 12), kv_corrected=(8, 12), page_size=4
    )
    # Proxy plan: one row per request, dense length keys. Sparse GQA plan:
    # row-expanded per query token, tagged "MM-SA-Nv" with seqused_k.
    proxy_sub = {
        "kv_segment_lens": torch.zeros(2, dtype=torch.int32),
        "qo_offset": torch.zeros(2, dtype=torch.int32),
    }
    gqa_sub = {"MM-SA-Nv": True, "seqused_k": torch.zeros(5, dtype=torch.int32)}
    metadata._msa_proxy_plan = SimpleNamespace(plan=(False, 2, 2, proxy_sub, None))
    metadata._msa_gqa_plan = SimpleNamespace(plan=(False, 2, 2, gqa_sub, None))

    metadata.on_update_kv_lens()

    # Token positions: request 0 attends 8 tokens with 2 queries -> 6, 7;
    # request 1 attends 12 with 3 queries -> 9, 10, 11.
    assert metadata.msa_out_cache_loc[:5].tolist() == [6, 7, 109, 110, 111]
    assert metadata.msa_out_cache_loc[5:].tolist() == [-1, -1, -1]
    # ceil((pos + 1) / 4) per token.
    assert metadata.msa_n_valid_blocks[:5].tolist() == [2, 2, 3, 3, 3]
    assert proxy_sub["kv_segment_lens"].tolist() == [8, 12]
    assert proxy_sub["qo_offset"].tolist() == [6, 9]
    assert gqa_sub["seqused_k"].tolist() == [7, 8, 10, 11, 12]


def test_on_update_kv_lens_clamps_to_the_staged_lens(monkeypatch):
    """A correction can only shrink lengths; anything above the staged value is
    clamped.
    """
    metadata = _kv_lens_update_metadata(
        monkeypatch, qo_lens=(1, 1), kv_staged=(5, 7), kv_corrected=(9, 7), page_size=4
    )
    dense_sub = {
        "kv_segment_lens": torch.zeros(2, dtype=torch.int32),
        "qo_offset": torch.zeros(2, dtype=torch.int32),
    }
    metadata._msa_eager_dense_plan = (False, 2, 2, dense_sub, None)

    metadata.on_update_kv_lens()

    assert dense_sub["kv_segment_lens"].tolist() == [5, 7]
    assert dense_sub["qo_offset"].tolist() == [4, 6]
    assert metadata.msa_out_cache_loc[:2].tolist() == [4, 106]


def test_on_update_kv_lens_patches_mixed_plans_per_sub_plan(monkeypatch):
    """A mixed plan has two sub-plans over rows [0, split) and [split, batch); each is
    patched over its own rows, per request or per query token.
    """
    metadata = _kv_lens_update_metadata(
        monkeypatch, qo_lens=(1, 3), kv_staged=(6, 12), kv_corrected=(6, 11), page_size=4
    )
    decode_sub = {
        "kv_segment_lens": torch.zeros(1, dtype=torch.int32),
        "qo_offset": torch.zeros(1, dtype=torch.int32),
    }
    prefill_sub = {
        "kv_segment_lens": torch.zeros(3, dtype=torch.int32),
        "qo_offset": torch.zeros(3, dtype=torch.int32),
    }
    metadata._msa_eager_gqa_plan = (True, 1, 2, decode_sub, prefill_sub)

    metadata.on_update_kv_lens()

    assert decode_sub["kv_segment_lens"].tolist() == [6]
    assert decode_sub["qo_offset"].tolist() == [5]
    # Request 1 attends 11 tokens with 3 queries: positions 8, 9, 10.
    assert prefill_sub["kv_segment_lens"].tolist() == [11, 11, 11]
    assert prefill_sub["qo_offset"].tolist() == [8, 9, 10]


def test_on_update_kv_lens_is_a_noop_without_speculative_decoding(monkeypatch):
    """Without speculative decoding the patch and its staging are skipped. The gate
    also sees max_total_draft_tokens, set by update_spec_dec_param, which
    covers draft_len=1 under trtllm-gen (no extra KV tokens, linear-tree spec
    decoding reported disabled).
    """
    metadata = _kv_lens_update_metadata(
        monkeypatch, qo_lens=(1, 1), kv_staged=(4, 6), kv_corrected=(3, 5), page_size=4
    )
    metadata.max_total_draft_tokens = None
    metadata.draft_kv_cache_manager = None
    metadata.is_spec_decoding_enabled = False
    metadata.kv_cache_params = SimpleNamespace(num_extra_kv_tokens=0)
    metadata.runtime_features = None
    assert metadata._msa_kv_lens_may_change() is False
    metadata.max_total_draft_tokens = 1
    assert metadata._msa_kv_lens_may_change() is True
    metadata.max_total_draft_tokens = None
    metadata.kv_cache_params = SimpleNamespace(num_extra_kv_tokens=2)
    assert metadata._msa_kv_lens_may_change() is True

    metadata._msa_kv_lens_dynamic = False
    dense_sub = {
        "kv_segment_lens": torch.zeros(2, dtype=torch.int32),
        "qo_offset": torch.zeros(2, dtype=torch.int32),
    }
    metadata._msa_eager_dense_plan = (False, 2, 2, dense_sub, None)

    metadata.on_update_kv_lens()

    assert metadata.msa_out_cache_loc.tolist() == [-1] * 5
    assert dense_sub["kv_segment_lens"].tolist() == [0, 0]
