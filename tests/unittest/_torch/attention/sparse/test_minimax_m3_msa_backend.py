# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate backend selection and decode scratch-buffer sizing without
launching kernels. Numerical parity against the Triton reference is covered
by the SM100 integration accuracy test.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


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


def test_msa_fp8_indexer_config_is_explicit_and_lowered():
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
def test_msa_fp8_cache_converts_live_index_query_before_scoring():
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
    captured = {}

    class FakeIndexer:
        def select_blocks(self, idx_q, idx_k, **kwargs):
            captured["idx_q"] = idx_q
            captured["idx_k"] = idx_k
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

        def __init__(self):
            self.cache = torch.empty(2, 1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")

        def msa_write_idx_k(self, layer_idx, idx_k):
            captured["write"] = (layer_idx, idx_k)

        def msa_idx_k_cache(self, layer_idx):
            captured["read_layer"] = layer_idx
            return self.cache

    idx_q = torch.randn(2, 4 * 128, dtype=torch.bfloat16, device="cuda")
    idx_k = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    result = attention.run_indexer(idx_q, idx_k, FakeMetadata())

    assert result.shape == (2, 4, 16)
    assert captured["idx_q"].dtype == torch.float8_e4m3fn
    assert captured["idx_k"].dtype == torch.float8_e4m3fn
    assert captured["idx_k"].stride(0) > captured["idx_k"].shape[-1]
    assert captured["write"][0] == 3
    assert captured["write"][1].data_ptr() == idx_k.data_ptr()

    # The production fused producer has already inserted K and passes no live
    # K tensor; E4M3 Q must flow to the scorer without a duplicate cache write.
    captured.pop("write")
    fused_q = idx_q.to(torch.float8_e4m3fn)
    result = attention.run_indexer(fused_q, None, FakeMetadata())
    assert result.shape == (2, 4, 16)
    assert captured["idx_q"].data_ptr() == fused_q.data_ptr()
    assert "write" not in captured


@pytest.mark.parametrize(
    ("num_contexts", "num_generations", "expected_head_major"),
    [(2, 0, True), (1, 1, False), (0, 2, False)],
)
def test_run_indexer_routes_head_major_output_by_batch_mode(
    num_contexts, num_generations, expected_head_major
):
    num_tokens, num_index_heads, sparse_index_dim = 3, 4, 128
    captured = {}

    class FakeIndexer:
        def select_blocks(self, *args, **kwargs):
            del args
            captured["head_major_output"] = kwargs["head_major_output"]
            return torch.zeros(num_tokens, 1, 16, dtype=torch.int32)

    class FakeMetadata:
        msa_decode_proxy_plan = None
        msa_eager_proxy_plan = ("eager",)
        msa_eager_all_blocks_empty = False
        msa_eager_n_valid_blocks = torch.ones(num_tokens, dtype=torch.int32)
        msa_kv_indices = torch.arange(num_tokens, dtype=torch.int32)
        msa_qo_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_kv_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_qo_offset_cpu = torch.tensor([0], dtype=torch.int32)

        def __init__(self):
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            self.idx_k_cache = None

        def msa_write_idx_k(self, layer_idx, idx_k):
            del layer_idx
            self.idx_k_cache = idx_k

        def msa_idx_k_cache(self, layer_idx):
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
    )
    metadata = FakeMetadata()

    result = MiniMaxM3MsaSparseAttention.run_indexer(
        attention,
        torch.zeros(num_tokens, num_index_heads * sparse_index_dim),
        torch.zeros(num_tokens, sparse_index_dim),
        metadata,
    )

    assert result.shape == (num_tokens, 1, 16)
    assert captured["head_major_output"] is expected_head_major


def test_msa_proxy_max_score_strided_index_k_matches_packed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import _proxy_max_score
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
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
    )
    index_k_strided = index_k_pool[:, 0]
    index_k_packed = index_k_strided.contiguous()
    index_q = torch.randn(
        kv_lens_cpu.numel(),
        num_index_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
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


def test_msa_scratch_sizing_covers_spec_verify_tokens():
    """Under one-model Eagle3 spec verify a decode step carries
    1 + draft_len query tokens per request, so the proxy scratch must be
    sized by the worst-case decode TOKEN count, not the batch size.
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


def test_per_token_valid_blocks_multi_token_decode():
    """Spec-verify decode rows expose one entry per query TOKEN, walking the
    causal ladder within the verify window."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
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
