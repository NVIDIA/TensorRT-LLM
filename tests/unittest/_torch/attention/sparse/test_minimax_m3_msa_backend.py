# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

Most of these validate backend selection and decode scratch-buffer sizing
without launching kernels; the CUDA-gated tests cover the fused cache-scatter
parity and strided-cache kernel paths. Numerical parity against the Triton
reference is covered by the SM100 integration accuracy test.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import write_kv_slots
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_scatter import (
    fused_write_layer_caches,
)
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


def test_fused_qkv_index_projection_is_explicit_and_shards_index_heads():
    cfg = MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        fuse_qkv_index_projection=True,
    )
    sparse_params = cfg.to_sparse_params()
    metadata_params = cfg.to_sparse_metadata_params(
        pretrained_config=SimpleNamespace(num_attention_heads=64, num_key_value_heads=4)
    )
    mapping = SimpleNamespace(tp_size=2, enable_attention_dp=False)

    assert sparse_params.fuse_qkv_index_projection is True
    assert metadata_params.sharded_head_counts(mapping) == (32, 2)
    assert metadata_params.sharded_index_head_count(mapping) == 2

    compatibility = MiniMaxM3SparseAttentionConfig(implementation="msa").to_sparse_metadata_params(
        pretrained_config=SimpleNamespace(num_attention_heads=64, num_key_value_heads=4)
    )
    assert compatibility.sharded_index_head_count(mapping) == 4

    with pytest.raises(ValueError, match=r"requires the 'msa' implementation"):
        MiniMaxM3SparseAttentionConfig(
            implementation="triton",
            fuse_qkv_index_projection=True,
        )


def test_prepopulated_kv_dispatches_compact_q_and_consumes_marker(monkeypatch):
    from tensorrt_llm._torch.attention_backend.fmha import msa_sparse_gqa

    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    q = torch.empty(3, 8 * 128)
    output = torch.empty_like(q)
    topk = torch.zeros(3, 1, 16, dtype=torch.int32)
    eager_plan = object()

    class FakeMetadata:
        msa_decode_gqa_plan = None
        msa_eager_gqa_plan = eager_plan
        msa_decode_dense_plan = None
        msa_eager_dense_plan = object()
        _msa_prewritten_layer = 7

    captured = {}

    def fake_run(attn, q_arg, k, v, metadata, output_arg, **kwargs):
        captured.update(
            attn=attn,
            q=q_arg,
            k=k,
            v=v,
            metadata=metadata,
            output=output_arg,
            kwargs=kwargs,
        )
        metadata._msa_prewritten_layer = None

    monkeypatch.setattr(msa_sparse_gqa, "run_msa_paged_gqa", fake_run)
    metadata = FakeMetadata()
    attention.forward_prepopulated_kv(
        q,
        metadata,
        AttentionForwardArgs(output=output, topk_indices=topk),
    )

    assert captured["attn"] is attention
    assert captured["q"] is q
    assert captured["k"] is None and captured["v"] is None
    assert captured["metadata"] is metadata
    assert captured["output"] is output
    assert captured["kwargs"]["kv_block_indexes"] is topk
    assert captured["kwargs"]["plan"] is eager_plan
    assert metadata._msa_prewritten_layer is None


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_msa_buffers_include_graph_stable_block_table():
    """The 2-D page table and per-request length the ported decode kernels take
    must come from the graph buffer pool at the manager's worst-case geometry,
    so their addresses survive capture."""
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    max_num_sequences, max_blocks_per_seq = 8, 64

    metadata.kv_cache_manager = SimpleNamespace(
        max_blocks_per_seq=max_blocks_per_seq,
        tokens_per_block=128,
        get_index_k_buffer=lambda layer_idx, kv_layout=None: None,
    )

    class RecordingBuffers:
        def __init__(self):
            self.requested = {}

        def get_buffer(self, tensor_shape, dtype, cache_name, capture_graph):
            self.requested[cache_name] = (tuple(tensor_shape), dtype, capture_graph)
            return torch.zeros(tensor_shape, device="cuda", dtype=dtype)

    buffers = RecordingBuffers()
    metadata.is_cuda_graph = True
    metadata.cuda_graph_buffers = buffers
    metadata.max_num_sequences = max_num_sequences
    metadata.max_num_tokens = 512
    # No sparse params, so the fmha_sm100 proxy scratch is skipped and this
    # exercises only the layer-invariant buffers.
    metadata._msa_params = None

    metadata._create_msa_buffers()

    assert metadata._msa_buffers_ready
    assert metadata.msa_block_table.shape == (max_num_sequences, max_blocks_per_seq)
    assert metadata.msa_seq_lens_cuda.shape == (max_num_sequences,)
    # Reserved from the graph pool at the worst-case geometry, alongside the
    # flat page table the fmha_sm100 path uses.
    assert buffers.requested["msa_block_table"] == (
        (max_num_sequences, max_blocks_per_seq),
        torch.int32,
        True,
    )
    assert buffers.requested["msa_seq_lens_cuda"] == (
        (max_num_sequences,),
        torch.int32,
        True,
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
        # Two decode requests. run_indexer reads both counts to route the
        # selection output head-major or token-major.
        num_contexts = 0
        num_generations = 2
        msa_eager_n_valid_blocks = torch.ones(2, dtype=torch.int32, device="cuda")
        msa_kv_indices = torch.arange(2, dtype=torch.int32, device="cuda")
        msa_qo_lens_cpu = torch.ones(2, dtype=torch.int32)
        msa_kv_lens_cpu = torch.full((2,), 128, dtype=torch.int32)
        msa_qo_offset_cpu = torch.full((2,), 127, dtype=torch.int32)
        num_contexts = 0
        num_generations = 2

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
        msa_eager_n_valid_blocks = torch.ones(num_tokens, dtype=torch.int32)
        msa_kv_indices = torch.arange(num_tokens, dtype=torch.int32)
        msa_qo_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_kv_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
        msa_qo_offset_cpu = torch.tensor([0], dtype=torch.int32)

        def __init__(self):
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            # run_indexer reads the index-K cache before it writes this layer's
            # index-K, so the fake has to hold a tensor from the start, as the
            # persistent cache does in production.
            self.idx_k_cache = torch.zeros(num_tokens, 1, sparse_index_dim)

        def msa_write_idx_k(self, layer_idx, idx_k):
            del layer_idx
            self.idx_k_cache.copy_(idx_k)

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


def _resolution_metadata(
    *, num_contexts=0, qo_lens=(1, 1), kv_lens=(9, 11), is_cuda_graph=False, page_size=128
):
    """Metadata with just enough state for _resolve_decode_kernels.

    The resolver reads only host-side facts, so seq_lens/kv_lens are enough to
    drive the real msa_*_cpu length properties; no cache pool is needed.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata._msa_params = MiniMaxM3SparseAttentionConfig(
        implementation="msa"
    ).to_sparse_metadata_params()
    metadata.mapping = None
    # Assigned behind the seq_lens property, whose setter would stage a device
    # copy the resolver never reads; num_seqs derives from it. The num_contexts
    # setter then runs on_update() over both, as it does in a real step.
    metadata._seq_lens = torch.tensor(qo_lens, dtype=torch.int32)
    metadata.num_contexts = num_contexts
    metadata.kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    metadata.kv_cache_params = None
    metadata.is_cuda_graph = is_cuda_graph
    metadata._msa_captured_resolution = None
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=page_size,
        indexer_kv_dtype="bf16",
        # Present, so the trtllm-gen dense support check passes.
        get_kv_subpage_pool=lambda: None,
    )
    metadata.msa_block_table = torch.zeros(len(qo_lens), 4, dtype=torch.int32)
    metadata.msa_seq_lens_cuda = torch.zeros(len(qo_lens), dtype=torch.int32)
    return metadata


def test_resolve_decode_kernels_commits_on_uniform_decode(monkeypatch):
    """A uniform pure-decode step under the defaults hands every site to a
    ported kernel, which is what lets prepare() skip the fmha_sm100 plans."""
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    metadata = _resolution_metadata()
    # The CuTe DSL runner needs SM100; force the geometry verdict so the test
    # covers the resolution logic on any host.
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    metadata._resolve_decode_kernels()

    assert metadata._msa_use_cutedsl_indexer is True
    assert metadata._msa_use_triton_sparse is True
    assert metadata._msa_use_trtllm_gen_dense is True
    assert metadata._msa_runs_no_fmha() is True
    assert metadata.msa_decode_query_len == 1
    assert metadata.msa_max_kv_len == 11
    # The whole batch is the span, so nothing is left for fmha_sm100.
    span = metadata.msa_decode_span
    assert (span.row_first, span.row_last) == (0, 2)
    assert (span.token_first, span.token_last) == (0, 2)
    assert span.is_mixed is False


def test_resolve_decode_kernels_commits_the_generation_span_of_a_mixed_step(monkeypatch):
    """A context request no longer disqualifies the whole step.

    The generation requests are the batch's row and token suffix, so the ported
    main-attention kernels take that span and fmha_sm100 keeps the context
    prefix. The indexer scorer stays on the proxy for the whole batch, which is
    Phase 1's deliberate limit.
    """
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    # Two context requests (7 and 5 query tokens, the first a chunk of a long
    # prompt) ahead of two decode rows.
    metadata = _resolution_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    metadata._resolve_decode_kernels()

    span = metadata.msa_decode_span
    assert (span.row_first, span.row_last) == (2, 4)
    assert (span.token_first, span.token_last) == (12, 14)
    assert span.query_len == 1
    assert span.is_mixed is True
    assert metadata._msa_use_triton_sparse is True
    assert metadata._msa_use_trtllm_gen_dense is True
    # max_score carries the query tokens in its last dimension, so the context
    # and generation ranges are not separable sub-blocks the proxy and the
    # scorer could each write half of. The proxy keeps the whole batch.
    assert metadata._msa_use_cutedsl_indexer is False
    # fmha_sm100 still runs the context prefix, so its page table stays live.
    assert metadata._msa_runs_no_fmha() is False
    # The trtllm-gen scheduling bound must come from the span's own rows: the
    # 4096-token context row here would inflate a whole-batch maximum by 100x.
    assert metadata.msa_max_kv_len == 40


@pytest.mark.parametrize(
    ("num_contexts", "qo_lens", "kv_lens"),
    [
        # Ragged decode: the ported kernels' token -> request mapping breaks.
        (0, (1, 3), (9, 11)),
        # Ragged generation rows behind a context request. The context request
        # is fine, but these rows still have no single query length.
        (1, (5, 1, 2), (5, 9, 11)),
        # Pure prefill: no generation row for a ported kernel to own.
        (2, (5, 7), (5, 7)),
    ],
    ids=["ragged-decode", "ragged-mixed", "pure-prefill"],
)
def test_resolve_decode_kernels_declines_without_a_uniform_span(
    monkeypatch, num_contexts, qo_lens, kv_lens
):
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    metadata = _resolution_metadata(num_contexts=num_contexts, qo_lens=qo_lens, kv_lens=kv_lens)
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    metadata._resolve_decode_kernels()

    assert metadata.msa_decode_span is None
    assert metadata.msa_decode_query_len is None
    assert metadata._msa_use_cutedsl_indexer is False
    assert metadata._msa_use_triton_sparse is False
    assert metadata._msa_use_trtllm_gen_dense is False
    # Every fmha_sm100 plan is still built, so the page table is still needed.
    assert metadata._msa_runs_no_fmha() is False


def test_fmha_plan_rows_narrow_to_the_context_prefix(monkeypatch):
    """Each site's plan must cover exactly the rows fmha_sm100 still runs.

    on_update_kv_lens patches a plan's length mirrors against the requests it
    was built from, so a plan that claimed the whole batch while only the
    context prefix ran would write the wrong lengths into the kernel.
    """
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        MiniMaxM3MsaSparseAttention.Metadata,
        "_cutedsl_indexer_supported",
        lambda self, **kw: True,
    )

    mixed = _resolution_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))
    mixed._msa_live_batch = 4
    mixed._resolve_decode_kernels()
    # A site fmha_sm100 still owns entirely is planned over the whole batch.
    assert mixed._msa_fmha_plan_rows(False) == (0, 4)
    # A site whose ported kernel took the span is planned over the prefix.
    assert mixed._msa_fmha_plan_rows(True) == (0, 2)

    decode = _resolution_metadata()
    decode._msa_live_batch = 2
    decode._resolve_decode_kernels()
    assert decode._msa_fmha_plan_rows(False) == (0, 2)
    # Nothing is left to plan on a pure-decode step the kernel fully owns.
    assert decode._msa_fmha_plan_rows(True) is None


def test_resolve_decode_kernels_honors_the_msa_kill_switch(monkeypatch):
    """TLLM_M3_*=msa must put the plans back, since it is the only way to
    recover the fmha_sm100 path once prepare() has learned to skip it."""
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.setenv(var, "msa")
    metadata = _resolution_metadata()
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    metadata._resolve_decode_kernels()

    assert metadata._msa_use_cutedsl_indexer is False
    assert metadata._msa_use_triton_sparse is False
    assert metadata._msa_use_trtllm_gen_dense is False
    assert metadata._msa_runs_no_fmha() is False


def test_resolve_decode_kernels_declines_dense_without_subpage_pool(monkeypatch):
    """trtllm-gen needs the flat sub-page pool, and a manager without one must
    keep its dense plan even though the other two sites are ported."""
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    metadata = _resolution_metadata()
    del metadata.kv_cache_manager.get_kv_subpage_pool
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    metadata._resolve_decode_kernels()

    assert metadata._msa_use_trtllm_gen_dense is False
    # One site still on fmha_sm100 keeps the shared page table alive.
    assert metadata._msa_runs_no_fmha() is False


def test_resolution_must_not_change_under_a_captured_graph(monkeypatch):
    """The kernels inside a captured graph are fixed, so a later step that
    resolves differently would stage inputs for a kernel that never runs."""
    for var in ("TLLM_M3_INDEXER_SCORE", "TLLM_M3_SPARSE_DECODE", "TLLM_M3_DENSE_DECODE"):
        monkeypatch.delenv(var, raising=False)
    metadata = _resolution_metadata(is_cuda_graph=True)
    monkeypatch.setattr(type(metadata), "_cutedsl_indexer_supported", lambda self, **kw: True)

    def _step():
        metadata._resolve_decode_kernels()
        metadata._check_capture_stable_resolution()

    _step()
    # Same inputs: the replay agrees with the capture.
    _step()

    monkeypatch.setenv("TLLM_M3_SPARSE_DECODE", "msa")
    with pytest.raises(RuntimeError, match=r"changed under a captured CUDA graph"):
        _step()


def test_indexer_raises_when_a_committed_cutedsl_scorer_declines():
    """prepare() skipped the proxy plan on this step, so a decline has no
    fallback and must surface instead of silently reading a stale page table."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseConfig
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import MsaIndexer

    indexer = MsaIndexer(
        MiniMaxM3SparseConfig(
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=128,
            num_index_heads=4,
            sparse_index_dim=128,
            block_size=128,
            topk=16,
        )
    )
    # block_table/seq_lens left None, so the scorer cannot even be attempted.
    with pytest.raises(RuntimeError, match=r"resolved this step to the CuTe DSL"):
        indexer.select_blocks(
            torch.zeros(2, 4, 128),
            torch.zeros(4, 1, 128, 128),
            idx_sm_scale=1.0,
            kv_indices=torch.zeros(4, dtype=torch.int32),
            max_score=torch.zeros(4, 8, 2),
            require_cutedsl=True,
        )


def test_paged_gqa_raises_when_a_committed_dense_step_declines():
    """The mirror of the indexer guard on the attention side. prepare()
    promised trtllm-gen and dropped the dense plan, so a call site that finds
    the geometry unsupported has nothing left to fall back to."""
    from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_paged_gqa

    num_heads, head_dim, num_pages, page_size = 8, 128, 4, 16
    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 3
    attention.head_dim = head_dim
    attention.num_heads = num_heads
    attention.q_scaling = 1.0

    metadata = SimpleNamespace(
        # No get_kv_subpage_pool, so trtllm-gen declines at the call site.
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda layer_idx, kv_layout=None: torch.zeros(
                num_pages, 2, 1, page_size, head_dim
            )
        ),
        _msa_use_trtllm_gen_dense=True,
        # A resolved pure-decode span, so the decline is the only thing that
        # can send this call to fmha_sm100.
        msa_decode_query_len=1,
    )

    with pytest.raises(RuntimeError, match=r"skipped the fmha_sm100 dense plan"):
        run_msa_paged_gqa(
            attention,
            torch.zeros(2, num_heads * head_dim),
            None,
            None,
            metadata,
            torch.zeros(2, num_heads * head_dim),
            kv_block_indexes=None,
            plan=None,
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
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
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
    # placeholder that on_update_kv_lens re-derives, so it only has to stay
    # inside the row rather than index off the table.
    padded = build_paged_kv_slot_mapping(
        kv_cache_manager=FakeCacheManager(),
        request_ids=[0, 1, 2],
        qo_lens_cpu=torch.tensor([1, 1, 1], dtype=torch.int32),
        qo_offset_cpu=torch.tensor([0, -1, -1], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    for b, slot in enumerate(padded.out_cache_loc.tolist()):
        assert slot in req_to_token[b].tolist()


def _reference_scatter_write(k_cache, v_cache, idx_cache, slots, k, v, idx_k):
    num_tokens = int(slots.shape[0])
    num_heads, head_dim = int(k_cache.shape[1]), int(k_cache.shape[3])
    write_kv_slots(k_cache, slots, k.reshape(num_tokens, num_heads, head_dim), layout="HND")
    write_kv_slots(v_cache, slots, v.reshape(num_tokens, num_heads, head_dim), layout="HND")
    if idx_k is not None:
        write_kv_slots(idx_cache, slots, idx_k.reshape(num_tokens, 1, head_dim), layout="HND")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("with_idx", [True, False])
def test_fused_scatter_matches_reference(cache_dtype, num_kv_heads, with_idx):
    """The fused per-layer cache scatter must match the legacy write_kv_slots
    path exactly on production-shaped inputs: non-contiguous HND cache views
    carved from a pooled allocation and strided source rows sliced from a fused
    projection, including the bf16 -> fp8 cache cast. Asserting on the whole
    pool also catches stray writes outside the targeted slots."""
    torch.manual_seed(0)
    device = "cuda"
    num_pages, tokens_per_block, head_dim = 6, 32, 128
    num_tokens = 17
    inner = num_kv_heads * head_dim

    # Paged HND caches carved from a pool with a coalescing axis, so the
    # views are non-contiguous like production get_buffers(...) output.
    pool = torch.zeros(
        num_pages, 2, num_kv_heads, tokens_per_block, head_dim, dtype=cache_dtype, device=device
    )
    k_cache, v_cache = pool[:, 0], pool[:, 1]
    idx_pool = torch.zeros(
        num_pages, 2, 1, tokens_per_block, head_dim, dtype=torch.bfloat16, device=device
    )
    idx_cache = idx_pool[:, 0]

    # Strided sources: rows sliced out of a wider fused-projection tensor.
    qkv = torch.randn(num_tokens, 3 * inner + 64, dtype=torch.bfloat16, device=device)
    k = qkv[:, :inner]
    v = qkv[:, inner : 2 * inner]
    idx_k = qkv[:, 2 * inner : 2 * inner + head_dim] if with_idx else None

    slots = torch.randperm(num_pages * tokens_per_block, device=device)[:num_tokens].to(torch.int32)

    ref_pool = pool.clone()
    ref_idx_pool = idx_pool.clone()
    _reference_scatter_write(ref_pool[:, 0], ref_pool[:, 1], ref_idx_pool[:, 0], slots, k, v, idx_k)

    wrote = fused_write_layer_caches(
        k_cache, v_cache, idx_cache if with_idx else None, slots, k, v, idx_k
    )
    assert wrote

    torch.testing.assert_close(pool.to(torch.float32), ref_pool.to(torch.float32))
    torch.testing.assert_close(idx_pool, ref_idx_pool)


def _mixed_batch_sparse_gqa_case(*, page_size, head_dim, num_kv_heads, group, topk, seed):
    """A one-context-plus-three-decode batch for run_msa_paged_gqa.

    Returns the attention stub, the metadata fields both runs share, q, the
    per-query top-k table, and the batch's context token count. Pages are
    shuffled so a kernel that ignored the block table and indexed the cache by
    logical block would not pass.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
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
    # Head-major backing, so the .permute(1, 0, 2) in run_msa_paged_gqa is the
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

    The generation rows move off fmha_sm100 and onto the Triton sparse decode
    kernel while the context rows stay behind under a context-only plan, so the
    correctness gate is that both halves still agree with the whole-batch
    fmha_sm100 run that TLLM_M3_SPARSE_DECODE=msa forces. This is the only test
    that covers which kernel produced which output rows.
    """
    from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_paged_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        MSA_REQUIRED_TOPK,
        msa_package_available,
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

    def run(**resolution):
        output = torch.zeros_like(q)
        run_msa_paged_gqa(
            attention,
            q,
            None,
            None,
            SimpleNamespace(**fields, **resolution),
            output,
            kv_block_indexes=kv_block_indexes,
            plan=None,
        )
        torch.cuda.synchronize()
        return output.view(total_q, attention.num_heads, head_dim).float()

    reference = run(_msa_use_triton_sparse=False, msa_decode_span=None)
    split = run(
        _msa_use_triton_sparse=True,
        msa_decode_span=_MsaDecodeSpan(
            row_first=1,
            row_last=4,
            token_first=num_ctx_tokens,
            token_last=total_q,
            query_len=1,
        ),
    )

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
