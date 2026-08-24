# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

Most of these validate backend selection and decode scratch-buffer sizing
without launching kernels; the CUDA-gated tests cover the fused cache-scatter
parity, strided-cache kernel paths, and sparse block-index stride contract.
Numerical parity against the Triton reference is covered by the SM100
integration accuracy test.
"""

import copy
from inspect import signature
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import write_kv_slots
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_scatter import (
    fused_write_layer_caches,
    fused_write_layer_caches_nvfp4,
    fused_write_subpaged_layer_caches,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    msa_ported_decode_active,
)
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_sparse_decode_fixed_stride_page_indptr_matches_expanded_rows():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        _msa_fixed_stride_page_indptr,
    )

    indptr = _msa_fixed_stride_page_indptr(
        torch.tensor([2, 1], dtype=torch.int32), page_table_stride=8
    )
    assert indptr.tolist() == [0, 0, 8, 16]


def test_graph_safe_plan_owners_do_not_alias_shared_buffer_names():
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        _MsaGraphSafePlan,
    )

    class ReusingMetadata:
        def __init__(self):
            self.cuda_graph_buffers = {}
            self.buffers = {}

        def get_empty(self, buffers, shape, *, cache_name, dtype, capture_graph):
            del buffers, capture_graph
            if cache_name not in self.buffers:
                self.buffers[cache_name] = torch.empty(shape, dtype=dtype)
            return self.buffers[cache_name]

    metadata = ReusingMetadata()
    first = _MsaGraphSafePlan(metadata, "msa_gqa_plan", max_batch=4, num_ctas=8, capture_graph=True)
    second = _MsaGraphSafePlan(
        metadata, "msa_gqa_plan", max_batch=4, num_ctas=8, capture_graph=True
    )

    first_ptrs = {tensor.data_ptr() for tensor in first._buf.values()}
    second_ptrs = {tensor.data_ptr() for tensor in second._buf.values()}
    assert first_ptrs.isdisjoint(second_ptrs)


def test_post_init_drops_shallow_copied_plan_and_step_state(monkeypatch):
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        MiniMaxM3MsaSparseAttentionMetadata,
    )
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

    monkeypatch.setattr(TrtllmAttentionMetadata, "__post_init__", lambda self: None)
    monkeypatch.setattr(
        MiniMaxM3MsaSparseAttentionMetadata, "_create_msa_buffers", lambda self: None
    )
    source = MiniMaxM3MsaSparseAttentionMetadata.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    sentinel = object()
    source.sparse_metadata_params = None
    source._msa_gqa_plan = sentinel
    source._msa_eager_gqa_plan = sentinel
    source._msa_decode_span = sentinel
    source._msa_captured_resolution = sentinel

    graph_metadata = copy.copy(source)
    graph_metadata.__post_init__()

    assert source._msa_gqa_plan is sentinel
    assert graph_metadata._msa_gqa_plan is None
    assert graph_metadata._msa_eager_gqa_plan is None
    assert graph_metadata._msa_decode_span is None
    assert graph_metadata._msa_captured_resolution is None


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(implementation="msa").to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is MiniMaxM3MsaSparseAttention


def test_adaptive_decode_requires_msa_implementation():
    with pytest.raises(ValueError, match=r"requires implementation='msa'"):
        MiniMaxM3SparseAttentionConfig(implementation="triton", decode_backend="adaptive")


@pytest.mark.parametrize("tactic", [-1, "msa"])
def test_sparse_decode_tunable_runner_dispatches_and_falls_back_to_triton(monkeypatch, tactic):
    from tensorrt_llm._torch.attention_backend.fmha import msa_sparse_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        sparse_decode_autotuner,
        triton_sparse_decode,
    )
    from tensorrt_llm._torch.autotuner import DistributedTuningStrategy

    calls = []
    monkeypatch.setattr(
        triton_sparse_decode,
        "minimax_m3_sparse_attn_decode",
        lambda q_arg, *args, **kwargs: calls.append(("triton", tuple(q_arg.shape))),
    )
    monkeypatch.setattr(
        msa_sparse_gqa,
        "run_msa_sparse_gqa",
        lambda q_arg, *args, **kwargs: calls.append(("msa", tuple(q_arg.shape))),
    )

    batch_size = 2
    decode_query_len = 4
    total_q = batch_size * decode_query_len
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = page_size = 128
    topk = 16
    q = torch.zeros(total_q, num_q_heads, head_dim, dtype=torch.bfloat16)
    k_paged = torch.zeros(2, num_kv_heads, page_size, head_dim, dtype=torch.bfloat16)
    v_paged = torch.zeros_like(k_paged)
    block_indexes = torch.zeros(total_q, num_kv_heads, topk, dtype=torch.int32)
    block_table = torch.zeros(batch_size, 2, dtype=torch.int32)
    seq_lens = torch.ones(batch_size, dtype=torch.int32)
    output = torch.empty_like(q)
    inputs = [q, k_paged, v_paged, block_indexes, block_table, seq_lens, output]
    runner = sparse_decode_autotuner.MiniMaxM3SparseDecodeRunner(
        decode_query_len=decode_query_len,
        input_layouts=tuple((tensor.dtype, tuple(tensor.stride())) for tensor in inputs),
        sm_scale=head_dim**-0.5,
    )
    assert (
        runner.tuning_config.distributed_tuning_strategy
        == DistributedTuningStrategy.BROADCAST
    )

    runner(
        inputs,
        tactic=tactic,
        plan=object(),
    )

    expected = "triton" if tactic == -1 else tactic
    assert calls == [(expected, (total_q, num_q_heads, head_dim))]


@pytest.mark.parametrize(
    ("decode_backend", "rank_local_batch_size", "expected_backend"),
    [
        ("default", 16, "triton"),
        ("adaptive", 1, "adaptive"),
        ("adaptive", 16, "adaptive"),
    ],
)
def test_pure_decode_dispatches_by_configured_policy(
    monkeypatch, decode_backend, rank_local_batch_size, expected_backend
):
    from tensorrt_llm._torch.attention_backend.fmha import msa_sparse_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
        msa_utils,
        sparse_decode_autotuner,
        triton_sparse_decode,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan

    calls = []
    page_size = head_dim = 128
    num_heads = num_kv_heads = 1
    q = torch.zeros(rank_local_batch_size, num_heads * head_dim)
    output = torch.empty_like(q)
    k_paged = torch.zeros(1, num_kv_heads, page_size, head_dim)
    v_paged = torch.zeros_like(k_paged)
    monkeypatch.setattr(msa_utils, "msa_paged_kv", lambda manager, layer_idx: (k_paged, v_paged))
    monkeypatch.setattr(
        triton_sparse_decode,
        "minimax_m3_sparse_attn_decode",
        lambda q_arg, *args, **kwargs: calls.append(("triton", int(q_arg.shape[0]))),
    )
    monkeypatch.setattr(
        msa_sparse_gqa,
        "run_msa_sparse_gqa",
        lambda q_arg, *args, **kwargs: calls.append(("msa", int(q_arg.shape[0]))),
    )
    monkeypatch.setattr(
        sparse_decode_autotuner,
        "run_adaptive_sparse_decode",
        lambda q_arg, *args, **kwargs: calls.append(("adaptive", int(q_arg.shape[0]))),
    )

    attention = SimpleNamespace(
        layer_idx=0,
        head_dim=head_dim,
        num_heads=num_heads,
        q_scaling=1.0,
        sparse_params=MiniMaxM3SparseAttentionConfig(
            implementation="msa", decode_backend=decode_backend
        ).to_sparse_params(),
    )
    block_table = torch.zeros(rank_local_batch_size, 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        is_cuda_graph=True,
        kv_cache_manager=object(),
        _msa_prewritten_layer=None,
        msa_decode_query_len=1,
        msa_decode_span=_MsaDecodeSpan(
            0,
            rank_local_batch_size,
            0,
            rank_local_batch_size,
            1,
        ),
        msa_block_table=block_table,
        msa_seq_lens_cuda=torch.ones(rank_local_batch_size, dtype=torch.int32),
        msa_kv_indices=block_table.flatten(),
        msa_qo_lens_cpu=torch.ones(rank_local_batch_size, dtype=torch.int32),
        msa_kv_lens_cpu=torch.ones(rank_local_batch_size, dtype=torch.int32),
        msa_qo_offset_cpu=torch.zeros(rank_local_batch_size, dtype=torch.int32),
    )
    topk = torch.zeros(rank_local_batch_size, num_kv_heads, 16, dtype=torch.int32)
    plan = object() if expected_backend in ("msa", "adaptive") else None

    msa_sparse_gqa.run_msa_paged_gqa(
        attention,
        q,
        None,
        None,
        metadata,
        output,
        kv_block_indexes=topk,
        plan=plan,
    )

    assert calls == [(expected_backend, rank_local_batch_size)]


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


def test_msa_main_kv_fp8_probe_retries_legacy_layout_signature() -> None:
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        MiniMaxM3MsaSparseAttentionMetadata,
    )

    class _LegacyManager:
        def get_buffers(self, layer_idx: int) -> torch.Tensor:
            assert layer_idx == 0
            return torch.empty(1, 2, 1, 1, 1, dtype=torch.float8_e4m3fn)

    metadata = SimpleNamespace(kv_cache_manager=_LegacyManager())
    assert MiniMaxM3MsaSparseAttentionMetadata._msa_main_kv_is_fp8(metadata)


def test_msa_main_kv_fp8_probe_does_not_hide_manager_errors() -> None:
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        MiniMaxM3MsaSparseAttentionMetadata,
    )

    class _BrokenManager:
        def get_buffers(self, layer_idx: int, kv_layout: str | None = None) -> torch.Tensor:
            raise RuntimeError(f"broken layer {layer_idx} layout {kv_layout}")

    metadata = SimpleNamespace(kv_cache_manager=_BrokenManager())
    with pytest.raises(RuntimeError, match="broken layer 0 layout HND"):
        MiniMaxM3MsaSparseAttentionMetadata._msa_main_kv_is_fp8(metadata)


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


def test_msa_metadata_clears_padded_cache_slot_tail():
    """A smaller piecewise replay must not reuse a prior step's cache slots."""
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)

    class FakeCacheManager:
        tokens_per_block = 4

        @staticmethod
        def get_buffers(_layer_idx):
            return torch.empty(1)

        @staticmethod
        def get_block_ids_per_seq(_request_ids):
            return torch.tensor([[3]], dtype=torch.int32)

    metadata._msa_buffers_ready = True
    metadata.request_ids = [0]
    metadata.seq_lens = torch.tensor([2], dtype=torch.int32)
    metadata.kv_lens = torch.tensor([2], dtype=torch.int32)
    metadata.kv_cache_params = None
    metadata.kv_cache_manager = FakeCacheManager()
    metadata.msa_out_cache_loc = torch.full((4,), 99, dtype=torch.int32)
    metadata.msa_block_table = torch.zeros((1, 1), dtype=torch.int32)
    metadata.msa_seq_lens_cuda = torch.zeros(1, dtype=torch.int32)
    metadata.msa_cu_q_lens = torch.zeros(2, dtype=torch.int32)
    metadata.msa_cu_kv_lens = torch.zeros(2, dtype=torch.int32)
    metadata.msa_subpage_block_table = None
    metadata.msa_req_to_token = torch.zeros((1, 4), dtype=torch.int32)
    metadata.msa_q_batch_row = torch.zeros(4, dtype=torch.int32)
    metadata.msa_q_intra = torch.zeros(4, dtype=torch.int32)
    metadata.msa_qo_lens_dev = torch.zeros(1, dtype=torch.int32)
    metadata.kv_lens_cuda = None
    metadata._msa_uses_fixed_stride_page_table = lambda: True

    metadata._build_msa_fields()

    assert metadata.msa_out_cache_loc.tolist() == [12, 13, -1, -1]
    assert metadata._msa_fields_ready


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
    """The 2-D page table and per-request length the ported decode kernels take
    must come from the graph buffer pool at the manager's worst-case geometry,
    so their addresses survive capture."""
    metadata = _buffer_metadata()

    metadata._create_msa_buffers()

    assert metadata._msa_buffers_ready
    assert metadata.msa_block_table.shape == (MAX_NUM_SEQUENCES, MAX_BLOCKS_PER_SEQ)
    assert metadata.msa_seq_lens_cuda.shape == (MAX_NUM_SEQUENCES,)
    # Reserved from the graph pool at the worst-case geometry, alongside the
    # flat page table the fmha_sm100 path uses.
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
    """The sub-page expansion is hoisted out of the dense layers into prepare(),
    which runs before any layer is named. That is sound only where every layer
    of the pool packs the same number of sub-pages per slot; where they
    disagree, no table is staged and each dense layer expands its own.
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
    """A mixed step hands the dense kernel only the span's rows, and its block
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
    # test what is done with the number it returns.
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
    assert metadata.msa_proxy_max_score_view(4, 32, 8).shape == (4, 32, 8)


def _resolution_metadata(
    *,
    num_contexts=0,
    qo_lens=(1, 1),
    kv_lens=(9, 11),
    is_cuda_graph=False,
    page_size=128,
    decode_backend="msa",
):
    """Metadata with just enough state for _resolve_decode_kernels.

    The resolver reads only host-side facts, so seq_lens/kv_lens are enough to
    drive the real msa_*_cpu length properties; no cache pool is needed.
    """
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata._msa_params = MiniMaxM3SparseAttentionConfig(
        implementation="msa", decode_backend=decode_backend
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


def _force_cutedsl_supported(monkeypatch):
    """Force the CuTe DSL geometry verdict, which otherwise needs an SM100 host.

    The resolver raises on an unsupported geometry, so without this every
    resolution test would be a test of the runner's availability.
    """
    monkeypatch.setattr(
        MiniMaxM3MsaSparseAttention.Metadata,
        "_cutedsl_indexer_supported",
        lambda self, **kw: True,
    )


def test_resolve_decode_kernels_commits_on_uniform_decode(monkeypatch):
    """A uniform pure-decode step resolves a span over the whole batch, which is
    what lets the CuTe scorer, MSA sparse GQA, and trtllm-gen dense paths agree."""
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata()

    metadata._resolve_decode_kernels()

    assert msa_ported_decode_active(metadata) is True
    assert metadata._msa_runs_no_fmha() is False
    assert metadata._msa_uses_fixed_stride_page_table() is True
    assert metadata.msa_decode_query_len == 1
    assert metadata.msa_max_kv_len == 11
    # The whole batch is the span; sparse GQA still runs through fmha_sm100.
    span = metadata.msa_decode_span
    assert (span.row_first, span.row_last) == (0, 2)
    assert (span.token_first, span.token_last) == (0, 2)
    assert span.is_mixed is False


@pytest.mark.parametrize(
    ("decode_backend", "uses_msa"),
    [
        ("default", False),
        ("adaptive", True),
    ],
)
def test_decode_policy_controls_plan_and_page_table_preparation(
    monkeypatch, decode_backend, uses_msa
):
    batch_size = 2
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata(
        qo_lens=(1,) * batch_size,
        kv_lens=(11,) * batch_size,
        decode_backend=decode_backend,
    )
    metadata._msa_live_batch = batch_size

    metadata._resolve_decode_kernels()

    assert metadata._msa_uses_fixed_stride_page_table() is uses_msa
    assert metadata._msa_runs_no_fmha() is not uses_msa
    assert metadata._msa_fmha_plan_rows() == ((0, batch_size) if uses_msa else None)


def test_resolve_decode_kernels_commits_the_generation_span_of_a_mixed_step(monkeypatch):
    """A context request does not disqualify the step.

    The generation requests are the batch's row and token suffix, so the ported
    kernels take that span and fmha_sm100 keeps the context prefix.
    """
    _force_cutedsl_supported(monkeypatch)
    # Two context requests (7 and 5 query tokens, the first a chunk of a long
    # prompt) ahead of two decode rows.
    metadata = _resolution_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))

    metadata._resolve_decode_kernels()

    span = metadata.msa_decode_span
    assert (span.row_first, span.row_last) == (2, 4)
    assert (span.token_first, span.token_last) == (12, 14)
    assert span.query_len == 1
    assert span.is_mixed is True
    assert msa_ported_decode_active(metadata) is True
    # fmha_sm100 still runs the context prefix, so its page table stays live.
    assert metadata._msa_runs_no_fmha() is False
    assert metadata._msa_uses_fixed_stride_page_table() is False
    # The trtllm-gen scheduling bound must come from the span's own rows: the
    # 4096-token context row here would inflate a whole-batch maximum by 100x.
    assert metadata.msa_max_kv_len == 40


def test_resolve_decode_kernels_resolves_no_span_for_a_pure_prefill(monkeypatch):
    """A step with no generation row has nothing for the ported kernels, and
    fmha_sm100 keeps every plan and the page table they read."""
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata(num_contexts=2, qo_lens=(5, 7), kv_lens=(5, 7))

    metadata._resolve_decode_kernels()

    assert metadata.msa_decode_span is None
    assert metadata.msa_decode_query_len is None
    assert msa_ported_decode_active(metadata) is False
    assert metadata._msa_runs_no_fmha() is False
    assert metadata._msa_uses_fixed_stride_page_table() is False


def test_a_span_without_its_buffers_is_not_active():
    """The ported kernels address the page table and per-request lengths
    directly, so a metadata carrying a query length but neither (the standalone
    kernel tests, which never run prepare()) has not resolved a span."""
    metadata = SimpleNamespace(
        msa_decode_query_len=1, msa_block_table=None, msa_seq_lens_cuda=torch.zeros(1)
    )
    assert msa_ported_decode_active(metadata) is False

    metadata.msa_block_table = torch.zeros(1)
    assert msa_ported_decode_active(metadata) is True

    metadata.msa_seq_lens_cuda = None
    assert msa_ported_decode_active(metadata) is False


def test_span_bounds_stop_at_the_span_rather_than_the_padded_token_count():
    """token_last is the span's own end, not the length of the tensor it came
    with, so a caller can tell a padded step from the bounds alone."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        msa_decode_span_bounds,
    )

    # Eleven speculative decode requests of 4 query tokens, padded to 512.
    metadata = SimpleNamespace(
        msa_decode_span=_MsaDecodeSpan(
            row_first=0, row_last=11, token_first=0, token_last=44, query_len=4
        )
    )
    bounds = msa_decode_span_bounds(metadata, 512)
    assert bounds == (0, 44, 0, 11, 4)
    # Labelled, so a call site taking only some of the five cannot reorder them.
    assert (bounds.token_first, bounds.token_last) == (0, 44)
    assert (bounds.row_first, bounds.row_last, bounds.query_len) == (0, 11, 4)

    # No span (the standalone kernel tests, which never run prepare()): the
    # whole batch is the span, and token_last follows the rows the query length
    # implies rather than a count that does not divide into them.
    assert msa_decode_span_bounds(SimpleNamespace(msa_decode_query_len=4), 513) == (
        0,
        512,
        0,
        128,
        4,
    )

    # Neither: every caller is on the fmha_sm100 path and ignores the bounds.
    assert msa_decode_span_bounds(SimpleNamespace(), 512) == (0, 0, 0, 0, 0)


def test_decode_span_shape_check_names_the_kernel_that_rejected_the_q():
    """The guard both ported decode kernels share. Naming the kernel is the
    point: the alternative is an assert several frames inside one of them."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        check_decode_span_shape,
    )

    check_decode_span_shape("kernel", 44, 11, 4)

    # A piecewise CUDA graph's pad folded into the batch: 128 rows, not 11.
    with pytest.raises(ValueError, match=r"kernel: total_q \(512\) must be batch \(11\)"):
        check_decode_span_shape("kernel", 512, 11, 4)


@pytest.mark.parametrize(
    ("num_contexts", "qo_lens", "kv_lens"),
    [
        # Ragged decode: the ported kernels' token -> request mapping breaks.
        (0, (1, 3), (9, 11)),
        # Ragged generation rows behind a context request. The context request
        # is fine, but these rows still have no single query length.
        (1, (5, 1, 2), (5, 9, 11)),
    ],
    ids=["ragged-decode", "ragged-mixed"],
)
def test_resolve_decode_kernels_raises_on_ragged_generation_rows(
    monkeypatch, num_contexts, qo_lens, kv_lens
):
    """There is no fmha_sm100 decode path left to fall back to, so a span the
    ported kernels cannot serve has to surface rather than cost the step its
    decode throughput silently."""
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata(num_contexts=num_contexts, qo_lens=qo_lens, kv_lens=kv_lens)

    with pytest.raises(NotImplementedError, match=r"one query length"):
        metadata._resolve_decode_kernels()

    assert metadata.msa_decode_span is None


def test_resolve_decode_kernels_raises_without_the_dense_subpage_pool(monkeypatch):
    """trtllm-gen needs the flat sub-page pool, and its dense plan is gone, so a
    manager without one cannot serve the dense layers at all."""
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata()
    del metadata.kv_cache_manager.get_kv_subpage_pool

    with pytest.raises(NotImplementedError, match=r"sub-page pool"):
        metadata._resolve_decode_kernels()


def test_resolve_decode_kernels_raises_when_the_scorer_declines_the_geometry(monkeypatch):
    """Same for the indexer: the proxy pass over the span is gone with it."""
    monkeypatch.setattr(
        MiniMaxM3MsaSparseAttention.Metadata,
        "_cutedsl_indexer_supported",
        lambda self, **kw: False,
    )
    metadata = _resolution_metadata()

    with pytest.raises(NotImplementedError, match=r"CuTe DSL indexer scorer"):
        metadata._resolve_decode_kernels()


def test_fmha_plan_rows_narrow_to_the_context_prefix(monkeypatch):
    """The plans must cover exactly the rows fmha_sm100 still runs.

    on_update_kv_lens patches a plan's length mirrors against the requests it
    was built from, so a plan that claimed the whole batch while only the
    context prefix ran would write the wrong lengths into the kernel.
    """
    _force_cutedsl_supported(monkeypatch)

    mixed = _resolution_metadata(num_contexts=2, qo_lens=(7, 5, 1, 1), kv_lens=(4096, 5, 40, 33))
    mixed._msa_live_batch = 4
    mixed._resolve_decode_kernels()
    # The span took the generation suffix, so the plans cover the prefix.
    assert mixed._msa_fmha_plan_rows() == (0, 2)

    decode = _resolution_metadata()
    decode._msa_live_batch = 2
    decode._resolve_decode_kernels()
    # Sparse GQA uses the whole-batch decode plan; proxy and dense remain ported.
    assert decode._msa_fmha_plan_rows() == (0, 2)

    prefill = _resolution_metadata(num_contexts=2, qo_lens=(5, 7), kv_lens=(5, 7))
    prefill._msa_live_batch = 2
    prefill._resolve_decode_kernels()
    # No span, so fmha_sm100 runs every row and is planned over all of them.
    assert prefill._msa_fmha_plan_rows() == (0, 2)


def test_resolution_must_not_change_under_a_captured_graph(monkeypatch):
    """The kernels inside a captured graph are fixed, so a later step that
    resolves differently would stage inputs for a kernel that never runs."""
    _force_cutedsl_supported(monkeypatch)
    metadata = _resolution_metadata(is_cuda_graph=True)

    def _step():
        metadata._resolve_decode_kernels()
        metadata._check_capture_stable_resolution()

    _step()
    # Same inputs: the replay agrees with the capture.
    _step()

    # A replay whose batch turned mixed. The graph's sparse plan covers a
    # different row range and the eager context plans were not captured.
    metadata._seq_lens = torch.tensor([5, 1, 1], dtype=torch.int32)
    metadata.kv_lens = torch.tensor([5, 9, 11], dtype=torch.int32)
    metadata.num_contexts = 1
    metadata.msa_block_table = torch.zeros(3, 4, dtype=torch.int32)
    metadata.msa_seq_lens_cuda = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match=r"changed under a captured CUDA graph"):
        _step()


def test_indexer_raises_when_a_committed_cutedsl_scorer_declines():
    """prepare() left the proxy plan covering nothing but this step's context
    prefix, so a decline has no fallback for the generation span and must
    surface instead of silently reading a stale page table."""
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
    with pytest.raises(RuntimeError, match=r"CuTe DSL indexer scorer declined the span"):
        indexer.select_blocks(
            torch.zeros(2, 4, 128),
            torch.zeros(4, 1, 128, 128),
            idx_sm_scale=1.0,
            kv_indices=torch.zeros(4, dtype=torch.int32),
            max_score=torch.zeros(4, 8, 2),
            require_cutedsl=True,
        )


def test_select_blocks_scores_only_the_span_it_was_given(monkeypatch):
    """The scorer reads the page table row a query token maps to, so it sees
    the span alone even when idx_q runs longer. Tokens past the span keep a row
    in the returned table, selecting nothing."""
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
    captured = {}

    def fake_cutedsl_score(idx_q, idx_k_paged, max_score, **kwargs):
        del idx_k_paged, max_score, kwargs
        captured["scored_tokens"] = int(idx_q.shape[0])
        return True

    def fake_select_blocks_from_maxscore(max_score_kv, *, topk, n_valid_blocks, **kwargs):
        del max_score_kv, kwargs
        captured["selected_tokens"] = int(n_valid_blocks.shape[0])
        return torch.zeros(n_valid_blocks.shape[0], config.num_kv_heads, topk, dtype=torch.int32)

    monkeypatch.setattr(indexer_module, "_cutedsl_score", fake_cutedsl_score)
    monkeypatch.setattr(
        indexer_module, "select_blocks_from_maxscore", fake_select_blocks_from_maxscore
    )

    # Three decode requests of 4 query tokens inside a 32-token idx_q.
    total_q, span_tokens = 32, 12
    table = indexer.select_blocks(
        torch.zeros(total_q, 4, 128, dtype=torch.bfloat16),
        torch.zeros(3, 1, 128, 128, dtype=torch.bfloat16),
        idx_sm_scale=1.0,
        kv_indices=torch.zeros(3, dtype=torch.int32),
        max_score=torch.zeros(4, 2, span_tokens),
        n_valid_blocks=torch.ones(total_q, dtype=torch.int32),
        block_table=torch.zeros(3, 2, dtype=torch.int32),
        seq_lens_cuda=torch.full((3,), 8, dtype=torch.int32),
        decode_query_len=4,
        require_cutedsl=True,
        gen_token_last=span_tokens,
    )

    assert captured["scored_tokens"] == span_tokens
    assert captured["selected_tokens"] == span_tokens
    assert table.shape == (total_q, config.num_kv_heads, config.topk)
    assert torch.equal(table[span_tokens:], torch.full_like(table[span_tokens:], -1))


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

    combined = _combined_topk_table(ctx, gen, total_q=8, head_major=head_major)

    assert combined.shape == (8, num_kv_heads, topk)
    assert torch.equal(combined[:5], ctx)
    assert torch.equal(combined[5:], gen)
    assert combined.permute(1, 0, 2).is_contiguous() is head_major
    assert combined.is_contiguous() is not head_major


@pytest.mark.parametrize("head_major", [False, True])
def test_combined_topk_table_empties_the_tokens_past_the_span(head_major):
    """The table keeps a row per token so consumers can address it absolutely.
    A row beyond the two halves selects nothing rather than carrying whatever
    the scratch buffer held."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import (
        _combined_topk_table,
    )

    num_kv_heads, topk = 2, 16
    gen = torch.arange(3 * num_kv_heads * topk, dtype=torch.int32).reshape(3, num_kv_heads, topk)

    # No context prefix, three scored tokens, eight rows to fill.
    combined = _combined_topk_table(None, gen, total_q=8, head_major=head_major)

    assert combined.shape == (8, num_kv_heads, topk)
    assert torch.equal(combined[:3], gen)
    assert torch.equal(combined[3:], torch.full_like(combined[3:], -1))
    assert combined.permute(1, 0, 2).is_contiguous() is head_major


def test_paged_gqa_raises_when_a_committed_dense_step_declines():
    """The mirror of the indexer guard on the attention side. The step resolved
    a span and dropped the dense plan, so a call site that finds the geometry
    unsupported has nothing left to fall back to."""
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
        # A resolved pure-decode span, so the decline is the only thing that
        # can send this call to fmha_sm100.
        msa_decode_query_len=1,
        msa_block_table=torch.zeros(2, 1, dtype=torch.int32),
        msa_seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
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


def test_paged_gqa_rejects_a_q_that_outruns_the_span():
    """A q longer than the span no longer describes the batch, so the ported
    kernels would read rows it does not have. Name that here rather than let it
    surface as a kernel-internal assert."""
    from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_paged_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan

    num_heads, head_dim, num_pages, page_size = 8, 128, 4, 16
    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 3
    attention.head_dim = head_dim
    attention.num_heads = num_heads
    attention.q_scaling = 1.0
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda layer_idx, kv_layout=None: torch.zeros(
                num_pages, 2, 1, page_size, head_dim
            )
        ),
        # Two decode requests of one token each, padded up to five tokens.
        msa_decode_span=_MsaDecodeSpan(
            row_first=0, row_last=2, token_first=0, token_last=2, query_len=1
        ),
        msa_decode_query_len=1,
        msa_block_table=torch.zeros(2, 1, dtype=torch.int32),
        msa_seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )

    with pytest.raises(RuntimeError, match=r"5 query tokens for a span ending at token 2"):
        run_msa_paged_gqa(
            attention,
            torch.zeros(5, num_heads * head_dim),
            None,
            None,
            metadata,
            torch.zeros(5, num_heads * head_dim),
            kv_block_indexes=None,
            plan=None,
        )


def test_nvfp4_sparse_dispatch_uses_only_the_msa_csr_path(monkeypatch):
    """Packed NVFP4 bytes must never reach the Triton sparse decode path."""
    import tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa as msa_gqa

    monkeypatch.setattr(msa_gqa, "_MSA_NVFP4_STANDARD_STAGE_ENABLED", False)
    num_tokens, num_heads, head_dim = 2, 8, 128
    pages, kv_heads, page_size = 4, 1, 128
    packed = torch.zeros(pages, 2, kv_heads, page_size, head_dim // 2, dtype=torch.int8)
    scales = torch.zeros(pages, 2, kv_heads, page_size, head_dim // 16, dtype=torch.uint8)
    manager = SimpleNamespace(
        get_buffers=lambda layer_idx, kv_layout=None: packed,
        get_block_scale_buffers=lambda layer_idx, kv_layout=None: scales,
        is_nvfp4_layer=lambda layer_idx: layer_idx == 3,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        _msa_prewritten_layer=3,
        _msa_main_kv_is_nvfp4=lambda: True,
    )
    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 3
    attention.head_dim = head_dim
    attention.num_heads = num_heads
    attention.q_scaling = 1.0
    called = {}

    def fake_nvfp4(q, k, v, scale_buffers, indexes, meta, **kwargs):
        called.update(q=q, k=k, v=v, scales=scale_buffers, indexes=indexes, kwargs=kwargs)

    monkeypatch.setattr(msa_gqa, "run_msa_nvfp4_sparse_gqa", fake_nvfp4)
    q = torch.zeros(num_tokens, num_heads * head_dim)
    output = torch.empty_like(q)
    indexes = torch.zeros(num_tokens, kv_heads, 16, dtype=torch.int32)
    dequant = torch.ones(3, dtype=torch.float32)

    msa_gqa.run_msa_paged_gqa(
        attention,
        q,
        None,
        None,
        metadata,
        output,
        kv_block_indexes=indexes,
        plan=None,
        kv_scale_orig_quant=torch.ones_like(dequant),
        kv_scale_quant_orig=dequant,
    )

    assert called["k"].shape[-1] == head_dim // 2
    assert called["scales"] is scales
    assert called["indexes"] is indexes
    k_scale = called["kwargs"]["k_global_scale"]
    v_scale = called["kwargs"]["v_global_scale"]
    assert k_scale.item() == dequant[1].item()
    assert v_scale.item() == dequant[2].item()
    assert k_scale.data_ptr() % 16 == 0
    assert v_scale.data_ptr() % 16 == 0
    assert called["kwargs"]["k_global_scale_value"] is None
    assert called["kwargs"]["v_global_scale_value"] is None

    # The padded aligned storage is persistent on the layer object rather than
    # allocated once per sparse layer call or once per CUDA-graph replay.
    first_ptrs = (k_scale.data_ptr(), v_scale.data_ptr())
    msa_gqa.run_msa_paged_gqa(
        attention,
        q,
        None,
        None,
        metadata,
        output,
        kv_block_indexes=indexes,
        plan=None,
        kv_scale_orig_quant=torch.ones_like(dequant),
        kv_scale_quant_orig=dequant,
    )
    assert (
        called["kwargs"]["k_global_scale"].data_ptr(),
        called["kwargs"]["v_global_scale"].data_ptr(),
    ) == first_ptrs


def test_fp8_subpaged_dispatch_forwards_dequant_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared Eagle cache path must preserve its FP8 dequant scale."""
    import tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa as msa_gqa
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils as msa_utils
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.trtllm_gen_dense_decode as dense_decode

    monkeypatch.setattr(
        msa_utils, "msa_decode_span_bounds", lambda metadata, num_tokens: (0, 0, 0, 0, 0)
    )
    parameter = signature(dense_decode.minimax_m3_trtllm_gen_dense_attention).parameters[
        "kv_scale_quant_orig"
    ]
    assert parameter.default is None
    captured: dict[str, object] = {}

    def fake_dense_attention(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        dense_decode,
        "minimax_m3_trtllm_gen_dense_attention",
        fake_dense_attention,
    )

    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 3
    attention.head_dim = 128
    attention.num_heads = 8
    attention.q_scaling = 1.0
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            is_nvfp4_layer=lambda layer_idx: False,
            is_fp8_subpaged_layer=lambda layer_idx: layer_idx == 3,
        ),
        _msa_prewritten_layer=3,
    )
    q = torch.zeros(2, attention.num_heads * attention.head_dim)
    output = torch.empty_like(q)
    dequant_scale = torch.ones(3, dtype=torch.float32)

    msa_gqa.run_msa_paged_gqa(
        attention,
        q,
        None,
        None,
        metadata,
        output,
        kv_block_indexes=None,
        plan=None,
        kv_scale_quant_orig=dequant_scale,
    )

    assert captured["kv_scale_quant_orig"] is dequant_scale


@pytest.mark.parametrize("q_heads,kv_heads", [(64, 4), (16, 1)])
def test_nvfp4_standard_stage_uses_preplanned_msa_and_stable_scratch(
    monkeypatch, q_heads, kv_heads
):
    """The opt-in decode route stages selected pages and reuses fixed scratch."""
    import tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa as msa_gqa
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils as msa_utils

    monkeypatch.setattr(msa_gqa, "_MSA_NVFP4_STANDARD_STAGE_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    batch, decode_query_len = 2, 3
    total_q, capacity, head_dim = batch * decode_query_len, 8, 128
    pages, page_size, topk = 16, 128, 16
    q = torch.zeros(total_q, q_heads, head_dim, dtype=torch.float8_e4m3fn)
    packed = torch.zeros(pages, kv_heads, page_size, head_dim // 2, dtype=torch.uint8)
    scales = torch.zeros(pages, 2, kv_heads, page_size, head_dim // 16, dtype=torch.uint8)
    indexes = torch.zeros(total_q, kv_heads, topk, dtype=torch.int32)
    packed_mask = torch.tensor([[[1], [3], [5]], [[1], [3], [7]]], dtype=torch.int32)
    q_batch_row = torch.zeros(capacity, dtype=torch.int32)
    q_batch_row[:total_q] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    q_intra = torch.zeros(capacity, dtype=torch.int32)
    q_intra[:total_q] = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32)
    plan = object()
    stage_calls = []
    fmha_calls = []

    def fail_csr(*args, **kwargs):
        raise AssertionError("standard staged M3 decode must not build CSR metadata")

    def stage_selected_nvfp4_to_fp8(*args, **kwargs):
        stage_calls.append((args, kwargs))

    def standard_fmha(*args, **kwargs):
        fmha_calls.append((args, kwargs))
        return kwargs["out"], None

    fake_sparse = SimpleNamespace(
        build_k2q_csr=fail_csr,
        sparse_atten_nvfp4_kv_func=fail_csr,
        stage_selected_nvfp4_to_fp8=stage_selected_nvfp4_to_fp8,
    )
    fake_msa = SimpleNamespace(sparse=fake_sparse, fmha_sm100=standard_fmha)
    monkeypatch.setattr(msa_utils, "require_msa_module", lambda: fake_msa)
    metadata = SimpleNamespace(
        _msa_live_batch=batch,
        msa_cu_q_lens=torch.tensor([0, 3, 6], dtype=torch.int32),
        msa_cu_kv_lens=torch.tensor([0, 640, 1280], dtype=torch.int32),
        _msa_max_q_len=decode_query_len,
        msa_block_table=torch.zeros(batch, 8, dtype=torch.int32),
        msa_seq_lens_cuda=torch.tensor([640, 640], dtype=torch.int32),
        msa_q_batch_row=q_batch_row,
        msa_q_intra=q_intra,
        spec_decoding_packed_mask=packed_mask,
        is_spec_dec_dynamic_tree=True,
        num_contexts=0,
        num_generations=batch,
        kv_cache_manager=SimpleNamespace(),
    )
    global_scales = torch.ones(2, 4, dtype=torch.float32)
    output = torch.empty(total_q, q_heads, head_dim, dtype=torch.bfloat16)

    msa_gqa.run_msa_nvfp4_sparse_gqa(
        q,
        packed,
        packed,
        scales,
        indexes,
        metadata,
        sm_scale=head_dim**-0.5,
        k_global_scale=global_scales[0, :1],
        v_global_scale=global_scales[1, :1],
        k_global_scale_value=1.25,
        v_global_scale_value=0.75,
        plan=plan,
        out=output,
    )

    assert len(stage_calls) == 2
    assert stage_calls[0][1]["is_v"] is False
    assert stage_calls[1][1]["is_v"] is True
    assert tuple(stage_calls[0][0][5].shape) == (
        total_q * topk,
        kv_heads,
        page_size,
        head_dim,
    )
    assert len(fmha_calls) == 1
    args, kwargs = fmha_calls[0]
    assert args[3] is plan
    assert kwargs["kv_block_indexes"] is indexes
    assert kwargs["sparse_custom_mask"] is packed_mask
    assert kwargs["sparse_custom_mask_q_indices"].data_ptr() == q_intra.data_ptr()
    assert kwargs["sparse_custom_mask_batch_indices"].data_ptr() == q_batch_row.data_ptr()
    assert kwargs["k_scale"] == 1.25
    assert kwargs["v_scale"] == 0.75
    assert kwargs["out"] is output
    physical = kwargs["kv_physical_block_indexes"]
    assert tuple(physical.shape) == (total_q, kv_heads, topk)
    assert torch.equal(physical[:, 0], torch.arange(total_q * topk).view(total_q, topk))

    first_k_scratch = stage_calls[0][0][5]
    first_v_scratch = stage_calls[1][0][5]
    first_physical = physical
    metadata._msa_live_batch = 1
    metadata.num_generations = 1
    smaller_q = q[:decode_query_len]
    smaller_indexes = indexes[:decode_query_len]
    smaller_output = output[:decode_query_len]
    msa_gqa.run_msa_nvfp4_sparse_gqa(
        smaller_q,
        packed,
        packed,
        scales,
        smaller_indexes,
        metadata,
        sm_scale=head_dim**-0.5,
        k_global_scale=global_scales[0, :1],
        v_global_scale=global_scales[1, :1],
        k_global_scale_value=1.25,
        v_global_scale_value=0.75,
        plan=plan,
        out=smaller_output,
    )
    assert stage_calls[2][0][5].data_ptr() == first_k_scratch.data_ptr()
    assert stage_calls[3][0][5].data_ptr() == first_v_scratch.data_ptr()
    assert tuple(stage_calls[2][0][5].shape) == (
        decode_query_len * topk,
        kv_heads,
        page_size,
        head_dim,
    )
    smaller_physical = fmha_calls[1][1]["kv_physical_block_indexes"]
    assert smaller_physical.data_ptr() == first_physical.data_ptr()
    assert tuple(smaller_physical.shape) == (decode_query_len, kv_heads, topk)
    assert len(metadata.kv_cache_manager._msa_nvfp4_selected_scratch_cache) == 1


def test_nvfp4_standard_stage_capacity_ignores_context_token_capacity():
    """Pure-decode scratch is request-bound, not chunked-prefill-bound."""
    import tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa as msa_gqa

    metadata = SimpleNamespace(
        msa_q_batch_row=torch.empty(16384, dtype=torch.int32),
        msa_block_table=torch.empty(128, 1024, dtype=torch.int32),
    )

    assert msa_gqa._nvfp4_standard_stage_capacity(metadata) == 128 * 8


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
    """The fused per-layer cache scatter must match the per-cache write_kv_slots
    writes exactly on production-shaped inputs: non-contiguous HND cache views
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_subpage_scatter_places_tokens_inside_p32_pages():
    torch.manual_seed(17)
    num_slots, pages_per_role, num_heads = 3, 4, 2
    physical_page, head_dim = 32, 128
    k_cache = torch.zeros(
        (num_slots, pages_per_role, num_heads, physical_page, head_dim),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    slots = torch.tensor([0, 31, 32, 127, 128, 255, -1], dtype=torch.int32, device="cuda")
    qkv = torch.randn(
        slots.numel(), 2 * num_heads * head_dim + 13, dtype=torch.bfloat16, device="cuda"
    )
    k = qkv[:, : num_heads * head_dim]
    v = qkv[:, num_heads * head_dim : 2 * num_heads * head_dim]

    assert fused_write_subpaged_layer_caches(k_cache, v_cache, slots, k, v)
    expected_k = k.reshape(slots.numel(), num_heads, head_dim).to(torch.float8_e4m3fn)
    expected_v = v.reshape(slots.numel(), num_heads, head_dim).to(torch.float8_e4m3fn)
    for row, slot in enumerate(slots[:-1].tolist()):
        page, logical_within = divmod(slot, 128)
        subpage, within = divmod(logical_within, physical_page)
        assert torch.equal(k_cache[page, subpage, :, within], expected_k[row])
        assert torch.equal(v_cache[page, subpage, :, within], expected_v[row])
    # The invalid row is masked and therefore cannot touch any cache location.
    assert torch.count_nonzero(k_cache[2]).item() == 0
    assert torch.count_nonzero(v_cache[2]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nvfp4_scatter_writes_physical_p32_data_and_scale_layouts():
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("NVFP4 quantization requires Blackwell")
    torch.manual_seed(13)
    num_slots, pages_per_role, num_heads = 2, 4, 1
    physical_page, head_dim = 32, 128
    packed_dim, scale_cols = head_dim // 2, head_dim // 16
    shape = (num_slots, pages_per_role, num_heads, physical_page, packed_dim)
    scale_shape = (num_slots, pages_per_role, num_heads, physical_page, scale_cols)
    k_cache = torch.zeros(shape, dtype=torch.uint8, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(scale_shape, dtype=torch.uint8, device="cuda")
    v_scale_cache = torch.zeros_like(k_scale_cache)
    slots = torch.tensor([0, 31, 32, 127, 128, -1], dtype=torch.int32, device="cuda")
    k = torch.randn(slots.numel(), head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    inv_scales = torch.ones(3, dtype=torch.float32, device="cuda")

    # The kernel flattens token-row scale offsets, so an exact shape and
    # contiguous columns are insufficient when rows contain hidden padding.
    padded_k_scale_cache = torch.zeros(
        (*scale_shape[:-1], scale_cols + 1), dtype=torch.uint8, device="cuda"
    )[..., :scale_cols]
    padded_v_scale_cache = torch.zeros(
        (*scale_shape[:-1], scale_cols + 1), dtype=torch.uint8, device="cuda"
    )[..., :scale_cols]
    assert padded_k_scale_cache.stride(-1) == 1
    assert padded_k_scale_cache.stride(-2) == scale_cols + 1
    assert padded_v_scale_cache.stride(-2) == scale_cols + 1
    assert not fused_write_layer_caches_nvfp4(
        k_cache,
        v_cache,
        padded_k_scale_cache,
        v_scale_cache,
        None,
        slots,
        k,
        v,
        None,
        inv_scales,
    )
    assert not fused_write_layer_caches_nvfp4(
        k_cache,
        v_cache,
        k_scale_cache,
        padded_v_scale_cache,
        None,
        slots,
        k,
        v,
        None,
        inv_scales,
    )

    wrote = fused_write_layer_caches_nvfp4(
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        None,
        slots,
        k,
        v,
        None,
        inv_scales,
    )
    assert wrote
    expected_k, expected_ksf = torch.ops.trtllm.fp4_quantize(
        k.view(slots.numel(), 1, head_dim), inv_scales[1:2], 16, False, False
    )
    expected_v, expected_vsf = torch.ops.trtllm.fp4_quantize(
        v.view(slots.numel(), 1, head_dim), inv_scales[2:3], 16, False, False
    )
    expected_k = expected_k.view(torch.uint8)
    expected_v = expected_v.view(torch.uint8)
    expected_ksf = expected_ksf.view(slots.numel(), 1, scale_cols)
    expected_vsf = expected_vsf.view(slots.numel(), 1, scale_cols)

    for row, slot in enumerate(slots[:-1].tolist()):
        logical_page, logical_within = divmod(slot, 128)
        subpage, within = divmod(logical_within, physical_page)
        assert torch.equal(k_cache[logical_page, subpage, 0, within], expected_k[row, 0])
        assert torch.equal(v_cache[logical_page, subpage, 0, within], expected_v[row, 0])
        assert torch.equal(k_scale_cache[logical_page, subpage, 0, within], expected_ksf[row, 0])
        v_region = v_scale_cache[logical_page, subpage, 0].view(-1)
        offsets = torch.arange(scale_cols, device="cuda") * 4
        offsets += (within // 4) * (4 * scale_cols) + within % 4
        assert torch.equal(v_region[offsets], expected_vsf[row, 0])


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
    attention.sparse_params = MiniMaxM3SparseAttentionConfig(
        implementation="msa", decode_backend="msa"
    ).to_sparse_params()

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
def test_pure_decode_sparse_gqa_uses_preplanned_msa_and_matches_triton(monkeypatch):
    """The production pure-decode dispatcher must take the MSA sparse plan.

    Exercise AgentX DQL4, a production-strided FP8 Q view, distinct per-token
    block-index slices, and CUDA-graph replay. The Triton result remains the
    numerical reference but is made unavailable at dispatch time, so a silent
    routing fallback fails the test.
    """
    from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_paged_gqa
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import triton_sparse_decode
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import _MsaDecodeSpan
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        MSA_REQUIRED_TOPK,
        msa_package_available,
        require_msa_module,
    )
    from tensorrt_llm._utils import get_sm_version

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA submodule) required")
    if get_sm_version() not in (100, 103):
        pytest.skip("fmha_sm100 requires SM100/SM103")

    batch, dql = 8, 4
    num_heads, num_kv_heads = 64, 4
    page_size = head_dim = 128
    seq_len = 4096
    num_blocks = seq_len // page_size
    total_q = batch * dql
    num_pages = batch * num_blocks
    generator = torch.Generator(device="cuda").manual_seed(20260814)

    block_table = (
        torch.randperm(num_pages, device="cuda", generator=generator)
        .to(torch.int32)
        .reshape(batch, num_blocks)
    )
    pool = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    fused_qkv = torch.randn(
        total_q,
        q_width + 2 * kv_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    q = fused_qkv[:, :q_width]
    q_view = q.reshape(total_q, num_heads, head_dim)
    assert not q_view.is_contiguous()

    offsets = torch.arange(total_q, device="cuda", dtype=torch.int32) % 4
    selected = (
        offsets[:, None]
        + torch.arange(MSA_REQUIRED_TOPK, device="cuda", dtype=torch.int32)[None, :]
    )
    token_major = selected[:, None, :].expand(total_q, num_kv_heads, -1)
    head_major = token_major.permute(1, 0, 2).contiguous()
    topk_indices = head_major.permute(1, 0, 2)
    assert not topk_indices.is_contiguous()

    seq_lens_cuda = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    qo_lens_cpu = torch.full((batch,), dql, dtype=torch.int32)
    kv_lens_cpu = torch.full((batch,), seq_len, dtype=torch.int32)
    qo_offset_cpu = kv_lens_cpu - qo_lens_cpu
    fmha_sm100 = require_msa_module()
    plan = fmha_sm100.fmha_sm100_plan(
        qo_lens_cpu,
        kv_lens_cpu,
        num_heads,
        num_kv_heads=num_kv_heads,
        qo_offset=qo_offset_cpu,
        page_size=page_size,
        kv_block_num=MSA_REQUIRED_TOPK,
        causal=True,
        num_kv_splits=1,
        use_fp8_kvcache=True,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    reference = torch.empty(total_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    triton_sparse_decode.minimax_m3_sparse_attn_decode(
        q_view,
        pool[:, 0],
        pool[:, 1],
        head_major,
        block_table,
        seq_lens_cuda,
        sm_scale=head_dim**-0.5,
        output=reference,
        decode_query_len=dql,
    )

    attention = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    attention.layer_idx = 0
    attention.head_dim = head_dim
    attention.num_heads = num_heads
    attention.q_scaling = 1.0
    attention.sparse_params = MiniMaxM3SparseAttentionConfig(
        implementation="msa", decode_backend="msa"
    ).to_sparse_params()
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=page_size,
            get_buffers=lambda layer_idx, kv_layout=None: pool,
        ),
        _msa_prewritten_layer=None,
        msa_decode_query_len=dql,
        msa_decode_span=_MsaDecodeSpan(0, batch, 0, total_q, dql),
        msa_block_table=block_table,
        msa_seq_lens_cuda=seq_lens_cuda,
        msa_kv_indices=block_table.flatten(),
        msa_qo_lens_cpu=qo_lens_cpu,
        msa_kv_lens_cpu=kv_lens_cpu,
        msa_qo_offset_cpu=qo_offset_cpu,
    )
    output = torch.empty(total_q, q_width, device="cuda", dtype=torch.bfloat16)

    def run_candidate():
        run_msa_paged_gqa(
            attention,
            q,
            None,
            None,
            metadata,
            output,
            kv_block_indexes=topk_indices,
            plan=plan,
        )

    # The reference is complete; any dispatch through Triton from here is a
    # routing regression rather than a numerical fallback.
    monkeypatch.setattr(
        triton_sparse_decode,
        "minimax_m3_sparse_attn_decode",
        lambda *args, **kwargs: pytest.fail("pure decode routed to Triton"),
    )
    for _ in range(3):
        run_candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_candidate()
    graph.replay()
    torch.cuda.synchronize()

    actual = output.view(total_q, num_heads, head_dim)
    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparse_decode_cached_worklist_matches_fresh_plan_at_new_lengths():
    """A cached decode worklist plus patched live fields must equal replanning."""
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        _msa_fixed_stride_page_indptr,
        _MsaGraphSafePlan,
    )
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        MSA_REQUIRED_TOPK,
        build_kv_page_indices,
        msa_package_available,
        require_msa_module,
    )
    from tensorrt_llm._utils import get_sm_version

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA submodule) required")
    if get_sm_version() not in (100, 103):
        pytest.skip("fmha_sm100 requires SM100/SM103")

    batch, dql = 2, 4
    num_heads, num_kv_heads = 64, 4
    page_size = head_dim = 128
    total_q = batch * dql
    first_lens = torch.tensor([4096, 4096], dtype=torch.int32)
    live_lens = torch.tensor([4096, 8192], dtype=torch.int32)
    qo_lens = torch.full((batch,), dql, dtype=torch.int32)
    max_blocks = int((int(live_lens.max()) + page_size - 1) // page_size)
    num_pages = batch * max_blocks
    block_table = torch.arange(num_pages, dtype=torch.int32).reshape(batch, max_blocks)
    compact_indices = build_kv_page_indices(block_table, live_lens, page_size).cuda()
    fixed_indices = block_table.cuda().flatten()

    generator = torch.Generator(device="cuda").manual_seed(20260814)
    q = torch.randn(
        total_q,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=generator).to(
        torch.float8_e4m3fn
    )
    selected = torch.arange(MSA_REQUIRED_TOPK, device="cuda", dtype=torch.int32)
    topk = selected.expand(total_q, num_kv_heads, -1)

    fmha_sm100 = require_msa_module()

    def plan_for(kv_lens):
        return fmha_sm100.fmha_sm100_plan(
            qo_lens,
            kv_lens,
            num_heads,
            num_kv_heads=num_kv_heads,
            qo_offset=kv_lens - qo_lens,
            page_size=page_size,
            kv_block_num=MSA_REQUIRED_TOPK,
            causal=True,
            num_kv_splits=1,
            use_fp8_kvcache=True,
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    class FakeMetadata:
        cuda_graph_buffers = {}

        @staticmethod
        def get_empty(buffers, shape, *, cache_name, dtype, capture_graph):
            del buffers, cache_name, capture_graph
            return torch.empty(shape, dtype=dtype, device="cuda")

    first_plan = plan_for(first_lens)
    fresh_plan = plan_for(live_lens)
    signature = (
        tuple(qo_lens.tolist()),
        num_heads,
        num_kv_heads,
        MSA_REQUIRED_TOPK,
        page_size,
        True,
        max_blocks,
    )
    owner = _MsaGraphSafePlan(
        FakeMetadata(),
        "test_gqa_plan",
        max_batch=total_q,
        num_ctas=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        capture_graph=False,
    )
    fixed_indptr = _msa_fixed_stride_page_indptr(qo_lens, max_blocks)
    owner.refresh(
        first_plan,
        cache_signature=signature,
        stable_overrides={"kv_page_indptr": fixed_indptr},
    )
    cached_plan = owner.reuse_sparse_decode(signature)
    assert cached_plan is not None
    # Production's captured on_update_kv_lens() patches these two tensors on
    # device before forward. Simulate that boundary directly; deliberately do
    # not patch kv_segment_offsets, which paged sparse load does not consume.
    for key in ("kv_segment_lens", "qo_offset"):
        cached_plan[3][key].copy_(fresh_plan[3][key])
    torch.cuda.synchronize()
    torch.testing.assert_close(cached_plan[3]["kv_page_indptr"], fixed_indptr.cuda())

    fresh_out = torch.empty(total_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cached_out = torch.empty_like(fresh_out)
    common = dict(
        kv_block_indexes=topk,
        sm_scale=head_dim**-0.5,
        output_maxscore=False,
    )
    fmha_sm100.fmha_sm100(q, k, v, fresh_plan, kv_indices=compact_indices, out=fresh_out, **common)
    fmha_sm100.fmha_sm100(q, k, v, cached_plan, kv_indices=fixed_indices, out=cached_out, **common)
    torch.cuda.synchronize()
    torch.testing.assert_close(cached_out, fresh_out, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_batch_generation_span_matches_the_whole_batch_msa_path():
    """Splitting a mixed batch by phase must not change any row's answer.

    The generation rows move off fmha_sm100 and onto the Triton sparse decode
    kernel while the context rows stay behind under a context-only plan, so the
    correctness gate is that both halves still agree with a whole-batch
    fmha_sm100 run, which a metadata carrying no span still takes. This is the
    only test that covers which kernel produced which output rows.
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

    reference = run(msa_decode_span=None)
    split = run(
        # A property of the span on real metadata, and what
        # msa_ported_decode_active reads; this fake carries both.
        msa_decode_query_len=1,
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


def _run_msa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: tuple,
    kv_indices: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    from fmha_sm100.api import fmha_sm100

    out = torch.empty_like(q)
    returned, _ = fmha_sm100(
        q,
        k,
        v,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=selected,
        out=out,
        sm_scale=128.0**-0.5,
        output_maxscore=False,
    )
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(out).all()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("query_len", [1, 5])
@pytest.mark.parametrize("num_kv_splits", [1, 2])
def test_msa_sparse_attention_honors_noncontiguous_block_indexes(
    query_len: int,
    num_kv_splits: int,
) -> None:
    from fmha_sm100.api import fmha_sm100_plan

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    total_selector_rows = query_len + 2
    q = torch.zeros((query_len, 32, 128), dtype=torch.bfloat16, device=device)
    k = torch.zeros((8, 2, 128, 128), dtype=torch.bfloat16, device=device)
    v = torch.empty_like(k)
    for page in range(8):
        v[page].fill_(page + 1)
    kv_indices = torch.arange(8, dtype=torch.int32, device=device)

    intended = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    poison = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device)
    logical = intended.expand(total_selector_rows, 2, 4).clone()
    # These rows are outside the logical slice but occupy addresses reached by
    # a pointer-only contiguous read for later query/head pairs.
    logical[:2, 1, :] = poison
    backing = logical.permute(1, 0, 2).contiguous()
    selected_strided = backing.permute(1, 0, 2)[-query_len:]
    selected_contiguous = logical[-query_len:].contiguous()

    assert selected_strided.stride() == (4, total_selector_rows * 4, 1)
    assert not selected_strided.is_contiguous()
    assert torch.equal(selected_strided, selected_contiguous)

    plan = fmha_sm100_plan(
        torch.tensor([query_len], dtype=torch.int32),
        torch.tensor([1024], dtype=torch.int32),
        32,
        num_kv_heads=2,
        qo_offset=torch.tensor([1024 - query_len], dtype=torch.int32),
        num_kv_splits=num_kv_splits,
        page_size=128,
        output_maxscore=False,
        kv_block_num=4,
        causal=True,
        device=device,
    )
    short_plan = plan[3]
    assert short_plan["MM-SA-Nv"] is False
    assert short_plan["num_kv_splits"] == num_kv_splits

    expected = _run_msa_sparse_attention(q, k, v, plan, kv_indices, selected_contiguous)
    actual = _run_msa_sparse_attention(q, k, v, plan, kv_indices, selected_strided)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
