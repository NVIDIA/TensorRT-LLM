# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in MSA-v1 q8kv4 decode indexer over the NVFP4 index-K mirror.

Covers the mirror store (against TRT-LLM's fp4_quantize), the selection contract the sparse
decode kernels rely on, agreement with the E4M3 CuTe DSL scorer + select (NVFP4 index-K
changes scores, so agreement is reported and bounded rather than exact), CUDA graph replay,
and the decode attention output end to end. Selection tests skip unless MiniMax MSA is
importable on SM100/SM103.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_v1_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
    MsaIndexer,
    _cutedsl_score,
)
from tensorrt_llm._utils import get_sm_version

PAGE = 128
D = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="The MSA-v1 q8kv4 indexer requires SM100/SM103.",
)

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@pytest.fixture
def msa_enabled(monkeypatch):
    monkeypatch.setenv(msa_v1_indexer.ENV_VAR, "1")
    msa_v1_indexer._msa_classes.cache_clear()
    if not msa_v1_indexer.is_enabled():
        msa_v1_indexer._msa_classes.cache_clear()
        pytest.skip("MiniMax MSA is not importable.")
    yield
    msa_v1_indexer._msa_classes.cache_clear()


def _mirrored_pool(num_pages, *, structured, gen):
    """A mirrored INDEX_KEY pool with E4M3 index-K written and the NVFP4 mirror refreshed."""
    raw = torch.zeros(
        num_pages, 1, PAGE, msa_v1_indexer.MIRRORED_BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda"
    )
    fp8, data, scale = msa_v1_indexer.split_mirrored_index_k(raw)
    k = torch.randn(num_pages, PAGE, D, device="cuda", generator=gen)
    if structured:
        # Pages carry a shared direction, as normalized index keys of related tokens do.
        center = torch.randn(num_pages, 1, D, device="cuda", generator=gen)
        k = 0.8 * center + 0.6 * k
    k = k / k.norm(dim=-1, keepdim=True) * math.sqrt(D) * 0.5
    fp8[:, 0] = k.to(torch.float8_e4m3fn)
    slots = torch.arange(num_pages * PAGE, dtype=torch.int32, device="cuda")
    msa_v1_indexer.quantize_index_k_nvfp4(fp8, data, scale, slots)
    return fp8, data, scale


def _dequant_mirror(data, scale):
    table = torch.tensor(E2M1, device=data.device)
    vals = torch.stack([table[(data & 0xF).long()], table[(data >> 4).long()]], -1).flatten(-2)
    return vals * scale.float().repeat_interleave(16, dim=-1)


def test_mirror_store_matches_fp4_quantize():
    gen = torch.Generator(device="cuda").manual_seed(0)
    fp8, data, scale = _mirrored_pool(6, structured=False, gen=gen)
    rows = fp8[:, 0].reshape(-1, D)
    ref_data, ref_scale = torch.ops.trtllm.fp4_quantize(
        rows.to(torch.bfloat16), torch.ones(1, device="cuda"), 16, False, False
    )
    got_scale = scale.reshape(-1, D // 16).view(torch.uint8)
    scale_mismatch = (got_scale != ref_scale.view(torch.uint8).reshape(-1, D // 16)).float().mean()
    assert scale_mismatch.item() < 5e-3
    got = data.reshape(-1, D // 2)
    mismatch = (got != ref_data.view(torch.uint8).reshape(-1, D // 2)).float().mean().item()
    # fp4_quantize scales by an approximate reciprocal, so a few values on a rounding
    # boundary land on the neighbouring code.
    assert mismatch < 5e-3
    # Round trip: the mirror dequantizes to the E4M3 values within NVFP4 precision.
    deq = _dequant_mirror(data, scale).reshape(-1, D)
    rel = ((deq - rows.float()).norm() / rows.float().norm()).item()
    assert rel < 0.1


def test_mirror_store_skips_negative_slots():
    gen = torch.Generator(device="cuda").manual_seed(1)
    fp8, data, scale = _mirrored_pool(2, structured=False, gen=gen)
    before = data.clone()
    msa_v1_indexer.quantize_index_k_nvfp4(
        fp8, data, scale, torch.full((4,), -1, dtype=torch.int32, device="cuda")
    )
    assert torch.equal(before, data)


def _decode_case(batch, q_len, heads, ctx, *, structured, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_req = -(-ctx // PAGE)
    num_pages = batch * pages_per_req + 2
    fp8, data, scale = _mirrored_pool(num_pages, structured=structured, gen=gen)
    block_table = (
        torch.randperm(num_pages, device="cuda", generator=gen)[: batch * pages_per_req]
        .to(torch.int32)
        .view(batch, pages_per_req)
        .contiguous()
    )
    seq_lens = torch.randint(
        ctx // 2, ctx + 1, (batch,), device="cuda", generator=gen, dtype=torch.int32
    )
    idx_q = torch.randn(batch * q_len, heads, D, device="cuda", generator=gen).to(
        torch.float8_e4m3fn
    )
    return fp8, data, scale, block_table, seq_lens, idx_q


def _ours(fp8, idx_q, block_table, seq_lens, q_len, heads):
    cfg = MiniMaxM3SparseConfig(
        num_q_heads=16 * heads,
        num_kv_heads=heads,
        head_dim=D,
        num_index_heads=heads,
        sparse_index_dim=D,
        block_size=PAGE,
        topk=16,
    )
    total_q = idx_q.shape[0]
    max_score = torch.empty(heads, block_table.shape[1], total_q, device="cuda")
    assert _cutedsl_score(
        idx_q,
        fp8,
        max_score,
        block_table=block_table,
        seq_lens_cuda=seq_lens,
        decode_query_len=q_len,
    )
    qi = torch.arange(q_len, device="cuda", dtype=torch.int32)
    n_valid = ((seq_lens[:, None] - q_len + qi[None]) // PAGE + 1).reshape(-1).to(torch.int32)
    return MsaIndexer(cfg)._select(max_score, n_valid)


def _msa(data, scale, idx_q, block_table, seq_lens, q_len, heads, plan_cache, layer_idx=0):
    return msa_v1_indexer.try_select(
        idx_q,
        data,
        scale,
        block_table=block_table,
        seq_lens=seq_lens,
        decode_query_len=q_len,
        num_kv_heads=heads,
        init_blocks=0,
        local_blocks=1,
        layer_idx=layer_idx,
        plan_cache=plan_cache,
    )


def _check_contract(table, seq_lens, q_len):
    """Ascending valid prefix, -1 tail, local page last, count = min(16, valid blocks)."""
    t = table.permute(1, 0, 2).cpu()  # [H, T, 16]
    for h in range(t.shape[0]):
        for row in range(t.shape[1]):
            r = t[h, row]
            valid = r[r >= 0]
            assert torch.all(r[: valid.numel()] >= 0) and torch.all(r[valid.numel() :] == -1)
            assert torch.all(valid[1:] > valid[:-1])
            b, qi = divmod(row, q_len)
            local = (int(seq_lens[b]) - q_len + qi) // PAGE
            assert int(valid[-1]) == local
            assert valid.numel() == min(16, local + 1)


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("q_len", [1, 4, 8])
def test_selection_contract_and_agreement(msa_enabled, heads, q_len):
    fp8, data, scale, bt, sl, idx_q = _decode_case(
        8, q_len, heads, 40_000, structured=True, seed=10 + q_len
    )
    ours = _ours(fp8, idx_q, bt, sl, q_len, heads)
    got = _msa(data, scale, idx_q, bt, sl, q_len, heads, {})
    assert got.shape == ours.shape and got.dtype == torch.int32
    _check_contract(got, sl.cpu(), q_len)
    # Short requests (every block selected) agree exactly; long ones overlap in most entries.
    overlap = torch.zeros((), device="cuda")
    for h in range(heads):
        a, b = ours[:, h], got[:, h]
        hits = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
        overlap += hits.any(-1).sum() / (a >= 0).sum()
    overlap = (overlap / heads).item()
    print(f"selection overlap heads={heads} q_len={q_len}: {overlap:.4f}")
    assert overlap > 0.85


def test_short_contexts_select_identically(msa_enabled):
    fp8, data, scale, bt, sl, idx_q = _decode_case(4, 4, 2, 1_500, structured=False, seed=3)
    ours = _ours(fp8, idx_q, bt, sl, 4, 2)
    got = _msa(data, scale, idx_q, bt, sl, 4, 2, {})
    assert torch.equal(ours, got)


def test_cuda_graph_replay_tracks_lengths(msa_enabled):
    fp8, data, scale, bt, sl, idx_q = _decode_case(4, 4, 2, 20_000, structured=True, seed=7)
    plan_cache = {}
    eager = _msa(data, scale, idx_q, bt, sl, 4, 2, plan_cache).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _msa(data, scale, idx_q, bt, sl, 4, 2, plan_cache)
    graph.replay()
    assert torch.equal(captured, eager)
    sl.sub_(5_000)
    graph.replay()
    expected = _msa(data, scale, idx_q, bt, sl, 4, 2, {}).clone()
    assert torch.equal(captured, expected)
    _check_contract(captured, sl.cpu(), 4)


def test_declines_outside_contract(msa_enabled):
    fp8, data, scale, bt, sl, idx_q = _decode_case(2, 4, 2, 5_000, structured=False, seed=5)
    kwargs = dict(block_table=bt, seq_lens=sl, decode_query_len=4, layer_idx=0, plan_cache={})
    assert (
        msa_v1_indexer.try_select(
            idx_q, data, scale, num_kv_heads=2, init_blocks=1, local_blocks=1, **kwargs
        )
        is None
    )
    assert (
        msa_v1_indexer.try_select(
            idx_q, data, scale, num_kv_heads=1, init_blocks=0, local_blocks=1, **kwargs
        )
        is None
    )
    assert (
        msa_v1_indexer.try_select(
            idx_q.to(torch.bfloat16), data, scale, num_kv_heads=2, init_blocks=0,
            local_blocks=1, **kwargs,
        )
        is None
    )


@pytest.mark.parametrize("ctx", [1_900, 30_000])
def test_decode_attention_end_to_end(msa_enabled, ctx):
    """Index selection then NVFP4 sparse attention: MSA-v1 indexer vs the E4M3 path.

    Rows whose selections agree attend identical pages and must match bit for bit; at
    1.9K tokens every block is selected, so the whole decode output must match.
    """
    batch, q_len, hkv, group = 8, 4, 2, 16
    fp8, data, scale, bt, sl, idx_q = _decode_case(
        batch, q_len, hkv, ctx, structured=True, seed=21
    )
    gen = torch.Generator(device="cuda").manual_seed(22)
    num_pages = fp8.shape[0]
    kv = torch.randint(0, 256, (num_pages, 2, hkv, PAGE, D // 2), dtype=torch.uint8,
                       device="cuda", generator=gen)
    sc = (torch.rand(num_pages, 2, hkv, PAGE, D // 16, device="cuda", generator=gen) * 0.3
          + 0.05).to(torch.float8_e4m3fn)
    q = torch.randn(batch * q_len, hkv * group, D, device="cuda", generator=gen).to(
        torch.float8_e4m3fn
    )
    one = torch.ones(1, device="cuda")

    def attend(table):
        out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
        minimax_m3_sparse_attn_decode(
            q, kv[:, 0], kv[:, 1], table.permute(1, 0, 2), bt, sl, sm_scale=D**-0.5, output=out,
            decode_query_len=q_len, k_block_scale=sc[:, 0], v_block_scale=sc[:, 1],
            k_global_scale=one, v_global_scale=one.clone(),
        )
        return out.float()

    ours_table = _ours(fp8, idx_q, bt, sl, q_len, hkv)
    msa_table = _msa(data, scale, idx_q, bt, sl, q_len, hkv, {})
    ref, got = attend(ours_table), attend(msa_table)
    same = torch.equal(
        torch.sort(ours_table, -1).values, torch.sort(msa_table, -1).values
    )
    rel = ((got - ref).norm() / ref.norm()).item()
    rows_equal = (torch.sort(ours_table, -1).values == torch.sort(msa_table, -1).values).all(-1)
    rows_equal = rows_equal.all(-1)
    print(
        f"end-to-end decode ctx={ctx}: rel L2 {rel:.4f}, rows with identical selection "
        f"{rows_equal.float().mean().item():.3f}"
    )
    assert torch.isfinite(got).all()
    assert torch.equal(got[rows_equal], ref[rows_equal])
    if ctx <= 16 * PAGE:
        assert same and torch.equal(got, ref)


def _make_manager(num_layers=4):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
        MiniMaxM3KVCacheManagerV2,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            use_kv_cache_manager_v2=True,
        ),
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=D,
        tokens_per_block=PAGE,
        max_seq_len=4096,
        max_batch_size=2,
        max_num_tokens=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        vocab_size=65536,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa",
            indexer_kv_dtype="fp8",
            sparse_disable_index_value=True,
            fuse_qkv_index_projection=False,
        ),
        sparse_layer_ids=list(range(num_layers)),
        sparse_index_dim=D,
    )


def test_cache_manager_mirror_geometry(msa_enabled):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    manager = _make_manager()
    try:
        assert manager.index_k_nvfp4_mirror
        assert manager.get_layer_bytes_per_token(0, Role.INDEX_KEY) == 200
        fp8 = manager.get_index_k_buffer(0)
        data, scale = manager.get_index_k_nvfp4_buffers(0)
        assert fp8.dtype == torch.float8_e4m3fn and tuple(fp8.shape[1:]) == (1, PAGE, D)
        assert tuple(data.shape[1:]) == (PAGE, D // 2) and tuple(scale.shape[1:]) == (PAGE, 8)
        assert fp8.shape[0] == data.shape[0] == scale.shape[0]
        # The three regions of a page are disjoint: a mirror refresh leaves E4M3 rows intact.
        rows = torch.randn(2 * PAGE, D, device="cuda").to(torch.float8_e4m3fn)
        fp8[:2, 0] = rows.view(2, PAGE, D)
        slots = torch.arange(2 * PAGE, dtype=torch.int32, device="cuda")
        msa_v1_indexer.quantize_index_k_nvfp4(fp8, data, scale, slots)
        assert torch.equal(fp8[:2, 0].reshape(-1, D).view(torch.uint8), rows.view(torch.uint8))
        deq = _dequant_mirror(data[:2], scale[:2]).reshape(-1, D)
        assert ((deq - rows.float()).norm() / rows.float().norm()).item() < 0.1
    finally:
        manager.shutdown()


def test_cache_manager_without_env_keeps_fp8_geometry(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

    monkeypatch.delenv(msa_v1_indexer.ENV_VAR, raising=False)
    msa_v1_indexer._msa_classes.cache_clear()
    manager = _make_manager()
    try:
        assert not manager.index_k_nvfp4_mirror
        assert manager.get_layer_bytes_per_token(0, Role.INDEX_KEY) == D
        assert manager.get_index_k_nvfp4_buffers(0) is None
    finally:
        manager.shutdown()
        msa_v1_indexer._msa_classes.cache_clear()
