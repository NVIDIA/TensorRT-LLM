# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA context-row select and mixed-step host trims, bitwise against the stock paths.

* the prefill tile select (minimax_m31_select_prefill_tile, sm_103) against minimax_m3_select_blocks;
* the pure-prefill select_blocks path against the stock path;
* the reused k2q CSR workspace against sparse.build_k2q_csr;
* the NumPy per_token_valid_blocks against the torch expansion it replaced;
"""

import math
import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa import msa_indexer
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    msa_package_available,
    per_token_valid_blocks,
    select_blocks_from_maxscore,
)

PAGE = 128
CUDA = torch.cuda.is_available()
SM103 = CUDA and torch.cuda.get_device_capability() == (10, 3)
SM100F = CUDA and torch.cuda.get_device_capability()[0] == 10

needs_sm100f = pytest.mark.skipif(
    not (SM100F and msa_package_available()),
    reason="fmha_sm100 needs SM100-family and the MSA package",
)


def _tile_op_available() -> bool:
    return SM103 and msa_indexer._family_op("minimax_m31_select_prefill_tile") is not None


needs_tile = pytest.mark.skipif(
    not _tile_op_available(), reason="minimax_m31_select_prefill_tile needs sm_103"
)


def _per_token_valid_blocks_torch(qo, kv, off, *, causal, block_size):
    """The torch expansion per_token_valid_blocks used before its NumPy port."""
    qo = qo.to(torch.long)
    kv = kv.to(torch.long)
    batch = int(qo.shape[0])
    total = int(qo.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.long)
    batch_row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), qo)
    starts = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        starts[1:] = torch.cumsum(qo, 0)[:-1]
    intra = torch.arange(total, dtype=torch.long) - starts[batch_row]
    kv_per = kv[batch_row]
    if causal:
        o = off.to(torch.long)[batch_row] if off is not None else (kv - qo)[batch_row]
        eff = torch.minimum(o + intra + 1, kv_per)
    else:
        eff = kv_per
    return (eff + block_size - 1) // block_size


def _rows(prefixes, chunks):
    qo = torch.tensor(chunks, dtype=torch.int32)
    off = torch.tensor(prefixes, dtype=torch.int32)
    return qo, off + qo, off


def _plan_k_tiles(max_kv_len: int) -> int:
    """fmha_sm100's max_k_tiles: the block count rounded up to a multiple of 128."""
    return math.ceil(math.ceil(max_kv_len / PAGE) / 128) * 128


# --------------------------------------------------------------------------------------------- per_token_valid_blocks


@pytest.mark.cpu_only
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("with_offset", [True, False])
def test_numpy_per_token_valid_blocks_matches_torch(causal, with_offset):
    gen = torch.Generator().manual_seed(7)
    for _ in range(50):
        batch = int(torch.randint(1, 9, (1,), generator=gen))
        qo = torch.randint(0, 300, (batch,), generator=gen, dtype=torch.int32)
        off = torch.randint(0, 700_000, (batch,), generator=gen, dtype=torch.int32)
        kv = off + qo + torch.randint(0, 3, (batch,), generator=gen, dtype=torch.int32)
        args = (qo, kv, off if with_offset else None)
        got = per_token_valid_blocks(*args, causal=causal, block_size=PAGE)
        want = _per_token_valid_blocks_torch(*args, causal=causal, block_size=PAGE)
        assert got.dtype == torch.long
        assert torch.equal(got, want)
    empty = torch.zeros(2, dtype=torch.int32)
    assert per_token_valid_blocks(empty, empty, None, causal=True, block_size=PAGE).numel() == 0


# ------------------------------------------------------------------------------------------------- prefill tile select


def _scores(
    kind: str, num_kv_heads: int, num_blocks: int, total_q: int, n_valid: torch.Tensor, seed: int
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (num_kv_heads, num_blocks, total_q)
    if kind == "randn":
        scores = torch.randn(shape, generator=gen, device="cuda")
    elif kind == "ties":
        # Four distinct values: hundreds of blocks share the 16th-largest score.
        scores = torch.randint(0, 4, shape, generator=gen, device="cuda").float()
    elif kind == "fp16ties":
        # Proxy-like magnitudes rounded to a coarse grid, ties across the whole range.
        scores = (torch.randn(shape, generator=gen, device="cuda") * 8).half().float()
    elif kind == "nonfinite":
        scores = torch.randn(shape, generator=gen, device="cuda")
        pick = torch.rand(shape, generator=gen, device="cuda")
        scores[pick < 0.01] = float("inf")
        scores[(pick >= 0.01) & (pick < 0.02)] = float("-inf")
        scores[(pick >= 0.02) & (pick < 0.025)] = float("nan")
        scores[(pick >= 0.025) & (pick < 0.03)] = -0.0
        scores[(pick >= 0.03) & (pick < 0.035)] = 0.0
    else:
        raise AssertionError(kind)
    # Entries past each query's valid blocks are never read; poison them.
    invalid = torch.arange(num_blocks, device="cuda")[:, None] >= n_valid.to("cuda")[None, :]
    scores.masked_fill_(invalid[None], float("nan"))
    return scores


def _assert_tile_matches_stock(scores, n_valid, init_blocks, local_blocks):
    nvb = n_valid.to(device="cuda", dtype=torch.int32)
    stock = select_blocks_from_maxscore(
        scores,
        topk=MSA_REQUIRED_TOPK,
        n_valid_blocks=nvb,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        head_major_output=True,
    )
    tile = msa_indexer._prefill_tile_select(scores, nvb, init_blocks, local_blocks)
    assert tile is not None, "the tile select declined a supported geometry"
    assert tile.shape == stock.shape and tile.stride() == stock.stride()
    mismatch = (tile != stock).any(dim=-1)
    assert not bool(mismatch.any()), (
        f"{int(mismatch.sum())} (query, head) rows differ; first at {mismatch.nonzero()[0].tolist()}: "
        f"tile {tile[mismatch][0].tolist()} stock {stock[mismatch][0].tolist()}"
    )


@pytest.fixture
def tile_all_rows(monkeypatch):
    """Let the tile select take any row count, so short chunks are compared too."""
    monkeypatch.setattr(msa_indexer, "PREFILL_SELECT_TILE", True)
    monkeypatch.setattr(msa_indexer, "PREFILL_SELECT_TILE_MIN_ROWS", 1)
    monkeypatch.setattr(msa_indexer, "PREFILL_SELECT_TILE_MIN_BLOCKS", 1)


@needs_tile
@pytest.mark.parametrize("prefix", [1024, 600_000])
@pytest.mark.parametrize("chunk", [1, 256, 16_384])
@pytest.mark.parametrize("kind", ["ties"])
def test_tile_select_matches_stock_causal_chunk(tile_all_rows, prefix, chunk, kind):
    """One context row of `chunk` tokens on a `prefix`-token prefix, the M3.1 forcing (init 0, local 1)."""
    qo, kv, off = _rows([prefix], [chunk])
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=PAGE)
    num_blocks = _plan_k_tiles(int(kv.max()))
    scores = _scores(kind, 2, num_blocks, chunk, n_valid, seed=prefix + chunk)
    _assert_tile_matches_stock(scores, n_valid, 0, 1)


@needs_tile
@pytest.mark.parametrize(("init_blocks", "local_blocks"), [(0, 0), (0, 1), (20, 8)])
@pytest.mark.parametrize("kind", ["randn", "ties", "fp16ties", "nonfinite"])
@pytest.mark.parametrize("num_kv_heads", [2])
def test_tile_select_matches_stock_forcing(
    tile_all_rows, init_blocks, local_blocks, kind, num_kv_heads
):
    """A mixed batch of context rows, short and long, under every init / local forcing combination."""
    qo, kv, off = _rows([0, 1000, 70_000, 400_000], [33, 2048, 777, 3000])
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=PAGE)
    num_blocks = _plan_k_tiles(int(kv.max()))
    scores = _scores(
        kind, num_kv_heads, num_blocks, int(qo.sum()), n_valid, seed=init_blocks * 31 + local_blocks
    )
    _assert_tile_matches_stock(scores, n_valid, init_blocks, local_blocks)


# ------------------------------------------------------------------------ MsaIndexer: proxy + select, end to end


def _index_case(prefixes, chunks, *, dtype, seed):
    """Random index Q / paged index-K for these context rows, pages scattered over a shuffled pool."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qo, kv, off = _rows(prefixes, chunks)
    pages = [(int(k) + PAGE - 1) // PAGE for k in kv]
    num_pages = sum(pages) + 3
    perm = torch.randperm(num_pages, generator=gen, device="cuda").to(torch.int32)
    kv_indices = perm[: sum(pages)].contiguous()
    total_q = int(qo.sum())
    idx_q = torch.randn((total_q, 2, 128), generator=gen, device="cuda").to(dtype)
    idx_k = torch.randn((num_pages, 1, PAGE, 128), generator=gen, device="cuda").to(dtype)
    return idx_q, idx_k, kv_indices, qo, kv, off


def _indexer(init_blocks=0, local_blocks=1):
    config = types.SimpleNamespace(
        num_index_heads=2, num_kv_heads=2, init_blocks=init_blocks, local_blocks=local_blocks
    )
    return msa_indexer.MsaIndexer(config)


def _proxy_plan(fmha_sm100, qo, kv, off, *, segmented, kv_indices):
    if segmented:
        planned = msa_indexer.segmented_proxy_plan(
            fmha_sm100, qo, off, num_index_heads=2, page_size=PAGE, kv_indices=kv_indices
        )
        if planned is not None:
            return planned
    plan = fmha_sm100.fmha_sm100_plan(
        qo,
        kv,
        qo_offset=off,
        page_size=PAGE,
        num_kv_splits=1,
        causal=True,
        num_qo_heads=2,
        num_kv_heads=1,
        output_maxscore=True,
    )
    return plan, kv_indices


@needs_sm100f
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    ("prefixes", "chunks"),
    [
        ([0], [300]),
        ([20_000, 0], [1291, 700]),
        ([150_000], [16_384]),
        ([700_000], [2048]),
        ([750_000], [512]),
    ],
)
def test_pure_prefill_select_blocks_new_path_matches_stock(monkeypatch, dtype, prefixes, chunks):
    """MsaIndexer.select_blocks over a pure-prefill step: tile + unfilled scores against the stock path.

    750K tokens (5888 padded blocks) is past the tile's bound and must fall back.
    """
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
        require_msa_module,
    )

    fmha_sm100 = require_msa_module()
    idx_q, idx_k, kv_indices, qo, kv, off = _index_case(
        prefixes, chunks, dtype=dtype, seed=len(chunks) + chunks[0]
    )
    plan, plan_kv_indices = _proxy_plan(
        fmha_sm100, qo, kv, off, segmented=True, kv_indices=kv_indices
    )
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=PAGE).to(
        "cuda", torch.int32
    )
    indexer = _indexer()

    def run(new_paths: bool) -> torch.Tensor:
        monkeypatch.setattr(msa_indexer, "PREFILL_SELECT_TILE", new_paths)
        monkeypatch.setattr(msa_indexer, "PREFILL_HOST_TRIMS", new_paths)
        return indexer.select_blocks(
            idx_q,
            idx_k,
            idx_sm_scale=1.0,
            kv_indices=plan_kv_indices,
            proxy_plan=plan,
            n_valid_blocks=n_valid,
        )

    stock = run(False)
    new = run(True)
    assert new.shape == stock.shape and new.stride() == stock.stride()
    assert torch.equal(new, stock)


# ---------------------------------------------------------------------------------------------- k2q CSR workspace reuse


def _csr_case(prefixes, chunks, *, seed):
    """An NVFP4 paged cache, FP8 q and a real top-16 table for these context rows."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qo, kv, off = _rows(prefixes, chunks)
    batch, num_kv_heads, q_heads = len(chunks), 2, 32
    pages = [(int(k) + PAGE - 1) // PAGE for k in kv]
    num_pages = sum(pages) + 2
    perm = torch.randperm(num_pages, generator=gen, device="cuda").to(torch.int32)
    block_table = torch.zeros((batch, max(pages)), dtype=torch.int32, device="cuda")
    first = 0
    for b, n in enumerate(pages):
        block_table[b, :n] = perm[first : first + n]
        first += n
    data = torch.randint(
        0, 256, (num_pages, 2, num_kv_heads, PAGE, 64), generator=gen, device="cuda"
    ).to(torch.uint8)
    scale_values = (
        torch.rand((num_pages, 2, num_kv_heads, PAGE, 8), generator=gen, device="cuda") + 0.5
    )
    scales = scale_values.to(torch.float8_e4m3fn).view(torch.uint8)
    total_q = int(qo.sum())
    q = torch.randn((total_q, q_heads, 128), generator=gen, device="cuda").to(torch.float8_e4m3fn)
    n_valid = per_token_valid_blocks(qo, kv, off, causal=True, block_size=PAGE).to(
        "cuda", torch.int32
    )
    tables = []
    for _ in range(2):  # two layers' selections over the same step
        scores = torch.randn(
            (num_kv_heads, _plan_k_tiles(int(kv.max())), total_q), generator=gen, device="cuda"
        )
        tables.append(
            select_blocks_from_maxscore(
                scores,
                topk=16,
                n_valid_blocks=n_valid,
                init_blocks=0,
                local_blocks=1,
                head_major_output=True,
            )
        )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa.msa_backend import (
        MiniMaxM31MsaSparseAttentionMetadata,
    )

    metadata = object.__new__(MiniMaxM31MsaSparseAttentionMetadata)
    cu_q = torch.zeros(batch + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(qo, 0)
    cu_kv = torch.zeros(batch + 1, dtype=torch.int32)
    cu_kv[1:] = torch.cumsum(kv, 0)
    metadata.__dict__.update(
        _msa_live_batch=batch,
        msa_cu_q_lens=cu_q.cuda(),
        msa_cu_kv_lens=cu_kv.cuda(),
        _msa_max_q_len=int(qo.max()),
        _msa_max_kv_len_all=int(kv.max()),
        _msa_total_k=int(kv.sum()),
        _msa_total_k_rows=sum(pages),
        msa_block_table=block_table,
        msa_seq_lens_cuda=kv.cuda(),
    )
    k_scale = torch.tensor([0.75], device="cuda")
    v_scale = torch.tensor([1.25], device="cuda")
    return q, data, scales, tables, metadata, k_scale, v_scale


def _run_csr_attention(case, topk, monkeypatch, reuse: bool) -> torch.Tensor:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa.msa_prefill import (
        run_msa_nvfp4_sparse_gqa,
    )

    q, data, scales, _, metadata, k_scale, v_scale = case
    monkeypatch.setattr(msa_indexer, "PREFILL_HOST_TRIMS", reuse)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda")
    run_msa_nvfp4_sparse_gqa(
        q,
        data[:, 0],
        data[:, 1],
        scales,
        topk,
        metadata,
        sm_scale=0.088,
        k_global_scale=k_scale,
        v_global_scale=v_scale,
        out=out,
    )
    return out


@needs_sm100f
@pytest.mark.parametrize(
    ("prefixes", "chunks"),
    [([0], [300]), ([1024, 9000], [2048, 513]), ([150_000], [16_384]), ([600_000], [1291])],
)
def test_reused_k2q_csr_workspace_matches_build_k2q_csr(monkeypatch, prefixes, chunks):
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.msa import msa_prefill

    case = _csr_case(prefixes, chunks, seed=len(chunks) * 7 + chunks[0])
    q, data, _, (topk, other), metadata, _, _ = case
    sparse = msa_prefill._msa_sparse_api()
    assert msa_prefill._k2q_csr_parts() is not None, (
        "the packaged builder moved; the reuse path is dead"
    )
    q2k = topk.permute(1, 0, 2).contiguous()
    args = dict(
        total_k=metadata._msa_total_k,
        max_seqlen_k=metadata._msa_max_kv_len_all,
        max_seqlen_q=metadata._msa_max_q_len,
        total_rows=metadata._msa_total_k_rows,
        qhead_per_kv=int(q.shape[1]) // int(data.shape[2]),
    )
    cu_q, cu_kv = metadata.msa_cu_q_lens, metadata.msa_cu_kv_lens
    want = sparse.build_k2q_csr(q2k, cu_q, cu_kv, PAGE, return_schedule=True, **args)
    monkeypatch.setattr(msa_indexer, "PREFILL_HOST_TRIMS", True)
    for _ in range(2):  # the second call runs on the cached workspace
        got = msa_prefill._k2q_csr_with_schedule(
            sparse, q2k, cu_q, cu_kv, PAGE, metadata=metadata, **args
        )
        assert torch.equal(got[0], want[0])
        assert torch.equal(got[1], want[1])
        sched, ref = got[2], want[2]
        assert sched.target_q_per_cta == ref.target_q_per_cta
        assert sched.work_capacity == ref.work_capacity
        assert torch.equal(sched.work_count, ref.work_count)
        assert torch.equal(sched.split_counts, ref.split_counts)
        # Work items are appended in atomic order, so the list is compared as a set; the attention
        # outputs below are compared bitwise.
        work = int(ref.work_count.item())
        assert sorted(map(tuple, sched.scheduler_metadata[:work].tolist())) == sorted(
            map(tuple, ref.scheduler_metadata[:work].tolist())
        )
    assert metadata._msa_csr_workspace is not None

    stock = _run_csr_attention(case, topk, monkeypatch, reuse=False)
    assert torch.equal(_run_csr_attention(case, topk, monkeypatch, reuse=False), stock), (
        "stock path not deterministic"
    )
    assert torch.equal(_run_csr_attention(case, topk, monkeypatch, reuse=True), stock)
    # A second layer's table on the same (now warm) workspace.
    want_other = _run_csr_attention(case, other, monkeypatch, reuse=False)
    assert torch.equal(_run_csr_attention(case, other, monkeypatch, reuse=True), want_other)
    assert torch.equal(_run_csr_attention(case, topk, monkeypatch, reuse=True), stock)


# ------------------------------------------------------------------------ NVFP4 attention combined straight into place
