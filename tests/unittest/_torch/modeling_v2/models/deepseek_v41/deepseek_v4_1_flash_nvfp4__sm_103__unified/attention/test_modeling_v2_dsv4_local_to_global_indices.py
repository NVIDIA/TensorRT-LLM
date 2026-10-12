# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dsv4_local_to_global_indices catalog entry.

Certified column (DeepSeek-V4.1 values): a 128-column SWA segment over 128-token
pages, and a 512-column compressed top-k segment over 128 // ratio entries per page
for ratio 1 (128) and ratio 2 (64); ratio-0 layers call it without a compressed
segment. The original cases use pools of POOL_PAGES = 4096 pages -- chosen, plainly
larger than any batch here needs, before a runtime-value dump existed -- and block
tables of TABLE_ROWS = 64 rows with the batch at scattered rows, so a table-row or
page addressing defect cannot hide behind row == batch index. The cases after those
use the core engine's pool sizes and then every pool, width and batch an engine can
be configured with.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are
the gate. This file turns a cell's spec into tensors and keeps the checks a cell
cannot carry: strided views, past-the-table indices, a -1 page, T = 0, PDL, capture
and replay, the in-tree V4 launcher, the wrong-variant controls, the op's own
rejections, and the engine-size sweeps.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.dsv4_local_to_global_indices import (  # noqa: E501
    dsv4_local_to_global_indices as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "dsv4_local_to_global_indices requires a CUDA device"

DEV = "cuda"
WINDOW = 128
TOPK = 512
SWA_TPB = 128
POOL_PAGES = 4096
TABLE_ROWS = 64


def _reference(req_id, table, local, tpb, offset):
    """One segment's pool rows: `op.reference` on that segment alone, as int32 like the op's
    output (every row these cases produce is far below 2**31)."""
    return op.reference(req_id, table, local, tpb, offset).to(torch.int32)


def _table(width, g):
    """[TABLE_ROWS, width] int32; each row holds distinct pages of a POOL_PAGES pool."""
    return torch.stack(
        [torch.randperm(POOL_PAGES, generator=g, device=DEV)[:width] for _ in range(TABLE_ROWS)]
    ).to(torch.int32)


def _window(positions):
    """The causal window of each query: positions max(0, p-127) .. p in order, then -1."""
    start = (positions[:, None] - WINDOW + 1).clamp(min=0)
    cols = start + torch.arange(WINDOW, device=DEV, dtype=torch.int32)
    return torch.where(cols > positions[:, None], -1, cols).to(torch.int32)


def _topk_local(positions, ratio, g):
    """V4.1 index top-k shape: min(512, (p+1)//ratio) distinct compressed positions in
    [0, (p+1)//ratio), in score order (not sorted by position), then -1."""
    visible = (positions.long() + 1) // ratio
    nmax = int(visible.max()) if positions.numel() else 0
    out = torch.full((positions.shape[0], TOPK), -1, dtype=torch.int32, device=DEV)
    if nmax == 0:
        return out
    scores = torch.rand(positions.shape[0], nmax, generator=g, device=DEV)
    col = torch.arange(nmax, device=DEV)
    scores = scores.masked_fill(col[None, :] >= visible[:, None], float("-inf"))
    k = min(TOPK, nmax)
    idx = scores.topk(k, dim=1).indices
    rank = torch.arange(k, device=DEV)
    out[:, :k] = torch.where(rank[None, :] < visible[:, None], idx, -1).to(torch.int32)
    return out


def _batch(ratio, seqs, max_seq_len, seed, replay_tail=None):
    """Tokens of requests given as (cached, new); returns the op's inputs plus positions."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = torch.randperm(TABLE_ROWS, generator=g, device=DEV)[: len(seqs)].to(torch.int32)
    req_id = torch.cat([rows[b].repeat(new) for b, (_, new) in enumerate(seqs)]).contiguous()
    positions = torch.cat(
        [torch.arange(c, c + new, device=DEV, dtype=torch.int32) for c, new in seqs]
    )
    swa_table = _table(math.ceil(max_seq_len / SWA_TPB), g)
    swa_local = _window(positions)
    if replay_tail is not None:
        # Bounded replay: the window is truncated to the replayed segment.
        swa_local = torch.where(swa_local < replay_tail, -1, swa_local).to(torch.int32)
    if ratio == 0:
        return req_id, swa_table, swa_local, None, None, 0, positions
    tpb_c = SWA_TPB // ratio
    comp_table = _table(math.ceil(max_seq_len // ratio / tpb_c), g)
    comp_local = _topk_local(positions, ratio, g)
    return req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, positions


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    ratio = spec["ratio"]
    assert spec["pool_pages"] == POOL_PAGES and spec["table_rows"] == TABLE_ROWS
    assert spec["swa_tokens_per_block"] == SWA_TPB
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, positions = _batch(
        ratio,
        spec["seqs"],
        spec["max_seq_len"],
        seed=zlib.crc32(cell.why.encode()),
        replay_tail=spec["replay_tail"],
    )
    assert tpb_c == spec["compressed_tokens_per_block"]
    args = (
        req_id,
        swa_table,
        swa_local,
        spec["swa_tokens_per_block"],
        spec["swa_row_offset"],
        comp_table,
        comp_local,
        spec["compressed_tokens_per_block"],
        spec["compressed_row_offset"],
    )
    out = op(*args)
    t = positions.shape[0]
    assert out.shape == (t, WINDOW + (TOPK if ratio else 0)) and out.is_contiguous()
    ref = op.reference(*args)
    # The inputs are the V4.1 window: min(128, p + 1) valid rows per query (fewer under
    # replay), and min(512, (p + 1) // ratio) valid compressed entries.
    if spec["replay_tail"] is None:
        want_swa = (positions.long() + 1).clamp(max=WINDOW)
        assert int((ref[:, :WINDOW] >= 0).sum()) == int(want_swa.sum())
    if ratio:
        want_c = ((positions.long() + 1) // ratio).clamp(max=TOPK)
        assert int((ref[:, WINDOW:] >= 0).sum()) == int(want_c.sum())
    op.compare(out, ref)


@pytest.mark.parametrize("ratio", [1, 2])
def test_strided_views_are_read_through_their_strides(ratio):
    """Block tables as column slices of wider tables, index rows as non-contiguous views."""
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = _batch(
        ratio, [(0, 700), (5000, 3), (130, 1)], 16384, seed=7
    )
    t = req_id.shape[0]
    wide_swa = torch.full((TABLE_ROWS, swa_table.shape[1] + 37), -7, dtype=torch.int32, device=DEV)
    wide_swa[:, : swa_table.shape[1]] = swa_table
    wide_c = torch.full((TABLE_ROWS, comp_table.shape[1] + 5), -7, dtype=torch.int32, device=DEV)
    wide_c[:, 5:] = comp_table
    swa_t = torch.empty((WINDOW, t), dtype=torch.int32, device=DEV)
    swa_t.copy_(swa_local.t())
    comp_wide = torch.full((t, WINDOW + TOPK), -9, dtype=torch.int32, device=DEV)
    comp_wide[:, WINDOW:] = comp_local
    swa_view, comp_view = swa_t.t(), comp_wide[:, WINDOW:]
    bt_swa_view, bt_c_view = wide_swa[:, : swa_table.shape[1]], wide_c[:, 5:]
    assert not swa_view.is_contiguous() and not comp_view.is_contiguous()
    assert not bt_swa_view.is_contiguous() and not bt_c_view.is_contiguous()
    out = op(req_id, bt_swa_view, swa_view, SWA_TPB, 0, bt_c_view, comp_view, tpb_c, 0)
    assert torch.equal(out[:, :WINDOW], _reference(req_id, swa_table, swa_local, SWA_TPB, 0))
    assert torch.equal(out[:, WINDOW:], _reference(req_id, comp_table, comp_local, tpb_c, 0))


def test_local_index_past_the_table_is_minus_one():
    g = torch.Generator(device=DEV).manual_seed(11)
    req_id = torch.tensor([3, 3, 60], dtype=torch.int32, device=DEV)
    swa_table = _table(4, g)  # covers local indices 0 .. 511
    comp_table = _table(2, g)  # covers 0 .. 127 at 64 per page
    swa_local = torch.tensor([[511, 512, 1000, -1] * 32] * 3, dtype=torch.int32, device=DEV)
    comp_local = torch.tensor([[127, 128, 5000, -1] * 128] * 3, dtype=torch.int32, device=DEV)
    out = op(req_id, swa_table, swa_local, 128, 0, comp_table, comp_local, 64, 0)
    assert bool((out[:, 0:WINDOW:4] >= 0).all()) and bool((out[:, WINDOW::4] >= 0).all())
    for c in (1, 2, 3):
        assert bool((out[:, c:WINDOW:4] == -1).all())
        assert bool((out[:, WINDOW + c :: 4] == -1).all())


def test_unallocated_page_is_not_masked():
    """A -1 page in a table is used as page -1: the op does not validate page ids."""
    req_id = torch.tensor([0], dtype=torch.int32, device=DEV)
    table = torch.full((1, 4), -1, dtype=torch.int32, device=DEV)
    local = torch.arange(WINDOW, dtype=torch.int32, device=DEV)[None, :] + 5
    out = op(req_id, table, local, SWA_TPB, 1000)
    expect = 1000 - SWA_TPB + (local % SWA_TPB)
    assert torch.equal(out, expect.to(torch.int32))


@pytest.mark.parametrize("ratio", [0, 2])
def test_zero_tokens(ratio):
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = _batch(
        ratio, [(0, 4)], 1024, seed=3
    )
    out = op(
        req_id[:0],
        swa_table,
        swa_local[:0],
        SWA_TPB,
        0,
        comp_table,
        None if comp_local is None else comp_local[:0],
        tpb_c,
        0,
    )
    assert out.shape == (0, WINDOW + (TOPK if ratio else 0)) and out.dtype == torch.int32


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_pdl_launch_gives_the_same_table(ratio):
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = _batch(
        ratio, [(0, 1000), (9000, 6)], 16384, seed=5
    )
    args = (req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, comp_local, tpb_c, 0)
    assert torch.equal(
        op(*args, enable_pdl=True),
        op(*args, enable_pdl=False),
    )


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_cuda_graph_capture_and_replay(ratio, enable_pdl):
    """Capture a decode-shaped call; replay it over new requests, positions and tables."""
    seqs = [(c, 6) for c in range(100, 16000, 530)]
    first = _batch(ratio, seqs, 16384, seed=21)
    static = [x.clone() if isinstance(x, torch.Tensor) else x for x in first[:5]]
    tpb_c = first[5]

    def call():
        return op(
            static[0],
            static[1],
            static[2],
            SWA_TPB,
            0,
            static[3],
            static[4],
            tpb_c,
            0,
            enable_pdl=enable_pdl,
        )

    call()  # JIT outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for seed in (22, 23):
        new = _batch(ratio, [(c + seed, 6) for c, _ in seqs], 16384, seed=seed)
        for dst, src in zip(static, new[:5]):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:, :WINDOW], _reference(new[0], new[1], new[2], SWA_TPB, 0))
        assert torch.equal(out[:, WINDOW:], _reference(new[0], new[3], new[4], tpb_c, 0))


def test_matches_the_in_tree_v4_launcher_where_it_applies():
    """Same kernel: ratio 0 and ratio 2 agree bit for bit with the V4 Python launcher.

    (That launcher gates its compressed segment on compress_ratio > 1, so it cannot
    express a 1:1 compressed pool; the contract records the probe.)
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
        deepseek_v4_local_to_global_indices,
    )

    for ratio in (0, 2):
        req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = _batch(
            ratio, [(0, 900), (7000, 6), (15000, 1)], 16384, seed=31
        )
        ours = op(req_id, swa_table, swa_local, SWA_TPB, 256, comp_table, comp_local, tpb_c, 64)
        theirs = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=swa_table,
            swa_local_indices=swa_local,
            swa_pool_base_ptr=0,
            swa_buffer_ptr=256,
            tokens_per_block=SWA_TPB,
            token_stride=1,
            block_table_compressed=comp_table,
            compressed_local_indices=comp_local,
            compress_pool_base_ptr=0,
            compressed_buffer_ptr=64,
            compress_ratio=max(ratio, 1),
            num_compressed_indices=TOPK if ratio else 0,
        )
        assert torch.equal(ours, theirs)


def test_deliberately_wrong_controls_are_rejected():
    """Each wrong variant is run through the op and fails the same exact-equality gate."""
    report = {}
    for ratio in (1, 2):
        req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, positions = _batch(
            ratio, [(0, 1000), (0, 129), (9000, 6), (16000, 1)], 16384, seed=41
        )
        swa_off, comp_off = 3 * POOL_PAGES * SWA_TPB, 2 * POOL_PAGES * tpb_c
        ref = torch.cat(
            [
                _reference(req_id, swa_table, swa_local, SWA_TPB, swa_off),
                _reference(req_id, comp_table, comp_local, tpb_c, comp_off),
            ],
            dim=1,
        )
        good = op(
            req_id, swa_table, swa_local, SWA_TPB, swa_off, comp_table, comp_local, tpb_c, comp_off
        )
        assert torch.equal(good, ref)

        # Off-by-one window: positions p-128 .. p-1 (the query's own row missing).
        shifted = _window(positions - 1)
        variants = {
            "window_off_by_one": (
                req_id,
                swa_table,
                shifted,
                SWA_TPB,
                swa_off,
                comp_table,
                comp_local,
                tpb_c,
                comp_off,
            ),
            # The other ratio's compressed page size: 64 on a ratio-1 layer, 128 on a ratio-2 layer.
            "wrong_ratio_page_size": (
                req_id,
                swa_table,
                swa_local,
                SWA_TPB,
                swa_off,
                comp_table,
                comp_local,
                64 if ratio == 1 else 128,
                comp_off,
            ),
            "neighbouring_table_row": (
                (req_id + 1) % TABLE_ROWS,
                swa_table,
                swa_local,
                SWA_TPB,
                swa_off,
                comp_table,
                comp_local,
                tpb_c,
                comp_off,
            ),
            "pool_offsets_swapped": (
                req_id,
                swa_table,
                swa_local,
                SWA_TPB,
                comp_off,
                comp_table,
                comp_local,
                tpb_c,
                swa_off,
            ),
        }
        for name, args in variants.items():
            wrong = op(*args)
            bad = int((wrong != ref).sum())
            report[(ratio, name)] = (bad, ref.numel())
            assert not torch.equal(wrong, ref), f"control {name} at ratio {ratio} was not rejected"
    for (ratio, name), (bad, total) in report.items():
        print(f"control ratio={ratio} {name}: {bad} of {total} elements differ")


def test_rejections_are_loud():
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = _batch(
        2, [(0, 300)], 16384, seed=51
    )
    with pytest.raises(ValueError, match="must be int32 tensors"):
        op(req_id.long(), swa_table, swa_local, SWA_TPB, 0)
    with pytest.raises(ValueError, match="must be int32 tensors"):
        op(req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, comp_local.long(), tpb_c, 0)
    with pytest.raises(ValueError, match="W_compressed must be a power of two, got 500"):
        op(req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, comp_local[:, :500], tpb_c, 0)
    with pytest.raises(ValueError, match="W_swa must be a power of two, got 127"):
        op(req_id, swa_table, swa_local[:, :127], SWA_TPB, 0)
    with pytest.raises(ValueError, match="given together or not at all"):
        op(req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, None, tpb_c, 0)
    with pytest.raises(ValueError, match="req_id must be a contiguous 1-D tensor"):
        wide = torch.stack([req_id, req_id], dim=1)
        op(wide[:, 0], swa_table, swa_local, SWA_TPB, 0)
    with pytest.raises(ValueError, match="compressed_tokens_per_block must be positive"):
        op(req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, comp_local, 0, 0)
    with pytest.raises(ValueError, match=r"swa_row_offset must be in \[0, 2\*\*31\)"):
        op(req_id, swa_table, swa_local, SWA_TPB, -1)


# ─── The DeepSeek-V4.1 core call at engine-size tables ─────────────────────────────────────────
# The core target calls this op once per layer per step: ratio 0 as
# (req_idx_per_token[:T], SWA table, swa_local_indices_cuda[:T], 128, 0) and ratios 1 / 2 with the
# ratio's compress block table, the layer's [T, 512] index top-k and 128 // ratio entries per page,
# offsets 0. T is every row count 1..8192 (max_num_tokens) in eager steps, and in captured decode
# steps every batch 1..8 (boot, padding off), {1, 2, 4, 8, 16, 24, 32} (accuracy, padding on) and
# 6 x those batches (DSpark target verify). T enters the kernel only as the grid size.
# Tables are sized from the engine's pools: SWA 188160 pages x 128 rows, COMPRESS 453120 x 64
# (ratio 2) and 151040 x 128 (ratio 1); block tables are 64 rows x 129 pages (chosen: the gate's
# max_seq_len 16384 needs 128 pages of 128 tokens per request, plus one), and every page id comes
# from the top 64 * 129 pages of its pool, so every produced row lies at the pool's far end.
import tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels as _l2g_kernels  # noqa: E402

ENGINE_SWA_PAGES = 188160
ENGINE_COMP_PAGES = {1: 151040, 2: 453120}
ENGINE_ROWS = 64
ENGINE_BLOCKS = 129


def _engine_table(pages, g):
    """[ENGINE_ROWS, ENGINE_BLOCKS] page ids, a random arrangement of the pool's top
    ENGINE_ROWS * ENGINE_BLOCKS pages (the last page included)."""
    band = ENGINE_ROWS * ENGINE_BLOCKS
    ids = pages - band + torch.randperm(band, generator=g, device=DEV)
    return ids.view(ENGINE_ROWS, ENGINE_BLOCKS).to(torch.int32)


def _topk_local_with_newest(positions, ratio, g):
    """Like _topk_local, and the newest visible compressed position is always among the picks."""
    visible = (positions.long() + 1) // ratio
    nmax = max(int(visible.max()), 1)
    scores = torch.rand(positions.shape[0], nmax, generator=g, device=DEV)
    col = torch.arange(nmax, device=DEV)
    scores = scores.masked_fill(col[None, :] == (visible - 1)[:, None], float("inf"))
    scores = scores.masked_fill(col[None, :] >= visible[:, None], float("-inf"))
    k = min(TOPK, nmax)
    idx = scores.topk(k, dim=1).indices
    out = torch.full((positions.shape[0], TOPK), -1, dtype=torch.int32, device=DEV)
    rank = torch.arange(k, device=DEV)
    out[:, :k] = torch.where(rank[None, :] < visible[:, None], idx, -1).to(torch.int32)
    return out


def _engine_step(ratio, seqs, seed):
    """One step's inputs on engine-size tables. seqs: (cached, new) per request in batch order.
    Request b's newest SWA page and newest compressed page are the pool's last pages
    (page P - 1 - b), so the pool's last rows are produced."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = torch.randperm(ENGINE_ROWS, generator=g, device=DEV)[: len(seqs)].to(torch.int32)
    req_id = torch.cat([rows[b].repeat(new) for b, (_, new) in enumerate(seqs)]).contiguous()
    positions = torch.cat(
        [torch.arange(c, c + new, device=DEV, dtype=torch.int32) for c, new in seqs]
    )
    swa_table = _engine_table(ENGINE_SWA_PAGES, g)
    for b, (c, new) in enumerate(seqs):
        swa_table[rows[b], (c + new - 1) // SWA_TPB] = ENGINE_SWA_PAGES - 1 - b
    swa_local = _window(positions)
    if ratio == 0:
        return req_id, swa_table, swa_local, None, None, 0, positions
    tpb_c = SWA_TPB // ratio
    pages = ENGINE_COMP_PAGES[ratio]
    comp_table = _engine_table(pages, g)
    for b, (c, new) in enumerate(seqs):
        vis = (c + new) // ratio
        if vis:
            comp_table[rows[b], (vis - 1) // tpb_c] = pages - 1 - b
    comp_local = _topk_local_with_newest(positions, ratio, g)
    return req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, positions


def _step_reference(ratio, step):
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, _ = step
    ref = _reference(req_id, swa_table, swa_local, SWA_TPB, 0)
    if ratio:
        ref = torch.cat([ref, _reference(req_id, comp_table, comp_local, tpb_c, 0)], dim=1)
    return ref


def _target_call(ratio, req_id, swa_table, swa_local, comp_table, comp_local, tpb_c):
    """The target's call forms (offsets 0)."""
    if ratio == 0:
        return op(req_id, swa_table, swa_local, SWA_TPB, 0)
    return op(req_id, swa_table, swa_local, SWA_TPB, 0, comp_table, comp_local, tpb_c, 0)


def _compiled_variants():
    """Number of compiled variants of the op's Triton kernel held by this process."""
    kern = _l2g_kernels._deepseek_v4_local_to_global_kernel
    caches = getattr(kern, "device_caches", None)
    if caches is not None:
        return sum(len(v[0] if isinstance(v, tuple) else v) for v in caches.values())
    return sum(len(v) for v in kern.cache.values())


# One mixed step of 8192 tokens: contexts (fresh, chunked on a cached prefix, 1-token, up to
# position 16383) ahead of 30 decode rows at positions 0 .. 16383.
_CORE_STEP_DECODE = [0, 1, 126, 127, 128, 129, 16383] + list(range(300, 16000, 600))[:23]
CORE_STEP = [
    (0, 3000),
    (8192, 2048),
    (0, 1),
    (0, 129),
    (130, 5),
    (2, 1570),
    (4000, 430),
    (15405, 979),
] + [(c, 1) for c in _CORE_STEP_DECODE]


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_rows_prefix_bitwise(ratio):
    """Every T = 1..8192, not a sample: the 8192-row call equals the reference (engine-size tables,
    the pools' last rows produced), every call on the first T rows (prefix views of the same
    buffers, as the target slices its metadata) is bitwise equal to the first T rows of it, and the
    sweep adds no compiled variant of the Triton kernel. A counter control shows a new constexpr
    combination does add one."""
    step = _engine_step(ratio, CORE_STEP, seed=9000 + ratio)
    req_id, swa_table, swa_local, comp_table, comp_local, tpb_c, positions = step
    n = req_id.numel()
    assert n == 8192
    full = _target_call(ratio, *step[:6])
    ref = _step_reference(ratio, step)
    assert torch.equal(full, ref)
    assert int(full[:, :WINDOW].max()) >= (ENGINE_SWA_PAGES - 1) * SWA_TPB
    if ratio:
        assert int(full[:, WINDOW:].max()) >= (ENGINE_COMP_PAGES[ratio] - 1) * tpb_c
    before = _compiled_variants()
    assert before > 0
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for t in range(1, n + 1):
        out = _target_call(
            ratio,
            req_id[:t],
            swa_table,
            swa_local[:t],
            comp_table,
            None if comp_local is None else comp_local[:t],
            tpb_c,
        )
        bad += (out != full[:t]).any(1).sum()
    torch.cuda.synchronize()
    after = _compiled_variants()
    print(
        f"ratio={ratio}: prefixes T=1..{n}: rows differing {int(bad)}; compiled variants {before} -> {after}"
    )
    assert int(bad) == 0
    assert after == before
    # Counter control: a page size never used elsewhere is a new constexpr, hence a new variant.
    op(req_id[:4], swa_table, swa_local[:4], 1000 + ratio, 0)
    torch.cuda.synchronize()
    assert _compiled_variants() == after + 1


# Captured decode steps (the union of the gates' capture sets per rank): the graph batches
# G of the boot gates (every G in 1..8: max_batch_size 8, padding off) and of the accuracy gates
# ({1, 2, 4, 8, 16, 24, 32}: padding on), each at next_n 1 (T = G rows) and at next_n 6 (the DSpark
# target-verify step, T = 6 G rows: the DSpark boot gates capture every G 1..8 at next_n 6, so
# T = 18, 30, 36 and 42 are in the union too).
CAPTURE_UNION_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)
CAPTURE_UNION = sorted(
    {(b, 1) for b in CAPTURE_UNION_BATCHES} | {(6 * b, 6) for b in CAPTURE_UNION_BATCHES}
)


def _decode_seqs(rows, next_n, g, step):
    """rows // next_n requests, each verifying next_n positions at random positions; on even steps
    request 0 ends at position 16383 and request 1 (when present) starts at position 0."""
    batch = rows // next_n
    cached = torch.randint(0, 16384 - next_n + 1, (batch,), generator=g).tolist()
    if step % 2 == 0:
        cached[0] = 16384 - next_n
        if batch > 1:
            cached[1] = 0
    return [(c, next_n) for c in cached]


@pytest.mark.parametrize("rows,next_n", CAPTURE_UNION, ids=[f"T{r}_n{n}" for r, n in CAPTURE_UNION])
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_cuda_graph_capture_sets(ratio, rows, next_n):
    """Captured as the target captures it: prefix views [:T] of max_num_tokens-row metadata
    buffers, static tables, a static [T, 512] top-k buffer, after an eager warm-up at that shape.
    Three replays over new requests, positions, tables and top-k: each is bitwise equal to an eager
    call on the same inputs and to the reference, and differs from the previous step's reference."""
    g = torch.Generator(device="cpu").manual_seed(1000 * ratio + rows)
    max_tokens = 8192
    req_buf = torch.zeros(max_tokens, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((max_tokens, WINDOW), -1, dtype=torch.int32, device=DEV)
    first = _engine_step(ratio, _decode_seqs(rows, next_n, g, 0), seed=11000 + 100 * ratio + rows)
    swa_table = first[1].clone()
    comp_table = None if ratio == 0 else first[3].clone()
    comp_buf = None if ratio == 0 else torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    tpb_c = first[5]

    def load(step):
        req_buf[:rows].copy_(step[0])
        swa_buf[:rows].copy_(step[2])
        swa_table.copy_(step[1])
        if ratio:
            comp_table.copy_(step[3])
            comp_buf.copy_(step[4])
        return _step_reference(ratio, step)

    def call():
        return _target_call(
            ratio, req_buf[:rows], swa_table, swa_buf[:rows], comp_table, comp_buf, tpb_c
        )

    prev = load(first)
    assert torch.equal(call(), prev)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for it in range(1, 4):
        ref = load(
            _engine_step(
                ratio, _decode_seqs(rows, next_n, g, it), seed=11000 + 100 * ratio + rows + 7 * it
            )
        )
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"replay {it}"
        assert torch.equal(out, call()), f"replay {it} vs eager"
        assert not torch.equal(out, prev), f"replay {it} equals the previous step"
        prev = ref


@pytest.mark.parametrize("ratio", [1, 2])
def test_results_up_to_int32_max_are_exact(ratio):
    """offset + page * tpb + l % tpb reaching exactly 2**31 - 1 at a pool's last page: the kernel's
    int32 arithmetic is exact up to the top of the int32 range."""
    tpb_c = SWA_TPB // ratio
    comp_pages = ENGINE_COMP_PAGES[ratio]
    req_id = torch.zeros(1, dtype=torch.int32, device=DEV)
    swa_table = torch.tensor([[ENGINE_SWA_PAGES - 1]], dtype=torch.int32, device=DEV)
    comp_table = torch.tensor([[comp_pages - 1]], dtype=torch.int32, device=DEV)
    swa_local = torch.arange(WINDOW, dtype=torch.int32, device=DEV)[None, :]
    comp_local = torch.full((1, TOPK), -1, dtype=torch.int32, device=DEV)
    comp_local[0, :tpb_c] = torch.arange(tpb_c, dtype=torch.int32, device=DEV)
    swa_off = 2**31 - ENGINE_SWA_PAGES * SWA_TPB
    comp_off = 2**31 - comp_pages * tpb_c
    out = op(
        req_id, swa_table, swa_local, SWA_TPB, swa_off, comp_table, comp_local, tpb_c, comp_off
    )
    ref = torch.cat(
        [
            _reference(req_id, swa_table, swa_local, SWA_TPB, swa_off),
            _reference(req_id, comp_table, comp_local, tpb_c, comp_off),
        ],
        dim=1,
    )
    assert int(out[0, WINDOW - 1]) == 2**31 - 1 and int(out[0, WINDOW + tpb_c - 1]) == 2**31 - 1
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------------------------
# The multi-request engine envelope. One call covers ALL rows of a step, context rows
# first and then generation rows (T <= 8192, <= 32 requests, per-request length <= 16384), with
# req_idx_per_token naming each token's block-table row, the SWA window (positions
# max(0, p-127) .. p, then -1) and, at ratios 1 and 2, the layer's index top-k (valid compressed
# positions < (p+1)//ratio packed first, then -1). Block tables are engine-size: ENGINE_ROWS x
# ENGINE_BLOCKS page ids drawn as a random permutation of the whole engine pool (SWA 188160 pages;
# COMPRESS 453120 x 64 rows at ratio 2, 151040 x 128 at ratio 1).
import base64  # noqa: E402
import random  # noqa: E402

from torch.profiler import ProfilerActivity, profile  # noqa: E402

KERNEL_NAME = "_deepseek_v4_local_to_global_kernel"
MAX_REQUEST_LEN = 16384


def _multi_request_tables(ratio, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = ENGINE_ROWS * ENGINE_BLOCKS
    swa = torch.randperm(ENGINE_SWA_PAGES, generator=g, device=DEV)[:n].view(
        ENGINE_ROWS, ENGINE_BLOCKS
    )
    comp = None
    if ratio:
        comp = torch.randperm(ENGINE_COMP_PAGES[ratio], generator=g, device=DEV)[:n].view(
            ENGINE_ROWS, ENGINE_BLOCKS
        )
        comp = comp.to(torch.int32)
    return swa.to(torch.int32), comp


def _multi_request_step(ratio, ctx, gen, next_n, seed):
    """Per-token inputs of one step: ctx = [(cached, new)] context requests, then gen = [cached]
    generation requests each verifying next_n positions cached .. cached + next_n - 1.
    Returns (req_id, swa_local, comp_local, first row of each request, table row of each)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seqs = list(ctx) + [(c, next_n) for c in gen]
    rows = torch.randperm(ENGINE_ROWS, generator=g, device=DEV)[: len(seqs)].to(torch.int32)
    req_id = torch.cat([rows[b].repeat(new) for b, (_, new) in enumerate(seqs)]).contiguous()
    positions = torch.cat(
        [torch.arange(c, c + new, device=DEV, dtype=torch.int32) for c, new in seqs]
    )
    assert int(positions.max()) < MAX_REQUEST_LEN
    swa_local = _window(positions)
    comp_local = _topk_local(positions, ratio, g) if ratio else None
    starts = [0]
    for _, new in seqs[:-1]:
        starts.append(starts[-1] + new)
    return req_id, swa_local, comp_local, starts, rows


def _multi_request_reference(ratio, req_id, swa, swa_local, comp, comp_local):
    ref = _reference(req_id, swa, swa_local, SWA_TPB, 0)
    if ratio:
        ref = torch.cat([ref, _reference(req_id, comp, comp_local, SWA_TPB // ratio, 0)], dim=1)
    return ref


def _multi_request_random_mixed(rng):
    """k in [1, 32] requests, a random number of them context requests (random cached prefix and
    new length, T - generation rows split at random) and the rest generation requests with next_n
    1 or 6 (DSpark verify), T in [1, 8192]."""
    k = rng.randint(1, 32)
    next_n = rng.choice((1, 6))
    n_gen = rng.randint(0, k)
    n_ctx = k - n_gen
    gen_rows = n_gen * next_n
    ctx = []
    if n_ctx:
        ctx_rows = rng.randint(gen_rows + n_ctx, 8192) - gen_rows
        cuts = sorted(rng.sample(range(1, ctx_rows), n_ctx - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        ctx = [(rng.randint(0, MAX_REQUEST_LEN - n), n) for n in news]
    gen = [rng.randint(0, MAX_REQUEST_LEN - next_n) for _ in range(n_gen)]
    return ctx, gen, next_n


def _profiled_calls(prof):
    calls = [e for e in prof.events() if e.name == "trtllm::dsv4_local_to_global_indices"]
    return len(calls), {kern.name for e in calls for kern in e.kernels}


def _check_variants(prof, n_calls, before):
    seen, names = _profiled_calls(prof)
    after = _compiled_variants()
    print(
        f"kernel-variant map over {seen} calls: {sorted(names)}; compiled variants {before} -> {after}"
    )
    assert seen == n_calls and names == {KERNEL_NAME}, (seen, names)
    assert after == before


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_random_mixed_partitions(ratio):
    """200 seeded random mixed steps per ratio on engine-size tables, each bitwise equal to the
    native-torch table lookup, under torch.profiler: every call launches the one Triton kernel,
    and no compiled variant is added after one warm-up call of the ratio's form. Every 20th step
    with >= 2 requests also runs a control with req_idx_per_token off by one request at a request
    boundary (the next request's first token given the previous request's table row); the same
    exact check must see it."""
    rng = random.Random(23100 + ratio)
    swa, comp = _multi_request_tables(ratio, seed=23200 + ratio)
    tpb_c = SWA_TPB // ratio if ratio else 0
    steps = [_multi_request_random_mixed(rng) for _ in range(200)]
    warm = _multi_request_step(ratio, [(0, 4)], [], 1, seed=1)
    _target_call(ratio, warm[0], swa, warm[1], comp, warm[2], tpb_c)
    torch.cuda.synchronize()
    before = _compiled_variants()
    controls, rows_total, kinds = [], 0, set()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (ctx, gen, next_n) in enumerate(steps):
            req_id, swa_local, comp_local, starts, rows = _multi_request_step(
                ratio, ctx, gen, next_n, seed=ratio * 1000 + i
            )
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            ref = _multi_request_reference(ratio, req_id, swa, swa_local, comp, comp_local)
            assert torch.equal(out, ref), (i, ctx, gen, next_n)
            rows_total += req_id.numel()
            kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
            if i % 20 == 0 and len(starts) >= 2:
                b = rng.randint(1, len(starts) - 1)
                bad_id = req_id.clone()
                bad_id[starts[b]] = rows[b - 1]
                wrong = _target_call(ratio, bad_id, swa, swa_local, comp, comp_local, tpb_c)
                controls.append(int((wrong != ref).sum()))
    torch.cuda.synchronize()
    print(
        f"ratio={ratio}: {len(steps)} steps, {rows_total} rows, step kinds (ctx, gen, next_n) "
        f"{sorted(kinds)}; control (req_id off by one request at a boundary) elements differing {controls}"
    )
    assert len(controls) >= 8 and all(c > 0 for c in controls), controls
    assert {
        (True, True, 1),
        (True, True, 6),
        (False, True, 1),
        (False, True, 6),
        (True, False, 0),
    } <= kinds
    _check_variants(prof, len(steps) + len(controls), before)


# Every (T, ratio) cell of dsv4_local_to_global_indices in the core runtime-value dumps (boot,
# GSM8K, 24K passkey, gate and accuracy-preview logs). The dump records only T,
# which counts context AND generation rows, so each cell is given one consistent step, as
# "T,ratio,tc,nc,max_q_len,G,next_n" records (space-separated, zlib-compressed, base64):
# - 1577 cells: a trtllm_sparse_mla_dsv4.varlen record at tc == T (nc context requests, one with
#   max_q_len new tokens), no generation rows;
# - 25 cells: a deepgemm_fp8_fp4_paged_mqa_logits record with G * next_n == T, generation only;
# - 1094 cells: the largest varlen tc < T whose remainder T - tc is a dumped decode batch G x
#   next_n with nc + G <= 32 (context rows first, then generation rows).
# 7 dumped records are garbles and dropped: T 155512 and 673966 (above max_num_tokens 8192), and
# five whose fields are not the call's (e.g. (1849, 128128, 512, 128), (8192, 384, 0, 576)).
DUMPED_CELLS = (
    "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"
    "sa+YSfeRSc9qNRRjXzGT7iOTntVqKMa+YibdRyY9q9VQjH3FRLmPTHpWq6EY+4ppdR/XRM9qNRRjXzGT7iOTntVqKMa+Yibd"
    "RyY9q9VQjH3FTLqPTHpWq6EY+4qZdL/7a2EMuzGH7syk+8ikZ7UaisXy3BgWxlCMfcVMuo9MelaroRj7itdE93FN9KxWQzH2"
    "FfPsPvLsWa2GYuwr5tl95NmzWg3F2FfMs/vIs2e1Goqxr5hn95Fnz2o1FGvLc2NYGMNuzKE7M+k+MulZrYZisTw3hoUxFGNv"
    "MJPuI5Oe1Wooxt5gJt1HJj2r1VCMvcFMuo9Melarodi1PDeGhTEUY28wk+4jk57VaijG3mAm3UcmPavVUIy9wUy6j0x6Vquh"
    "GLc5zKT7yKRntRp2Yw7dmUn3kUnPajUUY18xk+4jk57VaijGvmIm3UcmPavVUIx9xUy6j0x6VquhGPuKmXQfmfSsVkOxa3lu"
    "DAtjKMa+4vXUfVxPPavVUIx9xTy7jzx7VquhGPuKeXYfefasVkMx9hXz7J559luths3ioxD7w0Q7qz+/9c5wHNsgkRye4fjQ"
    "IIf4yLHfemc4PjXIKT4nh2c4vjTIJb4mh2c4vrdBIjk8w/GjQR7xk2O/9c5w3DRIE7fJ4RmG47MNguTwDMfqk1BfjKz6t94Z"
    "jtUnoTUfx+TwDMfqE80T+smZQn/qneFYfaL5HUb157feGY61ijX96PglDs9wrFUcWt7xrrS33hmOuwbp4j45PMOwQuaPrnUe"
    "1Z/femc4Vp9Aax4xOTzDMTQIxJgcnuFYzaZpFEf157feGY7PbZBIDs9wrGaD2gfZPrPeGY7VbFD7INtHszA6hmM1G9Q+yPaZ"
    "9c5w3LdBIjk8w/Dx2QZBcniGYzXbofY5sn1mvTMcq08OtY9+icMzHKtPDvXFkZuZWe8Mx9c2SCSHZzjWh9ehNT8umv6td4Zj"
    "9cmh5X20yeEZjtUnh/ri6JPDMwwrDf6cWt4nJodnONYq1kwTo/rzW+8Mx+c2SCSHZzjWKj61Ss98c896ZzjWKtYkm6P681vv"
    "DMfPNkgkh2c41io+tebPNjk8w7AC4c+lt/6VnySz3hmOYxskksMzHGMbBMnhGY6PbZAjOTzD8bkNciaHZzhWn1xqnyv3TDXh"
    "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"
)


def _dumped_cell_step(t, tc, nc, max_q_len, g_reqs, next_n, rng):
    news = []
    if nc:
        rest = tc - max_q_len
        news = [max_q_len] + [
            rest // (nc - 1) + (1 if i < rest % (nc - 1) else 0) for i in range(nc - 1)
        ]
    assert sum(news) + g_reqs * next_n == t and all(n >= 1 for n in news)
    ctx = [(rng.randint(0, MAX_REQUEST_LEN - n), n) for n in news]
    gen = [rng.randint(0, MAX_REQUEST_LEN - next_n) for _ in range(g_reqs)]
    return ctx, gen, next_n


def test_dumped_partitions():
    """Every dumped (T, ratio) cell as a consistent step, bitwise equal to the reference, under
    torch.profiler (the one Triton kernel, no new compiled variant)."""
    cells = [
        tuple(map(int, c.split(",")))
        for c in zlib.decompress(base64.b64decode("".join(DUMPED_CELLS))).decode().split()
    ]
    assert len(cells) == 2696
    rng = random.Random(23300)
    tables = {r: _multi_request_tables(r, seed=23400 + r) for r in (0, 1, 2)}
    for r in (0, 1, 2):
        warm = _multi_request_step(r, [(0, 4)], [], 1, seed=2)
        _target_call(
            r, warm[0], tables[r][0], warm[1], tables[r][1], warm[2], SWA_TPB // r if r else 0
        )
    torch.cuda.synchronize()
    before = _compiled_variants()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (t, ratio, tc, nc, mq, g_reqs, next_n) in enumerate(cells):
            ctx, gen, nn = _dumped_cell_step(t, tc, nc, mq, g_reqs, next_n, rng)
            swa, comp = tables[ratio]
            req_id, swa_local, comp_local, _, _ = _multi_request_step(
                ratio, ctx, gen, nn, seed=60000 + i
            )
            assert req_id.numel() == t
            tpb_c = SWA_TPB // ratio if ratio else 0
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            assert torch.equal(
                out, _multi_request_reference(ratio, req_id, swa, swa_local, comp, comp_local)
            ), cells[i]
    torch.cuda.synchronize()
    print(
        f"{len(cells)} dumped cells exact; ratios {sorted({c[1] for c in cells})}, "
        f"T {min(c[0] for c in cells)}..{max(c[0] for c in cells)}"
    )
    _check_variants(prof, len(cells), before)


# ---------------------------------------------------------------------------------------------
# Pool bound: page ids up to each pool's B_max, for every role the op
# maps. The op takes no pool tensor. A pool's size reaches it only as the magnitude of the page ids
# in the block tables, and every result, page * tpb + slot, is formed in int32 arithmetic. So the
# pool-size axis is exactly the addressing axis, and it is run here at every named bound up to the
# edge of the int32 result range, against an int64 reference (no cast that could hide a wrap).
# Page counts are pages of the role's pool view; the ratio-2 COMPRESS view holds 3 pages per block,
# so its count is 3 x the block count.
# - SWA (128-row pages, every ratio): 188,160 (the engine the dumps came from); 199,920 / 238,560 /
#   360,440 / 468,720 (batch 8 at max_num_tokens 1024 / 2048 / 4096 / 8192, quota
#   148,510,474,240 B); 261,600 (accuracy gate); 491,000 (DSpark included).
# - COMPRESS ratio 1 (128-row pages): 151,040; 175,786 / 231,680 / 268,288 / 300,032 (batch 8 at
#   max_num_tokens 8192 / 4096 / 2048 / 1024); 287,573 (the SWA 491,000 configuration); 333,824
#   (batch 32 at 2048).
# - COMPRESS ratio 2 (64-row pages): 453,120; 804,864; 862,719; 1,001,472.
# - The GB300 block ceiling, 2,615,990 blocks (7,847,970 ratio-2 pages), for every role.
# - The int32 edge: 2^24 pages of 128 rows and 2^25 pages of 64 rows, whose last row is 2^31 - 1.
#   A pool past it would need 2^31 rows in 298,013,687,808 B of device memory, under 139 B a row;
#   a V4.1 KV row holds 512 elements.
# Block tables are the engine's width, max_blocks_per_seq = 132 at max_seq_len 16384 (the SWA
# table a [layer, type] view of a 4-D buffer, as the metadata holds it).
import itertools  # noqa: E402

POOL_BLOCK_CEILING = 2_615_990
POOL_INT32_EDGE = {128: 2**24, 64: 2**25}
POOL_BOUNDS = {
    "swa": [
        188_160,
        199_920,
        238_560,
        261_600,
        360_440,
        468_720,
        491_000,
        POOL_BLOCK_CEILING,
        POOL_INT32_EDGE[128],
    ],
    1: [
        151_040,
        175_786,
        231_680,
        268_288,
        287_573,
        300_032,
        333_824,
        POOL_BLOCK_CEILING,
        POOL_INT32_EDGE[128],
    ],
    2: [
        453_120,
        804_864,
        862_719,
        1_001_472,
        POOL_BLOCK_CEILING,
        3 * POOL_BLOCK_CEILING,
        POOL_INT32_EDGE[64],
    ],
}
POOL_BOUND_WIDTH = 132
POOL_STEPS_PER_BOUND = 25


def _reference64(req_id, table, local, tpb):
    """One segment's pool rows in int64 (offset 0), never cast: a result at or past 2**31
    stays visible. This is `op.reference` on that segment alone."""
    ref = op.reference(req_id, table, local, tpb, 0)
    assert ref.dtype == torch.int64
    return ref


def _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local):
    ref = _reference64(req_id, swa, swa_local, SWA_TPB)
    if ratio:
        ref = torch.cat([ref, _reference64(req_id, comp, comp_local, SWA_TPB // ratio)], dim=1)
    return ref


def _pool_bound_pairs(ratio):
    """(SWA bound, COMPRESS bound) per sweep point: every bound of each role the ratio maps."""
    swa = POOL_BOUNDS["swa"]
    if ratio == 0:
        return [(b, None) for b in swa]
    comp = POOL_BOUNDS[ratio]
    return list(itertools.zip_longest(swa, comp, fillvalue=None))


def _pool_bound_fill(pairs):
    """Pad a zip_longest column with the role's last (largest) bound."""
    out, last = [], None
    for b in pairs:
        last = b if b is not None else last
        out.append(last)
    return out


def _pool_bound_table(pages, g, layered):
    """[ENGINE_ROWS, POOL_BOUND_WIDTH] page ids drawn uniformly from [0, pages). layered: the view
    [1, 0] of a [2, 2, ENGINE_ROWS, POOL_BOUND_WIDTH] buffer (the metadata's sliding_block_tables)."""
    ids = torch.randint(
        0, pages, (ENGINE_ROWS, POOL_BOUND_WIDTH), generator=g, device=DEV, dtype=torch.int64
    )
    ids = ids.to(torch.int32)
    if not layered:
        return ids
    buf = torch.full((2, 2, ENGINE_ROWS, POOL_BOUND_WIDTH), -1, dtype=torch.int32, device=DEV)
    buf[1, 0] = ids
    return buf[1, 0]


def _pool_bound_far_newest(table, rows, starts, local, tpb, pages, total):
    """Each request's newest page (the page of the largest local index its last token names) is
    pages - 1 - (b % 4), so the pool's last pages appear in every step."""
    ends = starts[1:] + [total]
    for b, end in enumerate(ends):
        newest = int(local[end - 1].max())
        if newest >= 0:
            table[rows[b], newest // tpb] = pages - 1 - (b % 4)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_page_ids_up_to_every_pool_bound(ratio):
    """POOL_STEPS_PER_BOUND seeded random mixed steps (multi-request partitions: k 1..32, T 1..8192, context
    then generation rows, next_n 1 or 6) at every (SWA, COMPRESS) bound of the ratio, on width-132
    tables whose page ids are drawn from the whole of [0, B) with each request's newest page among
    the pool's last four. Every call equals the int64 reference, so no result wrapped. Pool
    translation: step 0's page ids folded into the smallest pool (mod 188160 / the role's smallest
    COMPRESS pool) and moved up by delta = B - B_min give outputs that differ by exactly
    delta * tpb at every valid entry, with the same -1s. Under torch.profiler: only the one
    kernel, and no compiled variant added by the sweep (page magnitude is data). Control: page ids
    truncated to their low 18 bits must fail."""
    rng = random.Random(24100 + ratio)
    g = torch.Generator(device=DEV).manual_seed(24200 + ratio)
    tpb_c = SWA_TPB // ratio if ratio else 0
    pairs = _pool_bound_pairs(ratio)
    swa_bounds = _pool_bound_fill([p[0] for p in pairs])
    comp_bounds = _pool_bound_fill([p[1] for p in pairs]) if ratio else [None] * len(pairs)
    warm_swa = _pool_bound_table(1024, g, layered=True)
    warm_comp = _pool_bound_table(1024, g, layered=False) if ratio else None
    warm = _multi_request_step(ratio, [(0, 4)], [], 1, seed=3)
    _target_call(ratio, warm[0], warm_swa, warm[1], warm_comp, warm[2], tpb_c)
    torch.cuda.synchronize()
    before = _compiled_variants()
    n_calls, rows_total, kinds, controls, top = 0, 0, set(), [], []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for bi, (b_swa, b_comp) in enumerate(zip(swa_bounds, comp_bounds)):
            swa = _pool_bound_table(b_swa, g, layered=True)
            comp = _pool_bound_table(b_comp, g, layered=False) if ratio else None
            max_row = [0, 0]
            for s in range(POOL_STEPS_PER_BOUND):
                ctx, gen, next_n = _multi_request_random_mixed(rng)
                req_id, swa_local, comp_local, starts, rows = _multi_request_step(
                    ratio, ctx, gen, next_n, seed=24300 + 1000 * ratio + 100 * bi + s
                )
                T = req_id.numel()
                _pool_bound_far_newest(swa, rows, starts, swa_local, SWA_TPB, b_swa, T)
                if ratio:
                    _pool_bound_far_newest(comp, rows, starts, comp_local, tpb_c, b_comp, T)
                out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
                ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
                n_calls += 1
                assert int(ref.max()) < 2**31
                assert torch.equal(out.long(), ref), (ratio, b_swa, b_comp, s)
                max_row[0] = max(max_row[0], int(ref[:, :WINDOW].max()))
                if ratio:
                    max_row[1] = max(max_row[1], int(ref[:, WINDOW:].max()))
                rows_total += T
                kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
                if s == 0:
                    # Pool translation: the step's page ids folded into the smallest pool
                    # (page mod B_min) and the same ids moved to the top of this bound's pool
                    # (+ delta, the largest landing on B - 1): both equal the reference, and they
                    # differ by exactly delta * tpb at every valid entry, with the same -1s.
                    d_swa = b_swa - ENGINE_SWA_PAGES
                    lo_swa = torch.remainder(swa, ENGINE_SWA_PAGES)
                    lo_comp, hi_comp, d_comp = None, None, 0
                    if ratio:
                        base = POOL_BOUNDS[ratio][0]
                        d_comp = b_comp - base
                        lo_comp = torch.remainder(comp, base)
                        hi_comp = lo_comp + d_comp
                    hi_swa = lo_swa + d_swa
                    lo = _target_call(ratio, req_id, lo_swa, swa_local, lo_comp, comp_local, tpb_c)
                    hi = _target_call(ratio, req_id, hi_swa, swa_local, hi_comp, comp_local, tpb_c)
                    n_calls += 2
                    ref_hi = _step_reference64(
                        ratio, req_id, hi_swa, swa_local, hi_comp, comp_local
                    )
                    assert torch.equal(
                        lo.long(),
                        _step_reference64(ratio, req_id, lo_swa, swa_local, lo_comp, comp_local),
                    )
                    assert torch.equal(hi.long(), ref_hi)
                    shift = torch.full_like(ref_hi, d_swa * SWA_TPB)
                    if ratio:
                        shift[:, WINDOW:] = d_comp * tpb_c
                    assert torch.equal(torch.where(lo >= 0, lo.long() + shift, -1), hi.long()), (
                        b_swa,
                        b_comp,
                    )
                    # Control: page ids truncated to their low 18 bits (a narrow page-id path).
                    if b_swa > 2**18 or (ratio and b_comp > 2**18):
                        cut_swa = swa & (2**18 - 1)
                        cut_comp = comp & (2**18 - 1) if ratio else None
                        wrong = _target_call(
                            ratio, req_id, cut_swa, swa_local, cut_comp, comp_local, tpb_c
                        )
                        n_calls += 1
                        controls.append(int((wrong.long() != ref).sum()))
            top.append((b_swa, b_comp, max_row[0], max_row[1] if ratio else None))
    torch.cuda.synchronize()
    for b_swa, b_comp, m_swa, m_comp in top:
        print(
            f"ratio={ratio} SWA B {b_swa}: largest row {m_swa} (last {b_swa * SWA_TPB - 1})"
            + (
                f"; COMPRESS B {b_comp}: largest row {m_comp} (last {b_comp * tpb_c - 1})"
                if ratio
                else ""
            )
        )
    print(
        f"ratio={ratio}: {n_calls} calls, {rows_total} rows, kinds {sorted(kinds)}; "
        f"control (page ids cut to 18 bits) elements differing {controls}"
    )
    assert len(controls) >= 4 and all(c > 0 for c in controls), controls
    assert {
        (True, True, 1),
        (True, True, 6),
        (False, True, 1),
        (False, True, 6),
        (True, False, 0),
    } <= kinds
    _check_variants(prof, n_calls, before)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_far_end_rows_of_every_bound(ratio):
    """At every bound of each role the ratio maps: one token whose table row names the pool's last
    pages B-1, B-2, ... and whose local indices enumerate every slot of them (window: positions
    0..127 on page B-1; compressed: 0..511 over pages B-1 .. B-512/tpb). Exact vs int64, and the
    pool's last row B * tpb - 1 is produced. At the int32 edge that row is 2**31 - 1."""
    tpb_c = SWA_TPB // ratio if ratio else 0
    req_id = torch.zeros(1, dtype=torch.int32, device=DEV)
    swa_local = torch.arange(WINDOW, dtype=torch.int32, device=DEV)[None, :]
    comp_local = torch.arange(TOPK, dtype=torch.int32, device=DEV)[None, :] if ratio else None
    cols = torch.arange(POOL_BOUND_WIDTH, device=DEV)
    pairs = _pool_bound_pairs(ratio)
    for b_swa, b_comp in pairs:
        for role, B in (("swa", b_swa), ("comp", b_comp)):
            if B is None:
                continue
            swa_B = B if role == "swa" else ENGINE_SWA_PAGES
            comp_B = B if role == "comp" else (POOL_BOUNDS[ratio][0] if ratio else None)
            swa = (swa_B - 1 - cols).to(torch.int32)[None, :]
            comp = (comp_B - 1 - cols).to(torch.int32)[None, :] if ratio else None
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
            assert torch.equal(out.long(), ref), (ratio, role, B)
            tpb = SWA_TPB if role == "swa" else tpb_c
            seg = out[0, :WINDOW] if role == "swa" else out[0, WINDOW:]
            assert int(seg.max()) == B * tpb - 1, (role, B, int(seg.max()))
    swa = torch.tensor([[POOL_INT32_EDGE[SWA_TPB] - 1]], dtype=torch.int32, device=DEV)
    comp = (
        torch.tensor([[POOL_INT32_EDGE[tpb_c] - 1] + [0] * 7], dtype=torch.int32, device=DEV)
        if ratio
        else None
    )
    edge_local = torch.where(comp_local < tpb_c, comp_local, -1).to(torch.int32) if ratio else None
    out = _target_call(ratio, req_id, swa, swa_local, comp, edge_local, tpb_c)
    assert torch.equal(
        out.long(), _step_reference64(ratio, req_id, swa, swa_local, comp, edge_local)
    )
    assert int(out[0, WINDOW - 1]) == 2**31 - 1
    if ratio:
        assert int(out[0, WINDOW + tpb_c - 1]) == 2**31 - 1


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_control_one_page_past_the_int32_edge_wraps(ratio):
    """Wrong control: page ids one past the int32 edge (2**24 at 128 rows, 2**25 at 64 rows) give
    results of 2**31 and up, which the op does not check (Precondition 5): they come out wrapped
    negative, and the same int64 comparison the bound tests use rejects them."""
    tpb_c = SWA_TPB // ratio if ratio else 0
    req_id = torch.zeros(1, dtype=torch.int32, device=DEV)
    swa_local = torch.arange(WINDOW, dtype=torch.int32, device=DEV)[None, :]
    swa = torch.tensor([[POOL_INT32_EDGE[SWA_TPB]]], dtype=torch.int32, device=DEV)
    comp = (
        torch.tensor([[POOL_INT32_EDGE[tpb_c]]], dtype=torch.int32, device=DEV) if ratio else None
    )
    comp_local = None
    if ratio:
        comp_local = torch.full((1, TOPK), -1, dtype=torch.int32, device=DEV)
        comp_local[0, :tpb_c] = torch.arange(tpb_c, dtype=torch.int32, device=DEV)
    out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
    ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
    assert int(ref.max()) >= 2**31
    assert not torch.equal(out.long(), ref)
    assert int(out[0, 0]) == -(2**31)  # 2**31 wrapped
    print(f"ratio={ratio}: one page past the edge -> {int(out[0, 0])} .. {int(out[0, WINDOW - 1])}")


@pytest.mark.parametrize("rows,next_n", [(32, 1), (192, 6)], ids=["T32_n1", "T192_n6"])
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_captured_call_over_every_pool_bound(ratio, rows, next_n):
    """The largest graph batches (decode 32; DSpark verify 32 x 6), captured once on width-132
    tables as the target captures the call, then replayed with tables drawn from [0, B) at every
    bound of each role (newest pages at the far end): each replay equals the int64 reference and an
    eager call on the same inputs."""
    g = torch.Generator(device=DEV).manual_seed(24500 + 10 * ratio + next_n)
    gh = torch.Generator(device="cpu").manual_seed(24600 + 10 * ratio + next_n)
    tpb_c = SWA_TPB // ratio if ratio else 0
    max_tokens = 8192
    req_buf = torch.zeros(max_tokens, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((max_tokens, WINDOW), -1, dtype=torch.int32, device=DEV)
    comp_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV) if ratio else None
    swa_table = _pool_bound_table(1024, g, layered=True)
    comp_table = _pool_bound_table(1024, g, layered=False) if ratio else None

    def load(seqs, b_swa, b_comp, seed):
        req_id, swa_local, comp_local, starts, rrows = _multi_request_step(
            ratio, [], [c for c, _ in seqs], next_n, seed
        )
        swa_table.copy_(_pool_bound_table(b_swa, g, layered=True))
        _pool_bound_far_newest(swa_table, rrows, starts, swa_local, SWA_TPB, b_swa, rows)
        req_buf[:rows].copy_(req_id)
        swa_buf[:rows].copy_(swa_local)
        if ratio:
            comp_table.copy_(_pool_bound_table(b_comp, g, layered=False))
            _pool_bound_far_newest(comp_table, rrows, starts, comp_local, tpb_c, b_comp, rows)
            comp_buf.copy_(comp_local)
        return _step_reference64(ratio, req_id, swa_table, swa_local, comp_table, comp_local)

    def call():
        return _target_call(
            ratio, req_buf[:rows], swa_table, swa_buf[:rows], comp_table, comp_buf, tpb_c
        )

    pairs = _pool_bound_pairs(ratio)
    swa_bounds = _pool_bound_fill([p[0] for p in pairs])
    comp_bounds = _pool_bound_fill([p[1] for p in pairs]) if ratio else [None] * len(pairs)
    ref = load(_decode_seqs(rows, next_n, gh, 0), 1024, 1024 if ratio else None, seed=24700)
    assert torch.equal(call().long(), ref)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for i, (b_swa, b_comp) in enumerate(zip(swa_bounds, comp_bounds)):
        ref = load(_decode_seqs(rows, next_n, gh, i + 1), b_swa, b_comp, seed=24800 + i)
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.long(), ref), (b_swa, b_comp)
        assert torch.equal(out, call()), (b_swa, b_comp)


# ---------------------------------------------------------------------------------------------
# The DSpark envelope. With DSpark (max_draft_len 5) and the overlap scheduler
# on (every gate), the executor widens the engine's max_seq_len to 16384 + 5 + 4 + 5 = 16398
# (py_executor_creator: tokens_per_gen_step - 1 for the overlap scheduler, the one-engine extra
# KV tokens max_draft_len - 1, and tokens_per_gen_step - 1). A request's committed tokens stay
# <= 16384 (the executor keeps the net max_seq_len, and a prompt must leave max_tokens > 0), so
# context rows stay below position 16384. A verify step at the end of a full-length request writes
# generation positions up to 16397: SWA logical page 128 (table column 128), and compressed
# positions up to (16397 + 1) // r - 1 = 16397 (ratio 1) / 8198 (ratio 2), also column 128. The
# manager's tables stay max_blocks_per_seq = 132 columns wide (16398 + 10 -> 129 -> 132).
DSPARK_MAX_POS = 16397
DSPARK_BOUNDS = {
    # engine of the dumps, native-storage cross-product maximum, the GB300 block
    # ceiling, the int32 edge
    "swa": [188_160, 854_440, POOL_BLOCK_CEILING, POOL_INT32_EDGE[128]],
    1: [151_040, 868_673, POOL_BLOCK_CEILING, POOL_INT32_EDGE[128]],
    2: [453_120, 2_606_019, 3 * POOL_BLOCK_CEILING, POOL_INT32_EDGE[64]],
}
# Contexts ahead of the generation rows: a fresh prompt, a chunk on an 8192-token prefix, 1- and
# 5-token contexts, and chunks ending at positions 16382 and 16383 (the last context position).
DSPARK_CTX = [(0, 3000), (8192, 2048), (0, 1), (130, 5), (15405, 978), (16000, 384)]
# Generation requests at the DSpark tail, as `cached` (first position of the step):
# next_n 1: decode rows at every position 16383 .. 16397;
# next_n 6: verify runs ending at 16397, 16396, 16393, 16392, 16389, 16388, 16384 and 16383.
DSPARK_TAIL = {
    1: list(range(16383, DSPARK_MAX_POS + 1)),
    6: [e - 5 for e in (16397, 16396, 16393, 16392, 16389, 16388, 16384, 16383)],
}


def _dspark_bound_pairs(ratio):
    comp = DSPARK_BOUNDS[ratio] if ratio else [None] * len(DSPARK_BOUNDS["swa"])
    return list(zip(DSPARK_BOUNDS["swa"], comp))


def _dspark_step(ratio, ctx, gen, next_n, seed):
    """Like _multi_request_step, with generation positions up to DSPARK_MAX_POS (contexts below 16384), and the
    newest visible compressed position always among a row's top-k picks (so a tail row names
    compressed table column 128)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seqs = list(ctx) + [(c, next_n) for c in gen]
    assert len(seqs) <= 32
    assert all(c + n <= MAX_REQUEST_LEN for c, n in ctx)
    rows = torch.randperm(ENGINE_ROWS, generator=g, device=DEV)[: len(seqs)].to(torch.int32)
    req_id = torch.cat([rows[b].repeat(new) for b, (_, new) in enumerate(seqs)]).contiguous()
    positions = torch.cat(
        [torch.arange(c, c + new, device=DEV, dtype=torch.int32) for c, new in seqs]
    )
    assert req_id.numel() <= 8192 and int(positions.max()) <= DSPARK_MAX_POS
    swa_local = _window(positions)
    comp_local = _topk_local_with_newest(positions, ratio, g) if ratio else None
    starts = [0]
    for _, new in seqs[:-1]:
        starts.append(starts[-1] + new)
    return req_id, swa_local, comp_local, starts, rows


def _dspark_col128(local, tpb):
    """Valid local indices this call resolves through table column 128."""
    return int(((local >= 0) & (torch.div(local, tpb, rounding_mode="floor") == 128)).sum())


@pytest.mark.parametrize("next_n", [1, 6])
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_dspark_tail_mixed_steps(ratio, next_n):
    """Mixed steps at the DSpark tail: 6 contexts (positions < 16384) ahead of 26 generation
    requests, the tail cases of DSPARK_TAIL (decode rows at every position 16383..16397, or verify
    runs ending at 16383..16397, both parities) plus random others up to position 16397, on
    132-column engine tables at every DSpark pool bound (each request's newest page at the pool's far
    end). Every call equals the int64 reference, and table column 128 is read in the SWA and the
    compressed segment. Control: the same step on the tables cut to 128 columns (a pre-DSpark
    width) differs from the reference exactly at the column-128 entries, which come back -1."""
    g = torch.Generator(device=DEV).manual_seed(25100 + 10 * ratio + next_n)
    rng = random.Random(25200 + 10 * ratio + next_n)
    tpb_c = SWA_TPB // ratio if ratio else 0
    tail = DSPARK_TAIL[next_n]
    gen = list(tail) + [
        rng.randint(0, DSPARK_MAX_POS + 1 - next_n) for _ in range(32 - len(DSPARK_CTX) - len(tail))
    ]
    rng.shuffle(gen)
    req_id, swa_local, comp_local, starts, rows = _dspark_step(
        ratio, DSPARK_CTX, gen, next_n, seed=25300 + 10 * ratio + next_n
    )
    T = req_id.numel()
    col128 = [_dspark_col128(swa_local, SWA_TPB)] + (
        [_dspark_col128(comp_local, tpb_c)] if ratio else []
    )
    assert all(c > 0 for c in col128), col128
    for b_swa, b_comp in _dspark_bound_pairs(ratio):
        swa = _pool_bound_table(b_swa, g, layered=True)
        comp = _pool_bound_table(b_comp, g, layered=False) if ratio else None
        _pool_bound_far_newest(swa, rows, starts, swa_local, SWA_TPB, b_swa, T)
        if ratio:
            _pool_bound_far_newest(comp, rows, starts, comp_local, tpb_c, b_comp, T)
        out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
        ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
        assert int(ref.max()) < 2**31
        assert torch.equal(out.long(), ref), (ratio, next_n, b_swa, b_comp)
        # Control: tables one page short of column 128.
        cut = _target_call(
            ratio,
            req_id,
            swa[:, :128],
            swa_local,
            None if comp is None else comp[:, :128],
            comp_local,
            tpb_c,
        )
        diff = cut.long() != ref
        assert int(diff.sum()) == sum(col128), (int(diff.sum()), col128)
        assert bool((cut[diff] == -1).all())
    print(
        f"ratio={ratio} next_n={next_n}: T {T}, {len(gen)} generation requests, column-128 entries "
        f"(SWA, compressed) {col128} at bounds {_dspark_bound_pairs(ratio)}"
    )


def _dspark_random_mixed(rng):
    """_multi_request_random_mixed with generation requests allowed up to position 16397: each generation
    request's first position is drawn from the DSpark tail [16384 - next_n, 16398 - next_n] with
    probability 1/4 and from [0, 16398 - next_n] otherwise. Contexts stay below 16384."""
    k = rng.randint(1, 32)
    next_n = rng.choice((1, 6))
    n_gen = rng.randint(0, k)
    n_ctx = k - n_gen
    gen_rows = n_gen * next_n
    ctx = []
    if n_ctx:
        ctx_rows = rng.randint(gen_rows + n_ctx, 8192) - gen_rows
        cuts = sorted(rng.sample(range(1, ctx_rows), n_ctx - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        ctx = [(rng.randint(0, MAX_REQUEST_LEN - n), n) for n in news]
    top = DSPARK_MAX_POS + 1 - next_n
    gen = [
        rng.randint(MAX_REQUEST_LEN - next_n, top) if rng.random() < 0.25 else rng.randint(0, top)
        for _ in range(n_gen)
    ]
    return ctx, gen, next_n


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_random_mixed_partitions_with_dspark_tail(ratio):
    """200 seeded multi-request mixed steps (k 1..32, T 1..8192, contexts then generation rows, next_n 1 or
    6) with generation positions up to 16397, on 132-column tables drawn from [0, B) at the DSpark pool
    bounds in turn (newest pages at the far end): every call equals the int64 reference. Under
    torch.profiler only the one kernel launches, and the sweep adds no compiled variant after one
    warm-up call in the 132-column form."""
    rng = random.Random(25400 + ratio)
    g = torch.Generator(device=DEV).manual_seed(25500 + ratio)
    tpb_c = SWA_TPB // ratio if ratio else 0
    pairs = _dspark_bound_pairs(ratio)
    tables = [
        (
            b_swa,
            b_comp,
            _pool_bound_table(b_swa, g, layered=True),
            _pool_bound_table(b_comp, g, layered=False) if ratio else None,
        )
        for b_swa, b_comp in pairs
    ]
    warm = _dspark_step(ratio, [(0, 4)], [], 1, seed=5)
    _target_call(ratio, warm[0], tables[0][2], warm[1], tables[0][3], warm[2], tpb_c)
    torch.cuda.synchronize()
    before = _compiled_variants()
    n_calls, rows_total, tail_rows, col128, kinds = 0, 0, 0, [0, 0], set()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(200):
            ctx, gen, next_n = _dspark_random_mixed(rng)
            b_swa, b_comp, swa, comp = tables[i % len(tables)]
            req_id, swa_local, comp_local, starts, rows = _dspark_step(
                ratio, ctx, gen, next_n, seed=25600 + 1000 * ratio + i
            )
            T = req_id.numel()
            _pool_bound_far_newest(swa, rows, starts, swa_local, SWA_TPB, b_swa, T)
            if ratio:
                _pool_bound_far_newest(comp, rows, starts, comp_local, tpb_c, b_comp, T)
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
            n_calls += 1
            assert int(ref.max()) < 2**31
            assert torch.equal(out.long(), ref), (ratio, i, ctx, gen, next_n)
            rows_total += T
            tail_rows += sum(next_n for c in gen if c + next_n - 1 >= MAX_REQUEST_LEN)
            col128[0] += _dspark_col128(swa_local, SWA_TPB)
            if ratio:
                col128[1] += _dspark_col128(comp_local, tpb_c)
            kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
    torch.cuda.synchronize()
    print(
        f"ratio={ratio}: {n_calls} calls, {rows_total} rows, {tail_rows} rows of requests reaching "
        f"positions >= 16384, column-128 entries (SWA, compressed) {col128}, kinds {sorted(kinds)}"
    )
    assert tail_rows > 0 and col128[0] > 0 and (col128[1] > 0 or not ratio)
    assert {
        (True, True, 1),
        (True, True, 6),
        (False, True, 1),
        (False, True, 6),
        (True, False, 0),
    } <= kinds
    _check_variants(prof, n_calls, before)


def _dspark_decode_seqs(rows, next_n, rng, step):
    """rows // next_n generation requests up to position 16397; request 0 ends at 16397 and
    request 1 at 16392 (even steps) or 16396 / 16383 (odd steps), the rest random."""
    batch = rows // next_n
    top = DSPARK_MAX_POS + 1 - next_n
    cached = [rng.randint(0, top) for _ in range(batch)]
    ends = (DSPARK_MAX_POS, 16392) if step % 2 == 0 else (16396, 16383)
    cached[0] = ends[0] - next_n + 1
    if batch > 1:
        cached[1] = ends[1] - next_n + 1
    return cached


@pytest.mark.parametrize("rows,next_n", [(32, 1), (192, 6)], ids=["T32_n1", "T192_n6"])
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_dspark_tail_captured(ratio, rows, next_n):
    """The largest graph batches (decode 32; DSpark verify 32 x 6) captured once on 132-column
    tables, as the target captures the call (prefix views of max_num_tokens-row buffers, static
    tables and top-k), then replayed over DSpark-tail steps (request 0 ending at 16397) with tables
    drawn at every DSpark pool bound: each replay equals the int64 reference and an eager call on the same
    inputs, differs from the previous step's reference, and reads table column 128."""
    g = torch.Generator(device=DEV).manual_seed(25700 + 10 * ratio + next_n)
    rng = random.Random(25800 + 10 * ratio + next_n)
    tpb_c = SWA_TPB // ratio if ratio else 0
    req_buf = torch.zeros(8192, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((8192, WINDOW), -1, dtype=torch.int32, device=DEV)
    comp_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV) if ratio else None
    swa_table = _pool_bound_table(1024, g, layered=True)
    comp_table = _pool_bound_table(1024, g, layered=False) if ratio else None

    def load(step_i, b_swa, b_comp):
        gen = _dspark_decode_seqs(rows, next_n, rng, step_i)
        req_id, swa_local, comp_local, starts, rrows = _dspark_step(
            ratio, [], gen, next_n, seed=25900 + 10 * ratio + step_i
        )
        assert _dspark_col128(swa_local, SWA_TPB) > 0 and (
            not ratio or _dspark_col128(comp_local, tpb_c) > 0
        )
        swa_table.copy_(_pool_bound_table(b_swa, g, layered=True))
        _pool_bound_far_newest(swa_table, rrows, starts, swa_local, SWA_TPB, b_swa, rows)
        req_buf[:rows].copy_(req_id)
        swa_buf[:rows].copy_(swa_local)
        if ratio:
            comp_table.copy_(_pool_bound_table(b_comp, g, layered=False))
            _pool_bound_far_newest(comp_table, rrows, starts, comp_local, tpb_c, b_comp, rows)
            comp_buf.copy_(comp_local)
        return _step_reference64(ratio, req_id, swa_table, swa_local, comp_table, comp_local)

    def call():
        return _target_call(
            ratio, req_buf[:rows], swa_table, swa_buf[:rows], comp_table, comp_buf, tpb_c
        )

    pairs = _dspark_bound_pairs(ratio)
    prev = load(0, *pairs[0])
    assert torch.equal(call().long(), prev)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for i, (b_swa, b_comp) in enumerate(pairs + pairs[:1]):
        ref = load(i + 1, b_swa, b_comp)
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.long(), ref), (i, b_swa, b_comp)
        assert torch.equal(out, call()), (i, b_swa, b_comp)
        assert not torch.equal(out.long(), prev), i


# ---------------------------------------------------------------------------------------------
# The native-storage compact read. Under native KV storage the
# target dequantizes, every step, every compressed entry of every request of the step into ONE
# persistent bf16 buffer: request b's entry e (every e < kv_len // r) at row b * L_r + e, where
# L_r = 132 * 128 / r (16,896 at ratio 1, 8,448 at ratio 2) and b is the request's position
# 0..B-1 in the step (the row of md.compress_block_tables[r] the dequantizer reads for it). The
# attention's compressed columns are then built by this op with a SYNTHETIC page table in place of
# md.compress_block_tables[r]:
#   comp_tbl = torch.arange(max_batch_size * 132, dtype=torch.int32).view(max_batch_size, 132)
# allocated once at load and shared by BOTH ratios (one tensor), in the call
#   op(req_idx_per_token[:T], swa_table, swa_local[:T], 128, 0,
#                                comp_tbl, topk, 128 // r, 0)
# so entry e of request b maps to (b * 132 + e // tpb) * tpb + e % tpb = b * L_r + e. The reference
# below forms b * L_r + e directly, never through a table, so what is tested is that the table
# implies the buffer's numbering. The window half is the ordinary call on the real SWA table: the
# [layer, SWA] view [max_batch_size, 132] of a 4-D buffer, as the metadata holds it, with page ids
# up to the native SWA pool's largest size over the max_batch_size axis.
#
# max_batch_size is an axis here: the table is [max_batch_size, 132] for EVERY
# max_batch_size 1..32 (the metadata's compress_block_tables are [max_batch_size, 132] at every
# max_batch_size 1..32 x max_num_tokens 2048 / 8192 x DSpark off / on, max_seq_len 16384 / 16398,
# and the CUDA-graph metadata copy keeps that shape; measured in the dequantizer entries' native-storage
# tests). So the compact buffer the table numbers holds max_batch_size * L_r rows, and the largest
# row a step can produce is (max_batch_size - 1) * L_r + 16398 // r - 1. The witness
# configuration (max_num_tokens 2048, max_batch_size 1) runs as its own configuration.
KVQ_COLS = 132
KVQ_L = {1: KVQ_COLS * 128, 2: KVQ_COLS * 64}
KVQ_MBS = tuple(range(1, 33))  # every max_batch_size the engine configuration permits
KVQ_LISTED = (1, 2, 3, 4, 8, 16, 32)
KVQ_CFGS = [(8192, s) for s in KVQ_MBS] + [
    (2048, 1)
]  # (max_num_tokens, max_batch_size); (2048, 1): witness
KVQ_CFG_IDS = [f"T{t}_mbs{s}" for t, s in KVQ_CFGS]
# Native SWA pages over the max_batch_size axis at the gate memory quota: 941,800 at (max_num_tokens
# 8192, max_batch_size 1, no DSpark), above the 854,440 pages over max_batch_size 8 / 32 only. Page
# ids up to 2**24 are certified above (the pool bound); the native-storage cases draw from this.
KVQ_SWA_PAGES = 941_800


def _kvq_table(seqs):
    """The target's synthetic compressed page table for max_batch_size = seqs."""
    return torch.arange(seqs * KVQ_COLS, dtype=torch.int32, device=DEV).view(seqs, KVQ_COLS)


def _kvq_swa(seqs, g):
    """The real SWA table at max_batch_size = seqs: [seqs, 132] page ids drawn uniformly from the native
    SWA pool (KVQ_SWA_PAGES), as the [layer, SWA] view [1, 0] of a [2, 2, seqs, 132] buffer (the
    metadata's sliding_block_tables at max_num_sequences = max_batch_size)."""
    buf = torch.full((2, 2, seqs, KVQ_COLS), -1, dtype=torch.int32, device=DEV)
    buf[1, 0] = torch.randint(
        0, KVQ_SWA_PAGES, (seqs, KVQ_COLS), generator=g, device=DEV, dtype=torch.int64
    ).to(torch.int32)
    return buf[1, 0]


def _kvq_reference(ratio, req_id, swa, swa_local, comp_local):
    """int64: the window half through the real SWA table; the compressed half b * L_r + e for a valid
    local e (e >= 0 and within the table's 132 columns), -1 otherwise -- formed without a table."""
    tpb = SWA_TPB // ratio
    win = _reference64(req_id, swa, swa_local, SWA_TPB)
    e = comp_local.long()
    valid = (e >= 0) & (torch.div(e, tpb, rounding_mode="floor") < KVQ_COLS)
    comp = torch.where(valid, req_id.long()[:, None] * KVQ_L[ratio] + e, -1)
    return torch.cat([win, comp], 1)


def _kvq_step(ratio, ctx, gen, next_n, seed, max_t=8192):
    """Per-token inputs of a step as the metadata builds them: the requests at step positions
    0..B-1 in order (context requests first, then generation requests each verifying next_n
    positions), req_id = the token's request position, the causal window, and the layer's top-k
    (valid compressed positions < (p+1)//ratio first, the newest always among them, then -1).
    Generation positions reach DSPARK_MAX_POS (DSpark); contexts stay below 16384."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seqs = list(ctx) + [(c, next_n) for c in gen]
    assert all(c + n <= MAX_REQUEST_LEN for c, n in ctx)
    req_id = torch.cat(
        [torch.full((n,), b, dtype=torch.int32, device=DEV) for b, (_, n) in enumerate(seqs)]
    )
    positions = torch.cat([torch.arange(c, c + n, device=DEV, dtype=torch.int32) for c, n in seqs])
    assert req_id.numel() <= max_t and int(positions.max()) <= DSPARK_MAX_POS
    return (
        req_id.contiguous(),
        _window(positions),
        _topk_local_with_newest(positions, ratio, g),
        len(seqs),
    )


def _kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local, tpb=None):
    return op(
        req_id,
        swa,
        swa_local,
        SWA_TPB,
        0,
        tbl,
        comp_local,
        SWA_TPB // ratio if tpb is None else tpb,
        0,
    )


@pytest.mark.parametrize("seqs", KVQ_MBS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_kvq_synthetic_table_maps_every_row_of_the_buffer(ratio, seqs):
    """At max_batch_size = seqs (every value 1..32): every request position b in 0..seqs-1 with every
    compressed position e in 0..L_r-1 -- every row of the ratio's view of the compressed read buffer --
    plus positions past the table's 132 columns, through ONE synthetic table used at both ratios, on
    the real [seqs, 132] SWA table: the op gives b * L_r + e, each of the seqs * L_r buffer rows
    exactly once, and -1 at and past column 132."""
    g = torch.Generator(device=DEV).manual_seed(26000 + 100 * ratio + seqs)
    tbl = _kvq_table(seqs)
    L = KVQ_L[ratio]
    per_req = -(-(L + TOPK) // TOPK)  # 512-wide rows of positions 0 .. >= L + 511
    req_id = (
        torch.arange(seqs, dtype=torch.int32, device=DEV).repeat_interleave(per_req).contiguous()
    )
    j = torch.arange(per_req, device=DEV).repeat(seqs)
    comp_local = (j[:, None] * TOPK + torch.arange(TOPK, device=DEV)[None]).to(torch.int32)
    positions = torch.randint(
        0, DSPARK_MAX_POS + 1, (req_id.numel(),), generator=g, device=DEV, dtype=torch.int32
    )
    swa_local = _window(positions)
    swa = _kvq_swa(seqs, g)
    out = _kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local)
    ref = _kvq_reference(ratio, req_id, swa, swa_local, comp_local)
    assert torch.equal(out.long(), ref)
    comp = out[:, WINDOW:]
    rows = comp[comp >= 0].long().sort().values
    assert torch.equal(rows, torch.arange(seqs * L, device=DEV)), "every buffer row exactly once"
    assert int((comp < 0).sum()) == seqs * (per_req * TOPK - L)
    # The other ratio's call on the same table object gives that ratio's numbering.
    other = 3 - ratio
    out2 = _kvq_call(other, req_id, swa, swa_local, tbl, comp_local)
    assert torch.equal(out2.long(), _kvq_reference(other, req_id, swa, swa_local, comp_local))
    print(
        f"ratio={ratio} max_batch_size={seqs}: rows 0..{seqs * L - 1} each produced once over {req_id.numel()} "
        f"tokens; {int((comp < 0).sum())} entries past column 132 -> -1"
    )


def _kvq_random_mixed(rng, seqs, max_t=8192):
    """_dspark_random_mixed with at most `seqs` requests (max_batch_size) and T <= max_t."""
    k = rng.randint(1, seqs)
    next_n = rng.choice((1, 6))
    n_gen = rng.randint(0, k)
    n_ctx = k - n_gen
    gen_rows = n_gen * next_n
    ctx = []
    if n_ctx:
        ctx_rows = rng.randint(gen_rows + n_ctx, max_t) - gen_rows
        cuts = sorted(rng.sample(range(1, ctx_rows), n_ctx - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        ctx = [(rng.randint(0, MAX_REQUEST_LEN - n), n) for n in news]
    top = DSPARK_MAX_POS + 1 - next_n
    gen = [
        rng.randint(MAX_REQUEST_LEN - next_n, top) if rng.random() < 0.25 else rng.randint(0, top)
        for _ in range(n_gen)
    ]
    return ctx, gen, next_n


def _kvq_far_step(seqs):
    """seqs requests (the full batch) whose last is one DSpark verify ending at position 16397, so the
    largest row the target can produce, (seqs - 1) * L_r + 16398 // r - 1, is produced: at seqs 1 the
    verify alone; at 2 a 4000-token context ahead of it; from 3 on also seqs - 2 short contexts."""
    ctx = [] if seqs == 1 else [(16384 - 4000, 4000)] + [(97 * i, 3 + i) for i in range(seqs - 2)]
    return ctx, [DSPARK_MAX_POS - 5], 6


def _kvq_num_steps(cfg):
    """200 random steps at every listed max_batch_size and at the witness configuration, 50 elsewhere."""
    return 200 if cfg[1] in KVQ_LISTED or cfg[0] != 8192 else 50


@pytest.mark.parametrize("cfg", KVQ_CFGS, ids=KVQ_CFG_IDS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_kvq_synthetic_table_random_mixed_steps(ratio, cfg):
    """Random multi-request steps of the native-storage call at one (max_num_tokens, max_batch_size): 200
    seeded steps (50 at a max_batch_size outside KVQ_LISTED) of k <= max_batch_size requests, T
    1..max_num_tokens, contexts then generation rows at next_n 1 or 6, generation positions up to
    16397, plus the far-end step, on the real [max_batch_size, 132] SWA table (each request's newest
    window page among the pool's last four) and the synthetic [max_batch_size, 132] table: every
    call equals the reference (window half through the SWA table, compressed half b * L_r + e),
    every compressed row lies inside the max_batch_size * L_r-row buffer, and the far-end step
    produces the buffer's largest reachable row (max_batch_size - 1) * L_r + 16398 // r - 1. Under
    torch.profiler only the one kernel launches, and no variant is compiled after one warm-up call
    in the target's form at this max_batch_size."""
    max_t, seqs = cfg
    rng = random.Random(26100 + 1000 * ratio + 10 * seqs + (max_t == 2048))
    g = torch.Generator(device=DEV).manual_seed(26200 + 1000 * ratio + 10 * seqs + (max_t == 2048))
    tbl = _kvq_table(seqs)
    swa = _kvq_swa(seqs, g)
    steps = [_kvq_random_mixed(rng, seqs, max_t) for _ in range(_kvq_num_steps(cfg))] + [
        _kvq_far_step(seqs)
    ]
    warm = _kvq_step(ratio, [(0, 4)], [], 1, seed=7)
    _kvq_call(ratio, warm[0], swa, warm[1], tbl, warm[2])
    torch.cuda.synchronize()
    before = _compiled_variants()
    rows_total, kinds, top, ks = 0, set(), -1, set()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (ctx, gen, next_n) in enumerate(steps):
            req_id, swa_local, comp_local, B = _kvq_step(
                ratio,
                ctx,
                gen,
                next_n,
                seed=26300 + 100_000 * ratio + 1000 * seqs + i + 500 * (max_t == 2048),
                max_t=max_t,
            )
            assert B <= seqs
            T = req_id.numel()
            lens = [q for _, q in ctx] + [next_n] * len(gen)
            starts = [sum(lens[:b]) for b in range(B)]
            _pool_bound_far_newest(
                swa, list(range(B)), starts, swa_local, SWA_TPB, KVQ_SWA_PAGES, T
            )
            out = _kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local)
            ref = _kvq_reference(ratio, req_id, swa, swa_local, comp_local)
            assert torch.equal(out.long(), ref), (ratio, cfg, i, ctx, gen, next_n)
            comp = ref[:, WINDOW:]
            assert int(comp.max()) < seqs * KVQ_L[ratio]
            top = max(top, int(comp.max()))
            rows_total += T
            kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
            ks.add(B)
    torch.cuda.synchronize()
    far = (seqs - 1) * KVQ_L[ratio] + (DSPARK_MAX_POS + 1) // ratio - 1
    print(
        f"ratio={ratio} cfg={cfg}: {len(steps)} calls, {rows_total} rows, requests {min(ks)}..{max(ks)}, kinds "
        f"{sorted(kinds)}; largest compressed row {top} (far end {far})"
    )
    assert top == far, (top, far)
    if len(steps) > 100:  # the 200-step configurations reach every step kind
        assert {(False, True, 1), (False, True, 6), (True, False, 0)} <= kinds, kinds
        if seqs >= 2:
            assert {(True, True, 1), (True, True, 6)} <= kinds, kinds
    _check_variants(prof, len(steps), before)


@pytest.mark.parametrize("cfg", KVQ_CFGS, ids=KVQ_CFG_IDS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_kvq_synthetic_table_cuda_graph_capture_set(ratio, cfg):
    """The capture set at one (max_num_tokens, max_batch_size): decode G in EVERY 1..max_batch_size
    (padding off captures all of them; padding on a subset plus max_batch_size) at next_n 1 and 6
    (DSpark verify), each captured as the target captures the call (prefix views of max_num_tokens-row
    req / window buffers, the static [max_batch_size, 132] SWA and synthetic tables, a static [T, 512]
    top-k) after one eager call at that shape. Three replays over new steps (request 0 ending at 16397
    or 16392, a fresh SWA table): each equals the reference and an eager call bitwise, and differs from
    the previous step's reference."""
    max_t, seqs = cfg
    g = torch.Generator(device=DEV).manual_seed(26400 + 1000 * ratio + 10 * seqs + (max_t == 2048))
    rng = random.Random(26500 + 1000 * ratio + 10 * seqs + (max_t == 2048))
    req_buf = torch.zeros(max_t, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((max_t, WINDOW), -1, dtype=torch.int32, device=DEV)
    swa = _kvq_swa(seqs, g)
    tbl = _kvq_table(seqs)
    n_graphs = 0
    for next_n in (1, 6):
        for G in range(1, seqs + 1):
            rows = G * next_n
            top_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV)

            def load(step_i):
                gen = [rng.randint(0, DSPARK_MAX_POS + 1 - next_n) for _ in range(G)]
                gen[0] = (DSPARK_MAX_POS if step_i % 2 == 0 else 16392) - next_n + 1
                req_id, swa_local, comp_local, _ = _kvq_step(
                    ratio,
                    [],
                    gen,
                    next_n,
                    seed=26600 + 97 * step_i + 1000 * rows + seqs,
                    max_t=max_t,
                )
                swa.copy_(_kvq_swa(seqs, g))
                req_buf[:rows].copy_(req_id)
                swa_buf[:rows].copy_(swa_local)
                top_buf.copy_(comp_local)
                return _kvq_reference(ratio, req_id, swa, swa_local, comp_local)

            def call():
                return _kvq_call(ratio, req_buf[:rows], swa, swa_buf[:rows], tbl, top_buf)

            prev = load(0)
            assert torch.equal(call().long(), prev)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = call()
            for it in range(1, 4):
                ref = load(it)
                out.fill_(-7)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out.long(), ref), (cfg, G, next_n, it)
                assert torch.equal(out, call()), (cfg, G, next_n, it)
                assert not torch.equal(out.long(), prev), (cfg, G, next_n, it)
                prev = ref
            del graph
            n_graphs += 1
    print(
        f"ratio={ratio} cfg={cfg}: {n_graphs} captures (G 1..{seqs} x next_n 1, 6) x 3 replays, "
        f"replay = eager = reference"
    )
    assert n_graphs == 2 * seqs


@pytest.mark.parametrize("seqs", [1, 2, 32])
@pytest.mark.parametrize("ratio", [1, 2])
def test_kvq_synthetic_table_wrong_numberings_are_rejected(ratio, seqs):
    """The exact check sees a wrong compact numbering, on 20 random steps at max_batch_size `seqs`
    (1, 2, 32): the table off by one (arange + 1, every valid entry one page late), the other ratio's
    page size on the shared table, and req_idx_per_token off by one request at a request boundary.

    At max_batch_size 1 two of them are not wrong numberings, and the test pins that instead of
    counting them: there is no request boundary, and request 0's row of the table is arange(132), so the
    other ratio's page size gives the same row e wherever e < 132 * that page size (and -1 past it:
    ratio 1 entries e >= 8,448 under the 64-row page size); ratio 2's entries e <= 8,198 all map to e."""
    rng = random.Random(26700 + 10 * ratio + seqs)
    g = torch.Generator(device=DEV).manual_seed(26800 + 10 * ratio + seqs)
    swa = _kvq_swa(seqs, g)
    tbl = _kvq_table(seqs)
    counts = {"table_off_by_one": [], "other_ratio_page_size": [], "req_off_by_one_request": []}
    same_at_b0 = []
    n = 0
    while n < 20:
        ctx, gen, next_n = _kvq_random_mixed(rng, seqs)
        req_id, swa_local, comp_local, B = _kvq_step(
            ratio, ctx, gen, next_n, seed=26900 + 1000 * ratio + 10 * seqs + n
        )
        if not bool(((comp_local >= 0).any(1) & (req_id >= (1 if seqs > 1 else 0))).any()):
            continue  # the controls need a valid compressed entry (of a request at position >= 1 when there is one)
        ref = _kvq_reference(ratio, req_id, swa, swa_local, comp_local)
        assert torch.equal(_kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local).long(), ref)
        wrong = {"table_off_by_one": _kvq_call(ratio, req_id, swa, swa_local, tbl + 1, comp_local)}
        other = _kvq_call(
            ratio, req_id, swa, swa_local, tbl, comp_local, tpb=SWA_TPB // (3 - ratio)
        )
        if seqs == 1:
            # Not a wrong numbering at b = 0: exactly e where e < 132 * other page size, -1 past it.
            e = comp_local.long()
            tpb_o = SWA_TPB // (3 - ratio)
            exp = torch.where(
                (e >= 0) & (torch.div(e, tpb_o, rounding_mode="floor") < KVQ_COLS), e, -1
            )
            assert torch.equal(other[:, WINDOW:].long(), exp) and torch.equal(
                other[:, :WINDOW], ref[:, :WINDOW]
            )
            same_at_b0.append(int((other.long() != ref).sum()))
        else:
            wrong["other_ratio_page_size"] = other
            starts = torch.nonzero(req_id[1:] != req_id[:-1]).flatten() + 1
            bad = req_id.clone()
            bad[int(starts[rng.randrange(starts.numel())])] -= 1
            wrong["req_off_by_one_request"] = _kvq_call(ratio, bad, swa, swa_local, tbl, comp_local)
        for k, w in wrong.items():
            counts[k].append(int((w.long() != ref).sum()))
        n += 1
    print(
        f"ratio={ratio} max_batch_size={seqs}: elements differing from the reference per control over 20 steps: "
        f"{counts}"
        + (
            f"; other ratio's page size at b = 0 (not a wrong numbering) differs in {same_at_b0}"
            if seqs == 1
            else ""
        )
    )
    for k, c in counts.items():
        if seqs == 1 and k != "table_off_by_one":
            assert c == [], (k, c)  # not applicable at max_batch_size 1
        else:
            assert len(c) == 20 and all(x > 0 for x in c), (k, c)
    if seqs == 1 and ratio == 2:
        assert all(x == 0 for x in same_at_b0), same_at_b0


# ---------------------------------------------------------------------------------------------
# The capture union (CAPTURE_UNION) at ratio 0 in the native-storage form. The native-storage target's ratio-0 layers
# make the same window-only call as every storage mode does, on the real SWA table as the metadata
# holds it: the [layer, SWA] view [max_batch_size, 132] of a 4-D buffer, page ids from the native SWA
# pool (up to KVQ_SWA_PAGES), with req_idx_per_token the token's request position in the step. (The
# native-storage ratio-1/2 call is test_kvq_synthetic_table_cuda_graph_capture_set, which captures every G
# in 1..max_batch_size at next_n 1 and 6.)
@pytest.mark.parametrize("rows,next_n", CAPTURE_UNION, ids=[f"T{r}_n{n}" for r, n in CAPTURE_UNION])
def test_capture_union_ratio0_on_native_swa_tables(rows, next_n):
    """Every (T, next_n) of the capture union, captured as the target captures the ratio-0 call
    (prefix views [:T] of 8192-row req / window buffers, the static SWA table) after one eager call at
    that shape, at max_batch_size 8 (the boot gates; G <= 8 only) and 32 (the accuracy gates). Three
    replays over new steps: G generation requests at positions 0, 1, ... (request position b), each
    verifying next_n positions at random up to 16397, request 0 ending at 16397 or 16392 with its
    newest SWA page the native pool's last page, and a fresh SWA table. Each replay equals the int64
    reference and an eager call bitwise, produces the pool's last row, and differs from the previous
    step's reference."""
    G = rows // next_n
    n_graphs = 0
    for seqs in (8, 32):
        if G > seqs:
            continue
        g = torch.Generator(device=DEV).manual_seed(27000 + 100 * seqs + rows + next_n)
        rng = random.Random(27100 + 100 * seqs + rows + next_n)
        req_buf = torch.zeros(8192, dtype=torch.int32, device=DEV)
        swa_buf = torch.full((8192, WINDOW), -1, dtype=torch.int32, device=DEV)
        swa = _kvq_swa(seqs, g)

        def load(step_i):
            end0 = DSPARK_MAX_POS if step_i % 2 == 0 else 16392
            starts = [rng.randint(0, DSPARK_MAX_POS + 1 - next_n) for _ in range(G)]
            starts[0] = end0 - next_n + 1
            positions = torch.cat(
                [torch.arange(c, c + next_n, device=DEV, dtype=torch.int32) for c in starts]
            )
            req_id = (
                torch.arange(G, device=DEV, dtype=torch.int32)
                .repeat_interleave(next_n)
                .contiguous()
            )
            swa_local = _window(positions)
            swa.copy_(_kvq_swa(seqs, g))
            swa[0, end0 // SWA_TPB] = KVQ_SWA_PAGES - 1
            req_buf[:rows].copy_(req_id)
            swa_buf[:rows].copy_(swa_local)
            ref = _reference64(req_id, swa, swa_local, SWA_TPB)
            # Request 0's last token attends its own position, on the pool's last page.
            assert (
                int(ref[next_n - 1, WINDOW - 1]) == (KVQ_SWA_PAGES - 1) * SWA_TPB + end0 % SWA_TPB
            )
            return ref

        def call():
            return op(req_buf[:rows], swa, swa_buf[:rows], SWA_TPB, 0)

        prev = load(0)
        assert torch.equal(call().long(), prev)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for it in range(1, 4):
            ref = load(it)
            out.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.long(), ref), (seqs, it)
            assert torch.equal(out, call()), (seqs, it)
            assert not torch.equal(out.long(), prev), (seqs, it)
            prev = ref
        del graph
        n_graphs += 1
    print(
        f"ratio=0 native SWA table T={rows} next_n={next_n}: {n_graphs} captures (max_batch_size "
        f"{[s for s in (8, 32) if G <= s]}) x 3 replays, replay = eager = reference"
    )
    assert n_graphs == (2 if G <= 8 else 1)


# ---------------------------------------------------------------------------------------------
# The engine envelope (per rank): max_num_tokens up to 16384, max_batch_size up to 256,
# max_seq_len L up to 1,048,576 with DSpark on and off, and the CUDA-graph batches GSET (every G in
# 1..32 plus TRT-LLM's padded list at cuda_graph_config max_batch_size 256), each captured at
# next_n 1 (T = G) and 6 (the DSpark verify step, T = 6 G). The metadata's block tables are
# [max_batch_size, W] with W = ceil(Lm / 128) + 3 columns (Lm = L + 14 with DSpark, L without);
# generation positions reach Lm + 5 (the graph metadata's max_draft_tokens) and context positions
# stay below L. Under decoder bounded replay a context step's decoder pass holds at most window =
# 128 rows per finishing request (its stash included), so one call's rows reach 128 * 256 = 32768.
# Page ids are drawn from [0, B) at the DSpark pool bounds (the engine of the dumps, the native
# cross-product maximum, the GB300 block ceiling and the int32 edge), each request's newest page
# among the pool's last four; every comparison is against the int64 reference.
GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
ENVELOPE_CAPTURE = [(G, n) for n in (1, 6) for G in GSET]
ENVELOPE_MAX_TOKENS = 16384
ENVELOPE_MAX_BATCH = 256
ENVELOPE_MAX_ROWS = WINDOW * ENVELOPE_MAX_BATCH
ENVELOPE_LS = (16384, 32768, 65536, 131072, 262144, 1048576)
ENVELOPE_CFGS = [(L, ds) for L in ENVELOPE_LS for ds in (True, False)]
ENVELOPE_CFG_IDS = [f"L{L}_{'dspark' if ds else 'nodspark'}" for L, ds in ENVELOPE_CFGS]


def _envelope_cfg(L, dspark):
    lm = L + 14 if dspark else L
    return {"L": L, "Lm": lm, "W": -(-lm // SWA_TPB) + 3, "max_gen": lm + 5}


ENVELOPE_WIDTHS = {cfg: _envelope_cfg(*cfg)["W"] for cfg in ENVELOPE_CFGS}
assert sorted(ENVELOPE_WIDTHS.values()) == [
    131,
    132,
    259,
    260,
    515,
    516,
    1027,
    1028,
    2051,
    2052,
    8195,
    8196,
]


def _envelope_table(pages, g, width, rows=ENVELOPE_MAX_BATCH, layered=False):
    """[rows, width] page ids drawn uniformly from [0, pages). layered: the [layer, SWA] view [1, 0]
    of a [2, 2, rows, width] buffer (the metadata's sliding_block_tables)."""
    ids = torch.randint(0, pages, (rows, width), generator=g, device=DEV, dtype=torch.int64).to(
        torch.int32
    )
    if not layered:
        return ids
    buf = torch.full((2, 2, rows, width), -1, dtype=torch.int32, device=DEV)
    buf[1, 0] = ids
    return buf[1, 0]


def _envelope_topk(positions, ratio, g):
    """The V4.1 index top-k shape without a [T, visible] score matrix (visible reaches ~1M here):
    min(512, v) valid compressed positions in [0, v), v = (p + 1) // ratio, the newest v - 1 always
    among them at a random column, the others uniform in [0, v) (the op maps element by element, so
    a repeat changes nothing), then -1."""
    t = positions.shape[0]
    vis = (positions.long() + 1) // ratio
    k = vis.clamp(max=TOPK)
    col = torch.arange(TOPK, device=DEV)
    draw = (torch.rand(t, TOPK, generator=g, device=DEV, dtype=torch.float64) * vis[:, None]).long()
    draw = torch.minimum(draw, (vis - 1).clamp(min=0)[:, None])
    newest = (torch.rand(t, generator=g, device=DEV, dtype=torch.float64) * k).long()
    draw = torch.where(col[None] == newest[:, None], (vis - 1)[:, None], draw)
    return torch.where(col[None] < k[:, None], draw, -1).to(torch.int32)


def _envelope_step(ratio, seqs, g, rows=None, tail_t0=None):
    """Per-token inputs of requests seqs = [(cached, new)] in step order. rows: each request's
    block-table row (default: distinct rows drawn from ENVELOPE_MAX_BATCH). tail_t0: per request, None or
    the bounded-replay window start t0 -- that request's rows then carry the decoder pass's window,
    positions t0 .. p in columns 0 .. p - t0, then -1. Returns req_id, positions, the window, the
    top-k (ratio > 0), each request's first and last row, and the table rows."""
    n = len(seqs)
    if rows is None:
        rows = torch.randperm(ENVELOPE_MAX_BATCH, generator=g, device=DEV)[:n]
    rows = rows.to(torch.int32)
    total = sum(q for _, q in seqs)
    cached = torch.tensor([c for c, _ in seqs], device=DEV, dtype=torch.int64)
    lens = torch.tensor([q for _, q in seqs], device=DEV, dtype=torch.int64)
    ends = torch.cumsum(lens, 0)
    starts = ends - lens
    req_id = torch.repeat_interleave(rows, lens, output_size=total).contiguous()
    tok = torch.arange(total, device=DEV)
    positions = (torch.repeat_interleave(cached - starts, lens, output_size=total) + tok).to(
        torch.int32
    )
    swa_local = _window(positions)
    if tail_t0 is not None and any(t is not None for t in tail_t0):
        t0 = torch.tensor([-1 if t is None else t for t in tail_t0], device=DEV, dtype=torch.int64)
        t0r = torch.repeat_interleave(t0, lens, output_size=total)
        cols = torch.arange(WINDOW, device=DEV)
        tail_win = torch.where(
            cols[None] <= (positions.long() - t0r)[:, None], t0r[:, None] + cols[None], -1
        )
        swa_local = torch.where((t0r >= 0)[:, None], tail_win, swa_local.long()).to(torch.int32)
    comp_local = _envelope_topk(positions, ratio, g) if ratio else None
    return req_id, positions, swa_local, comp_local, starts, ends - 1, rows


def _envelope_far_newest(table, rows, last, local, tpb, pages):
    """Each request's newest page (the page of the largest local index its last row names) is one of
    the pool's last four: pages - 1 - (b % 4) for request b."""
    newest = local[last].max(1).values.long()
    b = torch.arange(newest.numel(), device=DEV)
    ok = newest >= 0
    table[rows.long()[ok], newest[ok] // tpb] = (pages - 1 - b[ok] % 4).to(torch.int32)


def _envelope_tables(ratio, W, b_swa, b_comp, g, step):
    """Engine tables [ENVELOPE_MAX_BATCH, W] at one bound pair, with the step's newest pages at the far end."""
    _, _, swa_local, comp_local, _, last, rows = step
    swa = _envelope_table(b_swa, g, W, layered=True)
    _envelope_far_newest(swa, rows, last, swa_local, SWA_TPB, b_swa)
    comp = None
    if ratio:
        comp = _envelope_table(b_comp, g, W)
        _envelope_far_newest(comp, rows, last, comp_local, SWA_TPB // ratio, b_comp)
    return swa, comp


def _envelope_random_mixed(rng, c, max_t=ENVELOPE_MAX_TOKENS, max_k=ENVELOPE_MAX_BATCH):
    """A multi-request step at the engine envelope: k in [1, max_k] requests, context rows first, then
    generation rows at next_n 1 or 6, T in [1, max_t]. One step in ten is generation-only and at least
    one in ten context-only; the rest mix both. Each context sits at a random cached prefix below L;
    each generation request starts in the tail [L - next_n, max_gen + 1 - next_n] with probability 1/4
    and anywhere up to max_gen + 1 - next_n otherwise."""
    L, top = c["L"], c["max_gen"]
    k = rng.randint(1, max_k)
    next_n = rng.choice((1, 6))
    u = rng.random()
    if u < 0.1:
        n_gen = k
    elif u < 0.2 or k == 1:
        n_gen = 0
    else:
        n_gen = rng.randint(1, k - 1)
    n_ctx = k - n_gen
    gen_rows = n_gen * next_n
    ctx = []
    if n_ctx:
        ctx_rows = rng.randint(gen_rows + n_ctx, max_t) - gen_rows
        cuts = sorted(rng.sample(range(1, ctx_rows), n_ctx - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        ctx = [(rng.randint(0, L - n), n) for n in news]
    g_top = top + 1 - next_n
    gen = [
        rng.randint(L - next_n, g_top) if rng.random() < 0.25 else rng.randint(0, g_top)
        for _ in range(n_gen)
    ]
    return ctx, gen, next_n


def _envelope_col_count(local, tpb, col):
    """Valid local indices this call resolves through table column `col`."""
    return int(((local >= 0) & (torch.div(local, tpb, rounding_mode="floor") == col)).sum())


def _envelope_big_step(c, rng):
    """One 32768-row step (the decoder-pass maximum, 128 rows x 256 requests, context rows first):
    8 contexts (a fresh prompt, a chunk on an 8192-token prefix, 1-, 5- and 129-token contexts, a
    chunk ending at L - 1), 190 decoder-tail requests of 128 rows at random positions below L with the
    bounded-replay window, then 47 DSpark verify runs (the first ending at max_gen) and 4 decode rows
    (at L - 1, max_gen and two random positions): 249 requests."""
    L, top = c["L"], c["max_gen"]
    ctx = [
        (0, 3000),
        (8192, 2048),
        (0, 1),
        (0, 129),
        (130, 5),
        (2, 1570),
        (4000, 430),
        (L - 979, 979),
    ]
    tails = [(rng.randint(0, L - 128), 128) for _ in range(190)]
    verify = [(top - 5, 6)] + [(rng.randint(0, top - 5), 6) for _ in range(46)]
    decode = [(L - 1, 1), (top, 1), (rng.randint(0, top), 1), (rng.randint(0, top), 1)]
    seqs = ctx + tails + verify + decode
    t0 = [None] * len(ctx) + [c0 for c0, _ in tails] + [None] * (len(verify) + len(decode))
    assert sum(q for _, q in seqs) == ENVELOPE_MAX_ROWS and len(seqs) <= ENVELOPE_MAX_BATCH
    return seqs, t0


@pytest.mark.parametrize(
    "cfg", [(16384, True), (1048576, True)], ids=["L16384_dspark", "L1048576_dspark"]
)
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_envelope_rows_prefix_bitwise(ratio, cfg):
    """Every T = 1..32768, not a sample (max_num_tokens 16384; decoder bounded replay's 128 x 256 rows):
    the 32768-row call equals the int64 reference on [256, W] tables at the GB300 block ceiling with
    the pools' last rows produced, every call on the first T rows (prefix views of the same buffers)
    is bitwise equal to the first T rows of it, and the sweep adds no compiled variant. Counter
    control: a new constexpr (an unused page size) adds exactly one."""
    c = _envelope_cfg(*cfg)
    W = c["W"]
    rng = random.Random(28000 + 10 * ratio + (cfg[0] > 16384))
    g = torch.Generator(device=DEV).manual_seed(28050 + 10 * ratio + (cfg[0] > 16384))
    tpb_c = SWA_TPB // ratio if ratio else 0
    seqs, t0 = _envelope_big_step(c, rng)
    step = _envelope_step(ratio, seqs, g, tail_t0=t0)
    req_id, positions, swa_local, comp_local = step[:4]
    b_swa, b_comp = _dspark_bound_pairs(ratio)[2]
    swa, comp = _envelope_tables(ratio, W, b_swa, b_comp, g, step)
    n = req_id.numel()
    assert n == ENVELOPE_MAX_ROWS and int(positions.max()) == c["max_gen"]
    full = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
    ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
    assert torch.equal(full.long(), ref)
    assert int(ref[:, :WINDOW].max()) >= (b_swa - 4) * SWA_TPB
    if ratio:
        assert int(ref[:, WINDOW:].max()) >= (b_comp - 4) * tpb_c
    torch.cuda.synchronize()
    before = _compiled_variants()
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for t in range(1, n + 1):
        out = _target_call(
            ratio,
            req_id[:t],
            swa,
            swa_local[:t],
            comp,
            None if comp_local is None else comp_local[:t],
            tpb_c,
        )
        bad += (out != full[:t]).any(1).sum()
    torch.cuda.synchronize()
    after = _compiled_variants()
    print(
        f"ratio={ratio} cfg={cfg} W={W}: prefixes T=1..{n}: rows differing {int(bad)}; compiled variants "
        f"{before} -> {after}; positions up to {int(positions.max())}"
    )
    assert int(bad) == 0
    assert after == before
    op(req_id[:4], swa, swa_local[:4], 3000 + 10 * ratio + (cfg[0] > 16384), 0)
    torch.cuda.synchronize()
    assert _compiled_variants() == after + 1


ENVELOPE_RANDOM_STEPS = 240


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_envelope_random_mixed_partitions(ratio):
    """Random multi-request steps at the engine envelope: 240 seeded steps per ratio, k in [1, 256] requests,
    T in [1, 16384], contexts at random cached prefixes below L then generation rows (next_n 1 or 6)
    up to max_gen, cycling through the 12 (L, DSpark) table widths (20 steps each) and the DSpark pool
    bound pairs, on [256, W] tables at random rows; each call bitwise equal to the int64 reference. Every
    20th step with >= 2 requests also runs req_idx_per_token off by one request at a request boundary,
    which the same check must reject. Every step also runs on the tables cut to the rows it names
    (fresh, aligned [m, W] buffers, m = the largest row + 1: a smaller max_batch_size's tables),
    bitwise equal. Under torch.profiler only the one kernel launches, and no compiled variant is added
    after one warm-up call per width."""
    rng = random.Random(28100 + ratio)
    g = torch.Generator(device=DEV).manual_seed(28200 + ratio)
    tpb_c = SWA_TPB // ratio if ratio else 0
    bounds = _dspark_bound_pairs(ratio)
    before_warm = _compiled_variants()
    for cfg in ENVELOPE_CFGS:
        W = ENVELOPE_WIDTHS[cfg]
        warm = _envelope_step(ratio, [(0, 4)], g)
        swa, comp = _envelope_tables(ratio, W, 1024, 1024, g, warm)
        _target_call(ratio, warm[0], swa, warm[2], comp, warm[3], tpb_c)
    torch.cuda.synchronize()
    before = _compiled_variants()
    n_calls, rows_total, kinds, controls, ks, ts, tail_rows = 0, 0, set(), [], [], [], 0
    far = {}
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(ENVELOPE_RANDOM_STEPS):
            cfg = ENVELOPE_CFGS[i % len(ENVELOPE_CFGS)]
            c = _envelope_cfg(*cfg)
            b_swa, b_comp = bounds[(i // len(ENVELOPE_CFGS)) % len(bounds)]
            ctx, gen, next_n = _envelope_random_mixed(rng, c)
            seqs = ctx + [(p, next_n) for p in gen]
            step = _envelope_step(ratio, seqs, g)
            req_id, positions, swa_local, comp_local, starts, _, rows = step
            swa, comp = _envelope_tables(ratio, c["W"], b_swa, b_comp, g, step)
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
            n_calls += 1
            assert int(ref.max()) < 2**31
            assert torch.equal(out.long(), ref), (ratio, i, cfg, next_n, len(ctx), len(gen))
            # Row-count invariance: the same tables cut to the rows the step names (a smaller
            # max_batch_size's tables, fresh 16-byte-aligned buffers) give the same output.
            m = int(rows.max()) + 1
            swa_m = swa[:m].clone()
            comp_m = comp[:m].clone() if ratio else None
            assert torch.equal(
                _target_call(ratio, req_id, swa_m, swa_local, comp_m, comp_local, tpb_c), out
            )
            n_calls += 1
            T = req_id.numel()
            rows_total += T
            ks.append(len(seqs))
            ts.append(T)
            tail_rows += sum(next_n for p in gen if p + next_n - 1 >= c["L"])
            far[cfg] = max(far.get(cfg, 0), int(positions.max()))
            kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
            if i % 20 == 0 and len(seqs) >= 2:
                b = rng.randint(1, len(seqs) - 1)
                bad_id = req_id.clone()
                bad_id[int(starts[b])] = rows[b - 1]
                wrong = _target_call(ratio, bad_id, swa, swa_local, comp, comp_local, tpb_c)
                n_calls += 1
                controls.append(int((wrong.long() != ref).sum()))
    torch.cuda.synchronize()
    print(
        f"ratio={ratio}: {n_calls} calls, {rows_total} rows, requests {min(ks)}..{max(ks)}, T {min(ts)}..{max(ts)}, "
        f"{tail_rows} rows at positions >= L, kinds {sorted(kinds)}; largest position per (L, DSpark) {far}; "
        f"width warm-ups added {before - before_warm} compiled variants; control (req_id off by one request) "
        f"elements differing {controls}"
    )
    assert len(controls) >= 8 and all(x > 0 for x in controls), controls
    assert {
        (True, True, 1),
        (True, True, 6),
        (False, True, 1),
        (False, True, 6),
        (True, False, 0),
    } <= kinds
    assert max(ks) > 200 and max(ts) > 15000 and tail_rows > 0
    _check_variants(prof, n_calls, before)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_envelope_unaligned_table_views(ratio):
    """The metadata's sliding_block_tables is one [layers, types, max_num_sequences, W] buffer, so the
    [layer, SWA] view the op reads starts (layer * types + SWA) * max_num_sequences * W * 4 bytes in.
    When max_num_sequences * W is not a multiple of 4 (an odd W with max_batch_size not a multiple
    of 4), some of those views are not 16-byte aligned, and Triton specializes pointer arguments on
    16-byte alignment. Here: the views [0, 1], [1, 0] and [1, 1] of a [2, 2, 7, W] buffer at W 131 and
    8195 (offsets 4, 8 and 12 mod 16 bytes), each on 5 random steps of up to 7 requests: every call
    equals the int64 reference and the same call on an aligned copy; the first unaligned call adds
    exactly one compiled variant and no later call adds any (no earlier case in this file hands the
    op an unaligned SWA table). Captured on the [1, 0] view after an eager
    call (T = 7 and 42), three replays over new tables and steps each equal the reference and an eager
    call."""
    rng = random.Random(30100 + ratio)
    g = torch.Generator(device=DEV).manual_seed(30000 + ratio)
    tpb_c = SWA_TPB // ratio if ratio else 0
    mbs = 7
    added, offsets = [], {}
    for cfg in ((16384, False), (1048576, False)):
        c = _envelope_cfg(*cfg)
        W = c["W"]
        buf = torch.randint(
            0, POOL_INT32_EDGE[128], (2, 2, mbs, W), generator=g, device=DEV, dtype=torch.int64
        ).to(torch.int32)
        comp = _envelope_table(DSPARK_BOUNDS[ratio][2], g, W, rows=mbs) if ratio else None
        views = {k: buf[k] for k in ((0, 1), (1, 0), (1, 1))}
        for k, v in views.items():
            offsets[(W, k)] = v.data_ptr() % 16
            assert v.data_ptr() % 16 != 0, (W, k)
        for s_i in range(5):
            ctx, gen, next_n = _envelope_random_mixed(rng, c, max_k=mbs)
            seqs = ctx + [(p, next_n) for p in gen]
            step = _envelope_step(ratio, seqs, g, rows=torch.arange(len(seqs), device=DEV))
            req_id, _, swa_local, comp_local = step[:4]
            for k, v in views.items():
                before = _compiled_variants()
                out = _target_call(ratio, req_id, v, swa_local, comp, comp_local, tpb_c)
                torch.cuda.synchronize()
                added.append(_compiled_variants() - before)
                assert torch.equal(
                    out.long(), _step_reference64(ratio, req_id, v, swa_local, comp, comp_local)
                ), (W, k, s_i)
                assert torch.equal(
                    _target_call(ratio, req_id, v.clone(), swa_local, comp, comp_local, tpb_c), out
                )
    assert added[0] == 1 and all(a == 0 for a in added[1:]), added
    c = _envelope_cfg(16384, False)
    W = c["W"]
    buf = torch.randint(
        0, POOL_INT32_EDGE[128], (2, 2, mbs, W), generator=g, device=DEV, dtype=torch.int64
    ).to(torch.int32)
    view = buf[1, 0]
    comp = _envelope_table(DSPARK_BOUNDS[ratio][2], g, W, rows=mbs) if ratio else None
    n_graphs = 0
    for next_n in (1, 6):
        rows = mbs * next_n
        req_buf = torch.zeros(ENVELOPE_MAX_TOKENS, dtype=torch.int32, device=DEV)
        swa_buf = torch.full((ENVELOPE_MAX_TOKENS, WINDOW), -1, dtype=torch.int32, device=DEV)
        top_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV) if ratio else None

        def load(it):
            starts = [rng.randint(0, c["max_gen"] + 1 - next_n) for _ in range(mbs)]
            starts[0] = c["max_gen"] - next_n + 1 - (it % 2)
            step = _envelope_step(
                ratio, [(p, next_n) for p in starts], g, rows=torch.arange(mbs, device=DEV)
            )
            req_id, _, swa_local, comp_local = step[:4]
            view.copy_(
                torch.randint(
                    0, POOL_INT32_EDGE[128], (mbs, W), generator=g, device=DEV, dtype=torch.int64
                ).to(torch.int32)
            )
            req_buf[:rows].copy_(req_id)
            swa_buf[:rows].copy_(swa_local)
            if ratio:
                comp.copy_(_envelope_table(DSPARK_BOUNDS[ratio][2], g, W, rows=mbs))
                top_buf.copy_(comp_local)
            return _step_reference64(ratio, req_id, view, swa_local, comp, comp_local)

        def call():
            return _target_call(ratio, req_buf[:rows], view, swa_buf[:rows], comp, top_buf, tpb_c)

        prev = load(0)
        assert torch.equal(call().long(), prev)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for it in range(1, 4):
            ref = load(it)
            out.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.long(), ref), (next_n, it)
            assert torch.equal(out, call()), (next_n, it)
            assert not torch.equal(out.long(), prev), (next_n, it)
            prev = ref
        del graph
        n_graphs += 1
    print(
        f"ratio={ratio}: unaligned [layer, SWA] views, byte offsets mod 16 {offsets}; compiled variants added per "
        f"call {added} (first call {added[0]}); {n_graphs} captures on the [1, 0] view x 3 replays exact"
    )


# Every (T, ratio) cell of dsv4_local_to_global_indices in the runtime-value dumps of the envelope
# engines (max_batch_size 256 with graphs over the padded list to 256; max_num_tokens 16384 / 8192 at
# L 32768 with DSpark): "T:ratios".
ENVELOPE_DUMPED_CELLS = (
    "1:012 2:012 5:012 6:012 7:012 12:012 16:012 24:012 34:012 48:012 96:012 128:01 134:1 144:012 192:012 "
    "240:012 288:012 336:012 384:012 432:012 480:012 528:012 576:012 597:012 624:012 672:012 720:012 768:012 "
    "1152:012 1536:012 2252:012 4559:012 8147:012 8192:012 8467:012 9193:02 9240:012 9524:012 10441:012 "
    "10444:012 10447:012 10451:012 16298:012 16327:012 16328:012 16378:012 16384:012"
)


def _envelope_partition_exact(rng, T, c):
    """A random step of exactly T rows: n_gen generation requests at next_n 1 or 6 and the rest of
    the rows over n_ctx context requests, n_ctx + n_gen <= 256."""
    L, top = c["L"], c["max_gen"]
    next_n = rng.choice((1, 6))
    n_gen = rng.randint(0, min(ENVELOPE_MAX_BATCH - 1, T // next_n))
    ctx_rows = T - n_gen * next_n
    ctx = []
    if ctx_rows:
        n_ctx = rng.randint(1, min(ctx_rows, ENVELOPE_MAX_BATCH - n_gen))
        cuts = sorted(rng.sample(range(1, ctx_rows), n_ctx - 1))
        news = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        ctx = [(rng.randint(0, L - n), n) for n in news]
    gen = [rng.randint(0, top + 1 - next_n) for _ in range(n_gen)]
    return ctx, gen, next_n


def test_envelope_dumped_cells():
    """Every dumped (T, ratio) cell of the envelope engines as a consistent random step of exactly
    T rows (<= 256 requests) at a (L, DSpark) width cycling over the 12, on [256, W] tables at the DSpark
    pool bounds: bitwise equal to the int64 reference."""
    rng = random.Random(28300)
    g = torch.Generator(device=DEV).manual_seed(28400)
    cells = [
        (int(t), int(r))
        for tok in ENVELOPE_DUMPED_CELLS.split()
        for t, rs in [tok.split(":")]
        for r in rs
    ]
    assert len(cells) == 137
    for i, (T, ratio) in enumerate(cells):
        c = _envelope_cfg(*ENVELOPE_CFGS[i % len(ENVELOPE_CFGS)])
        b_swa, b_comp = _dspark_bound_pairs(ratio)[i % 4]
        ctx, gen, next_n = _envelope_partition_exact(rng, T, c)
        step = _envelope_step(ratio, ctx + [(p, next_n) for p in gen], g)
        req_id, _, swa_local, comp_local = step[:4]
        assert req_id.numel() == T
        swa, comp = _envelope_tables(ratio, c["W"], b_swa, b_comp, g, step)
        tpb_c = SWA_TPB // ratio if ratio else 0
        out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
        assert torch.equal(
            out.long(), _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
        ), (T, ratio)
    print(
        f"{len(cells)} dumped cells exact; T {min(t for t, _ in cells)}..{max(t for t, _ in cells)}"
    )


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_envelope_widths_far_end(ratio, cfg):
    """Every (L, DSpark) table width with rows at its far end, at every DSpark pool bound pair.
    (a) Tail steps (next_n 1 and 6): 6 contexts (two ending at L - 1) ahead of 250 generation
    requests, which include decode rows at every position L - 1 .. max_gen (next_n 1) or verify runs
    ending at every one of them (next_n 6); every call equals the int64 reference. Control: the same
    call on tables cut just before the farthest column the step reads differs from the reference
    exactly at the entries that read that column, which come back -1.
    (b) Far-end enumeration: table rows 0 and 255 name pages B - W .. B - 1, so the last column is the
    pool's last page; one token per row reads every slot of the last column (window) and the last
    512 compressed entries of the table's range, and one reads positions at column W (past the table):
    exact, the pool's last row B * tpb - 1 is produced, and every past-the-table entry is -1."""
    c = _envelope_cfg(*cfg)
    L, W, top = c["L"], c["W"], c["max_gen"]
    rng = random.Random(28500 + 100 * ratio + ENVELOPE_CFGS.index(cfg))
    g = torch.Generator(device=DEV).manual_seed(28600 + 100 * ratio + ENVELOPE_CFGS.index(cfg))
    tpb_c = SWA_TPB // ratio if ratio else 0
    bounds = _dspark_bound_pairs(ratio)
    ctx = [(0, 3000), (L - 7048, 2048), (0, 1), (130, 5), (L - 979, 979), (L - 384, 384)]
    tail_ends = list(range(L - 1, top + 1))
    col_s = top // SWA_TPB
    col_c = ((top + 1) // ratio - 1) // tpb_c if ratio else None
    counts = []
    for next_n in (1, 6):
        starts = [e - next_n + 1 for e in tail_ends]
        starts += [
            rng.randint(0, top + 1 - next_n)
            for _ in range(ENVELOPE_MAX_BATCH - len(ctx) - len(starts))
        ]
        rng.shuffle(starts)
        step = _envelope_step(ratio, ctx + [(s, next_n) for s in starts], g)
        req_id, positions, swa_local, comp_local = step[:4]
        assert req_id.numel() <= ENVELOPE_MAX_TOKENS and int(positions.max()) == top
        n_s = _envelope_col_count(swa_local, SWA_TPB, col_s)
        n_c = _envelope_col_count(comp_local, tpb_c, col_c) if ratio else 0
        assert n_s > 0 and (n_c > 0 or not ratio)
        counts.append((next_n, n_s, n_c))
        for b_swa, b_comp in bounds:
            swa, comp = _envelope_tables(ratio, W, b_swa, b_comp, g, step)
            out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
            ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
            assert int(ref.max()) < 2**31
            assert torch.equal(out.long(), ref), (ratio, cfg, next_n, b_swa, b_comp)
            cut = _target_call(
                ratio,
                req_id,
                swa[:, :col_s],
                swa_local,
                None if comp is None else comp[:, :col_c],
                comp_local,
                tpb_c,
            )
            diff = cut.long() != ref
            assert int(diff.sum()) == n_s + n_c, (int(diff.sum()), n_s, n_c)
            assert bool((cut[diff] == -1).all())
    req_id = torch.tensor(
        [0, 0, ENVELOPE_MAX_BATCH - 1, ENVELOPE_MAX_BATCH - 1], dtype=torch.int32, device=DEV
    )
    ar = torch.arange(WINDOW, dtype=torch.int32, device=DEV)
    swa_local = torch.stack([(W - 1) * SWA_TPB + ar, W * SWA_TPB + ar] * 2)
    comp_local = None
    if ratio:
        ak = torch.arange(TOPK, dtype=torch.int32, device=DEV)
        comp_local = torch.stack([W * tpb_c - TOPK + ak, W * tpb_c + ak] * 2)
    for b_swa, b_comp in bounds:
        swa = _envelope_table(b_swa, g, W, layered=True)
        swa[0] = b_swa - W + torch.arange(W, device=DEV, dtype=torch.int32)
        swa[ENVELOPE_MAX_BATCH - 1] = swa[0]
        comp = None
        if ratio:
            comp = _envelope_table(b_comp, g, W)
            comp[0] = b_comp - W + torch.arange(W, device=DEV, dtype=torch.int32)
            comp[ENVELOPE_MAX_BATCH - 1] = comp[0]
        out = _target_call(ratio, req_id, swa, swa_local, comp, comp_local, tpb_c)
        ref = _step_reference64(ratio, req_id, swa, swa_local, comp, comp_local)
        assert torch.equal(out.long(), ref), (ratio, cfg, b_swa, b_comp)
        assert (
            int(out[0, WINDOW - 1]) == b_swa * SWA_TPB - 1
            and int(out[2, WINDOW - 1]) == b_swa * SWA_TPB - 1
        )
        assert bool((out[1] == -1).all()) and bool((out[3] == -1).all())
        if ratio:
            assert int(out[0, -1]) == b_comp * tpb_c - 1
    print(
        f"ratio={ratio} cfg={cfg} W={W}: tail positions {L - 1}..{top}; far column (SWA {col_s}, compressed "
        f"{col_c}) entries per step (next_n, SWA, compressed) {counts}; bounds {bounds}"
    )


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
def test_envelope_tail_slots_form(cfg):
    """The decoder-replay slot call (the target's _TailSlots): req_id = each tail row's request
    position, the layer's SWA table, and a [tcd, 128] local table of [row position, -1 x 127]. Each
    finishing request contributes at most window = 128 rows, so tcd reaches 128 x 256 = 32768. 20
    random steps (1..256 requests, 1..128 tail rows each, tails ending anywhere up to L - 1) and the
    full 32768-row step (every tail ending at L - 1 .. L - 7), on [256, W] SWA tables at the DSpark
    SWA pool bounds: each call equals the int64 reference, so column 0 is the row's own slot and every other
    column is -1."""
    c = _envelope_cfg(*cfg)
    L, W = c["L"], c["W"]
    idx = ENVELOPE_CFGS.index(cfg)
    rng = random.Random(28700 + idx)
    g = torch.Generator(device=DEV).manual_seed(28800 + idx)
    steps = []
    for _ in range(20):
        nreq = rng.randint(1, ENVELOPE_MAX_BATCH)
        steps.append(
            [
                (n, rng.randint(n, L))
                for n in (
                    WINDOW if rng.random() < 0.5 else rng.randint(1, WINDOW) for _ in range(nreq)
                )
            ]
        )
    steps.append([(WINDOW, L - b % 7) for b in range(ENVELOPE_MAX_BATCH)])
    sizes = []
    for i, tails in enumerate(steps):
        b_swa = DSPARK_BOUNDS["swa"][i % 4]
        seqs = [(N - n, n) for n, N in tails]
        step = _envelope_step(0, seqs, g, rows=torch.arange(len(seqs), device=DEV))
        req_id, positions = step[0], step[1]
        self_local = torch.full((req_id.numel(), WINDOW), -1, dtype=torch.int32, device=DEV)
        self_local[:, 0] = positions
        swa = _envelope_table(b_swa, g, W, layered=True)
        _envelope_far_newest(swa, step[6], step[5], self_local, SWA_TPB, b_swa)
        out = op(req_id, swa, self_local, SWA_TPB, 0)
        ref = _reference64(req_id, swa, self_local, SWA_TPB)
        assert torch.equal(out.long(), ref), (cfg, i)
        assert bool((out[:, 1:] == -1).all()) and bool((out[:, 0] >= 0).all())
        sizes.append(req_id.numel())
    assert sizes[-1] == ENVELOPE_MAX_ROWS
    print(f"cfg={cfg} W={W}: tail-slot calls of {sizes} rows exact")


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_envelope_capture_gset_runtime_tables(ratio, cfg):
    """CUDA-graph capture over GSET x next_n {1, 6} (T = G and 6 G, 1..1536) in the target's
    runtime-table forms (ratio 0: window only; ratios 1 and 2: the runtime compress table), at every
    (L, DSpark) width: prefix views [:T] of 16384-row req / window buffers, static [256, W] SWA (the
    [layer, SWA] view of a 4-D buffer) and compress tables, a static [T, 512] top-k, after one eager
    call at that shape. Three replays over new steps (request 0 ending at max_gen or L - 1, table rows
    random or the request positions, tables at a new DSpark pool bound pair): each replay equals the int64
    reference and an eager call bitwise, differs from the previous step's reference, and request 0's
    last row reads its own position on the pool's last page."""
    c = _envelope_cfg(*cfg)
    L, W, top = c["L"], c["W"], c["max_gen"]
    idx = ENVELOPE_CFGS.index(cfg)
    rng = random.Random(29000 + 100 * ratio + idx)
    g = torch.Generator(device=DEV).manual_seed(29100 + 100 * ratio + idx)
    tpb_c = SWA_TPB // ratio if ratio else 0
    bounds = _dspark_bound_pairs(ratio)
    req_buf = torch.zeros(ENVELOPE_MAX_TOKENS, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((ENVELOPE_MAX_TOKENS, WINDOW), -1, dtype=torch.int32, device=DEV)
    swa_table = _envelope_table(1024, g, W, layered=True)
    comp_table = _envelope_table(1024, g, W) if ratio else None
    n_graphs = 0
    for G, next_n in ENVELOPE_CAPTURE:
        rows = G * next_n
        top_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV) if ratio else None

        def load(it):
            b_swa, b_comp = bounds[(n_graphs + it) % len(bounds)]
            starts = [rng.randint(0, top + 1 - next_n) for _ in range(G)]
            end0 = top if it % 2 == 0 else L - 1
            starts[0] = end0 - next_n + 1
            trows = torch.arange(G, device=DEV) if it % 2 else None
            step = _envelope_step(ratio, [(s, next_n) for s in starts], g, rows=trows)
            req_id, _, swa_local, comp_local, _, last, trow = step
            swa_table.copy_(_envelope_table(b_swa, g, W, layered=True))
            _envelope_far_newest(swa_table, trow, last, swa_local, SWA_TPB, b_swa)
            req_buf[:rows].copy_(req_id)
            swa_buf[:rows].copy_(swa_local)
            if ratio:
                comp_table.copy_(_envelope_table(b_comp, g, W))
                _envelope_far_newest(comp_table, trow, last, comp_local, tpb_c, b_comp)
                top_buf.copy_(comp_local)
            ref = _step_reference64(ratio, req_id, swa_table, swa_local, comp_table, comp_local)
            assert int(ref[next_n - 1, WINDOW - 1]) == (b_swa - 1) * SWA_TPB + end0 % SWA_TPB
            return ref

        def call():
            return _target_call(
                ratio, req_buf[:rows], swa_table, swa_buf[:rows], comp_table, top_buf, tpb_c
            )

        prev = load(0)
        assert torch.equal(call().long(), prev)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for it in range(1, 4):
            ref = load(it)
            out.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.long(), ref), (cfg, G, next_n, it)
            assert torch.equal(out, call()), (cfg, G, next_n, it)
            assert not torch.equal(out.long(), prev), (cfg, G, next_n, it)
            prev = ref
        del graph
        n_graphs += 1
    print(
        f"ratio={ratio} cfg={cfg} W={W}: {n_graphs} captures (G in GSET x next_n 1, 6; T 1..1536) x 3 replays, "
        f"replay = eager = reference"
    )
    assert n_graphs == 2 * len(GSET)


# The native-storage synthetic table at the engine envelope: [max_batch_size, W] = arange(mbs * W)
# for max_batch_size up to 256 and every (L, DSpark) width W, numbering request b's compressed entry e
# as row b * L_r + e of the compact read buffer, L_r = W * 128 / r. The window half is the real SWA
# table [mbs, W] (the [layer, SWA] view of a 4-D buffer) with page ids drawn from [0, 2**24).
def _envelope_kvq_reference(ratio, W, req_id, swa, swa_local, comp_local):
    tpb = SWA_TPB // ratio
    win = _reference64(req_id, swa, swa_local, SWA_TPB)
    e = comp_local.long()
    valid = (e >= 0) & (torch.div(e, tpb, rounding_mode="floor") < W)
    return torch.cat([win, torch.where(valid, req_id.long()[:, None] * (W * tpb) + e, -1)], 1)


def _envelope_kvq_table(mbs, W):
    return torch.arange(mbs * W, dtype=torch.int32, device=DEV).view(mbs, W)


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_envelope_kvq_maps_every_row_of_the_buffer(ratio, cfg):
    """max_batch_size 256 at every (L, DSpark) width: every request position b < 256 with every
    compressed position e in 0 .. L_r + 511 (in calls of <= 16384 rows), through the synthetic table:
    each call equals the reference, each of the 256 * L_r buffer rows comes out exactly once, and every
    position at or past column W gives -1."""
    c = _envelope_cfg(*cfg)
    W = c["W"]
    idx = ENVELOPE_CFGS.index(cfg)
    g = torch.Generator(device=DEV).manual_seed(29300 + 100 * ratio + idx)
    mbs = ENVELOPE_MAX_BATCH
    tpb = SWA_TPB // ratio
    L_r = W * tpb
    tbl = _envelope_kvq_table(mbs, W)
    swa = _envelope_table(POOL_INT32_EDGE[128], g, W, rows=mbs, layered=True)
    per_req = -(-(L_r + TOPK) // TOPK)
    total = mbs * per_req
    counts = torch.zeros(mbs * L_r, dtype=torch.int32, device=DEV)
    n_minus, calls = 0, 0
    ak = torch.arange(TOPK, device=DEV)
    for t0 in range(0, total, ENVELOPE_MAX_TOKENS):
        tok = torch.arange(t0, min(t0 + ENVELOPE_MAX_TOKENS, total), device=DEV)
        req_id = (tok // per_req).to(torch.int32)
        comp_local = ((tok % per_req)[:, None] * TOPK + ak[None]).to(torch.int32)
        positions = torch.randint(
            0, c["max_gen"] + 1, (tok.numel(),), generator=g, device=DEV, dtype=torch.int32
        )
        swa_local = _window(positions)
        out = _kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local)
        assert torch.equal(
            out.long(), _envelope_kvq_reference(ratio, W, req_id, swa, swa_local, comp_local)
        ), (cfg, t0)
        comp = out[:, WINDOW:]
        vals = comp[comp >= 0].long()
        counts.index_add_(0, vals, torch.ones_like(vals, dtype=torch.int32))
        n_minus += int((comp < 0).sum())
        calls += 1
    assert bool((counts == 1).all()), "every buffer row exactly once"
    assert n_minus == mbs * (per_req * TOPK - L_r)
    print(
        f"ratio={ratio} cfg={cfg} W={W}: rows 0..{mbs * L_r - 1} each produced once over {total} tokens in "
        f"{calls} calls; {n_minus} entries past column {W} -> -1"
    )


def _envelope_kvq_far_step(c):
    """256 requests (the full batch) whose last is one DSpark verify ending at max_gen, so the largest
    row a step can produce, 255 * L_r + (max_gen + 1) // r - 1, is produced."""
    L = c["L"]
    ctx = [(L - 4000, 4000)] + [
        ((97 * i) % (L - 64), 3 + i % 50) for i in range(ENVELOPE_MAX_BATCH - 2)
    ]
    return ctx, [c["max_gen"] - 5], 6


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_envelope_kvq_random_mixed_steps(ratio, cfg):
    """Random multi-request steps of the native-storage call at max_batch_size 256 and every (L, DSpark)
    width: 200 seeded steps (k in [1, 256] requests at positions 0..k-1, T in [1, 16384], contexts
    below L then generation rows at next_n 1 or 6 up to max_gen) plus the far-end step; every call
    equals the reference (window half through the SWA table, compressed half b * L_r + e), every
    compressed row lies inside the 256 * L_r-row buffer, and the far-end step produces the largest
    reachable row. Row-count invariance: each step also runs with the synthetic table at
    max_batch_size k and at a random max_batch_size in [k, 256] (fresh tensors), bitwise equal.
    Under torch.profiler only the one kernel launches, and no variant is compiled after one warm-up
    call in this form."""
    c = _envelope_cfg(*cfg)
    W = c["W"]
    idx = ENVELOPE_CFGS.index(cfg)
    rng = random.Random(29500 + 100 * ratio + idx)
    g = torch.Generator(device=DEV).manual_seed(29600 + 100 * ratio + idx)
    mbs = ENVELOPE_MAX_BATCH
    tpb = SWA_TPB // ratio
    L_r = W * tpb
    tbl = _envelope_kvq_table(mbs, W)
    swa = _envelope_table(POOL_INT32_EDGE[128], g, W, rows=mbs, layered=True)
    steps = [_envelope_random_mixed(rng, c) for _ in range(200)] + [_envelope_kvq_far_step(c)]
    warm = _envelope_step(ratio, [(0, 4)], g, rows=torch.arange(1, device=DEV))
    _kvq_call(ratio, warm[0], swa, warm[2], tbl, warm[3])
    torch.cuda.synchronize()
    before = _compiled_variants()
    rows_total, kinds, top, ks, n_calls = 0, set(), -1, [], 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, (ctx, gen, next_n) in enumerate(steps):
            seqs = ctx + [(p, next_n) for p in gen]
            k = len(seqs)
            step = _envelope_step(ratio, seqs, g, rows=torch.arange(k, device=DEV))
            req_id, _, swa_local, comp_local, _, last, trow = step
            _envelope_far_newest(swa, trow, last, swa_local, SWA_TPB, POOL_INT32_EDGE[128])
            out = _kvq_call(ratio, req_id, swa, swa_local, tbl, comp_local)
            ref = _envelope_kvq_reference(ratio, W, req_id, swa, swa_local, comp_local)
            assert torch.equal(out.long(), ref), (ratio, cfg, i, len(ctx), len(gen), next_n)
            m = rng.randint(k, mbs)
            assert torch.equal(
                _kvq_call(ratio, req_id, swa, swa_local, _envelope_kvq_table(k, W), comp_local), out
            )
            assert torch.equal(
                _kvq_call(ratio, req_id, swa, swa_local, _envelope_kvq_table(m, W), comp_local), out
            )
            n_calls += 3
            cmax = int(ref[:, WINDOW:].max())
            assert cmax < mbs * L_r
            top = max(top, cmax)
            rows_total += req_id.numel()
            kinds.add((bool(ctx), bool(gen), next_n if gen else 0))
            ks.append(k)
    torch.cuda.synchronize()
    far = (mbs - 1) * L_r + (c["max_gen"] + 1) // ratio - 1
    print(
        f"ratio={ratio} cfg={cfg} W={W}: {len(steps)} steps ({n_calls} calls), {rows_total} rows, requests "
        f"{min(ks)}..{max(ks)}, kinds {sorted(kinds)}; largest compressed row {top} (far end {far})"
    )
    assert top == far, (top, far)
    assert {
        (True, True, 1),
        (True, True, 6),
        (False, True, 1),
        (False, True, 6),
        (True, False, 0),
    } <= kinds
    _check_variants(prof, n_calls, before)


# Capture of the synthetic-table call. The target builds the per-request read buffer (and with it the
# synthetic table) only when it holds no more rows than the per-query one: max_batch_size * W * 128 <=
# max_num_tokens * 512, i.e. max_batch_size * W <= 65536 at max_num_tokens 16384; above that it passes
# the runtime compress table (test_envelope_capture_gset_runtime_tables). So at each width: every
# max_batch_size in GSET with max_batch_size * W <= 65536, plus 256 (a superset, the numbering at the
# largest batch), each capturing every G in GSET up to max_batch_size at next_n 1 and 6. At W 132 the
# max_batch_size values 1..32 are test_kvq_synthetic_table_cuda_graph_capture_set.
ENVELOPE_KVQ_PER_REQUEST_MAX = 4 * ENVELOPE_MAX_TOKENS


def _envelope_kvq_capture_mbs(cfg):
    W = ENVELOPE_WIDTHS[cfg]
    out = [m for m in GSET if m * W <= ENVELOPE_KVQ_PER_REQUEST_MAX or m == ENVELOPE_MAX_BATCH]
    return [m for m in out if m > 32] if cfg == (16384, True) else out


@pytest.mark.parametrize("cfg", ENVELOPE_CFGS, ids=ENVELOPE_CFG_IDS)
@pytest.mark.parametrize("ratio", [1, 2])
def test_envelope_kvq_capture_gset(ratio, cfg):
    """The native-storage call captured as the target captures it, at every max_batch_size of
    _envelope_kvq_capture_mbs(cfg): every G in GSET up to max_batch_size at next_n 1 and 6, prefix views of
    16384-row req / window buffers, the static [mbs, W] SWA and synthetic tables and a static
    [T, 512] top-k, after one eager call at that shape. Three replays over new steps (request 0
    ending at max_gen or L - 1, a fresh SWA table): each equals the reference and an eager call
    bitwise, and differs from the previous step's."""
    c = _envelope_cfg(*cfg)
    L, W, top = c["L"], c["W"], c["max_gen"]
    rng = random.Random(29800 + 1000 * ratio + 10 * ENVELOPE_CFGS.index(cfg))
    g = torch.Generator(device=DEV).manual_seed(
        29900 + 1000 * ratio + 10 * ENVELOPE_CFGS.index(cfg)
    )
    req_buf = torch.zeros(ENVELOPE_MAX_TOKENS, dtype=torch.int32, device=DEV)
    swa_buf = torch.full((ENVELOPE_MAX_TOKENS, WINDOW), -1, dtype=torch.int32, device=DEV)
    mbs_list = _envelope_kvq_capture_mbs(cfg)
    n_graphs = 0
    for mbs, (G, next_n) in ((m, gn) for m in mbs_list for gn in ENVELOPE_CAPTURE):
        if G > mbs:
            continue
        if G == 1 and next_n == 1:
            tbl = _envelope_kvq_table(mbs, W)
            swa = _envelope_table(POOL_INT32_EDGE[128], g, W, rows=mbs, layered=True)
        rows = G * next_n
        top_buf = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV)

        def load(it):
            starts = [rng.randint(0, top + 1 - next_n) for _ in range(G)]
            starts[0] = (top if it % 2 == 0 else L - 1) - next_n + 1
            step = _envelope_step(
                ratio, [(s, next_n) for s in starts], g, rows=torch.arange(G, device=DEV)
            )
            req_id, _, swa_local, comp_local, _, last, trow = step
            swa.copy_(_envelope_table(POOL_INT32_EDGE[128], g, W, rows=mbs, layered=True))
            _envelope_far_newest(swa, trow, last, swa_local, SWA_TPB, POOL_INT32_EDGE[128])
            req_buf[:rows].copy_(req_id)
            swa_buf[:rows].copy_(swa_local)
            top_buf.copy_(comp_local)
            return _envelope_kvq_reference(ratio, W, req_id, swa, swa_local, comp_local)

        def call():
            return _kvq_call(ratio, req_buf[:rows], swa, swa_buf[:rows], tbl, top_buf)

        prev = load(0)
        assert torch.equal(call().long(), prev)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for it in range(1, 4):
            ref = load(it)
            out.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.long(), ref), (cfg, mbs, G, next_n, it)
            assert torch.equal(out, call()), (cfg, mbs, G, next_n, it)
            assert not torch.equal(out.long(), prev), (cfg, mbs, G, next_n, it)
            prev = ref
        del graph
        n_graphs += 1
    print(
        f"ratio={ratio} cfg={cfg} W={W}: max_batch_size {mbs_list}: {n_graphs} captures x 3 replays, "
        f"replay = eager = reference"
    )
    assert n_graphs == sum(2 * sum(1 for G in GSET if G <= m) for m in mbs_list)
