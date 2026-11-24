# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the key-sharded decode Top-K entries rubin_dsa_key_shard_pack / _merge and their
workspace.

CPU: the merge reference against the engine's merge_helix_topk algorithm over each rank's exact
local Top-K (distinct scores: the same sets), the pack reference's key order, the workspace
layout, the guards. GPU (sm_107): every rank of a group on one GPU over plain local buffers
(``key_shard_workspace``), packed then merged in group order, as the multi-rank run does over
MNNVL: each rank's rows, counts and mask against the merge reference, bit for bit; the exact
fallback rows are counted in the state; a second step reuses the workspace (the epoch advances);
``failing_only`` gives every rank the same bytes as streaming every rank's logits, on every cell.

CPU also checks the arithmetic behind ``failing_only``: the Top-K of the ranks' candidates whose
certificate holds plus every key of the ranks whose certificate failed is the row's global Top-K,
for any threshold at or below the candidates' K-th key (the kernel's radix prefix is one).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_dsa_key_shard import (  # noqa: E501
    KEY_SHARD_LOCAL_K,
    key_shard_bytes_needed,
    key_shard_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (  # noqa: E501
    rubin_dsa_key_shard as merge_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_merge as merge_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_pack as pack_op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)

TPB = 64


def _group(spec, gen):
    """Every rank's local logits ``[rows, width]`` and row lengths; distinct scores unless ``ties``."""
    world, rows = spec["world"], spec["rows"]
    lo, hi = spec["row_lens"]
    width = -(-hi // 256) * 256
    lens = [
        torch.randint(lo, hi + 1, (rows,), generator=gen, dtype=torch.int32) for _ in range(world)
    ]
    skewed = range(int(spec.get("skew", 0)))
    for w in skewed:
        # Rank 0 (ranks 0 .. skew - 1) holds the best keys of every row: more than its 512 make
        # the global Top-K.
        lens[w] = torch.full((rows,), hi, dtype=torch.int32)
    total = world * rows * width
    if spec.get("ties"):
        vals = torch.randint(-4, 5, (total,), generator=gen).float()
    else:
        vals = (torch.randperm(total, generator=gen).float() + 1) / (total + 1)
    logits = vals.view(world, rows, width).clone()
    for w in skewed:
        logits[w] += 2.0
    return logits, lens, width


def _local_topk(logits, lens, k):
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r])
        order = torch.sort(-logits[r, :n].double(), stable=True).indices[:k]
        out[r, : order.numel()] = order.to(torch.int32)
    return out


def _engine_merge(logits, lens, rank, k):
    """merge_helix_topk (dsa/indexer.py) over every rank's exact local Top-K."""
    world, rows, _ = logits.shape
    topk = [_local_topk(logits[w], lens[w], k) for w in range(world)]
    scores = []
    for w in range(world):
        valid = topk[w] >= 0
        s = torch.gather(logits[w], 1, topk[w].clamp(min=0).long())
        scores.append(s.masked_fill(~valid, float("-inf")))
    all_scores = torch.stack(scores).transpose(0, 1).reshape(rows, -1)
    top_scores, top_pos = torch.topk(all_scores, k, dim=1)
    owner = torch.div(top_pos, k, rounding_mode="floor")
    mine = (owner == rank) & torch.isfinite(top_scores)
    kept = []
    for r in range(rows):
        slots = top_pos[r][mine[r]] - rank * k
        kept.append(set(topk[rank][r, slots].tolist()))
    return kept


# --- CPU -------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_merge_reference_equals_the_engine_merge(seed):
    spec = dict(world=4, rows=6, row_lens=(100, 900), top_k=256)
    gen = torch.Generator().manual_seed(seed)
    logits, lens, _ = _group(spec, gen)
    for rank in range(spec["world"]):
        rows, counts, mask = merge_op.reference(list(logits), lens, rank, spec["top_k"], None)
        assert mask is None
        want = _engine_merge(logits, lens, rank, spec["top_k"])
        for r in range(spec["rows"]):
            got = rows[r][rows[r] >= 0]
            assert set(got.tolist()) == want[r] and int(counts[r]) == len(want[r])
            assert torch.equal(got, torch.sort(got).values), "ascending local positions"


def test_merge_reference_mask():
    spec = dict(world=2, rows=4, row_lens=(0, 50), top_k=64)
    logits, lens, _ = _group(spec, torch.Generator().manual_seed(3))
    bounds = torch.tensor([0, 7, 7, 7], dtype=torch.int32)
    _, counts, mask = merge_op.reference(list(logits), lens, 0, 64, bounds)
    assert torch.equal(mask, (bounds == 0) | (counts == 0))


def test_pack_reference_keys_order_by_score_then_index():
    logits = torch.tensor([[0.5, -0.0, 0.0, 0.5]])
    topk = torch.tensor([[3, 0, 2, 1]], dtype=torch.int32)
    keys = pack_op.reference([logits], [topk], [torch.tensor([4], dtype=torch.int32)])[0]
    # Ascending local position; equal scores: the smaller global index has the larger key;
    # -0.0 ranks below +0.0.
    assert keys[0, 0] > keys[0, 3] and keys[0, 2] > keys[0, 1]


def test_workspace_layout_fits_the_buffer():
    world, rows, top_k, width = 4, 16, 2048, 1000
    need = key_shard_bytes_needed(rows, top_k, world, KEY_SHARD_LOCAL_K, width)
    bufs = [torch.zeros(need, dtype=torch.uint8) for _ in range(world)]
    ws = key_shard_workspace(bufs, 1, rows, top_k, width)
    assert ws.local_width == 1024 and ws.local_logits.shape == (rows, 1024)
    assert ws.local_topk.shape == (rows, KEY_SHARD_LOCAL_K) and ws.counts.shape == (rows,)
    assert ws.rank == 1 and ws.world == world and ws.dk.numel() == world
    with pytest.raises(AssertionError):
        key_shard_workspace([b[:-1] for b in bufs], 0, rows, top_k, width)


def test_guards():
    bufs = [torch.zeros(key_shard_bytes_needed(4, 2048, 2, 512, 256), dtype=torch.uint8)] * 2
    ws = key_shard_workspace(bufs, 0, 4, 2048, 256)
    with pytest.raises(AssertionError, match="more rows"):
        pack_op.is_valid(
            torch.zeros(5, 512, dtype=torch.int32), torch.zeros(5, dtype=torch.int32), ws
        )
    with pytest.raises(AssertionError, match="mask_out goes with row_bounds"):
        merge_op.is_valid(
            torch.zeros(4, 2048, dtype=torch.int32),
            None,
            None,
            torch.zeros(4, dtype=torch.bool),
            ws,
        )


def _keys(scores, local, rank, world):
    return merge_mod._order_keys(scores, merge_mod._global_index(local, rank, world))


@pytest.mark.parametrize("family", ["uniform", "skew1", "skew2", "ties", "short", "zeros"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_failing_ranks_reselection_is_the_global_top_k(family, seed):
    """The re-selection ``failing_only`` makes: the K largest keys of (the candidates of the ranks
    whose certificate holds) + (every key of the ranks whose certificate failed) are the row's
    global Top-K, for the certificate taken at the candidates' K-th key or below it (the radix
    prefix), with any tie order in the local Top-L."""
    world, top_k, local_k = 8, 512, 128
    gen = torch.Generator().manual_seed(seed * 7 + len(family))
    for _ in range(12):
        lens = torch.randint(200, 900, (world,), generator=gen)
        if family == "short":
            lens = torch.randint(0, 120, (world,), generator=gen)
        logits = []
        for w in range(world):
            n = int(lens[w])
            if family in ("ties", "zeros"):
                x = torch.randint(-2, 3, (n,), generator=gen).float()
                if family == "zeros":
                    x = torch.where(torch.rand(n, generator=gen) < 0.9, torch.zeros(n), x.abs())
            else:
                x = torch.randn(n, generator=gen)
            if family == "skew1" and w == 3 or family == "skew2" and w in (2, 5):
                x = x + 4.0
            logits.append(x)
        keys = [_keys(logits[w], torch.arange(int(lens[w])), w, world) for w in range(world)]
        every = torch.cat(keys)
        want = set(torch.sort(every, descending=True).values[:top_k].tolist())
        # Each rank's local Top-L by score, equal scores in a random order (as the CuTe kernel).
        cands = []
        for w in range(world):
            perm = torch.randperm(int(lens[w]), generator=gen)
            order = torch.sort(-logits[w][perm].double(), stable=True).indices[:local_k]
            cands.append(keys[w][perm[order]])
        flat = torch.cat(cands)
        kth = torch.sort(flat, descending=True).values[min(top_k, flat.numel()) - 1]
        for thr in (kth, kth & ~0xFFFFFF, kth & ~((1 << 40) - 1)):
            if flat.numel() < top_k:
                thr = torch.tensor(-(2**63) + 1)
            fail = [
                bool(lens[w] > cands[w].numel() and (cands[w].min() >> 32) >= (thr >> 32))
                for w in range(world)
            ]
            pool = torch.cat([keys[w] if fail[w] else cands[w] for w in range(world)])
            got = set(torch.sort(pool, descending=True).values[:top_k].tolist())
            assert got == want, (family, fail)


# --- GPU: every rank on one GPU --------------------------------------------------------------


def _run_group(spec, logits, lens, width, workspaces, bounds):
    rows, top_k = spec["rows"], spec["top_k"]
    for r, ws in enumerate(workspaces):
        ws.local_logits[:rows, :width].copy_(logits[r].cuda())
        local = torch.full((rows, ws.local_k), -1, dtype=torch.int32)
        local.copy_(_local_topk(logits[r], lens[r], ws.local_k))
        ws.local_topk[:rows].copy_(local.cuda())
        pack_op(ws.local_topk[:rows], lens[r].cuda(), ws)
    results = []
    for r, ws in enumerate(workspaces):
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
        mask = torch.zeros(rows, dtype=torch.bool, device="cuda") if bounds is not None else None
        merge_op(
            out,
            counts,
            None if bounds is None else bounds.cuda(),
            mask,
            ws,
            failing_only=spec.get("failing_only", False),
        )
        results.append((out.cpu(), counts.cpu(), None if mask is None else mask.cpu()))
    torch.cuda.synchronize()
    return results


@on_sm107
@pytest.mark.parametrize("cell", merge_op.CELLS + pack_op.CELLS, ids=lambda c: c.why[:48])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(merge_op)
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["rows"] + spec["world"])
    logits, lens, width = _group(spec, gen)
    world, rows, top_k = spec["world"], spec["rows"], spec["top_k"]
    need = key_shard_bytes_needed(rows, top_k, world, KEY_SHARD_LOCAL_K, width)
    base = torch.zeros(world, -(-need // 256) * 256, dtype=torch.uint8, device="cuda")
    bufs = [base[r] for r in range(world)]
    workspaces = [
        key_shard_workspace(bufs, r, rows, top_k, width, max_spin=1 << 20) for r in range(world)
    ]
    bounds = None
    if spec.get("bounds", True):
        bounds = torch.randint(0, 3, (rows,), generator=gen, dtype=torch.int32)
    for step in range(2):
        got = _run_group(spec, logits, lens, width, workspaces, bounds)
        for r in range(world):
            ref_rows, ref_counts, ref_mask = merge_op.reference(
                list(logits), lens, r, top_k, bounds
            )
            out, counts, mask = got[r]
            assert torch.equal(out, ref_rows), f"rank {r} step {step}: rows"
            assert torch.equal(counts, ref_counts), f"rank {r} step {step}: counts"
            if bounds is not None:
                assert torch.equal(mask, ref_mask), f"rank {r} step {step}: mask"
        logits = logits.flip(-1)  # a new step's scores through the same workspace
    # The workspace state words: [merges, ticket, fallback rows, spin timeouts].
    stats = workspaces[0].state.tolist()
    assert stats[3] == 0, "a merge waited out its spin bound"
    if spec.get("skew"):
        assert stats[2] > 0, "the skewed rows take the exact fallback"


def _group_workspaces(spec, width):
    world, rows, top_k = spec["world"], spec["rows"], spec["top_k"]
    need = key_shard_bytes_needed(rows, top_k, world, KEY_SHARD_LOCAL_K, width)
    base = torch.zeros(world, -(-need // 256) * 256, dtype=torch.uint8, device="cuda")
    bufs = [base[r] for r in range(world)]
    return [
        key_shard_workspace(bufs, r, rows, top_k, width, max_spin=1 << 20) for r in range(world)
    ]


@on_sm107
@pytest.mark.parametrize("cell", merge_op.CELLS, ids=lambda c: c.why[:48])
def test_failing_only_gives_the_same_bytes(cell):
    """Every merge cell on the same inputs with ``failing_only`` off and on: every rank's rows,
    counts and mask byte for byte, and the same fallback-row count."""
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["rows"] + spec["world"] + 11)
    logits, lens, width = _group(spec, gen)
    bounds = None
    if spec.get("bounds", True):
        bounds = torch.randint(0, 3, (spec["rows"],), generator=gen, dtype=torch.int32)
    got, stats = [], []
    for failing_only in (False, True):
        ws = _group_workspaces(spec, width)
        got.append(
            _run_group(dict(spec, failing_only=failing_only), logits, lens, width, ws, bounds)
        )
        stats.append(ws[0].state.tolist())  # [merges, ticket, fallback rows, spin timeouts]
    for r in range(spec["world"]):
        for a, b in zip(got[0][r], got[1][r]):
            assert (a is None and b is None) or torch.equal(a, b), f"rank {r}"
    assert stats[0][2] == stats[1][2] and stats[0][3] == stats[1][3] == 0
    if spec.get("skew"):
        assert stats[0][2] > 0, "the skewed rows take the exact fallback"
