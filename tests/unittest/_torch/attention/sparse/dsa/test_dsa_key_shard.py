# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exactness of the key-sharded DSA decode Top-K merge (dsa/key_shard.py) with emulated ranks on one GPU.

Every rank's buffer is a local tensor, so the pack of all ranks runs before the merges. Each
rank's local selection is the exact (score desc, index asc) top-``local_k`` of its owned keys.
The merged rows must equal the unsharded selection ``(fkey(score) desc, token index asc)`` as
index sets, on every rank, over two calls (both buffer slots), and all ranks must write the same
rows; with sorted pushes every row is rank-major and ascending within each rank. ``local_k`` 32
forces the exact fallback (re-selection from every rank's local scores) on every long row.

The Helix form (``merge_owned``) must give every rank its own share of that selection as local token
indices; its gate, its local Top-L and the score comparison of its debug cross-check are tested too.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

skip_no_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="needs an SM90+ GPU (Triton atomics, sys scope)",
)
skip_no_cute_topk = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100 or not IS_CUTLASS_DSL_AVAILABLE,
    reason="needs a Blackwell GPU and the CuTe DSL (radix-filter Top-K)",
)

K = 2048
TPB = 64


def _fkey(scores: torch.Tensor) -> torch.Tensor:
    bits = scores.view(torch.int32).long()
    return torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


def _top(scores: torch.Tensor, positions: torch.Tensor, k: int, gen=None) -> torch.Tensor:
    """Exact (fkey desc, position asc) top-k of ``scores`` at ``positions``; returns positions. With ``gen``, ties
    are broken in a random order instead (a valid Top-K by score, not the deterministic one)."""
    if positions.numel() <= k:
        return positions
    key = _fkey(scores)
    if gen is None:
        order = torch.argsort(positions, stable=True)
    else:
        order = torch.randperm(positions.numel(), device=positions.device, generator=gen)
    order = order[torch.argsort(-key[order], stable=True)]
    return positions[order[:k]]


def _local_positions(n: int, rank: int, world: int) -> torch.Tensor:
    blocks = torch.arange(rank, max(rank, -(-n // TPB)), world, device="cuda")
    pos = (blocks[:, None] * TPB + torch.arange(TPB, device="cuda")[None, :]).reshape(-1)
    return pos[pos < n]


def _make_rows(kind: str, lens, gen):
    rows = []
    for n in lens:
        if kind == "normal":
            s = torch.randn(n, device="cuda", generator=gen)
        elif kind == "ties":
            # few distinct values: exact ties at the Top-K boundary
            s = torch.randint(0, 6, (n,), device="cuda", generator=gen).float() * 0.25
        elif kind == "zeros":
            s = torch.zeros(n, device="cuda")
            s[1::2] = -0.0
            s[: n // 3] = torch.randn(n // 3, device="cuda", generator=gen)
        elif kind == "inf":
            s = torch.full((n,), float("inf"), device="cuda")
            s[::3] = -float("inf")
            s[1::5] = 1.0
        else:  # local hot spot: the best keys are recent and contiguous
            s = torch.randn(n, device="cuda", generator=gen) * 0.1
            s[max(0, n - 3000) :] += 5.0
        rows.append(s)
    return rows


def _rank_major_ascending(out: torch.Tensor, world: int) -> bool:
    """Every row's valid entries run rank by rank (the owner of position p is (p // TPB) % world), ascending."""
    for row in out:
        v = row[row >= 0].long()
        if v.numel() < 2:
            continue
        owner = (v // TPB) % world
        same = owner[1:] == owner[:-1]
        if not (bool((owner[1:] >= owner[:-1]).all()) and bool(((v[1:] > v[:-1]) | ~same).all())):
            return False
    return True


def _run(ks, world, rows, local_k, calls=2, tie_gen=None, sorted_rows=True, pdl=False):
    R = len(rows)
    width = max(-(-max(s.numel() for s in rows) // (TPB * world)) * TPB, 4)
    need = ks.KeyShardTopKMerge.bytes_needed(R, K, world, local_k, width)
    bufs = [torch.zeros(need, dtype=torch.uint8, device="cuda") for _ in range(world)]
    merges = [
        ks.KeyShardTopKMerge(bufs, r, R, K, width, local_k=local_k, max_spin=1 << 20)
        for r in range(world)
    ]
    for m in merges:
        m.sorted_rows = sorted_rows
        m.pdl = pdl
    ref = [set(_top(s, torch.arange(s.numel(), device="cuda"), K).tolist()) for s in rows]
    for _ in range(calls):
        for r in range(world):
            # The local logits kernel writes into the merge workspace.
            logits = merges[r].local_logits[:R]
            logits.fill_(-1.0)
            local = torch.full((R, local_k), -1, dtype=torch.int32, device="cuda")
            lens = torch.zeros(R, dtype=torch.int32, device="cuda")
            for i, s in enumerate(rows):
                pos = _local_positions(s.numel(), r, world)
                logits[i, : pos.numel()] = s[pos]
                lens[i] = pos.numel()
                sel = _top(s[pos], torch.arange(pos.numel(), device="cuda"), local_k, tie_gen)
                local[i, : sel.numel()] = sel.int()
            merges[r].pack(local, lens)
        outs = []
        for r in range(world):
            out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
            merges[r].merge(out)
            outs.append(out)
        for out in outs:
            got = [set(x[x >= 0].tolist()) for x in out]
            assert got == ref
            nvalid = (out >= 0).sum(1)
            prefix = torch.arange(K, device="cuda")[None, :] < nvalid[:, None]
            assert torch.equal(out >= 0, prefix) and bool((out[~prefix] == -1).all())
            if sorted_rows:
                assert _rank_major_ascending(out, world)
        assert all(torch.equal(outs[0], o) for o in outs[1:]), "ranks disagree"
    return [m.stats() for m in merges]


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 16])
@pytest.mark.parametrize("kind", ["normal", "ties", "zeros", "inf", "hotspot"])
@pytest.mark.parametrize("local_k", [512, 256, 32])
@pytest.mark.parametrize("local_ties", ["det", "random"])
def test_key_shard_merge_exact(world, kind, local_k, local_ties):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world * 100 + len(kind))
    lens = [1, 63, 1500, 2048, 2049, 3000, 20000, 70000]
    tie_gen = gen if local_ties == "random" else None
    stats = _run(ks, world, _make_rows(kind, lens, gen), local_k, tie_gen=tie_gen)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"
    if local_k * world < K:
        # rows with more keys than the ranks send cannot pass the certificate
        assert stats[0][2] > 0


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("local_k", [512, 32])
def test_key_shard_merge_exact_pdl(world, local_k):
    """Programmatic dependent launch of the pack and merge gives the same rows as plain launches."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world + local_k)
    lens = [1, 63, 2049, 3000, 20000, 70000]
    stats = _run(ks, world, _make_rows("hotspot", lens, gen), local_k, pdl=True)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"


def _run_owned(ks, world, rows, local_k, owner_rows, calls=2):
    """Helix form of the merge: every rank keeps only its own winners, as local token indices, and their count.

    Rank ``r``'s rows must be the unsharded Top-K restricted to the keys ``r`` owns (page ``g`` on rank
    ``g % world`` as local page ``g // world``), kept entries first and ascending, the rest ``-1``; the kept counts
    must match and add up to the row's Top-K size over the ranks. Runs both receive slots (two calls).
    """
    R = len(rows)
    width = max(-(-max(s.numel() for s in rows) // (TPB * world)) * TPB, 4)
    need = ks.KeyShardTopKMerge.bytes_needed(R, K, world, local_k, width)
    bufs = [torch.zeros(need, dtype=torch.uint8, device="cuda") for _ in range(world)]
    merges = [
        ks.KeyShardTopKMerge(
            bufs,
            r,
            R,
            K,
            width,
            local_k=local_k,
            max_spin=1 << 20,
            owner_rows=owner_rows,
            pdl=False,
        )
        for r in range(world)
    ]
    ref = [_top(s, torch.arange(s.numel(), device="cuda"), K) for s in rows]
    want = []
    for r in range(world):
        per_row = []
        for sel in ref:
            block = sel // TPB
            mine = sel[block % world == r]
            per_row.append(sorted(((mine // TPB) // world * TPB + mine % TPB).tolist()))
        want.append(per_row)
    for _ in range(calls):
        for r in range(world):
            logits = merges[r].local_logits[:R]
            logits.fill_(-1.0)
            local = torch.full((R, local_k), -1, dtype=torch.int32, device="cuda")
            lens = torch.zeros(R, dtype=torch.int32, device="cuda")
            for i, s in enumerate(rows):
                pos = _local_positions(s.numel(), r, world)
                logits[i, : pos.numel()] = s[pos]
                lens[i] = pos.numel()
                sel = _top(s[pos], torch.arange(pos.numel(), device="cuda"), local_k)
                local[i, : sel.numel()] = sel.int()
            merges[r].pack(local, lens)
        outs = [torch.full((R, K), -7, dtype=torch.int32, device="cuda") for _ in range(world)]
        counts = [torch.full((R,), -7, dtype=torch.int32, device="cuda") for _ in range(world)]
        if owner_rows:
            # Every owner publishes its rows' thresholds before any rank selects (the ranks run concurrently in a
            # real group; here one GPU runs them in turn).
            for r in range(world):
                merges[r].merge_owned_begin(R)
            for r in range(world):
                merges[r].merge_owned_end(outs[r], counts[r])
        else:
            for r in range(world):
                merges[r].merge_owned(outs[r], counts[r])
        total = torch.zeros(R, dtype=torch.int64, device="cuda")
        for r in range(world):
            out, count = outs[r], counts[r]
            nvalid = (out >= 0).sum(1)
            assert torch.equal(nvalid.int(), count), (
                f"rank {r}: counts differ from the kept entries"
            )
            prefix = torch.arange(K, device="cuda")[None, :] < nvalid[:, None]
            assert torch.equal(out >= 0, prefix) and bool((out[~prefix] == -1).all())
            for i in range(R):
                assert out[i, : int(nvalid[i])].tolist() == want[r][i], f"rank {r} row {i}"
            total += nvalid
        assert total.tolist() == [min(K, s.numel()) for s in rows]
    return [m.stats() for m in merges]


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("kind", ["normal", "ties", "hotspot"])
@pytest.mark.parametrize("local_k", [512, 32])
@pytest.mark.parametrize("owner_rows", [False, True])
def test_key_shard_merge_owned_exact(world, kind, local_k, owner_rows):
    """The Helix merge (``merge_owned``): each rank's own share of the exact global Top-K and its count."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world * 1000 + len(kind) + local_k)
    # Includes a row with no key at all and rows shorter than one page per rank.
    lens = [0, 1, 63, 130, 1500, 2048, 2049, 3000, 20000, 70000]
    rows = _make_rows(kind, lens, gen)
    stats = _run_owned(ks, world, rows, local_k, owner_rows)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"
    if local_k * world < K:
        assert stats[0][2] > 0


@skip_no_sm90
def test_key_shard_merge_owned_more_rows_than_sms():
    """A full batch: more decode rows than SMs (and than a wave of the merge grid), short and long rows mixed."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    rows = torch.cuda.get_device_properties(0).multi_processor_count + 8
    lens = ([0, 1, 63, 130, 700, 2049, 5000] * rows)[:rows]
    stats = _run_owned(ks, 8, _make_rows("normal", lens, gen), 512, owner_rows=False, calls=1)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"


@pytest.mark.parametrize(
    "helix, num_gen, rows, capacity, ragged, spec_bounds, expected_next_n",
    [
        # Helix merges at every batch size that fits the workspace ...
        (True, 1, 6, 384, False, True, 6),
        (True, 64, 384, 384, False, True, 6),
        (True, 3, 3, 384, False, False, 1),
        # ... and leaves the step to merge_helix_topk otherwise.
        (True, 65, 390, 384, False, True, 0),
        (True, 4, 24, 384, True, True, 0),
        (True, 4, 22, 384, False, True, 0),
        (True, 4, 24, 384, False, False, 0),
        # The TP path keeps its batch floor.
        (False, 10, 60, 384, False, True, 0),
    ],
)
def test_key_shard_step_gate(helix, num_gen, rows, capacity, ragged, spec_bounds, expected_next_n):
    """``_prepare_key_shard_decode``: which decode steps take the key-sharded merge."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    meta = SimpleNamespace(
        is_ragged_verify=ragged,
        enable_helix=helix,
        num_generations=num_gen,
        num_ctx_tokens=0,
        num_tokens=rows,
        key_shard_next_n=-1,
        key_shard_row_lens=torch.empty(capacity, dtype=torch.int32),
        helix_spec_row_kv_lens=lambda n: torch.empty(n) if spec_bounds else None,
    )
    DSAtrtllmAttentionMetadata._prepare_key_shard_decode(meta, None)
    assert meta.key_shard_next_n == expected_next_n


@pytest.mark.parametrize(
    "k31, role, env, expected",
    [
        # Unset: only a Kimi K3.1 generation server of a disaggregated deployment.
        (True, "generation", None, True),
        (True, "context", None, False),
        (True, None, None, False),  # aggregated serving
        (False, "generation", None, False),
        # The env forces it either way.
        (False, "generation", "1", True),
        (True, "context", "1", True),
        (True, "generation", "0", False),
    ],
)
def test_helix_key_shard_enabled(monkeypatch, k31, role, env, expected):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import helix_key_shard as hks
    from tensorrt_llm._torch.attention.backends.sparse.dsa import params

    monkeypatch.setattr(params, "_KIMI_K31_DSA", k31)
    for name, value in (("TRTLLM_DISAGG_ROLE", role), (hks.HELIX_KEY_SHARD_ENV, env)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert hks.helix_key_shard_enabled() is expected


def test_helix_verify_score_comparison_ignores_ties():
    """The debug cross-check's score comparison: equal for two valid selections that break a tie differently,
    different when one selection keeps a worse key, with a tolerance for scores from two kernels."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.helix_key_shard import (
        count_score_mismatches,
        count_set_mismatches,
        kept_scores,
    )

    # Two ranks, one row, top-3 of 4 keys per rank. Scores: rank 0 [5, 3, 3, 1], rank 1 [4, 3, 0, 0];
    # the global top-3 is {5, 4, 3} and three keys tie at 3.
    logits = [torch.tensor([[5.0, 3.0, 3.0, 1.0]]), torch.tensor([[4.0, 3.0, 0.0, 0.0]])]
    pick_a = [torch.tensor([[0, 1, -1]]), torch.tensor([[0, -1, -1]])]  # the 3 from rank 0
    pick_b = [torch.tensor([[0, -1, -1]]), torch.tensor([[0, 1, -1]])]  # the 3 from rank 1
    # Keeps the 1 instead of a 3.
    pick_bad = [torch.tensor([[0, 3, -1]]), torch.tensor([[0, -1, -1]])]

    def scores(picks):
        return torch.stack([kept_scores(lg, rows) for lg, rows in zip(logits, picks)])

    assert count_set_mismatches(pick_a[0], pick_b[0])[0] == 1  # the index sets differ on rank 0
    assert count_score_mismatches(scores(pick_a), scores(pick_b)) == 0
    assert count_score_mismatches(scores(pick_a), scores(pick_bad)) == 1
    assert count_score_mismatches(scores(pick_a), scores(pick_bad), tolerance=2.5) == 0
    # A selection that keeps fewer keys differs however large the tolerance.
    short = [torch.tensor([[0, -1, -1]]), torch.tensor([[0, -1, -1]])]
    assert count_score_mismatches(scores(pick_a), scores(short), tolerance=100.0) == 1


@skip_no_cute_topk
@pytest.mark.parametrize("batch", ["one_sequence", "above_sm_count"])
def test_helix_local_topk_matches_reference(batch):
    """The Helix local Top-L keeps each row's largest scores: for one sequence (6 rows) and for more rows than
    SMs (the kernel's other occupancy variant), for empty, short and full rows, a run of equal scores across
    the cut-off, and scores read from rows that are wider than the scored width (the merge workspace)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.helix_key_shard import (
        helix_local_topk,
        kept_scores,
        reference_local_topk,
    )

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = 6 if batch == "one_sequence" else sms + 1
    gen = torch.Generator(device="cuda")
    gen.manual_seed(rows)
    width, stride, local_k = 16448, 16640, 512
    workspace = torch.randn(rows + 1, stride, device="cuda", generator=gen)
    logits = workspace[:rows, :width]
    lens = [0, 1, 511, 513, width]
    lens += torch.randint(0, width + 1, (rows - len(lens),), device="cuda", generator=gen).tolist()
    # The last row: 100 scores near 20, then 3000 equal scores that straddle the 512th place.
    lens[-1] = width
    logits[-1, :3000] = 7.0
    logits[-1, 3000:3100] += 20.0
    row_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.full((rows, local_k), 2**30, dtype=torch.int32, device="cuda")
    helix_local_topk(logits, row_lens, out)

    want = reference_local_topk(logits, row_lens, local_k)
    kept = out >= 0
    assert ((out == -1) | kept).all()
    assert torch.equal(kept.sum(1), row_lens.clamp(max=local_k).long())
    assert (out < row_lens[:, None]).all()
    order = torch.where(kept, out, torch.full_like(out, 2**30)).sort(dim=1).values
    assert ((order[:, 1:] != order[:, :-1]) | (order[:, 1:] == 2**30)).all()  # no position twice
    assert torch.equal(
        kept_scores(logits, out).sort(dim=1, descending=True).values,
        kept_scores(logits, want).sort(dim=1, descending=True).values,
    )


def test_key_shard_local_k():
    from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import key_shard_local_k

    # About 4x the fair share of a 2048 Top-K, within [256, 512].
    assert key_shard_local_k(2048, 8) == 512
    assert key_shard_local_k(2048, 16) == 512
    assert key_shard_local_k(2048, 32) == 256
    assert key_shard_local_k(2048, 64) == 256


def test_key_shard_local_count():
    from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import key_shard_local_count

    for world in (1, 2, 8, 16):
        n = torch.arange(0, 5000, 7)
        total = sum(key_shard_local_count(n, r, world) for r in range(world))
        assert torch.equal(total, n)
        for r in range(world):
            expect = torch.tensor(
                [int(((torch.arange(int(x)) // TPB) % world == r).sum()) for x in n]
            )
            assert torch.equal(key_shard_local_count(n, r, world), expect)


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 8, 16])
@pytest.mark.parametrize("next_n", [1, 4, 7, 8])
def test_key_shard_prepare_rows(world, next_n):
    from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import (
        key_shard_local_count,
        key_shard_prepare_rows,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world * 10 + next_n)
    seqs, width = 13, 2048
    kv_lens = torch.randint(
        next_n, width * TPB, (seqs,), device="cuda", generator=gen, dtype=torch.int32
    )
    kv_lens[0], kv_lens[1] = next_n, TPB * world * 3 + 5
    table = torch.randint(
        -1, 1 << 20, (seqs + 3, width), device="cuda", generator=gen, dtype=torch.int32
    )
    local_width = -(-width // world)
    for rank in range(world):
        local_table = torch.full((seqs + 2, local_width), -7, dtype=torch.int32, device="cuda")
        ctx = torch.zeros(seqs + 2, dtype=torch.int32, device="cuda")
        rows = torch.zeros((seqs + 2) * next_n, dtype=torch.int32, device="cuda")
        key_shard_prepare_rows(
            kv_lens, table[1 : seqs + 1], local_table, ctx, rows, rank, world, next_n
        )
        want_ctx = key_shard_local_count(kv_lens.long(), rank, world).int()
        offs = torch.arange(1 - next_n, 1, device="cuda")
        want_rows = key_shard_local_count(
            (kv_lens.long()[:, None] + offs[None, :]).reshape(-1), rank, world
        ).int()
        assert torch.equal(ctx[:seqs], want_ctx)
        assert torch.equal(rows[: seqs * next_n], want_rows)
        want_table = table[1 : seqs + 1, rank::world].clamp(min=0)
        for s in range(seqs):
            c = -(-int(want_ctx[s]) // TPB)
            assert torch.equal(local_table[s, :c], want_table[s, :c])
            assert bool((local_table[s, c:] == -7).all())
