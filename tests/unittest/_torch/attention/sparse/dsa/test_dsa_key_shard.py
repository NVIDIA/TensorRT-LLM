# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exactness of the key-sharded DSA decode Top-K merge (dsa/key_shard.py) with emulated ranks on one GPU.

Every rank's buffer is a local tensor, so the pack of all ranks runs before the merges. Each
rank's local selection is the exact (score desc, index asc) top-``local_k`` of its owned keys.
The merged rows must equal the unsharded selection ``(fkey(score) desc, token index asc)`` as
index sets, on every rank, over two calls (both buffer slots), and all ranks must write the same
rows; with sorted pushes every row is rank-major and ascending within each rank. ``local_k`` 32
forces the exact fallback (re-selection from every rank's local scores) on every long row.
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

skip_no_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="needs an SM90+ GPU (Triton atomics, sys scope)",
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
