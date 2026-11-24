# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exactness of the key-sharded DSA decode Top-K merge (dsa/key_shard.py) with emulated ranks on one GPU.

Every rank's buffer is a local tensor, so the pack of all ranks runs before the merges. Rank ``r``
of ``W`` owns the 64-token blocks ``b`` with ``b % W == r`` (the Helix page layout), and its local
selection is the exact (score desc, index asc) top-``local_k`` of its owned keys. Each rank's merge
keeps its own winners as local token indices (``merge_owned``); mapped back to global positions,
the ranks' rows must partition the unsharded selection ``(fkey(score) desc, token index asc)``, over
two calls (both buffer slots), and every row's count must be its number of kept entries. Each rank's
row is exactly its share of the selection in ascending local position (the pushes are sorted).
``local_k`` 32 forces the exact fallback (re-selection from the short ranks' local scores) on every long
row; ``reselect_cap`` 64 makes most re-selections overflow their candidate list and stream the scores.
With deterministic local ties the merge also runs with ``ties_exact`` (whole-key certificate).
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
        elif kind == "hotspot":  # local hot spot: the best keys are recent and contiguous
            s = torch.randn(n, device="cuda", generator=gen) * 0.1
            s[max(0, n - 3000) :] += 5.0
        elif kind == "hotrank":  # every page of rank 0 (of any world) beats every other key
            s = torch.randn(n, device="cuda", generator=gen) * 0.1
            s[(torch.arange(n, device="cuda") // TPB) % 16 == 0] += 5.0
        else:  # "tierank": rank 0's pages all tie at one score above every other key
            s = torch.randn(n, device="cuda", generator=gen) * 0.1
            s[(torch.arange(n, device="cuda") // TPB) % 16 == 0] = 5.0
        rows.append(s)
    return rows


def _owned_rows(selected, rank: int, world: int) -> torch.Tensor:
    """Rank ``rank``'s share of each row's selected global positions, as ascending local token indices, then -1."""
    out = torch.full((len(selected), K), -1, dtype=torch.int32, device="cuda")
    for i, sel in enumerate(selected):
        mine = sorted(p for p in sel if (p // TPB) % world == rank)
        local = [(p // TPB) // world * TPB + p % TPB for p in mine]
        out[i, : len(local)] = torch.tensor(local, dtype=torch.int32)
    return out


def _run(ks, world, rows, local_k, calls=2, tie_gen=None, pdl=False, reselect_cap=None):
    R = len(rows)
    width = max(-(-max(s.numel() for s in rows) // (TPB * world)) * TPB, 4)
    need = ks.KeyShardTopKMerge.bytes_needed(R, K, world, local_k, width)
    bufs = [torch.zeros(need, dtype=torch.uint8, device="cuda") for _ in range(world)]
    merges = [
        ks.KeyShardTopKMerge(
            bufs, r, R, K, width, local_k=local_k, max_spin=1 << 20, reselect_cap=reselect_cap
        )
        for r in range(world)
    ]
    for m in merges:
        m.pdl = pdl
    ref = [set(_top(s, torch.arange(s.numel(), device="cuda"), K).tolist()) for s in rows]
    want = [_owned_rows(ref, r, world) for r in range(world)]
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
        got = [set() for _ in rows]
        for r in range(world):
            out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
            counts = torch.full((R,), -3, dtype=torch.int32, device="cuda")
            merges[r].merge_owned(out, counts, ties_exact=tie_gen is None)
            nvalid = (out >= 0).sum(1)
            prefix = torch.arange(K, device="cuda")[None, :] < nvalid[:, None]
            assert torch.equal(out >= 0, prefix) and bool((out[~prefix] == -1).all())
            assert torch.equal(counts, nvalid.int()), f"rank {r}"
            assert torch.equal(out, want[r]), f"rank {r}"
            for i, s in enumerate(rows):
                pos = _local_positions(s.numel(), r, world)
                kept = pos[out[i][out[i] >= 0].long()].tolist()
                assert got[i].isdisjoint(kept), f"rank {r} row {i}: a position kept twice"
                got[i].update(kept)
        assert got == ref
    return [m.stats() for m in merges]


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "kind", ["normal", "ties", "zeros", "inf", "hotspot", "hotrank", "tierank"]
)
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
@pytest.mark.parametrize("world", [2, 16])
@pytest.mark.parametrize("kind", ["ties", "hotrank", "tierank"])
@pytest.mark.parametrize("local_ties", ["det", "random"])
def test_key_shard_merge_exact_list_overflow(world, kind, local_ties):
    """Re-selections whose candidate list overflows stream the short ranks' scores: same rows."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world * 7 + len(kind))
    lens = [2049, 3000, 20000, 70000]
    tie_gen = gen if local_ties == "random" else None
    stats = _run(ks, world, _make_rows(kind, lens, gen), 32, tie_gen=tie_gen, reselect_cap=64)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"
    assert stats[0][2] > 0 and stats[0][4] > 0


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("local_k", [512, 32])
def test_key_shard_merge_exact_pdl(world, local_k):
    """Programmatic dependent launch of the pack and merge (the fixed setting) gives the same rows."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks

    gen = torch.Generator(device="cuda")
    gen.manual_seed(world + local_k)
    lens = [1, 63, 2049, 3000, 20000, 70000]
    stats = _run(ks, world, _make_rows("hotspot", lens, gen), local_k, pdl=True)
    assert all(s[3] == 0 for s in stats), "spin timeout"
    assert all(s[1] == 0 for s in stats), "epoch ticket not reset"


def _key_shard_rows():
    """dsa/key_shard_rows.py, loaded by path from the source tree when it is there (no torch / tensorrt_llm
    import, so the row-capacity tests also run without a GPU build)."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[6]
        / "tensorrt_llm/_torch/attention/backends/sparse/dsa/key_shard_rows.py"
    )
    if not path.is_file():
        from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard_rows

        return key_shard_rows
    spec = importlib.util.spec_from_file_location("_dsa_key_shard_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "batch, draft_len, rows",
    [(8, 7, 64), (16, 5, 96), (24, 5, 144), (1, 0, 1)],  # helix8 GEN b8k7 / b16k5 / b24k5
)
def test_key_shard_max_rows(batch, draft_len, rows):
    """The merge workspace holds max_batch_size x (max_draft_len + 1) rows."""
    assert _key_shard_rows().key_shard_max_rows(batch, draft_len) == rows


@pytest.mark.parametrize("batch, draft_len", [(0, 5), (8, -1)])
def test_key_shard_max_rows_rejects_bad_limits(batch, draft_len):
    with pytest.raises(ValueError):
        _key_shard_rows().key_shard_max_rows(batch, draft_len)
