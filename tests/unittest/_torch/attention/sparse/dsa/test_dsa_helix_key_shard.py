# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded Top-K merge under Helix CP (dsa/helix_key_shard.py) with emulated ranks on one GPU.

Helix rank ``r`` of ``W`` holds the global pages ``g`` with ``g % W == r`` in page order. Every
rank packs the exact top-``local_k`` of its local scores and its merge keeps its winners of the
global Top-K as local token indices: bit for bit what ``_keep_owned_local_rows`` (below) makes of
the unsharded selection listed rank by rank in ascending position, the same in both buffer slots.
Mapped back to global positions, the ranks' kept rows must partition the unsharded selection, and
they must equal what ``merge_helix_topk`` (all-gather of the local Top-K) keeps.

The step-level decision (``RubinUllDSAMetadata._prepare_key_shard_decode``, the K3.1 ULL generation target's
DSA metadata, which keeps the key-shard state; CPU stand-ins): a Helix decode step takes the key-sharded merge
from one sequence on; one that takes merge_helix_topk says so once per reason (no verify-group bounds, a ragged
verify); a step with more rows than the merge workspace (max_batch_size x (max_draft_len + 1)) raises.
The workspace holds that many rows on the metadata the engine builds for that target (``_create_key_shard_buffers``,
``update_spec_dec_param``, CPU stand-ins): sized from the KV cache manager's draft ceiling before the first
draft length arrives, grown when a larger ceiling arrives, and shared with the CUDA graph metadata.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

K = 2048
TPB = 64

skip_no_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="needs an SM90+ GPU (Triton atomics, sys scope)",
)


def _owned_positions(n: int, rank: int, world: int, device="cuda") -> torch.Tensor:
    """Global positions of rank ``rank``'s local tokens, in local order."""
    pages = torch.arange(rank, max(rank, -(-n // TPB)), world, device=device)
    pos = (pages[:, None] * TPB + torch.arange(TPB, device=device)[None, :]).reshape(-1)
    return pos[pos < n]


def _top(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Exact (score desc, index asc) top-k indices of a 1-D score row."""
    if scores.numel() <= k:
        return torch.arange(scores.numel(), device=scores.device)
    order = torch.argsort(-scores, stable=True)
    return order[:k]


def _keep_owned_local_rows(rows: torch.Tensor, cp_rank: int, cp_size: int) -> None:
    """Rewrite merged global Top-K rows into rank ``cp_rank``'s local Top-K rows, in place.

    ``rows`` holds key-shard global token indices (``-1`` padded): index ``p`` belongs to rank
    ``(p // TPB) % cp_size`` at local position ``(p // TPB) // cp_size * TPB + p % TPB``. Entries of
    other ranks become ``-1`` and this rank's entries move to the front of each row in their original
    order, as ``merge_helix_topk`` leaves them.
    """
    if rows.numel() == 0:
        return
    block = torch.div(rows, TPB, rounding_mode="floor")
    mine = (rows >= 0) & (torch.remainder(block, cp_size) == cp_rank)
    local = torch.div(block, cp_size, rounding_mode="floor") * TPB + torch.remainder(rows, TPB)
    values = torch.where(mine, local, torch.full_like(rows, -1))
    order = torch.sort((~mine).to(torch.int8), dim=1, stable=True).indices
    rows.copy_(torch.gather(values, 1, order))


def _reference_local_topk(
    local_logits: torch.Tensor, row_lens: torch.Tensor, top_k: int
) -> torch.Tensor:
    """A rank's exact local Top-K rows (``-1`` padded) over each row's first ``row_lens`` scores: the
    candidates ``merge_helix_topk`` gathers from every rank.

    Args:
        local_logits: ``[num_rows, local_len]`` local indexer scores.
        row_lens: ``[num_rows]`` int32 number of valid local scores per row.
        top_k: Selection size.

    Returns:
        ``[num_rows, top_k]`` int32 local token indices, ``-1`` past the valid entries.
    """
    rows, cols = local_logits.shape
    pos = torch.arange(cols, device=local_logits.device)
    masked = local_logits.float().masked_fill(
        pos[None, :] >= row_lens[:, None].long(), float("-inf")
    )
    k = min(top_k, cols)
    values, indices = torch.topk(masked, k, dim=1)
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=local_logits.device)
    out[:, :k] = torch.where(torch.isfinite(values), indices.to(torch.int32), -1)
    return out


def _count_set_mismatches(got: torch.Tensor, want: torch.Tensor) -> tuple:
    """``(rows whose kept sets differ, entries in the symmetric difference, entries kept in want)``."""
    a = torch.sort(got, dim=1).values
    b = torch.sort(want, dim=1).values
    differs = (a != b).any(dim=1)
    kept_want = int((want >= 0).sum())
    sym_diff = 0
    for i in torch.nonzero(differs).flatten().tolist():
        sa = set(a[i][a[i] >= 0].tolist())
        sb = set(b[i][b[i] >= 0].tolist())
        sym_diff += len(sa ^ sb)
    return int(differs.sum()), sym_diff, kept_want


@pytest.mark.parametrize("world", [2, 4, 16])
def test_keep_owned_local_rows_cpu(world):
    """The reference the merge test below maps the unsharded selection to each rank's local rows with."""
    gen = torch.Generator().manual_seed(world)
    n = 40_000
    rows = torch.stack([torch.randperm(n, generator=gen)[:K] for _ in range(3)]).int()
    rows[1, 1500:] = -1
    for rank in range(world):
        got = rows.clone()
        _keep_owned_local_rows(got, rank, world)
        owned = _owned_positions(n, rank, world, device="cpu")
        local_of = {int(p): i for i, p in enumerate(owned)}
        for r in range(rows.shape[0]):
            want = [local_of[int(p)] for p in rows[r] if int(p) >= 0 and int(p) in local_of]
            assert got[r, : len(want)].tolist() == want
            assert bool((got[r, len(want) :] == -1).all())


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 4, 8, 16])
@pytest.mark.parametrize("local_k", [512, 32])
def test_helix_key_shard_matches_helix_merge(world, local_k):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import merge_helix_topk

    gen = torch.Generator(device="cuda").manual_seed(17 * world + local_k)
    lens = [1500, 2048, 70_001, 133_000, 158_000, 5_000]
    rows = [torch.randn(n, device="cuda", generator=gen) for n in lens]
    R = len(rows)
    width = max(-(-max(lens) // (TPB * world)) * TPB, 4)
    need = ks.KeyShardTopKMerge.bytes_needed(R, K, world, local_k, width)
    bufs = [torch.zeros(need, dtype=torch.uint8, device="cuda") for _ in range(world)]
    merges = [
        ks.KeyShardTopKMerge(bufs, r, R, K, width, local_k=local_k, max_spin=1 << 20)
        for r in range(world)
    ]
    local_scores = []
    for r in range(world):
        logits = merges[r].local_logits[:R]
        logits.fill_(-1.0)
        local = torch.full((R, local_k), -1, dtype=torch.int32, device="cuda")
        row_lens = torch.zeros(R, dtype=torch.int32, device="cuda")
        per_rank = []
        for i, s in enumerate(rows):
            pos = _owned_positions(s.numel(), r, world)
            logits[i, : pos.numel()] = s[pos]
            row_lens[i] = pos.numel()
            sel = _top(s[pos], local_k)
            local[i, : sel.numel()] = sel.int()
            per_rank.append(s[pos])
        local_scores.append(per_rank)
        merges[r].pack(local, row_lens)
    local_rows = []
    for r in range(world):
        local = torch.full((R, local_k), -1, dtype=torch.int32, device="cuda")
        for i, x in enumerate(local_scores[r]):
            sel = _top(x, local_k)
            local[i, : sel.numel()] = sel.int()
        local_rows.append(local)
    # The unsharded selection listed rank by rank (the owner of position p is (p // TPB) % world), ascending: the
    # pushes are sorted, so a rank keeps exactly its part of this list.
    listed = torch.full((R, K), -1, dtype=torch.int32, device="cuda")
    for i, s in enumerate(rows):
        sel = sorted(_top(s, K).tolist(), key=lambda p: ((p // TPB) % world, p))
        listed[i, : len(sel)] = torch.tensor(sel, dtype=torch.int32)
    kept = []
    for r in range(world):
        out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((R,), -3, dtype=torch.int32, device="cuda")
        merges[r].merge_owned(out, counts)
        want = listed.clone()
        _keep_owned_local_rows(want, r, world)
        assert torch.equal(out, want), f"rank {r}"
        assert torch.equal(counts, (out >= 0).sum(dim=1, dtype=torch.int32)), f"rank {r}"
        kept.append(out)

    # A second exchange of the same candidates (the other receive slot): the same rows, bitwise, and counts.
    for r in range(world):
        row_lens = torch.tensor(
            [x.numel() for x in local_scores[r]], dtype=torch.int32, device="cuda"
        )
        merges[r].pack(local_rows[r], row_lens)
    for r in range(world):
        out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((R,), -3, dtype=torch.int32, device="cuda")
        merges[r].merge_owned(out, counts)
        assert torch.equal(out, kept[r]), f"rank {r}"
        assert torch.equal(counts, (kept[r] >= 0).sum(dim=1, dtype=torch.int32)), f"rank {r}"

    # Reference: the Helix all-gather merge over every rank's local Top-K.
    ref_kept = []
    full_width = max(x.numel() for per_rank in local_scores for x in per_rank)
    padded = []
    for r in range(world):
        lg = torch.full((R, full_width), -float("inf"), device="cuda")
        for i, x in enumerate(local_scores[r]):
            lg[i, : x.numel()] = x
        row_lens = torch.tensor(
            [x.numel() for x in local_scores[r]], dtype=torch.int32, device="cuda"
        )
        padded.append((lg, _reference_local_topk(lg, row_lens, K)))
    stacked = torch.stack(
        [
            torch.gather(lq, 1, tq.clamp(min=0).long()).masked_fill(tq < 0, -float("inf"))
            for lq, tq in padded
        ]
    )
    for r in range(world):
        lg, tk = padded[r]
        rows_r = tk.clone()
        merge_helix_topk(lg, rows_r, r, lambda t, st=stacked: st)
        ref_kept.append(rows_r)

    for r in range(world):
        want = (0, 0, int((ref_kept[r] >= 0).sum()))
        assert _count_set_mismatches(kept[r], ref_kept[r]) == want, f"rank {r}"
    for i, s in enumerate(rows):
        union = set()
        for r in range(world):
            pos = _owned_positions(s.numel(), r, world)
            union |= set(pos[kept[r][i][kept[r][i] >= 0].long()].tolist())
        assert union == set(_top(s, K).tolist())


def test_reference_helpers_cpu():
    """The helpers the merge test above checks the key-sharded rows with: an exact local Top-K and the kept-set
    comparison."""
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 300, generator=gen)
    row_lens = torch.tensor([300, 10, 0, 150], dtype=torch.int32)
    got = _reference_local_topk(logits, row_lens, 64)
    assert got.shape == (4, 64)
    for r, n in enumerate(row_lens.tolist()):
        want = _top(logits[r, :n], 64).tolist() if n else []
        assert sorted(got[r][got[r] >= 0].tolist()) == sorted(want)
        assert int((got[r] >= 0).sum()) == min(n, 64)
    same = got[:, torch.randperm(64, generator=gen)]
    assert _count_set_mismatches(same, got) == (0, 0, int((got >= 0).sum()))
    off = got.clone()
    off[0, 0] = 299 if int(got[0, 0]) != 299 else 298
    rows, sym, kept = _count_set_mismatches(off, got)
    assert rows == 1 and kept == int((got >= 0).sum())
    assert sym in (0, 2)


@skip_no_sm90
@pytest.mark.parametrize("world", [2, 8])
def test_merge_owned_writes_the_helix_combine_mask(world):
    """With bounds, merge_owned also writes ``(bounds == 0) | (count == 0)`` bit for bit, and its rows and
    counts are the ones it writes without the mask."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import key_shard as ks
    from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import helix_zero_or_empty_rows

    local_k = 512
    gen = torch.Generator(device="cuda").manual_seed(31 * world)
    # A one-page row (only rank 0 owns keys: every other rank keeps nothing), short rows and long rows.
    lens = [64, 1500, 2048, 70_001, 9, 133_000]
    rows = [torch.randn(n, device="cuda", generator=gen) for n in lens]
    R = len(rows)
    width = max(-(-max(lens) // (TPB * world)) * TPB, 4)
    need = ks.KeyShardTopKMerge.bytes_needed(R, K, world, local_k, width)
    bufs = [torch.zeros(need, dtype=torch.uint8, device="cuda") for _ in range(world)]
    merges = [
        ks.KeyShardTopKMerge(bufs, r, R, K, width, local_k=local_k, max_spin=1 << 20)
        for r in range(world)
    ]

    def push():
        for r in range(world):
            logits = merges[r].local_logits[:R]
            logits.fill_(-1.0)
            local = torch.full((R, local_k), -1, dtype=torch.int32, device="cuda")
            row_lens = torch.zeros(R, dtype=torch.int32, device="cuda")
            for i, s in enumerate(rows):
                pos = _owned_positions(s.numel(), r, world)
                logits[i, : pos.numel()] = s[pos]
                row_lens[i] = pos.numel()
                sel = _top(s[pos], local_k)
                local[i, : sel.numel()] = sel.int()
            merges[r].pack(local, row_lens)

    push()
    plain = []
    for r in range(world):
        out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((R,), -3, dtype=torch.int32, device="cuda")
        merges[r].merge_owned(out, counts)
        plain.append((out, counts))
    push()
    bounds_dtypes = (torch.int32, torch.int64)
    for r in range(world):
        bounds = torch.randint(0, 3, (R,), device="cuda", generator=gen, dtype=bounds_dtypes[r % 2])
        out = torch.full((R, K), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((R,), -3, dtype=torch.int32, device="cuda")
        mask = torch.ones(R, dtype=torch.bool, device="cuda")
        merges[r].merge_owned(out, counts, bounds=bounds, mask_out=mask)
        assert torch.equal(out, plain[r][0]) and torch.equal(counts, plain[r][1]), f"rank {r}"
        assert torch.equal(mask, helix_zero_or_empty_rows(bounds, counts)), f"rank {r}"
        if r > 0:
            assert bool(mask[0]), "a rank that owns no page of the one-page row masks it"


def test_metadata_mask_prefers_the_merge_mask(monkeypatch):
    """``helix_zero_or_empty_rows`` returns the mask the key-sharded merge wrote for these rows, and computes it
    otherwise (another row count, or no merge mask)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as md

    launched = []
    bounds = torch.tensor([1, 0, 2, 3], dtype=torch.int32)
    lens = torch.tensor(
        [0, 0, 5, 1, 4, 4], dtype=torch.int32
    )  # two context rows, then four generation rows
    stand_in = SimpleNamespace(
        helix_kv_bounds=bounds,
        _helix_spec_tokens_valid=True,
        _helix_num_rows=6,
        _helix_rows_set=True,
        num_contexts=2,
        helix_row_kv_lens_cuda=lens,
        _helix_combine_mask=None,
    )
    monkeypatch.setattr(
        md, "helix_zero_or_empty_rows", lambda b, n: launched.append(1) or (b == 0) | (n == 0)
    )
    want = torch.tensor([False, True, False, False])
    assert torch.equal(md.DSAtrtllmAttentionMetadata.helix_zero_or_empty_rows(stand_in, 4), want)
    assert launched == [1]
    merged = torch.tensor([True, True, False, True])
    stand_in._helix_combine_mask = merged
    assert md.DSAtrtllmAttentionMetadata.helix_zero_or_empty_rows(stand_in, 4) is merged
    assert launched == [1]
    md.DSAtrtllmAttentionMetadata.helix_zero_or_empty_rows(stand_in, 3)
    assert launched == [1, 1]


# ---------------------------------------------------------------------------------------------
# The step-level decision
# ---------------------------------------------------------------------------------------------
def _ull_metadata_module():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
        rubin_ull_dsa_metadata,
    )

    return rubin_ull_dsa_metadata


def _fake(num_gen, next_n, rows_cap, bounds=True, ragged=False):
    cls = _ull_metadata_module().RubinUllDSAMetadata
    return SimpleNamespace(
        key_shard_next_n=-1,
        num_generations=num_gen,
        num_tokens=num_gen * next_n + 5,
        num_ctx_tokens=5,
        is_ragged_verify=ragged,
        key_shard_merge=SimpleNamespace(rows_max=rows_cap),
        helix_spec_row_kv_lens=lambda n: torch.ones(n, dtype=torch.int32) if bounds else None,
        _warn_helix_key_shard_skip=cls._warn_helix_key_shard_skip,
    )


@pytest.fixture
def warnings_seen(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.dsa import metadata as md

    seen = []
    monkeypatch.setattr(md.logger, "warning_once", lambda msg, key=None: seen.append((key, msg)))
    return seen


@pytest.mark.parametrize(
    "kwargs, want_next_n, want_kind",
    [
        (dict(num_gen=8, next_n=8, rows_cap=64), 8, None),  # engaged: b8 k7, 64 rows at the cap
        (dict(num_gen=24, next_n=6, rows_cap=144), 6, None),  # engaged: b24 k5, 144 rows at the cap
        (dict(num_gen=1, next_n=8, rows_cap=64), 8, None),  # engaged from one sequence on
        (dict(num_gen=8, next_n=8, rows_cap=64, bounds=False), 0, "bounds"),
        (dict(num_gen=8, next_n=1, rows_cap=64, bounds=False), 1, None),  # next_n 1 needs no bounds
        (dict(num_gen=8, next_n=6, rows_cap=64, ragged=True), 0, "ragged"),
    ],
)
def test_helix_skip_reasons(warnings_seen, kwargs, want_next_n, want_kind):
    fake = _fake(**kwargs)
    _ull_metadata_module().RubinUllDSAMetadata._prepare_key_shard_decode(fake)
    assert fake.key_shard_next_n == want_next_n
    if want_kind is None:
        assert warnings_seen == []
    else:
        assert [k for k, _ in warnings_seen] == [f"dsa_helix_key_shard_skip_{want_kind}"]
        assert "merge_helix_topk" in warnings_seen[0][1]


def test_rows_over_the_workspace_raise(warnings_seen):
    """The workspace holds max_batch_size x (max_draft_len + 1) rows; a step with more is an error, not a
    silent fallback to the unsharded merge."""
    fake = _fake(num_gen=25, next_n=6, rows_cap=144)  # 150 rows > b24 k5's 144
    with pytest.raises(RuntimeError, match="exceed the merge workspace"):
        _ull_metadata_module().RubinUllDSAMetadata._prepare_key_shard_decode(fake)
    assert warnings_seen == []


# ---------------------------------------------------------------------------------------------
# Workspace sizing on the metadata the engine builds
# ---------------------------------------------------------------------------------------------
class _Merge:
    """Stands in for KeyShardTopKMerge (the metadata reads ``rows_max`` and ``local_k``)."""

    def __init__(self, rows_max):
        self.rows_max, self.local_k = rows_max, 512


@pytest.fixture
def sizing(monkeypatch):
    """The real metadata class (the ULL target's, which keeps the key-shard state) without its device setup:
    CPU buffers, an eligible Helix cp8 layout with the self-sampling GVR, no base-class spec-dec update and no
    other draft-sized buffers, and the merge workspace cached on its row count as ``get_key_shard_merge`` does
    (every new workspace recorded)."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    md = _ull_metadata_module()
    cls = md.RubinUllDSAMetadata
    workspaces, cache = [], {}

    def get_merge(mapping, rows_max, top_k, local_width):
        key = (rows_max, top_k, local_width)
        if key not in cache:
            workspaces.append(rows_max)
            cache[key] = _Merge(rows_max)
        return cache[key]

    def cpu_empty(buffers, tensor_shape, dtype, cache_name, capture_graph=False):
        return torch.zeros(tensor_shape, dtype=dtype)

    def skip(self, capture_graph=False):
        return None

    monkeypatch.setattr(md, "get_key_shard_merge", get_merge)
    monkeypatch.setattr(md, "self_sampling_gvr", lambda metadata: True)
    monkeypatch.setattr(cls, "get_empty", staticmethod(cpu_empty))
    monkeypatch.setattr(
        TrtllmAttentionMetadata, "update_spec_dec_param", lambda self, *a, **k: None
    )
    for name in (
        "_create_kv_lens_2d_buffer",
        "create_expanded_buffers",
        "_create_helix_sparse_row_buffers",
        "_create_radix_aux_buffers",
    ):
        monkeypatch.setattr(cls, name, skip)

    def eager(batch, manager, eligible=True):
        """What the engine's ``setup_attention`` leaves: no draft length yet (``build_attention_metadata`` passes
        none) and the key-shard buffers of ``__post_init__``."""
        monkeypatch.setattr(cls, "_key_shard_eligible", lambda self: eligible)
        meta = cls.__new__(cls)
        meta.max_num_sequences = batch
        meta.max_draft_tokens = 0
        meta.kv_cache_manager = manager
        meta.mapping = SimpleNamespace(
            cp_size=8, tp_size=1, repurpose_helix_cp_to_tp=lambda: SimpleNamespace(tp_rank=0)
        )
        meta.enable_helix = True
        meta.is_cuda_graph = False
        meta.cuda_graph_buffers = None
        meta.num_sms = 4
        meta.num_sparse_topk = K
        meta.nvfp4_mla_fp8_scratch = None
        meta._radix_rows_per_sequence = 1
        meta._create_key_shard_buffers(capture_graph=False)
        return meta

    def verify_step(meta, batch, next_n):
        """One eager decode step of ``batch`` sequences verifying ``next_n`` rows each."""
        step = _fake(num_gen=batch, next_n=next_n, rows_cap=0)
        step.key_shard_merge = meta.key_shard_merge
        cls._prepare_key_shard_decode(step)
        return step.key_shard_next_n

    return SimpleNamespace(cls=cls, eager=eager, verify_step=verify_step, workspaces=workspaces)


@pytest.mark.parametrize("batch, draft", [(8, 7), (16, 5), (24, 5)], ids=["b8k7", "b16k5", "b24k5"])
def test_workspace_sized_from_the_server_draft_ceiling(sizing, warnings_seen, batch, draft):
    """The eager metadata is built before the engine passes a draft length; the workspace still holds the server's
    largest verify step, b x (k + 1) rows, from the draft ceiling the KV cache manager was built with. The
    engine's update keeps it, the CUDA graph metadata (a copy that runs ``__post_init__`` again) shares it, one
    workspace in all, and a full eager verify step (Helix's generation-only autotuner warmup) runs."""
    rows = batch * (draft + 1)
    manager = SimpleNamespace(
        max_blocks_per_seq=4, max_draft_len=draft, max_total_draft_tokens=draft
    )
    meta = sizing.eager(batch, manager)
    assert meta.key_shard_merge.rows_max == rows
    assert meta.key_shard_local_topk.shape[0] == rows
    meta.update_spec_dec_param(batch, True, False, False, draft, draft)
    assert meta.key_shard_merge.rows_max == rows
    graph = copy.copy(meta)
    graph.is_cuda_graph = True
    graph._create_key_shard_buffers(capture_graph=True)
    assert graph.key_shard_local_topk.shape[0] == rows
    assert graph.key_shard_merge is meta.key_shard_merge
    assert sizing.workspaces == [rows]
    assert sizing.verify_step(meta, batch, draft + 1) == draft + 1
    assert warnings_seen == []


def test_workspace_grows_with_a_larger_draft_ceiling(sizing, warnings_seen):
    """A KV cache manager without a draft ceiling: the eager metadata starts at b rows, and the engine's
    ``update_spec_dec_param`` grows the workspace to b x (k + 1) before a verify step needs it (b16k5: the 96-row
    step runs); later updates with the same ceiling keep it. Without the key shard the update allocates none."""
    meta = sizing.eager(16, SimpleNamespace(max_blocks_per_seq=4))
    assert meta.key_shard_merge.rows_max == 16
    for _ in range(2):
        meta.update_spec_dec_param(16, True, False, False, 5, 5)
    assert meta.key_shard_merge.rows_max == 96 and meta.key_shard_local_topk.shape[0] == 96
    assert sizing.workspaces == [16, 96]
    assert sizing.verify_step(meta, 16, 6) == 6
    off = sizing.eager(16, SimpleNamespace(max_blocks_per_seq=4), eligible=False)
    off.update_spec_dec_param(16, True, False, False, 5, 5)
    assert off.key_shard_merge is None
    assert sizing.workspaces == [16, 96]
    assert warnings_seen == []
