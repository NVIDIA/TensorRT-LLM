# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter takes the selector's candidates from its vocab shard.

The drafter head's logits stay this rank's ``[rows, 20480]`` shard. ``_dflash2_global_top_k``'s
sharded branch takes the shard's top-32, puts the 32 ``(id, value)`` pairs in id order
(``_pairs_in_id_order``), all-gathers them and merges, where the drafter used to all-gather the
full ``[rows, 163840]`` logits, cast them to FP32 and take their top-32. The candidates must be
the full-vocab top-32's ids, values and order bit for bit; this test holds it to that.

One sm_103 / sm_107 GPU, the family's kernels with 8 shards on one GPU (the all-gather a
concatenation in rank order): on rows with planted ties (inside a shard, across shard edges, at
the 32nd place, equal values with different ids) and NaN-free extremes (a ``-inf`` mask, ``+inf``
ties, the largest finite values, subnormals, ``+-0``), at a helix8 decode shape (24 requests,
K = 5) the sharded branch equals the full-vocab branch on the gathered FP32 logits bit for bit,
and so do the selector's greedy block logits and the argmax draft tokens.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import torch

__extra_import_path__ = [".."]
from _k31_utils import on_sm

_PKG = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"
_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"

TOP_K = 32
VOCAB = 163840
TP = 8
SHARD = VOCAB // TP
EDGES = [r * SHARD for r in range(1, TP)]
_BF16_MAX = float(torch.finfo(torch.bfloat16).max)
_BF16_TINY = float(torch.finfo(torch.bfloat16).tiny)


# --- rows with planted ties -------------------------------------------------------------------------

_KINDS = (
    "randn",
    "levels",
    "all_equal",
    "inside_shard",
    "across_edges",
    "kth_split",
    "dup_above",
    "masked_ninf",
    "pos_inf",
    "max_finite",
    "subnormal",
    "hot",
)


def _row(kind: str, rng: np.random.Generator) -> np.ndarray:
    """One FP32 row of BF16-representable logits with the kind's ties planted."""
    x = rng.standard_normal(VOCAB).astype(np.float32) * 2.0
    if kind == "levels":
        x = 0.25 * rng.integers(-3, 6, VOCAB).astype(np.float32)
    elif kind == "all_equal":
        x[:] = 10.0
    elif kind == "inside_shard":
        rank, n_hi = int(rng.integers(0, TP)), int(rng.integers(0, 28))
        x[rng.choice(VOCAB, n_hi, replace=False)] = 20.0 + 0.5 * rng.integers(0, 3, n_hi)
        x[rank * SHARD + rng.choice(SHARD, 45, replace=False)] = 12.0
    elif kind == "across_edges":
        n_hi = int(rng.integers(0, 31))
        x[rng.choice(VOCAB, n_hi, replace=False)] = 20.0 + rng.integers(0, 4, n_hi)
        x[[e + d for e in EDGES for d in (-3, -2, -1, 0, 1, 2)]] = 9.0
        x[rng.choice(VOCAB, 20, replace=False)] = 9.0
    elif kind == "kth_split":
        e = EDGES[int(rng.integers(0, len(EDGES)))]
        x[rng.choice(VOCAB, 29, replace=False)] = 30.0
        x[[e - 2, e - 1, e, e + 1, e + 2]] = 15.0
    elif kind == "dup_above":
        for v in (25.0, 24.0, 23.0):
            x[rng.choice(VOCAB, 7, replace=False)] = v
        x[EDGES] = 24.0
        x[[e - 1 for e in EDGES]] = 23.0
    elif kind == "masked_ninf":
        x[:] = -np.inf
        n_fin = int(rng.integers(0, 31))
        x[rng.choice(VOCAB, n_fin, replace=False)] = rng.standard_normal(n_fin)
    elif kind == "pos_inf":
        x[rng.choice(VOCAB, 40, replace=False)] = np.inf
        x[EDGES] = np.inf
    elif kind == "max_finite":
        x[rng.choice(VOCAB, 20, replace=False)] = _BF16_MAX
        x[rng.choice(VOCAB, 20, replace=False)] = _BF16_MAX * 0.99
        x[rng.choice(VOCAB, 50, replace=False)] = -_BF16_MAX
        x[EDGES] = _BF16_MAX
    elif kind == "subnormal":
        x = -np.abs(x) - 1.0
        x[rng.choice(VOCAB, 10, replace=False)] = _BF16_TINY / 4
        x[rng.choice(VOCAB, 10, replace=False)] = -_BF16_TINY / 4
        x[rng.choice(VOCAB, 12, replace=False)] = 0.0
        x[rng.choice(VOCAB, 12, replace=False)] = -0.0
        x[EDGES] = -0.0
        x[[e - 1 for e in EDGES]] = 0.0
    elif kind == "hot":
        x[rng.choice(VOCAB, 60, replace=False)] = 30.0 + 0.25 * rng.integers(0, 6, 60)
    else:
        assert kind == "randn", kind
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def _logits(n: int, slots: int, seed: int) -> torch.Tensor:
    """``[n, slots, VOCAB]`` BF16 logits, the kinds dealt over the rows."""
    rng = np.random.default_rng(seed)
    rows = [_row(_KINDS[i % len(_KINDS)], rng) for i in range(n * slots)]
    return torch.from_numpy(np.stack(rows)).to(torch.bfloat16).view(n, slots, VOCAB)


_INT_VIEW = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit patterns, so NaN payloads and signed zeros compare too."""
    return t.view(_INT_VIEW[t.dtype]) if t.dtype in _INT_VIEW else t


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


# --- one sm_103 / sm_107 GPU ---------------------------------------------------------------------------

_ON_GPU = on_sm(103, 107, reason="the family's top-32 kernels are certified on sm_103 and sm_107")


def _gpu_group(mod, monkeypatch, shards):
    recorded = []

    def record(tensor, mapping, dim, sizes):
        recorded.append(tensor.clone())
        return torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * TP), device=tensor.device)

    def gather(tensor, mapping, dim, sizes):
        return torch.cat(recorded, dim=-1)

    def worker(rank):
        return SimpleNamespace(
            _vocab_shards=TP,
            _vocab_shard=rank,
            mapping=None,
            _draft_logits_are_sharded=lambda logits, md: logits.shape[-1] < VOCAB,
        )

    out = []
    for fn in (record, gather):
        monkeypatch.setattr(mod, "allgather_module", fn)
        out = [
            mod.DFlashWorker._dflash2_global_top_k(worker(r), shards[r], None, TOP_K, VOCAB)
            for r in range(TP)
        ]
    return out


@_ON_GPU
def test_gpu_sharded_branch_is_the_full_vocab_branch(monkeypatch):
    mod = importlib.import_module(f"{_PKG}.{_TARGET}.modeling")
    n, slots = 24, 5
    logits = _logits(n, slots, seed=n * 100 + slots).cuda()
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    full = torch.cat(shards, dim=-1).float()  # the gathered head logits, cast by the processor
    today = SimpleNamespace(_vocab_shards=TP, _vocab_shard=0, mapping=None)
    want_ids, want_unary, want_block = mod.DFlashWorker._dflash2_global_top_k(
        today, full, None, TOP_K, VOCAB
    )
    assert want_block is full
    out = _gpu_group(mod, monkeypatch, shards)
    for rank, (ids, unary, block) in enumerate(out):
        assert _equal(ids, want_ids), f"rank {rank}: ids or their order differ"
        assert _equal(unary, want_unary), f"rank {rank}: values differ"
        assert block.dtype == torch.float32 and block.shape == full.shape

    # The selector's greedy walk scores both blocks the same way.
    g = torch.Generator(device="cuda").manual_seed(5)
    rank_dim, hidden = 256, 512
    pred = (torch.randn(VOCAB, rank_dim, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    succ = (torch.randn(VOCAB, rank_dim, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    proj = (torch.randn(rank_dim, hidden, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    block_hidden = torch.randn(n, slots, hidden, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda")
    walk = mod.k31_dflash2_selector_greedy_path
    want = walk(pred, succ, proj, want_ids, want_unary, block_hidden, anchor, want_block)
    ids, unary, block = out[TP - 1]
    block.fill_(float("nan"))  # whatever the empty block held, the walk overwrites it all
    got = walk(pred, succ, proj, ids, unary, block_hidden, anchor, block)
    assert _equal(got, want), "the selector's block logits differ"
    assert torch.equal(got.view(-1, VOCAB).argmax(-1), want.view(-1, VOCAB).argmax(-1))
