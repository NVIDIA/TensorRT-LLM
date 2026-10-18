# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and built-in references shared by the draft-head tests (fast path ``draft_head``).

Not a test module: the entry tests (``test_modeling_v2_draft_shard_topk.py``,
``test_modeling_v2_draft_topk_merge.py``, ``test_modeling_v2_draft_path_walk.py``)
and the draft-head test (``test_modeling_v2_draft_head.py``) import it. With
``MUSESPARK_REFERENCE_DRAFTER`` set, the selector is the published drafter's.
"""

import os
import zlib

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlash2CandidateSelector, DFlashForCausalLM

draft_shard_topk = draft_head.draft_shard_topk
draft_topk_merge = draft_head.draft_topk_merge
draft_path_walk = draft_head.draft_path_walk

VOCAB, HIDDEN, RANK, TOP_K, TP = 202048, 8192, 256, 16, 8
_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode())


def logits(kind: str, rows: int, vocab: int, seed: int) -> torch.Tensor:
    """bf16 draft logits ``[rows, vocab]`` of one value family."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def pick(n):
        return torch.randint(0, vocab, (rows, n), generator=g, device="cuda")

    if kind == "logits":
        x = torch.randn(rows, vocab, generator=g, device="cuda") * 3.0
    elif kind == "grid":
        x = torch.randint(0, 40, (rows, vocab), generator=g, device="cuda").float() * 0.125
    elif kind == "zeros":
        x = torch.zeros(rows, vocab, device="cuda")
    elif kind in ("special", "special_nan"):
        x = -(torch.rand(rows, vocab, generator=g, device="cuda") + 1.0)
        for value, n in ((0.0, 12), (-0.0, 12), (float("inf"), 3), (float("-inf"), 2000)):
            idx = pick(n)
            x.scatter_(-1, idx, torch.full(idx.shape, value, device="cuda"))
        if kind == "special_nan":
            idx = pick(4)
            x.scatter_(-1, idx, torch.full(idx.shape, float("nan"), device="cuda"))
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def shard_keys(full: torch.Tensor, ranks: int) -> torch.Tensor:
    """Every rank's shard keys, concatenated rank-major: what the all-gather returns."""
    shard = full.shape[1] // ranks
    return torch.cat(
        [draft_shard_topk(full[:, r * shard : (r + 1) * shard]) for r in range(ranks)], dim=0
    )


def packed_keys(full: torch.Tensor, ranks: int) -> torch.Tensor:
    """Every rank's fp32 key halves summed: what the all-reduce returns."""
    shard = full.shape[1] // ranks
    packed = [
        draft_shard_topk(full[:, r * shard : (r + 1) * shard], rank=r, num_ranks=ranks)
        for r in range(ranks)
    ]
    return torch.stack(packed).sum(dim=0)


def assert_same_floats(a: torch.Tensor, b: torch.Tensor) -> None:
    """Bitwise equality, NaN payloads aside (both NaN counts as equal)."""
    assert a.shape == b.shape and a.dtype == b.dtype
    both_nan = a.isnan() & b.isnan()
    assert torch.equal(a.isnan(), b.isnan())
    assert torch.equal(
        torch.where(both_nan, 0.0, a).view(torch.int32),
        torch.where(both_nan, 0.0, b).view(torch.int32),
    )


_SELECTOR = None


def selector() -> DFlash2CandidateSelector:
    """The published drafter's selector when available, else a random one of its shape."""
    global _SELECTOR
    if _SELECTOR is None:
        if _DRAFTER_DIR:
            from safetensors import safe_open

            with safe_open(os.path.join(_DRAFTER_DIR, "model.safetensors"), "pt", "cuda") as f:
                names = set(f.keys())

                def take(key):
                    (name,) = [n for n in names if n.endswith(key)]
                    return f.get_tensor(name)

                pred = take("candidate_selector.predecessor_codebook")
                succ = take("candidate_selector.successor_codebook")
                proj = take("candidate_selector.hidden_projection.weight")
        else:
            g = torch.Generator(device="cuda").manual_seed(7)
            pred = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16()
            succ = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16()
            proj = (0.01 * torch.randn(RANK, HIDDEN, generator=g, device="cuda")).bfloat16()
        _SELECTOR = DFlash2CandidateSelector(pred, succ, proj, TOP_K, VOCAB, RANK).cuda()
    return _SELECTOR


def draft_step(requests, positions, kind, seed, ranks=TP):
    """Per-rank shard logits of one draft step, plus its selector inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests * positions
    hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    shard = VOCAB // ranks
    if kind == "gemm":
        weight = (
            torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * (3.0 / HIDDEN**0.5)
        ).bfloat16()
        shards = [F.linear(hidden, weight[r * shard : (r + 1) * shard]) for r in range(ranks)]
        del weight
    else:
        full = logits(kind, rows, VOCAB, seed)
        shards = [full[:, r * shard : (r + 1) * shard].contiguous() for r in range(ranks)]
    anchor = torch.randint(0, VOCAB, (requests,), generator=g, device="cuda")
    return shards, hidden.view(requests, positions, HIDDEN), anchor


class DraftStub:
    """What the worker's selector path reads off the drafter."""

    select_candidate_path = DFlashForCausalLM.select_candidate_path

    def __init__(self, candidate_selector):
        self.candidate_selector = candidate_selector


def builtin_draft_tokens(shards, block_hidden, anchor):
    """The built-in path on the gathered logits: topk, selector, walk, scatter, argmax."""
    requests, positions, _ = block_hidden.shape
    full = torch.cat(shards, dim=-1).float().view(requests, positions, VOCAB)
    unary, ids = torch.topk(full, TOP_K, dim=-1)
    block = DFlashForCausalLM.select_candidate_path(
        DraftStub(selector()), full, ids, unary, block_hidden, anchor.long()
    )
    return torch.argmax(block.reshape(-1, VOCAB), dim=-1).to(torch.int32).view(requests, positions)
