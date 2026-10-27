# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and built-in references shared by the draft-head tests (fast path ``draft_head``).

Not a test module: the entry tests under ``ops/spec`` and the draft-head tests
here import it. Every reference transform is the drafter's own
``DFlashForCausalLM.apply_draft_logit_transform``, never a formula written
here: on the device that method divides by the softcap as a multiplication by
its fp32 reciprocal, which a hand-written ``x / softcap`` does not reproduce.
"""

from __future__ import annotations

import types
import zlib

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    logit_transform,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlash2CandidateSelector, DFlashForCausalLM

draft_shard_topk = draft_head.draft_shard_topk
draft_topk_merge = draft_head.draft_topk_merge
draft_path_walk = draft_head.draft_path_walk

VOCAB, HIDDEN, RANK, TOP_K, TP = 202048, 8192, 256, 16, 8

#: The Muse Spark Open DFlash 2 drafter's declared transform: (output_multiplier, softcap).
MUSESPARK = (0.1767766952966369, 20.0)

#: Transform configurations the table tests cover besides MUSESPARK.
OTHER_TRANSFORMS = {
    "multiplier only": (0.5, None),
    "softcap only": (1.0, 20.0),
    "softcap 30": (1.0, 30.0),
}

#: Logit families of the end-to-end and merge matrices.
FAMILIES = (
    "gemm1",
    "gemm10",
    "gemm100",
    "gemm400",
    "grid",
    "special_nan",
    "straddle_up",
    "straddle_down",
    "ladder",
    "saturated",
    "edges",
)


def seed_of(text: str) -> int:
    return zlib.crc32(text.encode())


# --------------------------------------------------------------------------- drafters


_DRAFTERS: dict = {}
_TRANSFORMS: dict = {}


def drafter(multiplier: float = 1.0, softcap: float | None = None) -> DFlashForCausalLM:
    """A ``DFlashForCausalLM`` holding only the logit transform attributes (kept alive here)."""
    key = (multiplier, softcap)
    if key not in _DRAFTERS:
        model = DFlashForCausalLM.__new__(DFlashForCausalLM)
        torch.nn.Module.__init__(model)
        model.logit_multiplier, model.logit_softcap = multiplier, softcap
        _DRAFTERS[key] = model
    return _DRAFTERS[key]


def transform(config=MUSESPARK) -> logit_transform.DraftLogitTransform:
    """The draft head's tables of ``config``'s transform on the current device (built once)."""
    if config not in _TRANSFORMS:
        built = logit_transform.build(drafter(*config), torch.device("cuda"))
        assert built is not None
        _TRANSFORMS[config] = built
    return _TRANSFORMS[config]


def builtin_transform(config, logits: torch.Tensor) -> torch.Tensor:
    """The built-in transform: the drafter's own method, the call the DFlash worker makes."""
    model = drafter(*config)
    return DFlashForCausalLM.apply_draft_logit_transform(model, logits)


# --------------------------------------------------------------------------- logits


def _keys_values(keys) -> torch.Tensor:
    return logit_transform.bf16_of_key16(torch.as_tensor(keys, device="cuda"))


def _groups(config):
    """(upper, lower): the largest tie group above +0.0 and below -0.0 of ``config``'s transform."""
    groups = logit_transform.tie_groups(transform(config))
    upper = max((g for g in groups if g[0] >= 0x8000), key=lambda g: g[1] - g[0])
    lower = max((g for g in groups if g[1] <= 0x7FFF), key=lambda g: g[1] - g[0])
    return upper, lower


def logits(kind: str, rows: int, vocab: int, seed: int, shard: int | None = None, config=MUSESPARK):
    """bf16 draft logits ``[rows, vocab]`` of one value family.

    The tie families take their values from ``config``'s tie groups:
    ``straddle_up`` ten copies each of two distinct members of the upper
    saturating group at low and at high ids (both orders across rows),
    ``straddle_down`` the same for the lower group above a row of lower-group
    members, ``ladder`` consecutive bf16 values across the upper edge of the
    identity band, ``saturated`` whole rows of upper-group members with +inf,
    ``edges`` upper-group members on every shard's first and last columns.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)

    def pick(n):
        return torch.randint(0, vocab, (rows, n), generator=g, device="cuda")

    def grid(levels=8):
        return (
            torch.randint(0, levels, (rows, vocab), generator=g, device="cuda").float() * 0.25 - 3.0
        )

    def members(group, shape, with_inf=False):
        first = group[0] if with_inf else max(group[0], 0x0080)
        last = group[1] if with_inf else min(group[1], 0xFF7F)
        keys = torch.randint(first, last + 1, shape, generator=g, device="cuda")
        return _keys_values(keys)

    if kind.startswith("gemm"):
        sigma = float(kind[4:])
        x = torch.randn(rows, vocab, generator=g, device="cuda") * sigma
    elif kind == "logits":
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
    elif kind in ("straddle_up", "straddle_down", "ladder", "saturated", "edges"):
        upper, lower = _groups(config)
        x = grid().to(torch.bfloat16)
        quarter = vocab // 4
        low = torch.randperm(quarter, generator=g, device="cuda")[:10]
        high = vocab - 1 - torch.randperm(quarter, generator=g, device="cuda")[:10]
        if kind in ("straddle_up", "straddle_down"):
            group = upper if kind == "straddle_up" else lower
            if kind == "straddle_down":
                x = members(lower, (rows, vocab), with_inf=True)
            a = _keys_values([max(group[0], 0x0080)])
            b = _keys_values([min(group[1], 0xFF7F)])
            for r in range(rows):
                first, second = (a, b) if r % 2 == 0 else (b, a)
                x[r, low] = first
                x[r, high] = second
        elif kind == "ladder":
            hi = transform(config).hi
            ladder = _keys_values(torch.arange(hi - 23, min(hi + 9, 0xFF80)))
            for r in range(rows):
                x[r, torch.randperm(vocab, generator=g, device="cuda")[: ladder.numel()]] = ladder
        elif kind == "saturated":
            x = members(upper, (rows, vocab), with_inf=True)
        else:
            assert shard is not None, "the edges family places its ties on shard edges"
            cols = torch.tensor(
                [c for r in range(vocab // shard) for c in (r * shard, (r + 1) * shard - 1)],
                device="cuda",
            )
            x[:, cols] = members(upper, (rows, cols.numel()))
        return x
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


# --------------------------------------------------------------------------- key exchange


def split(full: torch.Tensor, ranks: int) -> list:
    shard = full.shape[1] // ranks
    return [full[:, r * shard : (r + 1) * shard].contiguous() for r in range(ranks)]


def shard_keys(full: torch.Tensor, ranks: int, t=None) -> torch.Tensor:
    """Every rank's shard keys, concatenated rank-major: what the all-gather returns."""
    return torch.cat(
        [draft_shard_topk(s, **draft_head.canon_kwargs(t)) for s in split(full, ranks)], dim=0
    )


def packed_keys(full: torch.Tensor, ranks: int, t=None) -> torch.Tensor:
    """Every rank's fp32 key halves summed: what the all-reduce returns."""
    packed = [
        draft_shard_topk(s, rank=r, num_ranks=ranks, **draft_head.canon_kwargs(t))
        for r, s in enumerate(split(full, ranks))
    ]
    return torch.stack(packed).sum(dim=0)


def rank_keys(full: torch.Tensor, ranks: int, t=None) -> torch.Tensor:
    """Every rank's own top-k halves (``wide_draft_topk``) summed: what its all-reduce returns."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_draft_topk import (  # noqa: E501
        exchange,
    )

    ex = exchange.RankTopKExchange(None)
    return torch.stack(
        [
            ex.keys(s, TOP_K, r, ranks, **draft_head.canon_kwargs(t))
            for r, s in enumerate(split(full, ranks))
        ]
    ).sum(dim=0)


def layout_keys(layout: str, full: torch.Tensor, ranks: int, t=None) -> torch.Tensor:
    return {"int32": shard_keys, "halves": packed_keys, "rank": rank_keys}[layout](full, ranks, t)


def merge_launch(layout: str, rows: int) -> dict:
    """The merge geometry a step of ``rows`` rows runs with in ``layout``."""
    if layout != "rank":
        return {}
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_draft_topk import (  # noqa: E501
        exchange,
    )

    return exchange.RankTopKExchange(None).merge_launch(rows)


def assert_same_floats(a: torch.Tensor, b: torch.Tensor) -> None:
    """Bitwise equality, NaN payloads aside (both NaN counts as equal)."""
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert logit_transform.same_bits(a, b), "values differ in their bits"


# --------------------------------------------------------------------------- selector


_SELECTOR = None


def selector() -> DFlash2CandidateSelector:
    """A random selector of the published drafter's shape."""
    global _SELECTOR
    if _SELECTOR is None:
        g = torch.Generator(device="cuda").manual_seed(7)
        pred = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16()
        succ = (0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")).bfloat16()
        proj = (0.01 * torch.randn(RANK, HIDDEN, generator=g, device="cuda")).bfloat16()
        _SELECTOR = DFlash2CandidateSelector(pred, succ, proj, TOP_K, VOCAB, RANK).cuda()
    return _SELECTOR


def draft_step(requests, positions, kind, seed, ranks=TP):
    """Per-rank bf16 shard logits of one draft step, plus its selector inputs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests * positions
    hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
    shard = VOCAB // ranks
    if kind.startswith("gemm") and requests * positions <= 56:
        # The head's GEMM on a random weight of the given logit scale.
        sigma = float(kind[4:])
        shards = []
        for r in range(ranks):
            weight = (
                torch.randn(shard, HIDDEN, generator=g, device="cuda") * (sigma / HIDDEN**0.5)
            ).bfloat16()
            shards.append(F.linear(hidden, weight))
            del weight
    else:
        full = logits(kind, rows, VOCAB, seed, shard=shard)
        shards = split(full, ranks)
    anchor = torch.randint(0, VOCAB, (requests,), generator=g, device="cuda")
    return shards, hidden.view(requests, positions, HIDDEN), anchor


class DraftStub:
    """What the worker's selector path reads off the drafter."""

    select_candidate_path = DFlashForCausalLM.select_candidate_path

    def __init__(self, candidate_selector):
        self.candidate_selector = candidate_selector


def builtin_full_logits(shards, requests, positions, config=None) -> torch.Tensor:
    """The built-in ``[B, K, vocab]`` fp32 logits: each shard transformed, then all-gathered."""
    if config is None:
        parts = [s.float() for s in shards]
    else:
        parts = [builtin_transform(config, s) for s in shards]
    return torch.cat(parts, dim=-1).float().view(requests, positions, VOCAB)


def builtin_candidates(shards, requests, positions, config=None):
    """``(unary, ids)``: ``torch.topk`` as the built-in selector runs it."""
    return torch.topk(builtin_full_logits(shards, requests, positions, config), TOP_K, dim=-1)


def builtin_draft_tokens(shards, block_hidden, anchor, config=None):
    """The built-in path on the gathered logits: topk, selector, walk, scatter, argmax."""
    requests, positions, _ = block_hidden.shape
    full = builtin_full_logits(shards, requests, positions, config)
    unary, ids = torch.topk(full, TOP_K, dim=-1)
    block = DFlashForCausalLM.select_candidate_path(
        DraftStub(selector()), full, ids, unary, block_hidden, anchor.long()
    )
    return torch.argmax(block.reshape(-1, VOCAB), dim=-1).to(torch.int32).view(requests, positions)


def mapping(rank: int, ranks: int = TP):
    return types.SimpleNamespace(tp_size=ranks, tp_rank=rank)
