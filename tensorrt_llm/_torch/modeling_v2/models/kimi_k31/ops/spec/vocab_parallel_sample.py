# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temperature-only sampling of one token per row from this rank's vocab shard of the LM-head
logits by the Gumbel-max trick (Triton local Gumbel argmax with Philox noise, one float32
all-reduce of the (value, index) pairs, Triton pair reduction)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.speculative.vocab_parallel import (
    _MAX_VOCAB,
    _local_sample_kernel,
    _PairExchange,
    _select_tokens,
)

#: Kimi K3.1's vocabulary.
VOCAB = 163840

_MASK32 = 0xFFFFFFFF
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57


def _mul32(a: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(high, low) 32-bit words of ``a * b`` for ``a`` in [0, 2**32) (int64) and a 32-bit
    constant ``b``, in 16-bit limbs so no int64 product overflows."""
    a_lo, a_hi = a & 0xFFFF, a >> 16
    b_lo, b_hi = b & 0xFFFF, b >> 16
    p0, p1, p2, p3 = a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi
    mid = (p0 >> 16) + (p1 & 0xFFFF) + (p2 & 0xFFFF)
    high = p3 + (p1 >> 16) + (p2 >> 16) + (mid >> 16)
    low = ((mid & 0xFFFF) << 16) | (p0 & 0xFFFF)
    return high & _MASK32, low & _MASK32


def philox4x32(
    seed: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Philox-4x32-10 as ``tl.philox`` computes it, on int64 tensors holding uint32 values.

    The 64-bit ``seed`` is the key (low word ``k0``, high word ``k1``); every argument
    broadcasts against the others.
    """
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    for _ in range(10):
        hi_b, lo_b = _mul32(c2, _PHILOX_ROUND_B)
        hi_a, lo_a = _mul32(c0, _PHILOX_ROUND_A)
        c0, c1, c2, c3 = hi_b ^ c1 ^ k0, lo_b, hi_a ^ c3 ^ k1, lo_a
        k0 = (k0 + _PHILOX_KEY_A) & _MASK32
        k1 = (k1 + _PHILOX_KEY_B) & _MASK32
    return c0, c1, c2, c3


def gumbel_scores(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    greedy_threshold: float,
) -> torch.Tensor:
    """``[rows, vocab]`` float32 scores whose row argmax is the sampled token.

    Greedy rows (temperature at or below the threshold) score the logits themselves. The others
    score ``logits * (1 / T) + G`` with the standard Gumbel sample of global vocab index ``v``
    taken from output ``v % 4`` of ``Philox(seed, (v // 4, offset_lo, offset_hi, 0))`` and its
    top 23 bits: ``u = (r >> 9 + 0.5) / 2**23``, ``G = -log(-log(u))``.
    """
    rows, vocab = logits.shape
    device = logits.device
    x = logits.float()
    temperatures = temperatures[:rows].float()
    greedy = temperatures <= greedy_threshold
    inv_t = 1.0 / torch.where(greedy, torch.ones_like(temperatures), temperatures)
    groups = torch.arange((vocab + 3) // 4, dtype=torch.int64, device=device)
    seed = seeds[:rows].to(torch.int64).view(rows, 1)
    offset = offsets[:rows].to(torch.int64).view(rows, 1)
    lanes = philox4x32(
        seed,
        groups.view(1, -1),
        offset & _MASK32,
        (offset >> 32) & _MASK32,
        torch.zeros_like(offset),
    )
    bits = torch.stack(lanes, dim=-1).reshape(rows, -1)[:, :vocab]
    u = ((bits >> 9).to(torch.float32) + 0.5) * (1.0 / 8388608.0)
    noise = -torch.log(-torch.log(u))
    scores = x * inv_t.view(rows, 1) + noise
    return torch.where(greedy.view(rows, 1), x, scores)


def _spec(tp_size: int, rows: int, **overrides) -> dict:
    spec = dict(
        tp_size=tp_size,
        rows=rows,
        vocab=VOCAB,
        dtype=torch.bfloat16,
        temperature=1.0,
        greedy_rows=0,
        data="randn",
    )
    spec.update(overrides)
    return spec


class VocabParallelSample(OpWrapper):
    """One temperature-only draw per row from ``softmax(cat(local_0, ..., local_{tp-1}, -1) / T)``
    (the TP group's vocab shards; rank ``tp_rank`` holds global columns ``[tp_rank * shard,
    (tp_rank + 1) * shard)``), from this rank's ``local [rows, shard]`` alone. Returns ``[rows]``
    int64, the same ids on every rank: the argmax of :func:`gumbel_scores`, i.e. ``logits / T``
    plus Philox Gumbel noise keyed by the row's ``(seed, offset)``; rows whose temperature is at
    or below ``greedy_threshold`` take the plain argmax.

    ``temperatures`` / ``seeds`` / ``offsets`` hold at least one entry per row (float32 / int64 /
    int64). ``all_reduce`` sums a float32 ``[rows, width]`` buffer over the TP group; None with
    ``tp_size == 1`` exchanges nothing.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"attention TP{tp}: T={temp} target tokens of {rows} rows",
                spec=_spec(tp, rows, temperature=temp),
            )
            for tp, rows, temp in ((2, 6, 1.0), (4, 40, 0.6), (8, 72, 1.0), (16, 288, 0.7))
        ),
        Cell(
            why="one shard (no exchange): the Philox noise over the full vocab",
            spec=_spec(1, 40, temperature=1.0),
        ),
        Cell(
            why="a mixed batch: greedy rows (the 1e-5 sentinel) take the argmax",
            spec=_spec(8, 40, temperature=0.8, greedy_rows=13),
        ),
        Cell(
            why="large offsets: both 32-bit words of the Philox counter move",
            spec=_spec(8, 9, temperature=1.0, offset_base=(1 << 62) + 12345),
        ),
        Cell(
            why="a high temperature: the noise dominates, every rank's shard wins somewhere",
            spec=_spec(8, 256, temperature=4.0),
        ),
    )
    note = """
    The Kimi K3.1 target reaches it through its DFlash worker under attention TP with the
    worker's VocabParallelSelection (vocab-sharded gathered LM head, no penalties, no guided
    decoder) when `samples` holds: advanced_sampling_mode NO_TOPK_NO_TOPP and rejection
    sampling off, so the verified tokens are drawn temperature-only and checked by exact match.
    The call is `_sample_tokens_for_batch` on `VocabShardedLogits` of a batch that is not all
    greedy, with the per-token `temperatures` / `seeds` / `offsets` of the spec metadata.

    Exact up to the 23-bit uniform behind G (G lies in about [-2.8, 16.6], so a token more than
    ~19.4 below its row's maximum of logits / T is never drawn); the distribution matches the
    gathered sampler's but the individual draws do not (another random stream). Every rank
    computes the noise of its own columns from the global index, so the winner agrees across
    ranks. That needs the shard width a multiple of 4 (a Philox call yields 4 lanes for 4
    consecutive global ids) and the vocab below 2**24 (indices travel as float32).
    CUDA-graph capturable; `all_reduce` must be an exact float32 sum off the one-shot Lamport
    workspace, as for `vocab_parallel_argmax`.

    The reference reproduces the Philox stream in int64 torch arithmetic; the kernel's log is
    the fast approximate one, so the scores differ in the last bits and the test admits a
    different winner only where the reference's top two scores are that close.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]],
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        return _select_tokens(
            _local_sample_kernel,
            local,
            (temperatures, seeds, offsets, float(greedy_threshold)),
            tp_rank * local.shape[-1],
            _PairExchange(all_reduce, tp_size, tp_rank),
            torch.int64,
        )

    def reference(
        self,
        per_rank_local: Sequence[torch.Tensor],
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """The argmax of :func:`gumbel_scores` over the gathered logits.

        The documented collective exception to mirroring: the result depends on every rank's
        shard, so this takes the TP group's ``local`` tensors in rank order. ``all_reduce``,
        ``tp_size`` and ``tp_rank`` are accepted and ignored (every rank gets the same ids).
        """
        gathered = torch.cat(list(per_rank_local), dim=-1)
        scores = gumbel_scores(gathered, temperatures, seeds, offsets, greedy_threshold)
        return scores.argmax(dim=-1)

    def is_valid(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]],
        tp_size: int,
        tp_rank: int,
    ) -> None:
        rows, shard = local.shape
        # A shard that is not a multiple of 4 shifts the 4-lane Philox groups against the
        # global ids: two ranks then draw correlated noise for different tokens and the draw
        # is silently biased.
        assert tp_size == 1 or shard % 4 == 0, (
            f"shard width {shard} must be a multiple of 4 under TP; the Philox noise is drawn "
            "in 4-column groups of the global vocab"
        )
        assert shard * tp_size < _MAX_VOCAB, (
            f"vocab {shard * tp_size} must stay below 2**24; the pair exchange carries indices "
            "as float32"
        )
        # The kernel reads entry `row` of each per-row tensor with no bound: a short tensor
        # is read past its end.
        for name, t in (("temperatures", temperatures), ("seeds", seeds), ("offsets", offsets)):
            assert t.numel() >= rows and t.stride(-1) == 1, (
                f"{name} needs one entry per row ({rows}), unit stride; got shape {tuple(t.shape)}"
            )
        assert tp_size == 1 or all_reduce is not None, (
            "tp_size > 1 needs the group's all_reduce; without it each rank draws from its own "
            "shard only"
        )


vocab_parallel_sample = VocabParallelSample()
