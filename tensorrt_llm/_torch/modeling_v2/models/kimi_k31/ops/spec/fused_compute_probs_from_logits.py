# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target probabilities for speculative verification: temperature, min-p, top-k and top-p applied
together with a per-row on-device skip, in one fused C++ sampling op."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Kimi K3.1's vocabulary.
VOCAB = 163840


def _renorm(probs: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    kept = torch.where(keep, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True)


def filtered_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: Optional[torch.Tensor],
    top_ps: Optional[torch.Tensor],
    min_ps: Optional[torch.Tensor],
) -> torch.Tensor:
    """``[rows, vocab]`` float32: ``softmax(logits / T)`` filtered by min-p, top-k, top-p.

    The engine samplers' filter order (``docs/source/features/sampling.md``): temperature and
    softmax, then min-p, top-k and top-p, each renormalizing what it keeps. A disabled filter takes
    its sentinel: ``min_p <= 0``, ``top_k`` outside ``(0, vocab)``, ``top_p >= 1``.

    Rows apply a filter only where their value enables it; None skips the filter for every row.
    Top-k keeps the ``k`` most probable tokens and top-p the smallest prefix of the descending
    order whose mass reaches ``p`` (the token that crosses ``p`` included); ties at either cut are
    ordered by token id, which the kernels need not share -- a test compares by mass, not by id.
    """
    rows, vocab = logits.shape
    temperatures = temperatures[:rows].float().view(rows, 1)
    probs = torch.softmax(logits.float() / temperatures, dim=-1)

    if min_ps is not None:
        min_p = min_ps[:rows].float().view(rows, 1)
        keep = (min_p <= 0) | (probs >= min_p * probs.amax(dim=-1, keepdim=True))
        probs = _renorm(probs, keep)

    if top_ks is not None or top_ps is not None:
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
        rank = torch.arange(vocab, device=logits.device).view(1, vocab)
        keep_sorted = torch.ones_like(sorted_probs, dtype=torch.bool)
        if top_ks is not None:
            k = top_ks[:rows].to(torch.int64).view(rows, 1)
            enabled = (k > 0) & (k < vocab)
            keep_sorted = keep_sorted & (~enabled | (rank < k))
            sorted_probs = _renorm(sorted_probs, keep_sorted)
        if top_ps is not None:
            p = top_ps[:rows].float().view(rows, 1)
            exclusive = sorted_probs.cumsum(dim=-1) - sorted_probs
            keep_sorted = keep_sorted & ((p >= 1) | (exclusive < p))
            sorted_probs = _renorm(sorted_probs, keep_sorted)
        probs = torch.zeros_like(probs).scatter_(1, order, sorted_probs)
    return probs


def sampling_spec(rows: int, **overrides) -> dict:
    """A sampler cell: ``rows`` rows over K3.1's vocabulary, float32 gathered target logits."""
    spec = dict(
        rows=rows,
        vocab=VOCAB,
        dtype=torch.float32,
        temperature=1.0,
        top_k=None,
        top_p=None,
        min_p=None,
        greedy_rows=0,
        mixed=False,
    )
    spec.update(overrides)
    return spec


#: The K3.1 sampling cells the fused samplers are certified over (shared by the probs and the
#: token entry): DFlash2 w5 / w8 verify rows (generations x (K + 1)) and context rows, at the
#: model card's T = 1.0 / top-p 0.95 and the filters a request may add.
SAMPLING_CELLS: tuple[Cell, ...] = (
    Cell(
        why="w5 verify of 8 requests (40 rows) at the card's T=1.0, top-p 0.95",
        spec=sampling_spec(40, top_p=0.95),
    ),
    Cell(
        why="w8 verify of 64 requests (512 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(512, top_p=0.95),
    ),
    Cell(
        why="every filter neutral: plain temperature softmax (T=0.7)",
        spec=sampling_spec(40, temperature=0.7),
    ),
    Cell(
        why="top-k 20 with top-p 0.95 at T=0.6",
        spec=sampling_spec(40, temperature=0.6, top_k=20, top_p=0.95),
    ),
    Cell(
        why="min-p 0.05 alone (the fused op is the only sampler that takes it)",
        spec=sampling_spec(16, min_p=0.05),
    ),
    Cell(
        why="a mixed batch: per-row filters and greedy-sentinel rows (T=1e-5)",
        spec=sampling_spec(72, top_k=50, top_p=0.9, min_p=0.02, greedy_rows=9, mixed=True),
    ),
    Cell(
        why="context rows of a mixed step: one row per request",
        spec=sampling_spec(3, top_p=0.95),
    ),
)


class FusedComputeProbsFromLogits(OpWrapper):
    """``[rows, vocab]`` float32: each row's ``softmax(logits / T)`` filtered by min-p, top-k and
    top-p (each renormalizing; see :func:`filtered_probs`), newly allocated.

    ``logits [rows, vocab]`` (float32, bf16 or fp16); ``temperatures`` float32, ``top_ks`` int32,
    ``top_ps`` / ``min_ps`` float32, one entry per row. A row whose value is a disable sentinel
    (``top_k`` outside ``(0, vocab)``, e.g. INT32_MAX; ``top_p >= 1``; ``min_p <= 0``) skips that
    filter on the device, so one call serves a batch mixing filtered and unfiltered requests.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = SAMPLING_CELLS
    note = """
    The Kimi K3.1 target calls it for the target distributions of DFlash2 block verification
    (spec_dflash2_verify `dflash2_sample_and_accept`, through the worker's
    `_compute_probs_from_logits`) when advanced_sampling_mode is FULL, the default: the gathered
    float32 logits of the generation rows (`num_gens x (K + 1)`, padding rows zeroed) with the
    per-token sampling tensors of the spec metadata. Other modes take the flashinfer chain
    (`flashinfer_compute_probs_from_logits`). The tep target takes `fused_compute_probs_split`
    instead, the same sampler in <= 32-row chunks.

    Greedy requests of a sampling batch carry the sentinel temperature 1e-5 and come out one-hot
    (their verification uses the argmax path anyway). Top-p's cut is not run-to-run reproducible
    for a token sitting exactly on the boundary, and ties at either cut may be ordered
    differently from the reference, so the test bounds each row's L1 distance rather than
    comparing elements. No RNG, no workspace; CUDA-graph capturable. The op itself rejects
    per-row tensors shorter than the logits rows, so there is no input guard.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fused_compute_probs_from_logits(
            logits, temperatures, top_ks, top_ps, min_ps
        )

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> torch.Tensor:
        """:func:`filtered_probs` in fp32."""
        return filtered_probs(logits, temperatures, top_ks, top_ps, min_ps)


fused_compute_probs_from_logits = FusedComputeProbsFromLogits()
