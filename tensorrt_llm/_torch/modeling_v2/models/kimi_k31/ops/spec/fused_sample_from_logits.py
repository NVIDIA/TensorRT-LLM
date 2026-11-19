# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One sampled token per row under temperature, min-p, top-k and top-p, in one fused C++ sampling
op with Philox (seed, offset) randomness."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    SAMPLING_CELLS,
    filtered_probs,
)


class FusedSampleFromLogits(OpWrapper):
    """``[rows]`` int32: one token per row drawn from the row's filtered distribution (what
    :class:`FusedComputeProbsFromLogits` returns for the same arguments).

    ``seed`` / ``offset`` are int64 Philox state, one entry per row (each row draws from the
    subsequence its own pair names, independent of its batch position) or one entry shared by
    every row (the row index then separates the streams).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = SAMPLING_CELLS
    note = """
    The Kimi K3.1 target calls it through the DFlash worker's `_sample_tokens_for_batch` when
    advanced_sampling_mode is FULL (the default) and the batch is not all greedy: the first
    token of each context request in a DFlash2 sampling step (spec_dflash2_verify
    `dflash2_sample_and_accept`) and the target tokens of exact-match verification
    (`_sample_and_accept_draft_tokens_base`) when the vocab-parallel sampler does not take
    them; seed / offset are the spec metadata's per-token Philox state. Other modes take
    `flashinfer_sample_from_logits`.

    A draw has no closed form to compare against, so `reference` returns the distribution the
    token is drawn from, and the test gates the tokens by it: every token inside the row's
    support, greedy-sentinel rows on the argmax, the same tokens for the same (seed, offset),
    and draw frequencies matching the distribution. Deterministic for a given (seed, offset);
    CUDA-graph capturable. The op itself rejects short per-row tensors, a seed / offset that is
    neither one entry nor one per row, and non-int64 Philox state, so there is no input guard.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return torch.ops.trtllm.fused_sample_from_logits(
            logits, temperatures, top_ks, top_ps, min_ps, seed, offset
        )

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The ``[rows, vocab]`` float32 distribution each row's token is drawn from
        (:func:`filtered_probs`); ``seed`` and ``offset`` are accepted and ignored."""
        return filtered_probs(logits, temperatures, top_ks, top_ps, min_ps)


fused_sample_from_logits = FusedSampleFromLogits()
