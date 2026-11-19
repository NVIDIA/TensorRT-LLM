# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target probabilities for speculative verification through the engine's flashinfer chain (top-k
mask, temperature softmax, top-p renorm), the samplers of the non-fused advanced sampling modes."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    filtered_probs,
    sampling_spec,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling

#: The K3.1 cells of the flashinfer samplers, one group per non-fused advanced sampling mode
#: (a filter the mode disables is None for every row; min-p is not taken).
FLASHINFER_CELLS: tuple[Cell, ...] = (
    Cell(
        why="NO_TOPK: w5 verify of 8 requests (40 rows) at T=1.0, top-p 0.95",
        spec=sampling_spec(40, top_p=0.95),
    ),
    Cell(
        why="NO_TOPK: w8 verify of 64 requests (512 rows) at T=0.8, top-p 0.9",
        spec=sampling_spec(512, temperature=0.8, top_p=0.9),
    ),
    Cell(
        why="NO_TOPP: top-k 20 at T=0.6",
        spec=sampling_spec(40, temperature=0.6, top_k=20),
    ),
    Cell(
        why="NO_TOPK_NO_TOPP: plain temperature softmax (T=1.0)",
        spec=sampling_spec(72),
    ),
    Cell(
        why="NO_TOPK with per-row top-p (1.0 rows unfiltered) and greedy-sentinel rows",
        spec=sampling_spec(72, top_p=0.9, greedy_rows=9, mixed=True),
    ),
    Cell(
        why="NO_TOPP with per-row top-k (INT32_MAX rows unfiltered, sanitized to the vocab)",
        spec=sampling_spec(40, temperature=0.7, top_k=50, mixed=True),
    ),
)


class FlashinferComputeProbsFromLogits(OpWrapper):
    """``[rows, vocab]`` float32: each row's ``softmax(logits / T)`` restricted to its top-k and
    then renormalized over its top-p nucleus (:func:`filtered_probs` without min-p), newly
    allocated.

    ``top_k`` (int32) / ``top_p`` (float32) hold one entry per row, or are None to skip that
    filter for every row; ``top_k`` values outside ``(0, vocab)`` are sanitized to the vocab
    (keep everything) before the mask. ``temperatures`` float32, one per row.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = FLASHINFER_CELLS
    note = """
    The Kimi K3.1 target calls it for the target distributions of DFlash2 block verification
    (spec_dflash2_verify, through the worker's `_compute_probs_from_logits`) when
    advanced_sampling_mode is NO_TOPK, NO_TOPP or NO_TOPK_NO_TOPP: the mode resolves the filter
    it disables to None (`resolve_advanced_sampling_filters`) and this chain skips its kernel.
    FULL, the default, takes `fused_compute_probs_from_logits` instead.

    The entry is the engine's `sampler.ops.flashinfer.compute_probs_from_logits`: a
    torch.compile'd wrapper (the first call per shape compiles) around flashinfer's
    `top_k_mask_logits`, `softmax` (temperature fused, PDL per TRTLLM_ENABLE_PDL) and
    `top_p_renorm_probs`. Masking the logits before the temperature softmax keeps the same set
    as top-k on the probabilities. Top-p's cut can flip a token sitting on the boundary and
    ties at either cut may be ordered differently from the reference, so the test bounds each
    row's L1 distance. No RNG.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: Optional[torch.Tensor],
        top_p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return flashinfer_sampling.compute_probs_from_logits(logits, temperatures, top_k, top_p)

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: Optional[torch.Tensor],
        top_p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """:func:`filtered_probs` in fp32, without min-p."""
        return filtered_probs(logits, temperatures, top_k, top_p, None)


flashinfer_compute_probs_from_logits = FlashinferComputeProbsFromLogits()
