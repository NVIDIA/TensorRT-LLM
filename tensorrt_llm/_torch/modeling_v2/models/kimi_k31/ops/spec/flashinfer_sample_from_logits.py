# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One sampled token per row through the engine's flashinfer chain (top-k mask, temperature
softmax, top-p or plain sampling from the probabilities), the token sampler of the non-fused
advanced sampling modes."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.flashinfer_compute_probs_from_logits import (  # noqa: E501
    FLASHINFER_CELLS,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    filtered_probs,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling


class FlashinferSampleFromLogits(OpWrapper):
    """``[rows]`` int32: one token per row drawn from the distribution
    :class:`FlashinferComputeProbsFromLogits` returns for the same ``logits`` / ``temperatures``
    / ``top_k`` / ``top_p``.

    ``seed`` / ``offset`` are int64 Philox state for flashinfer's sampling kernels (one entry per
    row, as the spec metadata lays them out); None draws from torch's default generator.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = FLASHINFER_CELLS
    note = """
    The Kimi K3.1 target calls it through the DFlash worker's `_sample_tokens_for_batch` when
    advanced_sampling_mode is NO_TOPK, NO_TOPP or NO_TOPK_NO_TOPP and the batch is not all
    greedy (and the vocab-parallel sampler does not take it): the first token of each context
    request in a DFlash2 sampling step and the target tokens of exact-match verification. FULL,
    the default, takes `fused_sample_from_logits` instead.

    The entry is the engine's `sampler.ops.flashinfer.sample_from_logits_op`, a torch.compile'd
    chain (the first call per shape compiles): flashinfer `top_k_mask_logits`, `softmax`, then
    `top_p_sampling_from_probs` or `sampling_from_probs`. `reference` returns the distribution
    the token is drawn from; the test gates the tokens by its support, the greedy-sentinel rows'
    argmax, determinism under a fixed (seed, offset) and the draw frequencies.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: Optional[torch.Tensor],
        top_p: Optional[torch.Tensor],
        seed: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return flashinfer_sampling.sample_from_logits_op(
            logits, temperatures, top_k, top_p, seed=seed, offset=offset
        )

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: Optional[torch.Tensor],
        top_p: Optional[torch.Tensor],
        seed: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The ``[rows, vocab]`` float32 distribution each row's token is drawn from
        (:func:`filtered_probs` without min-p); ``seed`` and ``offset`` are ignored."""
        return filtered_probs(logits, temperatures, top_k, top_p, None)


flashinfer_sample_from_logits = FlashinferSampleFromLogits()
