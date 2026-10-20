# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .context_logits import project_context_logits
from .embedding import LMHead
from .linear import Linear
from .logits_cast import fast_logits_cast_enabled, widen_logits_to_fp32


def scale_and_soft_cap_logits(logits: torch.Tensor, multiplier: float,
                              soft_cap: Optional[float]) -> torch.Tensor:
    """Return `soft_cap * tanh(logits * multiplier / soft_cap)` in FP32.

    The logit epilogue of models that scale and soft-cap their LM head output,
    shared with the DFlash drafters trained on it. Works in place when `logits`
    is already FP32, so pass a tensor the caller owns; a `soft_cap` of None
    skips the cap.
    """
    logits = logits.float()
    if multiplier != 1.0:
        logits.mul_(multiplier)
    if soft_cap is not None:
        logits.div_(soft_cap).tanh_().mul_(soft_cap)
    return logits


class LogitsProcessor(nn.Module):

    def __init__(self, *, allow_context_logits_plan: bool = True):
        super().__init__()
        self.allow_context_logits_plan = allow_context_logits_plan

    @nvtx_range("LogitsProcessor")
    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False) -> torch.Tensor:

        if (self.allow_context_logits_plan and return_context_logits
                and isinstance(lm_head, LMHead)
                and lm_head.context_logits_workspace is not None):
            planned_logits = project_context_logits(hidden_states, lm_head)
            if planned_logits is not None:
                return planned_logits

        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]

        logits = lm_head(hidden_states)
        if fast_logits_cast_enabled():
            # Bit-identical to .float(); the vectorized kernel runs at HBM
            # bandwidth (see logits_cast.py).
            return widen_logits_to_fp32(logits)
        logits = logits.float()
        return logits
