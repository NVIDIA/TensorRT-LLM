# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .linear import Linear


def scale_and_soft_cap_logits(logits: torch.Tensor, multiplier: float,
                              soft_cap: Optional[float]) -> torch.Tensor:
    """Return `soft_cap * tanh(logits * multiplier / soft_cap)` in FP32.

    The logit epilogue of models that scale and soft-cap their LM head output.
    Works in place when `logits` is already FP32, so pass a tensor the caller
    owns; a `soft_cap` of None skips the cap.
    """
    logits = logits.float()
    if multiplier != 1.0:
        logits.mul_(multiplier)
    if soft_cap is not None:
        logits.div_(soft_cap).tanh_().mul_(soft_cap)
    return logits


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    @nvtx_range("LogitsProcessor")
    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False) -> torch.Tensor:

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
        logits = logits.float()
        return logits
