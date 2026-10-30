# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .context_logits import project_context_logits
from .embedding import LMHead
from .linear import Linear


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
        logits = logits.float()
        return logits
