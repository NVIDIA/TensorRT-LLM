# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention_backend import AttentionMetadata
from .linear import Linear
from .logits_cast import fast_logits_cast_enabled, widen_logits_to_fp32


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
        if fast_logits_cast_enabled():
            # Bit-identical to .float(); the vectorized kernel runs at HBM
            # bandwidth (see logits_cast.py).
            return widen_logits_to_fp32(logits)
        logits = logits.float()
        return logits
