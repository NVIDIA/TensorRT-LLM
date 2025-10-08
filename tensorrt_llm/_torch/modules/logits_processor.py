import torch
import torch.nn as nn

from ..attention_backend import AttentionMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

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
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits
