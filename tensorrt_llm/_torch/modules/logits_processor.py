import torch
import torch.nn as nn

from ..attention_backend import AttentionMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool | int = False) -> torch.Tensor:

        # hidden_states: [total_tokens, d_model]

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
        elif isinstance(return_context_logits, int):
            # We're doing block prediction, return for the last `return_context_logits` tokens
            if attn_metadata is not None:  # Multiple sequences, packed
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                )
                d_model = hidden_states.size(-1)
                offsets = torch.arange(-return_context_logits + 1, 0, device=hidden_states.device)
                gather_indices = last_tokens[:, None] + offsets[None, :]

                hidden_states = torch.gather(hidden_states, 0, gather_indices.flatten().repeat(1, d_model))
            else:  # Single sequence
                hidden_states = hidden_states[-return_context_logits:]

        logits = lm_head(hidden_states)
        logits = logits.float()
        return logits
