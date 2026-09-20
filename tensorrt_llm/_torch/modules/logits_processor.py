import torch
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    @nvtx_range("LogitsProcessor")
    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False,
                *,
                upcast_to_float: bool = True) -> torch.Tensor:
        """Project hidden states through the LM head.

        ``upcast_to_float`` materializes an fp32 copy of the whole
        ``[tokens, vocab]`` tensor. That is what almost every consumer expects
        (logprobs, penalties, softmax-based sampling), so it stays the default;
        a caller whose only consumer is an argmax -- which is invariant under
        the widening cast -- can pass False and keep the head's own dtype.
        """

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
        if upcast_to_float:
            logits = logits.float()
        return logits
