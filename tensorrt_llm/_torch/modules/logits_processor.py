import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

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

        token_count = hidden_states.view(-1, hidden_states.shape[-1]).shape[0]

        # Add pre-lm gather logic
        if (self.model_config.mapping.enable_attention_dp and getattr(
                self.model_config.mapping, 'enable_lm_tp_in_adp', False)):
            # ADP + LM TP mode: perform All-Gather before LM_head
            from ..distributed import allgather
            all_rank_max_num_tokens = attn_metadata.all_rank_max_num_tokens
            pad_len = all_rank_max_num_tokens - token_count
            if pad_len > 0:
                padded_hidden_states = F.pad(hidden_states.view(
                    -1, hidden_states.shape[-1]), (0, 0, 0, pad_len),
                                             mode="constant",
                                             value=0)
            else:
                padded_hidden_states = hidden_states.view(
                    -1, hidden_states.shape[-1])
            hidden_states = allgather(padded_hidden_states,
                                      self.model_config.mapping,
                                      dim=0)

        # Temporarily disable gather_output when not in ADP mode or (in ADP mode and LM TP is enabled)
        if (not self.model_config.mapping.enable_attention_dp) or (
                self.model_config.mapping.enable_attention_dp and getattr(
                    self.model_config.mapping, 'enable_lm_tp_in_adp', False)):
            lm_head.gather_output = False
        logits = lm_head(hidden_states)
        if (not self.model_config.mapping.enable_attention_dp) or (
                self.model_config.mapping.enable_attention_dp and getattr(
                    self.model_config.mapping, 'enable_lm_tp_in_adp', False)):
            lm_head.gather_output = True
        # print(f"In LogitsProcessor, lm_head.weight.data_ptr: {lm_head.weight.data_ptr()}")
        # print(f"In LogitsProcessor, lm_head.weight.shape: {lm_head.weight.shape}")
        # print(f"In LogitsProcessor, logits.shape: {logits.shape}")
        logits = allgather(logits, self.model_config.mapping, dim=-1)
        batch_size = logits.shape[0]
        local_batch_size = batch_size // self.model_config.mapping.tp_size
        logits = logits.view(self.model_config.mapping.tp_size,
                             local_batch_size, -1)
        logits = logits[self.model_config.mapping.tp_rank][:token_count]
        print(f"In LogitsProcessor, final logits.shape: {logits.shape}")
        logits = logits.float()
        return logits
