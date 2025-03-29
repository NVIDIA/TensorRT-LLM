import inspect
from typing import List, Optional

import torch
import torch.nn as nn

from tensorrt_llm.sampling_params import \
    LogitsProcessor as LogitsProcessorCallables

from ..attention_backend import AttentionMetadata
from ..metadata import LogitsProcessorMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: Linear,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        logits_processor_metadata: Optional[LogitsProcessorMetadata] = None,
    ) -> torch.Tensor:

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

        if logits_processor_metadata:
            logits = _apply_logits_processors(logits, input_ids,
                                              logits_processor_metadata)

        return logits


# TODO: add ManagedThreads
def _apply_logits_processors(
        logits: torch.Tensor, input_ids: torch.LongTensor,
        logits_processor_metadata: LogitsProcessorMetadata) -> torch.Tensor:

    for seq_group in logits_processor_metadata.seq_groups:
        logits_processors = seq_group.logits_processors

        if logits_processors:
            for i, (req_id, batch_idx) in enumerate(
                    zip(seq_group.request_ids, seq_group.batch_indices)):
                logits_row = logits[batch_idx]
                # TODO: slice tokenids

                logits[batch_idx] = \
                    _apply_logits_processors_single_seq(
                        logits_processors,
                        req_id,
                        logits_row,
                        [],  # TODO: WIP on token ids
                        None,
                        None)

    return logits


def _apply_logits_processors_single_seq(
        logits_processors: List[LogitsProcessorCallables],
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: int = None,
        client_id: int = None) -> torch.Tensor:
    for logits_processor in logits_processors:
        lp_params = inspect.signature(logits_processor).parameters

        # WIP
        assert len(lp_params) >= 4 and len(lp_params) <= 5

        logits_row = logits_processor(req_id, logits, token_ids, stream_ptr,
                                      client_id)

    return logits_row
