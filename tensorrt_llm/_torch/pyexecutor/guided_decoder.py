import math
from typing import List, Optional

import torch

from ...bindings.executor import GuidedDecodingConfig
from .grammar_matcher import (GrammarMatcher, GrammarMatcherFactory,
                              LLGuidanceMatcherFactory, XGrammarMatcherFactory)
from .scheduler import ScheduledRequests
from .seq_slot_manager import SeqSlotManager


class GuidedDecoder:
    bitmask_dtype = torch.int32

    def __init__(self, guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int, vocab_size_padded: int):
        self.guided_decoding_backend = guided_decoding_config.backend
        self.max_num_sequences = max_num_sequences
        self.vocab_size_padded = vocab_size_padded

        self.grammar_matcher_factory: Optional[GrammarMatcherFactory] = None
        self.grammar_matchers: List[
            Optional[GrammarMatcher]] = [None] * self.max_num_sequences

        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            self.grammar_matcher_factory = XGrammarMatcherFactory(
                guided_decoding_config, vocab_size_padded)
        elif self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE:
            self.grammar_matcher_factory = LLGuidanceMatcherFactory(
                guided_decoding_config, vocab_size_padded)
        else:
            raise ValueError(
                f"invalid guided_decoding_backend: {self.guided_decoding_backend}"
            )

        self.bitmask = torch.empty(self.max_num_sequences,
                                   self.bitmask_size,
                                   dtype=self.bitmask_dtype,
                                   device='cuda')
        self.bitmask_host = torch.empty(self.max_num_sequences,
                                        self.bitmask_size,
                                        dtype=self.bitmask_dtype,
                                        pin_memory=True)

        self._stream = torch.cuda.Stream()

    @property
    def bitmask_size(self) -> int:
        return math.ceil(self.vocab_size_padded / 32)

    def build(self, scheduled_requests: ScheduledRequests,
              resource_manager: SeqSlotManager) -> None:
        for llm_req in scheduled_requests.all_requests():
            if llm_req.guided_decoding_params is None:
                continue
            slot = resource_manager.slot_manager.get_slot(llm_req.request_id)
            if llm_req.is_context_init_state and llm_req.context_current_position == llm_req.prepopulated_prompt_len:
                self.grammar_matchers[
                    slot] = self.grammar_matcher_factory.create(
                        llm_req.guided_decoding_params)

            elif llm_req.is_generation_in_progress_state:
                # The request is in a generation forward step.
                # Currently, guided decoding does not support with beam search.
                self.grammar_matchers[slot].accept_token(
                    llm_req.get_last_tokens(0))
            else:
                continue

            # Fill the bitmask on host and asynchorously copy to device.
            self.grammar_matchers[slot].fill_next_token_bitmask(
                self.bitmask_host, slot)
            with torch.cuda.stream(self._stream):
                self.bitmask[slot].copy_(self.bitmask_host[slot],
                                         non_blocking=True)

    def execute(self, scheduled_requests: ScheduledRequests,
                logits: torch.Tensor, resource_manager: SeqSlotManager) -> None:
        assert logits.size(0) == len(scheduled_requests.context_requests) + len(
            scheduled_requests.generation_requests)
        torch.cuda.current_stream().wait_stream(self._stream)

        batched_logits, batched_bitmask = [], []
        for i, llm_req in enumerate(scheduled_requests.all_requests()):
            if llm_req.guided_decoding_params is None:
                continue
            if llm_req.is_context_init_state and not llm_req.is_last_context_chunk:
                continue
            batched_logits.append(logits[i])
            slot = resource_manager.slot_manager.get_slot(llm_req.request_id)
            batched_bitmask.append(self.bitmask[slot])

        if len(batched_logits) > 0:
            torch.ops.trtllm.logits_bitmask(batched_logits, batched_bitmask)
