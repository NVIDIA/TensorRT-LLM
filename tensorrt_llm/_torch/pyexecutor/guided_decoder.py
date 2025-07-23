import math
from typing import List, Optional

import torch

from ..._utils import nvtx_range
from ...bindings.executor import GuidedDecodingConfig
from .grammar_matcher import (GrammarMatcher, GrammarMatcherFactory,
                              LLGuidanceMatcherFactory, XGrammarMatcherFactory)
from .scheduler import ScheduledRequests


class GuidedDecoder:
    bitmask_dtype = torch.int32

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0):
        self.guided_decoding_backend = guided_decoding_config.backend
        self.max_num_sequences = max_num_sequences
        self.vocab_size_padded = vocab_size_padded
        self.max_num_draft_tokens = max_num_draft_tokens

        self.grammar_matcher_factory: Optional[GrammarMatcherFactory] = None
        self.grammar_matchers: List[
            Optional[GrammarMatcher]] = [None] * self.max_num_sequences

        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            self.grammar_matcher_factory = XGrammarMatcherFactory(
                guided_decoding_config,
                vocab_size_padded,
                max_num_draft_tokens=max_num_draft_tokens)
        elif self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE:
            self.grammar_matcher_factory = LLGuidanceMatcherFactory(
                guided_decoding_config, vocab_size_padded)
        else:
            raise ValueError(
                f"invalid guided_decoding_backend: {self.guided_decoding_backend}"
            )

        self.bitmask = torch.empty(self.max_num_sequences,
                                   self.max_num_draft_tokens + 1,
                                   self.bitmask_size,
                                   dtype=self.bitmask_dtype,
                                   device='cuda')
        self.bitmask_host = torch.empty(self.max_num_sequences,
                                        self.max_num_draft_tokens + 1,
                                        self.bitmask_size,
                                        dtype=self.bitmask_dtype,
                                        pin_memory=True)
        self.num_guided_tokens: List[int] = [0] * self.max_num_sequences
        self._stream = torch.cuda.Stream()

    @property
    def bitmask_size(self) -> int:
        return math.ceil(self.vocab_size_padded / 32)

    @nvtx_range("GuidedDecoder.build")
    def build(self, scheduled_requests: ScheduledRequests) -> None:
        for llm_req in scheduled_requests.all_requests():
            slot: int = llm_req.py_seq_slot
            require_guided: bool = True

            if llm_req.guided_decoding_params is None:
                require_guided = False
            else:
                if llm_req.is_context_init_state and llm_req.is_last_context_chunk:
                    # The request is in the last chunk of a context forward step.
                    matcher = self.grammar_matcher_factory.create(
                        llm_req.guided_decoding_params)
                    self.grammar_matchers[slot] = matcher
                elif llm_req.is_generation_in_progress_state:
                    # The request is in a generation forward step.
                    matcher = self.grammar_matchers[slot]
                    # Rollback the grammar matcher to the last accepted token.
                    num_rollback_tokens = self.num_guided_tokens[slot] - (
                        1 + llm_req.py_num_accepted_draft_tokens)
                    if num_rollback_tokens < 0:
                        raise ValueError(
                            f"Failed to rollback: num_guided_tokens={self.num_guided_tokens[slot]}, num_accepted_draft_tokens={llm_req.py_num_accepted_draft_tokens}, num_rollback_tokens={num_rollback_tokens}"
                        )
                    matcher.rollback(num_rollback_tokens)

                    # Currently, guided decoding does not support with beam search.
                    accepted = matcher.accept_token(llm_req.get_last_tokens(0))
                    # TODO: Make this an error response.
                    if not accepted:
                        raise ValueError(
                            f"Failed to accept new token: {llm_req.get_last_tokens(0)}."
                        )
                else:
                    require_guided = False

            num_guided_tokens: int = 0
            if require_guided:
                if not matcher.is_terminated():
                    matcher.fill_next_token_bitmask(self.bitmask_host[slot], 0)
                    num_guided_tokens += 1
                # Process draft tokens
                for i, tid in enumerate(llm_req.py_draft_tokens, 1):
                    accepted = matcher.accept_token(tid)
                    if matcher.is_terminated():
                        matcher.rollback(1)
                        accepted = False
                    if accepted:
                        matcher.fill_next_token_bitmask(self.bitmask_host[slot],
                                                        i)
                        num_guided_tokens += 1
                    else:
                        break

            self.num_guided_tokens[slot] = num_guided_tokens
            if num_guided_tokens > 0:
                with torch.cuda.stream(self._stream):
                    self.bitmask[slot, :num_guided_tokens].copy_(
                        self.bitmask_host[slot, :num_guided_tokens],
                        non_blocking=True)

    @nvtx_range("GuidedDecoder.execute")
    def execute(self, scheduled_requests: ScheduledRequests,
                logits: torch.Tensor) -> None:
        torch.cuda.current_stream().wait_stream(self._stream)

        batched_logits, batched_bitmask = [], []
        offset = 0
        for llm_req in scheduled_requests.all_requests():
            slot: int = llm_req.py_seq_slot
            num_guided_tokens: int = self.num_guided_tokens[slot]
            for i in range(num_guided_tokens):
                batched_logits.append(logits[offset + i])
                batched_bitmask.append(self.bitmask[slot, i])
            offset += len(llm_req.py_draft_tokens) + 1

        assert offset == logits.size(0)

        if len(batched_logits) > 0:
            torch.ops.trtllm.logits_bitmask(batched_logits, batched_bitmask)
