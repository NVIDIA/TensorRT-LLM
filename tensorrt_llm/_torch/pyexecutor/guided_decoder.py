import math
from typing import List, Optional

import torch

from ..._utils import nvtx_range
from ...bindings.executor import GuidedDecodingConfig
from ...logger import logger
from .grammar_matcher import (GrammarMatcher, GrammarMatcherFactory,
                              LLGuidanceMatcherFactory, XGrammarMatcherFactory)
from .llm_request import LlmRequest
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
                f"invalid guided decoding backend: {self.guided_decoding_backend}"
            )
        logger.info(
            f"Guided decoder initialized with backend: {self.guided_decoding_backend}"
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

        # The number of tokens accepted by the grammar matcher in a build step.
        self.num_advanced_tokens: List[int] = [0] * self.max_num_sequences
        # The number of tokens with filled bitmask in a build step.
        self.num_guided_tokens: List[int] = [0] * self.max_num_sequences
        # The accumulated number of tokens accepted by the grammar matcher in a drafting loop.
        self.num_advanced_draft_tokens: List[int] = [0] * self.max_num_sequences
        # Whether is guided drafting is terminated because of unacceptable drafted tokens.
        self.is_draft_terminated: List[bool] = [False] * self.max_num_sequences

        self._stream = torch.cuda.Stream()

    @property
    def bitmask_size(self) -> int:
        return math.ceil(self.vocab_size_padded / 32)

    def _is_matcher_init(self, llm_req: LlmRequest) -> bool:
        if llm_req.guided_decoding_params is None:
            return False
        if llm_req.py_is_draft:
            return False
        # The request is in the last chunk of a context forward step.
        return llm_req.is_context_init_state and llm_req.is_last_context_chunk

    def _is_matcher_in_progress(self, llm_req: LlmRequest) -> bool:
        if llm_req.guided_decoding_params is None:
            return False
        if llm_req.py_is_draft:
            return True
        # The request is in a generation forward step.
        return llm_req.is_generation_in_progress_state

    @torch.inference_mode()
    @nvtx_range("GuidedDecoder.build")
    def build(self, scheduled_requests: ScheduledRequests) -> None:
        for llm_req in scheduled_requests.all_requests():
            slot: int = llm_req.py_target_seq_slot if llm_req.py_is_draft else llm_req.py_seq_slot
            self.num_advanced_tokens[slot] = 0
            self.num_guided_tokens[slot] = 0

            if self._is_matcher_init(llm_req):
                matcher = self.grammar_matcher_factory.create(
                    llm_req.guided_decoding_params)
                self.grammar_matchers[slot] = matcher

            elif self._is_matcher_in_progress(llm_req):
                matcher = self.grammar_matchers[slot]
                # The last new token must be acceptable unless the matcher is terminated in a drafting loop.
                if llm_req.py_is_draft and (matcher.is_terminated()
                                            or self.is_draft_terminated[slot]):
                    continue
                # TODO: Fix this.
                last_new_token = llm_req.get_tokens(0)[-1]
                accepted = matcher.accept_token(last_new_token)
                if not accepted:
                    if llm_req.py_is_draft:
                        self.is_draft_terminated[slot] = True
                        logger.debug(
                            f"Draft request {llm_req.py_request_id} failed to accept last new token: {last_new_token}."
                        )
                        continue
                    # TODO: Make this an error response.
                    raise ValueError(
                        f"Request {llm_req.py_request_id} failed to accept last new token: {last_new_token}."
                    )

            else:
                continue

            self.num_advanced_tokens[slot] += 1
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.bitmask_host[slot], 0)
                self.num_guided_tokens[slot] += 1
                # Process draft tokens
                for i, tid in enumerate(llm_req.py_draft_tokens, 1):
                    accepted = matcher.accept_token(tid)
                    if not accepted:
                        break
                    self.num_advanced_tokens[slot] += 1
                    if matcher.is_terminated():
                        break
                    matcher.fill_next_token_bitmask(self.bitmask_host[slot], i)
                    self.num_guided_tokens[slot] += 1

            if llm_req.py_is_draft:
                assert len(llm_req.py_draft_tokens) == 0
                self.num_advanced_draft_tokens[
                    slot] += self.num_advanced_tokens[slot]

            if (num_guided_tokens := self.num_guided_tokens[slot]) > 0:
                with torch.cuda.stream(self._stream):
                    self.bitmask[slot, :num_guided_tokens].copy_(
                        self.bitmask_host[slot, :num_guided_tokens],
                        non_blocking=True)

    @torch.inference_mode()
    @nvtx_range("GuidedDecoder.execute")
    def execute(self,
                scheduled_requests: ScheduledRequests,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        torch.cuda.current_stream().wait_stream(self._stream)

        # TODO: Fuse this to logits_bitmask.
        if d2t is not None:
            draft_logits = logits
            d2t_mapping = d2t + torch.arange(d2t.size(0), device=d2t.device)
            logits = torch.empty(draft_logits.size(0),
                                 self.vocab_size_padded,
                                 dtype=draft_logits.dtype,
                                 device=draft_logits.device)
            logits.index_copy_(-1, d2t_mapping, draft_logits)

        batched_logits, batched_bitmask = [], []
        offset = 0
        for llm_req in scheduled_requests.all_requests():
            slot: int = llm_req.py_target_seq_slot if llm_req.py_is_draft else llm_req.py_seq_slot
            for i in range(self.num_guided_tokens[slot]):
                batched_logits.append(logits[offset + i])
                batched_bitmask.append(self.bitmask[slot, i])
            offset += len(llm_req.py_draft_tokens) + 1

        assert offset == logits.size(0)

        if len(batched_logits) > 0:
            torch.ops.trtllm.logits_bitmask(batched_logits, batched_bitmask)

        if d2t is not None:
            torch.index_select(logits, -1, d2t_mapping, out=draft_logits)

    @nvtx_range("GuidedDecoder.rollback_rejected_tokens")
    def rollback_rejected_tokens(self,
                                 scheduled_requests: ScheduledRequests) -> None:
        """This method should be called:
        - after the verification (so that the accepted tokens are ready) and
        - before the first guided decoding build of the next drafting loop.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for llm_req in scheduled_requests.all_requests():
            assert not llm_req.py_is_draft
            slot: int = llm_req.py_seq_slot
            if self.num_advanced_tokens[slot] <= 0:
                continue
            # Rollback the grammar matcher to the last accepted token.
            num_rollback_tokens = self.num_advanced_tokens[slot] - (
                1 + llm_req.py_num_accepted_draft_tokens)
            # TODO: Make this an error response.
            if num_rollback_tokens < 0:
                raise ValueError(
                    f"Failed to rollback: num_advanced_tokens={self.num_advanced_tokens[slot]}, num_accepted_draft_tokens={llm_req.py_num_accepted_draft_tokens}, num_rollback_tokens={num_rollback_tokens}"
                )
            self.grammar_matchers[slot].rollback(num_rollback_tokens)

    @nvtx_range("GuidedDecoder.rollback_draft_tokens")
    def rollback_draft_tokens(self,
                              scheduled_requests: ScheduledRequests) -> None:
        """This method should be called:
        - after the the drafting loop and
        - before the guided decoding build of the target model.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for llm_req in scheduled_requests.all_requests():
            assert not llm_req.py_is_draft
            slot: int = llm_req.py_seq_slot
            if self.num_advanced_draft_tokens[slot] <= 0:
                continue
            self.grammar_matchers[slot].rollback(
                self.num_advanced_draft_tokens[slot])
            # Reset the drafting states.
            self.num_advanced_draft_tokens[slot] = 0
            self.is_draft_terminated[slot] = False
