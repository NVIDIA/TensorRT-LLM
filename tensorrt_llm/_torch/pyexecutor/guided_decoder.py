import math
from dataclasses import dataclass, field
from queue import Queue
from typing import Iterable, List, Optional, Protocol, Tuple

import torch
import xgrammar

from ..._utils import nvtx_range
from ...bindings.executor import GuidedDecodingConfig, GuidedDecodingParams
from ...logger import logger
from ..hostfunc import hostfunc
from .grammar_matcher import (GrammarMatcher, LLGuidanceMatcherFactory,
                              XGrammarMatcherFactory)
from .llm_request import LlmRequest
from .scheduler import ScheduledRequests


@dataclass(slots=True)
class GuidedRequest:
    """A snapshot of an LlmRequest that contains relevant fields for guided decoding.

    The instances of this class should be produced on the host and consumed on the device (by hostfunc).
    """
    guided_decoding_params: Optional[GuidedDecodingParams] = field(default=None)

    request_id: Optional[int] = field(default=None)
    seq_slot: Optional[int] = field(default=None)
    is_context_init_state: bool = field(default=False)
    is_last_context_chunk: bool = field(default=False)
    is_generation_in_progress_state: bool = field(default=False)

    new_token: Optional[int] = field(default=None)
    is_draft: bool = field(default=False)
    draft_tokens: Optional[List[int]] = field(default=None)
    num_accepted_draft_tokens: Optional[int] = field(default=None)

    def require_matcher_init(self) -> bool:
        if self.guided_decoding_params is None:
            return False
        if self.is_draft:
            return False
        # The request is in the last chunk of a context forward step.
        return self.is_context_init_state and self.is_last_context_chunk

    def require_matcher_advance(self) -> bool:
        if self.guided_decoding_params is None:
            return False
        if self.is_draft:
            if self.is_context_init_state and self.is_last_context_chunk:
                return True
            if self.is_generation_in_progress_state:
                return True
            return False
        # The request is in a generation forward step.
        return self.is_generation_in_progress_state

    @classmethod
    def from_request(cls, request: LlmRequest):
        return cls(
            guided_decoding_params=request.guided_decoding_params,
            request_id=request.py_request_id,
            seq_slot=request.py_target_seq_slot
            if request.py_is_draft else request.py_seq_slot,
            is_context_init_state=request.is_context_init_state,
            is_last_context_chunk=request.is_context_init_state
            and request.is_last_context_chunk,
            is_generation_in_progress_state=request.
            is_generation_in_progress_state,
            new_token=request.get_last_tokens(0),
            is_draft=request.py_is_draft,
            draft_tokens=request.py_draft_tokens,
            num_accepted_draft_tokens=request.py_num_accepted_draft_tokens)


@dataclass(slots=True)
class GuidedRequests:
    requests: List[GuidedRequest]
    num_contexts: int
    num_generations: int
    max_num_draft_tokens: int

    @classmethod
    def from_scheduled_requests(cls,
                                scheduled_requests: ScheduledRequests,
                                max_num_draft_tokens: int = 0):
        batch = [
            GuidedRequest.from_request(req)
            for req in scheduled_requests.all_requests()
        ]
        return cls(batch,
                   num_contexts=len(scheduled_requests.context_requests),
                   num_generations=len(scheduled_requests.generation_requests),
                   max_num_draft_tokens=max_num_draft_tokens)

    @property
    def num_bitmask_tokens(self) -> int:
        return self.num_contexts + self.num_generations * (
            self.max_num_draft_tokens + 1)

    def request_with_offset(self) -> Iterable[Tuple[GuidedRequest, int]]:
        offset: int = 0
        for req in self.requests:
            yield req, offset
            offset += 1
            if req.is_generation_in_progress_state:
                offset += self.max_num_draft_tokens

    def __iter__(self) -> Iterable[GuidedRequest]:
        return iter(self.requests)


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
                f"Invalid guided decoding backend: {self.guided_decoding_backend}"
            )
        logger.info(
            f"Guided decoder initialized with backend: {self.guided_decoding_backend}"
        )
        self.grammar_matchers: List[
            Optional[GrammarMatcher]] = [None] * self.max_num_sequences

        self.bitmask = torch.empty(self.max_num_sequences *
                                   (self.max_num_draft_tokens + 1),
                                   self.bitmask_size,
                                   dtype=self.bitmask_dtype,
                                   device='cuda')
        self.bitmask_host = torch.empty_like(self.bitmask,
                                             device='cpu',
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

    def _build(self, requests: GuidedRequests) -> None:
        """Build the bitmask for requests with guided decoding enabled.

        Specifically, this method:
        - build and advance the grammar matcher for context and generation requests, respectively;
        - call the grammar matcher to fill the bitmask on CPU;
        - asynchronously copy the bitmask to GPU.
        """
        # Fix it.
        self.bitmask_host[:requests.num_bitmask_tokens].fill_(-1)

        for req, offset in requests.request_with_offset():
            if (slot := req.seq_slot) is None:
                continue
            self.num_advanced_tokens[slot] = 0
            self.num_guided_tokens[slot] = 0

            matcher_init: bool = req.require_matcher_init()
            matcher_advance: bool = req.require_matcher_advance()
            if not (matcher_init or matcher_advance):
                continue

            if matcher_init:
                matcher = self.grammar_matcher_factory.create(
                    req.guided_decoding_params)
                self.grammar_matchers[slot] = matcher

            if matcher_advance:
                matcher = self.grammar_matchers[slot]
                # The last new token must be acceptable unless the matcher is terminated:
                # 1. For the main model loop, when overlap scheduler is enabled, the matcher may have accepted the EOS token in the draft tokens at the previous iteration.
                # 2. For the draft model loop, the matcher may have accepted the EOS token at the previous drafting iteration.
                if matcher.is_terminated() or self.is_draft_terminated[slot]:
                    continue
                accepted = matcher.accept_token(req.new_token)
                if not accepted:
                    if req.is_draft:
                        self.is_draft_terminated[slot] = True
                        logger.debug(
                            f"Draft request {req.request_id} at slot {slot} failed to accept last new token: {req.new_token}."
                        )
                        continue
                    # TODO: Make this an error response.
                    raise ValueError(
                        f"Request {req.request_id} at slot {slot} failed to accept last new token: {req.new_token}."
                    )

            self.num_advanced_tokens[slot] += 1
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.bitmask_host, offset)
                self.num_guided_tokens[slot] += 1
                # Process draft tokens
                for i, tid in enumerate(req.draft_tokens, 1):
                    accepted = matcher.accept_token(tid)
                    if not accepted:
                        break
                    self.num_advanced_tokens[slot] += 1
                    if matcher.is_terminated():
                        break
                    matcher.fill_next_token_bitmask(self.bitmask_host,
                                                    offset + i)
                    self.num_guided_tokens[slot] += 1

            if req.is_draft:
                assert len(req.draft_tokens) == 0
                self.num_advanced_draft_tokens[
                    slot] += self.num_advanced_tokens[slot]

    def copy_bitmask(self, requests: GuidedRequests) -> None:
        self.bitmask[:requests.num_bitmask_tokens].copy_(
            self.bitmask_host[:requests.num_bitmask_tokens], non_blocking=True)

    @torch.inference_mode()
    def apply_bitmask(self,
                      requests: GuidedRequests,
                      logits: torch.Tensor,
                      d2t: Optional[torch.Tensor] = None) -> None:
        # TODO: Fuse index_copy and index_select to logits_bitmask.
        if d2t is not None:
            draft_logits = logits
            d2t_mapping = d2t + torch.arange(d2t.size(0), device=d2t.device)
            logits = torch.empty(draft_logits.size(0),
                                 self.vocab_size_padded,
                                 dtype=draft_logits.dtype,
                                 device=draft_logits.device)
            logits.index_copy_(-1, d2t_mapping, draft_logits)

        xgrammar.apply_token_bitmask_inplace(
            logits[:requests.num_bitmask_tokens],
            self.bitmask[:requests.num_bitmask_tokens])

        if d2t is not None:
            torch.index_select(logits, -1, d2t_mapping, out=draft_logits)

    @nvtx_range("GuidedDecoder.execute")
    def execute(self,
                scheduled_requests: ScheduledRequests,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        """Apply the bitmask to the corresponding logits for requests with guided decoding enabled.

        This method inplace modifies the logits tensor so that any tokens that violate the grammar constraints are masked out.
        """
        requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, self.max_num_draft_tokens)
        self._build(requests)

        with torch.cuda.stream(self._stream):
            self.copy_bitmask(requests)

        torch.cuda.current_stream().wait_stream(self._stream)
        self.apply_bitmask(requests, logits, d2t=d2t)

    def _rollback_rejected_tokens(self, requests: GuidedRequests) -> None:
        for req in requests:
            assert not req.is_draft
            if (slot := req.seq_slot) is None:
                continue
            if self.num_advanced_tokens[slot] <= 0:
                continue
            num_accepted_tokens = 1 + req.num_accepted_draft_tokens
            # Rollback the grammar matcher to the last accepted token.
            num_rollback_tokens = self.num_advanced_tokens[
                slot] - num_accepted_tokens
            # TODO: Make this an error response.
            if num_rollback_tokens < 0:
                raise ValueError(
                    f"Failed to rollback: num_advanced_tokens={self.num_advanced_tokens[slot]}, num_accepted_tokens={num_accepted_tokens}, num_rollback_tokens={num_rollback_tokens}"
                )
            self.grammar_matchers[slot].rollback(num_rollback_tokens)

    @nvtx_range("GuidedDecoder.rollback_rejected_tokens")
    def rollback_rejected_tokens(self,
                                 scheduled_requests: ScheduledRequests) -> None:
        """Rollback the grammar matcher for rejected tokens.

        This method should be called:
        - after the verification (so that the accepted tokens are ready) and
        - before the first guided decoding build of the next drafting loop.
        """
        if self.max_num_draft_tokens <= 0:
            return
        requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, self.max_num_draft_tokens)
        self._rollback_rejected_tokens(requests)

    @nvtx_range("GuidedDecoder.rollback_draft_tokens")
    def rollback_draft_tokens(self,
                              scheduled_requests: ScheduledRequests) -> None:
        """Rollback the grammar matcher for draft tokens.

        This method should be called:
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

    @nvtx_range("GuidedDecoder.init_disagg_gen_requests")
    def init_disagg_gen_requests(self,
                                 scheduled_requests: ScheduledRequests) -> None:
        """Initialize the grammar matchers for disagg gen requests.
        """
        for llm_req in scheduled_requests.generation_requests:
            if llm_req.guided_decoding_params is None:
                continue
            assert not llm_req.py_is_draft
            slot: int = llm_req.py_seq_slot
            if llm_req.context_phase_params is not None and llm_req.py_decoding_iter == 1:
                # The request is in the first generation forward step at the disagg gen instance.
                self.grammar_matchers[
                    slot] = self.grammar_matcher_factory.create(
                        llm_req.guided_decoding_params)

    @hostfunc
    def inc_bitmask_host(self):
        self.bitmask_host.add_(1)

    @hostfunc
    def run(self, token_ids: torch.Tensor):
        self.grammar_matchers[0].accept_token(token_ids[0].item())
        self.grammar_matchers[0].fill_next_token_bitmask(self.bitmask_host, 0)
        if not hasattr(self, "token_ids"):
            self.token_ids = []
        self.token_ids.append(token_ids[0].item())


class SampleStateTensors(Protocol):
    new_tokens: torch.Tensor
    new_tokens_lens: torch.Tensor


class GuidedWorker(GuidedDecoder):

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0):
        super().__init__(guided_decoding_config, max_num_sequences,
                         vocab_size_padded, max_num_draft_tokens)
        self.requests: Optional[GuidedRequests] = None
        self.requests_hostfunc: Optional[GuidedRequests] = None
        self.queue = Queue()

        self.new_tokens = torch.empty(self.max_num_sequences,
                                      self.max_num_draft_tokens + 1,
                                      dtype=torch.int32,
                                      pin_memory=True)
        self.new_tokens_lens = torch.empty(self.max_num_sequences,
                                           dtype=torch.int32,
                                           pin_memory=True)

        self.token_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

    def add_batch(self,
                  scheduled_requests: ScheduledRequests,
                  new_tensors: Optional[SampleStateTensors] = None) -> None:
        self.requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, self.max_num_draft_tokens)
        if new_tensors is not None:
            self.new_tokens.copy_(new_tensors.new_tokens.squeeze(-1).permute(
                1, 0),
                                  non_blocking=True)
            self.new_tokens_lens.copy_(new_tensors.new_tokens_lens,
                                       non_blocking=True)
        self.queue.put((self.requests, new_tensors is not None))

    @hostfunc
    def next_batch(self) -> None:
        # Fix it.
        if self.queue.empty():
            return
        self.requests_hostfunc, has_new_tensors = self.queue.get()
        if not has_new_tensors:
            return

        new_tokens_list = self.new_tokens.tolist()
        new_tokens_lens_list = self.new_tokens_lens.tolist()
        for req in self.requests_hostfunc:
            if (slot := req.seq_slot) is None:
                continue
            req.new_token, *req.draft_tokens = new_tokens_list[slot]
            req.num_accepted_draft_tokens = new_tokens_lens_list[slot] - 1

    @hostfunc
    def build(self) -> None:
        self._build(self.requests_hostfunc)

    @hostfunc
    def rollback_rejected_tokens(self) -> None:
        self._rollback_rejected_tokens(self.requests_hostfunc)

    def execute(self,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        with torch.cuda.stream(self._stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.next_batch()
            self.rollback_rejected_tokens()
            self.build()
            self.copy_bitmask(self.requests)
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(self.requests, logits, d2t=d2t)
