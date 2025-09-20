import math
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, List, Optional, Tuple

import torch

from ..._utils import nvtx_range
from ...bindings.executor import GuidedDecodingConfig, GuidedDecodingParams
from ...bindings.internal.batch_manager import LlmRequestType
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
    guided_decoding_params: Optional[GuidedDecodingParams] = None

    request_id: Optional[int] = None
    seq_slot: Optional[int] = None
    prev_seq_slot: Optional[int] = None
    is_context_init_state: bool = False
    is_last_context_chunk: bool = False
    is_generation_in_progress_state: bool = False
    is_generation_only_first_iteration: bool = False

    new_token: Optional[int] = None
    is_draft: bool = False
    draft_tokens: Optional[List[int]] = None
    num_accepted_draft_tokens: Optional[int] = None

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
    def from_llm_request(cls, request: LlmRequest):
        return cls(
            guided_decoding_params=request.guided_decoding_params,
            request_id=request.py_request_id,
            seq_slot=(request.py_target_seq_slot
                      if request.py_is_draft else request.py_seq_slot),
            prev_seq_slot=request.py_batch_idx,
            is_context_init_state=request.is_context_init_state,
            is_last_context_chunk=(request.is_context_init_state
                                   and request.is_last_context_chunk),
            is_generation_in_progress_state=request.
            is_generation_in_progress_state,
            is_generation_only_first_iteration=(
                request.llm_request_type
                == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
                and request.py_decoding_iter == 1
                and request.py_batch_idx is None),
            new_token=request.get_last_tokens(0),
            is_draft=request.py_is_draft,
            draft_tokens=request.py_draft_tokens,
            num_accepted_draft_tokens=request.py_num_accepted_draft_tokens)

    def cast_to_draft(self) -> None:
        self.is_draft = True
        self.draft_tokens = []


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
        requests = [
            GuidedRequest.from_llm_request(req)
            for req in scheduled_requests.all_requests()
        ]
        return cls(requests,
                   num_contexts=len(scheduled_requests.context_requests),
                   num_generations=len(scheduled_requests.generation_requests),
                   max_num_draft_tokens=max_num_draft_tokens)

    @property
    def num_bitmask_tokens(self) -> int:
        if self.requests[0].is_draft:
            return len(self.requests)
        else:
            return self.num_contexts + self.num_generations * (
                self.max_num_draft_tokens + 1)

    def valid_requests_with_offsets(
            self) -> Iterable[Tuple[GuidedRequest, int]]:
        offset: int = 0
        for req in self.requests:
            if req.guided_decoding_params is not None and req.seq_slot is not None:
                yield req, offset
            offset += 1
            if not req.is_draft and req.is_generation_in_progress_state:
                offset += self.max_num_draft_tokens

    def valid_requests(self) -> Iterable[GuidedRequest]:
        for req in self.requests:
            if req.guided_decoding_params is not None and req.seq_slot is not None:
                yield req

    def __iter__(self) -> Iterable[GuidedRequest]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


class GuidedDecoder:
    bitmask_dtype = torch.int32
    token_mask_dtype = torch.int32

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
        self.token_mask = torch.empty(self.max_num_sequences *
                                      (self.max_num_draft_tokens + 1),
                                      dtype=self.token_mask_dtype,
                                      device='cuda')
        self.token_mask_host = torch.empty_like(self.token_mask,
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

        self.requests: Optional[GuidedRequests] = None

        self.stream = torch.cuda.Stream()
        self.token_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

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
        self.token_mask_host[:requests.num_bitmask_tokens].fill_(0)

        for req, offset in requests.valid_requests_with_offsets():
            slot = req.seq_slot
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
                self.token_mask_host[offset] = 1
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
                    self.token_mask_host[offset + i] = 1
                    self.num_guided_tokens[slot] += 1

            if req.is_draft:
                assert len(req.draft_tokens) == 0
                self.num_advanced_draft_tokens[
                    slot] += self.num_advanced_tokens[slot]

    def _copy_bitmask(self,
                      requests: GuidedRequests,
                      num_bitmask_tokens: Optional[int] = None) -> None:
        if num_bitmask_tokens is None:
            num_bitmask_tokens = requests.num_bitmask_tokens
        self.bitmask[:num_bitmask_tokens].copy_(
            self.bitmask_host[:num_bitmask_tokens], non_blocking=True)
        self.token_mask[:num_bitmask_tokens].copy_(
            self.token_mask_host[:num_bitmask_tokens], non_blocking=True)

    @torch.inference_mode()
    def _apply_bitmask(self,
                       requests: GuidedRequests,
                       logits: torch.Tensor,
                       d2t: Optional[torch.Tensor] = None,
                       num_bitmask_tokens: Optional[int] = None) -> None:
        """Apply the bitmask to the corresponding logits for requests with guided decoding enabled.

        This method inplace modifies the logits tensor so that any tokens that violate the grammar constraints are masked out.
        """
        if num_bitmask_tokens is None:
            num_bitmask_tokens = requests.num_bitmask_tokens
        torch.ops.trtllm.logits_bitmask(
            logits[:num_bitmask_tokens],
            self.bitmask[:num_bitmask_tokens],
            token_mask=self.token_mask[:num_bitmask_tokens],
            d2t=d2t)

    @nvtx_range("GuidedDecoder.add_batch")
    def add_batch(self, scheduled_requests: ScheduledRequests) -> None:
        self.requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, self.max_num_draft_tokens)

    @nvtx_range("GuideDecoder.build")
    def build(self) -> None:
        self._build(self.requests)

    @nvtx_range("GuideDecoder.copy_bitmask")
    def copy_bitmask(self, num_bitmask_tokens: Optional[int] = None) -> None:
        self._copy_bitmask(self.requests, num_bitmask_tokens=num_bitmask_tokens)

    @nvtx_range("GuidedDecoder.apply_bitmask")
    def apply_bitmask(self,
                      logits: torch.Tensor,
                      d2t: Optional[torch.Tensor] = None,
                      num_bitmask_tokens: Optional[int] = None) -> None:
        self._apply_bitmask(self.requests,
                            logits,
                            d2t=d2t,
                            num_bitmask_tokens=num_bitmask_tokens)

    def execute(self,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        self.build()

        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.copy_bitmask()
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(logits, d2t=d2t)
        self.token_event.record()

    def _rollback_rejected_tokens(self, requests: GuidedRequests) -> None:
        """Rollback the grammar matcher for rejected tokens.

        This method should be called:
        - after the verification (so that the accepted tokens are ready) and
        - before the first guided decoding build of the next drafting loop.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for req in requests.valid_requests():
            slot = req.seq_slot
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

    def _rollback_draft_tokens(self, requests: GuidedRequests) -> None:
        """Rollback the grammar matcher for draft tokens.

        This method should be called:
        - after the the drafting loop and
        - before the guided decoding build of the target model.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for req in requests.valid_requests():
            slot = req.seq_slot
            if self.num_advanced_draft_tokens[slot] <= 0:
                continue
            self.grammar_matchers[slot].rollback(
                self.num_advanced_draft_tokens[slot])
            # Reset the drafting states.
            self.num_advanced_draft_tokens[slot] = 0
            self.is_draft_terminated[slot] = False

    @nvtx_range("GuidedDecoder.rollback_rejected_tokens")
    def rollback_rejected_tokens(self) -> None:
        self._rollback_rejected_tokens(self.requests)

    @nvtx_range("GuidedDecoder.rollback_draft_tokens")
    def rollback_draft_tokens(self) -> None:
        self._rollback_draft_tokens(self.requests)

    def _init_disagg_gen_requests(self, requests: GuidedRequests) -> None:
        """Initialize the grammar matchers for disagg gen requests.
        """
        for req in requests.valid_requests():
            if req.is_generation_only_first_iteration:
                self.grammar_matchers[
                    req.seq_slot] = self.grammar_matcher_factory.create(
                        req.guided_decoding_params)

    @nvtx_range("GuidedDecoder.init_disagg_gen_requests")
    def init_disagg_gen_requests(self) -> None:
        self._init_disagg_gen_requests(self.requests)


class CapturableGuidedDecoder(GuidedDecoder):

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0):
        super().__init__(guided_decoding_config, max_num_sequences,
                         vocab_size_padded, max_num_draft_tokens)
        # self.requests should be accessed by normal host code;
        # self.requests_hostfunc should be accessed by hostfunc (CUDA callback).
        self.requests_hostfunc: Optional[GuidedRequests] = None
        self.queue = Queue()

        self.new_tokens = torch.empty(self.max_num_draft_tokens + 1,
                                      self.max_num_sequences,
                                      dtype=torch.int32,
                                      pin_memory=True)
        self.num_accepted_tokens = torch.empty(self.max_num_sequences,
                                               dtype=torch.int32,
                                               pin_memory=True)

        # torch.compile kernels are called with GIL being held;
        # this could cause deadlock with CUDA callback to Python code.
        # See: https://github.com/pytorch/pytorch/issues/163061
        torch.compiler.set_stance("force_eager")

    @nvtx_range("GuidedDecoder.add_batch")
    def add_batch(self,
                  scheduled_requests: ScheduledRequests,
                  new_tokens: Optional[torch.Tensor] = None) -> None:
        self.requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, self.max_num_draft_tokens)
        if new_tokens is not None:
            self.new_tokens.copy_(new_tokens.squeeze(-1), non_blocking=True)
        self.queue.put((self.requests, new_tokens is not None))
        # self.token_event.record() should be called inside CUDA graph capturing;
        # currently, it is in PyTorchModelEngine._preprocess_inputs.

    @hostfunc
    def fetch_batch(self) -> None:
        # CUDA graph warmup calls model forward for multiple times for one prepared inputs
        if self.queue.empty():
            return
        self.requests_hostfunc, has_new_tokens = self.queue.get()
        if not has_new_tokens:
            return

        for req in self.requests_hostfunc.valid_requests():
            if req.prev_seq_slot is None:
                continue
            req.new_token, *req.draft_tokens = self.new_tokens[:, req.
                                                               seq_slot].tolist(
                                                               )

    @hostfunc
    def build(self) -> None:
        self._build(self.requests_hostfunc)

    def execute(self,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.fetch_batch()
            self.init_disagg_gen_requests()
            self.build()
            self.copy_bitmask()
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(logits, d2t=d2t)

    @hostfunc
    def rollback_rejected_tokens(self) -> None:
        self._rollback_rejected_tokens(self.requests_hostfunc)

    @hostfunc
    def rollback_draft_tokens(self) -> None:
        self._rollback_draft_tokens(self.requests_hostfunc)

    @hostfunc
    def init_disagg_gen_requests(self) -> None:
        self._init_disagg_gen_requests(self.requests_hostfunc)

    @nvtx_range("GuidedDecoder.add_draft_batch")
    def add_draft_batch(self,
                        new_tokens: torch.Tensor,
                        num_accepted_tokens: torch.Tensor,
                        draft_step: int = 0) -> None:
        batch_size = len(self.requests)
        assert new_tokens.size(0) == batch_size
        self.new_tokens[0, :batch_size].copy_(new_tokens, non_blocking=True)
        if draft_step == 0:
            assert num_accepted_tokens.size(0) == batch_size
            self.num_accepted_tokens[:batch_size].copy_(num_accepted_tokens,
                                                        non_blocking=True)
        self.token_event.record()

    @hostfunc
    def fetch_draft_batch(self, draft_step: int = 0) -> None:
        batch_size = len(self.requests_hostfunc)
        new_tokens_list = self.new_tokens[0, :batch_size].tolist()
        if draft_step == 0:
            num_accepted_tokens_list = self.num_accepted_tokens[:
                                                                batch_size].tolist(
                                                                )
        for i, req in enumerate(self.requests_hostfunc.requests):
            if req.guided_decoding_params is None or (slot :=
                                                      req.seq_slot) is None:
                continue
            req.new_token = new_tokens_list[i]
            if draft_step == 0:
                # When overlap scheduler is enabled, it is possible that
                # - The EOS token is in the draft tokens, and
                # - Some draft tokens after the EOS token are accepted by the target model.
                # These requests should be terminated at this executor iteration.
                req.num_accepted_draft_tokens = min(
                    num_accepted_tokens_list[i],
                    self.num_advanced_tokens[slot]) - 1
                assert not req.is_draft
                req.cast_to_draft()
            else:
                assert req.is_draft

    def execute_draft_batch(self,
                            logits: torch.Tensor,
                            d2t: Optional[torch.Tensor] = None,
                            draft_step: int = 0) -> None:
        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.fetch_draft_batch(draft_step=draft_step)
            if draft_step == 0:
                self.rollback_rejected_tokens()
            self.build()
            if draft_step == self.max_num_draft_tokens - 1:
                self.rollback_draft_tokens()
            # Overwrite num_bitmask_tokens since the request might not be updated on CUDA stream yet.
            self.copy_bitmask(num_bitmask_tokens=len(self.requests))
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        # Overwrite num_bitmask_tokens since the request might not be updated on CUDA stream yet.
        self.apply_bitmask(logits,
                           d2t=d2t,
                           num_bitmask_tokens=len(self.requests))
