import math
from dataclasses import dataclass, field
from queue import Queue
from typing import Iterable, List, Optional, Tuple

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
        seq_slot = request.py_target_seq_slot if request.py_is_draft else request.py_seq_slot
        return cls(
            guided_decoding_params=request.guided_decoding_params,
            request_id=request.py_request_id,
            seq_slot=seq_slot,
            is_context_init_state=request.is_context_init_state,
            is_last_context_chunk=request.is_context_init_state
            and request.is_last_context_chunk,
            is_generation_in_progress_state=request.
            is_generation_in_progress_state,
            new_token=request.get_last_tokens(0),
            is_draft=request.py_is_draft,
            draft_tokens=request.py_draft_tokens,
            num_accepted_draft_tokens=request.py_num_accepted_draft_tokens)


class GuidedRequestBatch:

    def __init__(self,
                 scheduled_requests: ScheduledRequests,
                 max_num_draft_tokens: int = 0):
        self.guided_requests = [
            GuidedRequest.from_request(req)
            for req in scheduled_requests.all_requests()
        ]
        self.num_contexts = len(scheduled_requests.context_requests)
        self.num_generations = len(scheduled_requests.generation_requests)
        self.max_num_draft_tokens = max_num_draft_tokens

    @property
    def num_bitmask_tokens(self) -> int:
        return self.num_contexts + self.num_generations * (
            self.max_num_draft_tokens + 1)

    def guided_request_with_offset(self) -> Iterable[Tuple[GuidedRequest, int]]:
        offset: int = 0
        for req in self.guided_requests:
            yield req, offset
            offset += 1
            if req.is_generation_in_progress_state:
                offset += self.max_num_draft_tokens


@dataclass(slots=True)
class GuidedMetadata:
    """Metadata for guided decoding."""
    # No race condition
    # 1. Overlap scheduler disabled: sample_state is synced to host, so the last-iteration hostfunc has finished.
    # 2. Overlap scheduler enabled: D2H copy depends on new_tensors_device, which depends on the last-iteration hostfunc results.
    max_num_draft_tokens: int = field(default=0)

    queue: Optional[Queue] = field(default=None, init=False)
    guided_requests_device: Optional[GuidedRequestBatch] = field(default=None,
                                                                 init=False)
    guided_requests_host: Optional[GuidedRequestBatch] = field(default=None,
                                                               init=False)

    # gathered input_ids at the current iteration.
    gathered_input_ids: Optional[torch.Tensor] = field(default=None, init=False)
    # num_accepted_tokens at the last iteration.
    new_tokens_lens: Optional[torch.Tensor] = field(default=None, init=False)

    token_event: Optional[torch.cuda.Event] = field(default=None, init=False)
    bitmask_event: Optional[torch.cuda.Event] = field(default=None, init=False)

    def __post_init__(self):
        self.queue = Queue()
        self.token_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

    def add_batch(self, scheduled_requests: ScheduledRequests):
        self.guided_requests_host = GuidedRequestBatch(
            scheduled_requests, self.max_num_draft_tokens)
        self.queue.put(self.guided_requests_host)

    @hostfunc
    def next_batch_hostfunc(self):
        # Fix it.
        if self.queue.empty():
            return
        self.guided_requests_device = self.queue.get()


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

    @torch.inference_mode()
    @nvtx_range("GuidedDecoder.build")
    def build(self, guided_metadata: GuidedMetadata) -> None:
        """Build the bitmask for requests with guided decoding enabled.

        Specifically, this method:
        - build and advance the grammar matcher for context and generation requests, respectively;
        - call the grammar matcher to fill the bitmask on CPU;
        - asynchronously copy the bitmask to GPU.
        """
        gathered_input_ids = guided_metadata.gathered_input_ids
        if gathered_input_ids is not None:
            gathered_input_ids = gathered_input_ids.tolist()

        # Fix it.
        num_bitmask_tokens = guided_metadata.guided_requests_device.num_bitmask_tokens
        self.bitmask_host[:num_bitmask_tokens].fill_(-1)

        for req, offset in guided_metadata.guided_requests_device.guided_request_with_offset(
        ):
            if (slot := req.seq_slot) is None:
                continue
            self.num_advanced_tokens[slot] = 0
            self.num_guided_tokens[slot] = 0

            if gathered_input_ids is None:
                new_token = req.new_token
                draft_tokens = req.draft_tokens
            else:
                new_token = gathered_input_ids[offset]
                draft_tokens = []
                if req.is_generation_in_progress_state:
                    draft_tokens = gathered_input_ids[offset + 1:offset + self.
                                                      max_num_draft_tokens + 1]

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
                accepted = matcher.accept_token(new_token)
                if not accepted:
                    if req.is_draft:
                        self.is_draft_terminated[slot] = True
                        logger.debug(
                            f"Draft request {req.request_id} at slot {slot} failed to accept last new token: {new_token}."
                        )
                        continue
                    # TODO: Make this an error response.
                    raise ValueError(
                        f"Request {req.request_id} at slot {slot} failed to accept last new token: {new_token}."
                    )

            self.num_advanced_tokens[slot] += 1
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.bitmask_host, offset)
                self.num_guided_tokens[slot] += 1
                # Process draft tokens
                for i, tid in enumerate(draft_tokens, 1):
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
                assert len(draft_tokens) == 0
                self.num_advanced_draft_tokens[
                    slot] += self.num_advanced_tokens[slot]

    def bitmask_copy(self, guided_metadata: GuidedMetadata) -> None:
        # Fix it.
        num_bitmask_tokens = guided_metadata.guided_requests_host.num_bitmask_tokens
        self.bitmask[:num_bitmask_tokens].copy_(
            self.bitmask_host[:num_bitmask_tokens], non_blocking=True)

    @torch.inference_mode()
    @nvtx_range("GuidedDecoder.execute")
    def execute(self,
                guided_metadata: GuidedMetadata,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        """Apply the bitmask to the corresponding logits for requests with guided decoding enabled.

        This method inplace modifies the logits tensor so that any tokens that violate the grammar constraints are masked out.
        """
        # TODO: Fuse index_copy and index_select to logits_bitmask.
        if d2t is not None:
            draft_logits = logits
            d2t_mapping = d2t + torch.arange(d2t.size(0), device=d2t.device)
            logits = torch.empty(draft_logits.size(0),
                                 self.vocab_size_padded,
                                 dtype=draft_logits.dtype,
                                 device=draft_logits.device)
            logits.index_copy_(-1, d2t_mapping, draft_logits)

        num_bitmask_tokens = guided_metadata.guided_requests_host.num_bitmask_tokens
        xgrammar.apply_token_bitmask_inplace(logits[:num_bitmask_tokens],
                                             self.bitmask[:num_bitmask_tokens])

        if d2t is not None:
            torch.index_select(logits, -1, d2t_mapping, out=draft_logits)

    @nvtx_range("GuidedDecoder.rollback_rejected_tokens")
    def rollback_rejected_tokens(self, guided_metadata: GuidedMetadata) -> None:
        """Rollback the grammar matcher for rejected tokens.

        This method should be called:
        - after the verification (so that the accepted tokens are ready) and
        - before the first guided decoding build of the next drafting loop.
        """
        if self.max_num_draft_tokens <= 0:
            return

        new_tokens_lens = guided_metadata.new_tokens_lens
        if new_tokens_lens is not None:
            new_tokens_lens = new_tokens_lens.tolist()

        for req in guided_metadata.guided_requests_device.guided_requests:
            assert not req.is_draft
            if (slot := req.seq_slot) is None:
                continue
            if self.num_advanced_tokens[slot] <= 0:
                continue
            if new_tokens_lens is None:
                num_accepted_tokens = 1 + req.num_accepted_draft_tokens
            else:
                num_accepted_tokens = new_tokens_lens[slot]
            # Rollback the grammar matcher to the last accepted token.
            num_rollback_tokens = self.num_advanced_tokens[
                slot] - num_accepted_tokens
            # TODO: Make this an error response.
            if num_rollback_tokens < 0:
                raise ValueError(
                    f"Failed to rollback: num_advanced_tokens={self.num_advanced_tokens[slot]}, num_accepted_tokens={num_accepted_tokens}, num_rollback_tokens={num_rollback_tokens}"
                )
            self.grammar_matchers[slot].rollback(num_rollback_tokens)

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
    def build_hostfunc(self, guided_metadata: GuidedMetadata):
        self.build(guided_metadata)

    @hostfunc
    def rollback_rejected_tokens_hostfunc(self,
                                          guided_metadata: GuidedMetadata):
        self.rollback_rejected_tokens(guided_metadata)

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
