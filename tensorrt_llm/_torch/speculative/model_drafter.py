from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.handle_logits import HandleLogits
from ..pyexecutor.llm_request import (LlmRequest, LlmRequestState,
                                      get_draft_token_length)
from ..pyexecutor.resource_manager import BaseResourceManager, ResourceManager
from ..pyexecutor.sampler import Sampler, SampleState, TorchSampler
from ..pyexecutor.scheduler import ScheduledRequests
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from .drafter import Drafter

if TYPE_CHECKING:
    from ..pyexecutor.model_engine import ModelEngine
    from .interface import SpeculativeDecodingMode


# Place the tool function here to avoid circular import
def get_draft_model_prompt(spec_dec_mode: SpeculativeDecodingMode,
                           input_tokens: torch.Tensor) -> torch.Tensor:
    """
    Can be used to modify prompts for speculative algorithms that need to update tokens
    before drafting.
    """
    if spec_dec_mode.is_eagle3():
        # EAGLE3 always throws away the first token when processing draft inputs
        return input_tokens[1:]
    return input_tokens


class ModelDrafter(Drafter):
    """Model-based drafter that uses a draft model to generate draft tokens."""

    def __init__(
        self,
        spec_config: "DecodingBaseConfig",
        draft_model_engine: "ModelEngine",
        max_draft_tokens: int,
        draft_seq_slot_manager: SeqSlotManager,
        sampler: Sampler,
        spec_resource_manager: Optional[BaseResourceManager] = None,
        guided_decoder: Optional[GuidedDecoder] = None,
    ):
        super().__init__(spec_config.max_concurrency)

        # Validate required parameters
        if draft_model_engine is None:
            raise ValueError("draft_model_engine cannot be None")
        if max_draft_tokens < 0:
            raise ValueError(f"max_draft_tokens must be >= 0")

        # Model and resource management
        self.draft_model_engine = draft_model_engine
        self.draft_seq_slot_manager = draft_seq_slot_manager
        self.spec_resource_manager = spec_resource_manager

        # Configuration
        self.spec_config = spec_config
        self.max_draft_tokens = max_draft_tokens
        # Sampling
        self.sampler = sampler
        self._request_draft_logits = False
        if isinstance(sampler, TorchSampler):
            self._request_draft_logits = sampler.enable_mixed_sampler
        self.guided_decoder = guided_decoder

    def _create_draft_request(self, request: LlmRequest,
                              input_tokens: Optional[List]) -> LlmRequest:
        """Create a draft request with common parameters."""
        return LlmRequest(input_tokens=input_tokens,
                          request_id=request.py_request_id,
                          max_new_tokens=request.py_max_new_tokens,
                          sampling_config=request.sampling_config,
                          guided_decoding_params=request.guided_decoding_params,
                          target_seq_slot=request.py_seq_slot,
                          return_perf_metrics=request.return_perf_metrics,
                          is_streaming=False,
                          is_draft=True,
                          return_generation_logits=self._request_draft_logits)

    def _initialize_draft_tokens(self, request: LlmRequest) -> Tuple[int, int]:
        """Initialize draft token tracking for a request."""
        num_draft_tokens = len(
            request.py_last_draft_tokens
        ) if request.py_last_draft_tokens is not None else 0
        request.py_draft_tokens = []

        num_accepted_tokens = request.py_num_accepted_draft_tokens
        num_rejected_tokens = num_draft_tokens - num_accepted_tokens
        assert num_rejected_tokens >= 0

        return num_draft_tokens, num_accepted_tokens

    def _create_context_request(self, request: LlmRequest,
                                input_tokens: Any) -> LlmRequest:
        """Create a context request for first-time drafting."""
        new_request = self._create_draft_request(request, input_tokens)

        begin_compute, end_compute = request.py_last_context_chunk
        if begin_compute is not None:
            new_request.context_current_position = begin_compute
            new_request.context_chunk_size = end_compute - begin_compute
        return new_request

    def _create_generation_request(self, request: LlmRequest,
                                   input_tokens: Any) -> LlmRequest:
        """Create a generation request when no tokens were accepted."""
        new_request = self._create_draft_request(request, input_tokens)
        new_request.state = LlmRequestState.GENERATION_IN_PROGRESS
        return new_request

    def _create_accepted_tokens_request(self, request: LlmRequest,
                                        input_tokens: Any,
                                        num_accepted_tokens: int) -> LlmRequest:
        """
        Create a chunked context request for accepted tokens.
        Only applicable if the draft model needs to recompute KV cache for accepted tokens (e.g. eagle 3)
        """
        new_request = self._create_draft_request(request, input_tokens)
        new_request.context_chunk_size = num_accepted_tokens + 1
        new_request.context_current_position = len(
            input_tokens) - num_accepted_tokens - 1
        return new_request

    def _create_draft_request_for_request(
            self, request: LlmRequest) -> Optional[LlmRequest]:
        """Create a draft request based on the original request state."""
        num_draft_tokens, num_accepted_tokens = self._initialize_draft_tokens(
            request)
        input_tokens = get_draft_model_prompt(self.spec_config.spec_dec_mode,
                                              request.get_tokens(0))

        # First time seeing this request - context request
        if request.max_beam_num_tokens - 1 == request.py_prompt_len:
            # This is the first time the draft model is seeing this request.
            # Prepare a context request. We discard the first token and take
            # the newly decoded one - this is the convention for EAGLE 2 and 3.
            assert num_draft_tokens == 0
            return self._create_context_request(request, input_tokens)

        # No tokens accepted - generation request
        elif num_accepted_tokens == 0:
            return self._create_generation_request(request, input_tokens)

        # Tokens accepted - chunked context request
        else:
            return self._create_accepted_tokens_request(request, input_tokens,
                                                        num_accepted_tokens)

    def _add_to_draft_batch(self, draft_batch: ScheduledRequests,
                            draft_request: LlmRequest,
                            original_request: LlmRequest) -> None:
        """Add the draft request to the appropriate batch list."""
        # Copy additional properties
        draft_request.py_stop_words_list = original_request.py_stop_words_list

        # Add to appropriate batch based on request type
        if draft_request.state == LlmRequestState.GENERATION_IN_PROGRESS:
            draft_batch.generation_requests.append(draft_request)
        else:
            draft_batch.context_requests.append(draft_request)

    @nvtx_range("_prepare_draft_batch")
    def _prepare_draft_batch(
            self, scheduled_requests: ScheduledRequests) -> ScheduledRequests:
        """
        Prepares a batch for the draft model engine. Draft tokens are only produced
        for generation requests.

        The requests are prepared as follows:
        1. The first time the draft engine sees a request, it's a context request.
        2. Otherwise, if draft tokens were accepted on the last target model decoding
        step, it's a chunked context request (we process all the accepted tokens together).
        3. Otherwise, it's a generation request.

        Args:
            scheduled_requests: The scheduled requests to prepare draft batch for

        Returns:
            ScheduledRequests: The prepared draft batch
        """
        try:
            draft_batch = ScheduledRequests()

            for request in scheduled_requests.context_requests:
                if request.is_first_context_chunk:
                    # Ignore requests which still need to be processed by the target model.
                    continue

                # We hit this path if we're doing chunked prefill. The target model processed
                # a prefill chunk on the last iteration. Now, we need to fill in the KV cache
                # for the draft model too.
                all_tokens = request.get_tokens(0)
                input_tokens = get_draft_model_prompt(
                    self.spec_config.spec_dec_mode, all_tokens)

                new_request = self._create_context_request(
                    request, input_tokens)
                self._add_to_draft_batch(draft_batch, new_request, request)

            for request in scheduled_requests.generation_requests:
                if request.py_draft_pages_allocated == 0:
                    # No space for draft tokens
                    continue
                # Stop drafting when we hit the max seqlen. We still need dummy draft
                # tokens attached to the requests to make sure everything works properly
                # with CUDA graph. These dummy tokens are already added by
                # _prepare_draft_requests to make the KV cache/scheduler aware of the fact
                # that we want to do spec decoding, so no need to do anything else here.
                # This makes the perf for this case suboptimal, but that's OK - this is
                # a corner case for weird models like the llama 3.1 8b EAGLE3 implementation.
                if request.max_beam_num_tokens - 1 >= self.draft_model_engine.max_seq_len:
                    continue

                draft_request = self._create_draft_request_for_request(request)
                if draft_request is not None:
                    self._add_to_draft_batch(draft_batch, draft_request,
                                             request)

            return draft_batch

        except Exception as e:
            logger.error(f"Error in _prepare_draft_batch: {str(e)}")
            traceback.print_exc()
            raise e

    def _should_disable_cuda_graph(
            self, previous_batch: Optional[SampleState]) -> bool:
        """Check if CUDA graph should be disabled for the current forward pass."""
        if previous_batch is not None:
            return False
        return self.spec_config.spec_dec_mode.needs_kv_cache_recompute()

    def _forward_draft_model(
            self,
            draft_batch: ScheduledRequests,
            resource_manager: ResourceManager,
            previous_batch: Optional[SampleState] = None) -> Dict[str, Any]:
        """Forward pass through the draft model."""
        if self._should_disable_cuda_graph(previous_batch):
            with self.draft_model_engine.no_cuda_graph():
                outputs = self.draft_model_engine.forward(
                    draft_batch, resource_manager)
        else:
            new_tensors_device = previous_batch.device if previous_batch else None
            outputs = self.draft_model_engine.forward(
                draft_batch,
                resource_manager,
                new_tensors_device=new_tensors_device)

        # Handle d2t data if available
        if hasattr(self.draft_model_engine.model.model, 'd2t'):
            outputs['d2t'] = self.draft_model_engine.model.model.d2t.data

        return outputs

    def _sample_async(self, draft_batch: ScheduledRequests,
                      outputs: Dict[str, Any]) -> Optional[SampleState]:
        """Sample tokens from draft model outputs."""
        try:
            if self.sampler is not None:
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                for request in draft_batch.context_requests:
                    prefix_sum += request.context_chunk_size if request.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)

                HandleLogits()(
                    draft_batch.context_requests,
                    draft_batch.generation_requests, outputs["logits"],
                    self.sampler.beam_width(draft_batch.all_requests()),
                    num_context_logits_prefix_sum,
                    self.sampler.is_generation_model())

                return self.sampler.sample_async(draft_batch, outputs,
                                                 num_context_logits_prefix_sum)
            return None
        except Exception as e:
            logger.error(f"Error in sampling: {str(e)}")
            return None

    def _update_request_states(self,
                               scheduled_requests: ScheduledRequests) -> None:
        """Update request states after processing."""
        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS

    def _update_requests(self, sample_state: SampleState) -> None:
        """Update requests with sample state."""
        if self.sampler is not None:
            self.sampler.update_requests(sample_state)

    def _process_decoded_tokens(
            self, draft_batch: ScheduledRequests,
            req_id_to_old_request: Dict[int, LlmRequest]) -> List[LlmRequest]:
        """Process decoded tokens and determine which requests to continue processing."""
        new_requests = []
        for req in draft_batch.all_requests():
            target_model_req = req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                # This is a chunked prefill request and we have more prefill chunks
                # to process. Defer adding draft tokens until the whole prompt is processed.
                self.draft_seq_slot_manager.free_resources(req)
                continue

            target_model_req.py_draft_tokens.append(req.get_last_tokens(0))
            if self._request_draft_logits:
                target_model_req.py_draft_logits = req.py_result.generation_logits
            if req.state != LlmRequestState.GENERATION_COMPLETE and len(
                    target_model_req.py_draft_tokens
            ) < target_model_req.py_draft_pages_allocated:
                new_requests.append(req)
            else:
                self.draft_seq_slot_manager.free_resources(req)

        return new_requests

    def _pad_to_max_draft_tokens(self,
                                 scheduled_requests: ScheduledRequests) -> None:
        """Pad draft tokens to maximum length for all generation requests."""
        for req in scheduled_requests.generation_requests:
            max_draft_tokens = self.max_draft_tokens
            num_draft_tokens = get_draft_token_length(req)
            req.py_draft_tokens.extend(
                0 for _ in range(max_draft_tokens - num_draft_tokens))

    def _execute_guided_decoder(self,
                                scheduled_batch: ScheduledRequests,
                                logits: torch.Tensor,
                                d2t: Optional[torch.Tensor] = None):
        if self.guided_decoder is not None:
            self.guided_decoder.build(scheduled_batch)
            self.guided_decoder.execute(scheduled_batch, logits, d2t=d2t)

    @nvtx_range("prepare_draft_tokens")
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        request_mapping: Optional[dict[int, LlmRequest]] = None,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare draft tokens for the scheduled requests.

        Args:
            scheduled_requests: The scheduled requests for this iteration
            resource_manager: The resource manager for this iteration
        """
        if not self.draft_model_engine:
            raise ValueError("Draft model engine is not set")

        if resource_manager is None:
            raise ValueError("Resource manager is required")

        try:
            draft_batch = self._prepare_draft_batch(scheduled_requests)

            if draft_batch.batch_size == 0:
                return

            self.draft_seq_slot_manager.prepare_resources(draft_batch)

            req_id_to_old_request = {
                req.py_request_id: req
                for req in scheduled_requests.all_requests()
            }

            # Initial forward pass
            outputs = self._forward_draft_model(draft_batch, resource_manager)
            self._execute_guided_decoder(draft_batch,
                                         outputs['logits'],
                                         d2t=outputs.get('d2t'))
            sample_state = self._sample_async(draft_batch, outputs)
            previous_batch = sample_state

            self._update_request_states(draft_batch)

            # Convert context requests to generation requests
            draft_batch.generation_requests = draft_batch.context_requests + draft_batch.generation_requests
            draft_batch.context_requests = []

            # Generate remaining draft tokens iteratively
            for i in range(self.max_draft_tokens - 1):
                if len(draft_batch.generation_requests) == 0:
                    break

                outputs = self._forward_draft_model(draft_batch,
                                                    resource_manager,
                                                    previous_batch)
                if previous_batch is not None:
                    self._update_requests(previous_batch)
                self._execute_guided_decoder(draft_batch,
                                             outputs['logits'],
                                             d2t=outputs.get('d2t'))
                sample_state = self._sample_async(draft_batch, outputs)
                self._update_request_states(draft_batch)
                if previous_batch is not None:
                    new_requests = self._process_decoded_tokens(
                        previous_batch.scheduled_requests,
                        req_id_to_old_request)
                else:
                    new_requests = []
                draft_batch.generation_requests = new_requests
                previous_batch = sample_state

            # Final cleanup
            if previous_batch is not None:
                self._update_requests(previous_batch)
                self._process_decoded_tokens(previous_batch.scheduled_requests,
                                             req_id_to_old_request)
            self._pad_to_max_draft_tokens(scheduled_requests)

            if self.guided_decoder is not None:
                self.guided_decoder.rollback_draft_tokens(scheduled_requests)

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            raise e
