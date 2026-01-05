from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from ..attention_backend.trtllm import TrtllmAttention
from ..pyexecutor.guided_decoder import GuidedDecoder
from ..pyexecutor.handle_logits import HandleLogits
from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import (BaseResourceManager, ResourceManager,
                                           ResourceManagerType)
from ..pyexecutor.sampler import Sampler, SampleState, SampleStateTensors
from ..pyexecutor.scheduler import ScheduledRequests
from ..pyexecutor.seq_slot_manager import SeqSlotManager
from ..speculative.mtp import SampleStateTensorsMTP
from .drafter import Drafter

if TYPE_CHECKING:
    from ...llmapi.llm_args import DecodingBaseConfig
    from ..pyexecutor.model_engine import ModelEngine
    from .interface import SpeculativeDecodingMode


# Place the tool function here to avoid circular import
def get_draft_model_prompt(spec_dec_mode: SpeculativeDecodingMode,
                           request: LlmRequest,
                           disable_overlap_scheduler: bool) -> List[int]:
    """
    Can be used to modify prompts for speculative algorithms that need to update tokens
    before drafting.
    """
    draft_input_tokens = request.get_tokens(0)
    if spec_dec_mode.is_eagle3() or spec_dec_mode.is_mtp_eagle():
        # EAGLE3 always throws away the first token when processing draft inputs
        if not disable_overlap_scheduler:
            # Add a fake golden token here since the real one has not been generated.
            draft_input_tokens.extend([0])
        draft_input_tokens = draft_input_tokens[1:]

    if request.is_context_init_state:
        # A draft request's prompt is its target request's prompt adding the first golden token.
        # Add a fake golden token here since the real one has not been generated.
        draft_input_tokens.append(0)
    return draft_input_tokens


class ModelDrafter(Drafter):
    """Model-based drafter that uses a draft model to generate draft tokens."""

    def __init__(
        self,
        spec_config: "DecodingBaseConfig",
        draft_model_engine: "ModelEngine",
        max_draft_len: int,
        max_total_draft_tokens: int,
        draft_seq_slot_manager: SeqSlotManager,
        sampler: Sampler,
        spec_resource_manager: Optional[BaseResourceManager] = None,
        guided_decoder: Optional[GuidedDecoder] = None,
    ):
        # Validate required parameters
        if draft_model_engine is None:
            raise ValueError("draft_model_engine cannot be None")
        if max_draft_len < 0:
            raise ValueError("max_draft_len must be >= 0")
        if max_total_draft_tokens < 0:
            raise ValueError("max_total_draft_tokens must be >= 0")
        assert max_draft_len <= max_total_draft_tokens

        super().__init__(max_draft_len=max_draft_len,
                         max_total_draft_tokens=max_total_draft_tokens,
                         max_concurrency=spec_config.max_concurrency,
                         draft_len_schedule=spec_config.draft_len_schedule)

        # Model and resource management
        self.draft_model_engine = draft_model_engine
        self.draft_seq_slot_manager = draft_seq_slot_manager
        self.spec_resource_manager = spec_resource_manager
        self.disable_overlap_scheduler = True

        # Configuration
        self.spec_config = spec_config

        # Sampling
        self.sampler = sampler
        self.guided_decoder = guided_decoder

        self.use_static_draft_loop = draft_model_engine.model_is_wrapped
        if self.use_static_draft_loop:
            # TODO: enable sampling/guided decoding on static draft loop
            assert guided_decoder is None
            assert spec_config._allow_greedy_draft_tokens
            assert spec_config.draft_len_schedule is None

        # Create accumulator for draft tokens in non-CDL mode
        self.draft_tokens_accumulator: Dict[int, List[int]] = {}

        # Initialize draft latency tracking for specDecodingStats
        self.last_draft_latency_ms = 0.0
        # Track previous draft batch for overlap scheduling
        self.previous_draft_batch: Optional[ScheduledRequests] = None
        self.previous_draft_outputs: Optional[Any] = None
        self.previous_scheduled_batch: Optional[ScheduledRequests] = None
        # Map from request ID to original request
        self.req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None

    def _create_draft_request(self, request: LlmRequest,
                              input_tokens: Optional[List]) -> LlmRequest:
        """Create a draft request with common parameters."""
        return LlmRequest(
            input_tokens=input_tokens,
            request_id=request.py_request_id,
            max_new_tokens=request.py_max_new_tokens,
            sampling_config=request.sampling_config,
            guided_decoding_params=request.guided_decoding_params,
            target_seq_slot=request.py_seq_slot,
            return_perf_metrics=request.return_perf_metrics,
            is_streaming=False,
            exclude_last_generation_logits=
            True,  # prepare_draft_tokens uses overlap scheduling
            is_draft=True,
            # NB: self.sampler is shared with PyExecutor
            return_generation_logits=self.sampler.should_provide_draft_probs(
                request))

    def _initialize_draft_tokens(self, request: LlmRequest) -> Tuple[int, int]:
        """Initialize draft token tracking for a request."""
        if not self.disable_overlap_scheduler:
            return self.max_draft_len, 0

        num_draft_tokens = len(
            request.py_last_draft_tokens
        ) if request.py_last_draft_tokens is not None else 0

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

    def _get_previous_draft_request(
            self, request: LlmRequest) -> Optional[LlmRequest]:
        """Get the previous draft request for the given request."""
        if self.previous_draft_batch is None:
            return None
        for req in self.previous_draft_batch.all_requests():
            if req.py_request_id == request.py_request_id:
                return req
        return None

    def _create_accepted_tokens_request_for_trtllm_attn(
            self, request: LlmRequest, input_tokens: Any,
            num_accepted_tokens: int) -> LlmRequest:
        """
        Create a chunked context request for accepted tokens.
        Only applicable if the draft model needs to recompute KV cache for accepted tokens (e.g. eagle 3)
        """
        # Pad input_tokens to max_draft_len
        # We use max_draft_len instead of max_total_draft_tokens here,
        # because at most max_draft_len draft tokens are accepted.
        input_tokens.extend(
            0 for _ in range(self.max_draft_len - num_accepted_tokens))

        # Reuse the previous draft request if it exists.
        # This can reduce host overhead significantly.
        draft_request = self._get_previous_draft_request(request)
        if draft_request is not None:
            generated_tokens = input_tokens[draft_request.py_prompt_len:]
            draft_request.set_generated_tokens([generated_tokens])
        else:
            draft_request = self._create_draft_request(request, input_tokens)

        draft_request.state = LlmRequestState.GENERATION_IN_PROGRESS
        draft_request.py_num_accepted_draft_tokens = request.py_num_accepted_draft_tokens
        draft_request.py_is_first_draft = True
        # For tree decoding, we need to store the accepted tokens indices for these requests,
        # which will be used to update the hidden_states_read_indices.
        draft_request.py_num_accepted_draft_tokens_indices = request.py_num_accepted_draft_tokens_indices

        return draft_request

    def _create_draft_request_for_request(
            self, request: LlmRequest) -> Optional[LlmRequest]:
        """Create a draft request based on the original request state."""
        num_draft_tokens, num_accepted_tokens = self._initialize_draft_tokens(
            request)

        input_tokens = get_draft_model_prompt(self.spec_config.spec_dec_mode,
                                              request,
                                              self.disable_overlap_scheduler)

        is_eagle_style = self.spec_config.spec_dec_mode.is_eagle3(
        ) or self.spec_config.spec_dec_mode.is_mtp_eagle()

        # First time seeing this request - context request
        num_overlap_tokens = 0 if self.disable_overlap_scheduler else 1
        if request.max_beam_num_tokens - 1 + num_overlap_tokens == request.py_prompt_len:
            # This is the first time the draft model is seeing this request.
            # Prepare a context request. We discard the first token and take
            # the newly decoded one - this is the convention for EAGLE 2 and 3.
            if self.disable_overlap_scheduler:
                assert num_draft_tokens == 0
            return self._create_context_request(request, input_tokens)

        # For TRTLLM attention backend, we need to create a generation request for both no tokens accepted and tokens accepted
        elif issubclass(self.draft_model_engine.attn_backend, TrtllmAttention
                        ) and self.use_static_draft_loop and is_eagle_style:
            return self._create_accepted_tokens_request_for_trtllm_attn(
                request, input_tokens, num_accepted_tokens)

        # No tokens accepted - generation request. This only applies to speculation algorithms
        # that need to recompute KV cache for accepted tokens like eagle3.
        elif num_accepted_tokens == 0 or not self.spec_config.spec_dec_mode.needs_kv_cache_recompute(
        ):
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

        # Add to appropriate batch based on request typetensorrt_llm/_torch/speculative/model_drafter.py
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
            for req in scheduled_requests.all_requests():
                draft_model = self.draft_model_engine.model.draft_model if self.use_static_draft_loop else self.draft_model_engine.model
                if hasattr(draft_model.model, "d2t"):
                    req.d2t = draft_model.model.d2t.data
                req.py_draft_use_greedy_sampling = self.use_static_draft_loop

            draft_batch = ScheduledRequests()

            for request in scheduled_requests.context_requests:
                if request.py_disable_speculative_decoding:
                    continue
                if request.is_first_context_chunk:
                    # Ignore requests which still need to be processed by the target model.
                    continue

                # We hit this path if we're doing chunked prefill. The target model processed
                # a prefill chunk on the last iteration. Now, we need to fill in the KV cache
                # for the draft model too.
                input_tokens = get_draft_model_prompt(
                    self.spec_config.spec_dec_mode, request,
                    self.disable_overlap_scheduler)

                new_request = self._create_context_request(
                    request, input_tokens)
                self._add_to_draft_batch(draft_batch, new_request, request)

            for request in scheduled_requests.generation_requests:
                if request.py_disable_speculative_decoding:
                    continue
                if request.state == LlmRequestState.GENERATION_COMPLETE:
                    # Skip generation complete requests. This could happen when enabling overlap scheduler.
                    continue

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
                num_overlap_tokens = 0 if self.disable_overlap_scheduler else 1
                if request.max_beam_num_tokens - 1 + num_overlap_tokens >= self.draft_model_engine.max_seq_len:
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

    def _should_disable_cuda_graph(self, is_first_draft_token: bool) -> bool:
        """Check if CUDA graph should be disabled for the current forward pass."""
        if not is_first_draft_token:
            return False
        if self.use_static_draft_loop:
            return False
        return self.spec_config.spec_dec_mode.needs_kv_cache_recompute()

    @nvtx_range("forward_draft_model")
    def forward_draft_model(
        self,
        draft_batch: ScheduledRequests,
        resource_manager: ResourceManager,
        is_first_draft_token: bool,
        previous_tensors: Optional[SampleStateTensors] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Forward pass through the draft model."""
        if self._should_disable_cuda_graph(is_first_draft_token):
            with self.draft_model_engine.no_cuda_graph():
                outputs = self.draft_model_engine.forward(
                    draft_batch,
                    resource_manager,
                    new_tensors_device=previous_tensors,
                    num_accepted_tokens_device=num_accepted_tokens_device,
                    req_id_to_old_request=self.req_id_to_old_request)
        else:
            outputs = self.draft_model_engine.forward(
                draft_batch,
                resource_manager,
                new_tensors_device=previous_tensors,
                num_accepted_tokens_device=num_accepted_tokens_device,
                req_id_to_old_request=self.req_id_to_old_request)

        # Handle d2t data if available. Static drafting loops should incorporate d2t
        # in their implementations.
        if not self.use_static_draft_loop and hasattr(
                self.draft_model_engine.model.model, 'd2t'):
            outputs['d2t'] = self.draft_model_engine.model.model.d2t.data

        return outputs

    @nvtx_range("sample_async")
    def sample_async(
        self,
        draft_batch: ScheduledRequests,
        outputs: Dict[str, Any],
        resource_manager: Optional[ResourceManager] = None
    ) -> Optional[SampleState]:
        """Sample tokens from draft model outputs."""
        try:
            num_context_logits_prefix_sum = [0]
            prefix_sum = 0
            for request in draft_batch.context_requests:
                prefix_sum += request.context_chunk_size if request.py_return_context_logits else 1
                num_context_logits_prefix_sum.append(prefix_sum)

            HandleLogits()(draft_batch.context_requests,
                           draft_batch.generation_requests, outputs["logits"],
                           self.sampler.beam_width(draft_batch.all_requests()),
                           num_context_logits_prefix_sum,
                           self.sampler.is_generation_model())

            return self.sampler.sample_async(draft_batch, outputs,
                                             num_context_logits_prefix_sum,
                                             resource_manager)
        except Exception as e:
            logger.error(f"Error in sampling: {str(e)}")
            return None

    def update_request_states(self,
                              scheduled_requests: ScheduledRequests) -> None:
        """Update request states after processing."""
        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS

    def update_cur_draft_layer_idx(
            self,
            cur_draft_layer_idx: int,
            resource_manager: Optional[ResourceManager] = None):
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if spec_resource_manager is None:
            return None

        spec_tree_manager = spec_resource_manager.spec_tree_manager
        if spec_tree_manager is not None:
            spec_tree_manager.cur_draft_layer_idx = cur_draft_layer_idx

    def update_requests(
            self,
            sample_state: SampleState,
            resource_manager: Optional[ResourceManager] = None) -> None:
        """Update requests with sample state."""
        self.sampler.update_requests(sample_state, resource_manager)

    def process_decoded_tokens(
            self,
            draft_batch: ScheduledRequests,
            draft_position: int,
            cleanup_resources: bool = True) -> List[LlmRequest]:
        """Process decoded tokens and determine which requests to continue processing."""
        new_requests = []
        for req in draft_batch.all_requests():
            target_model_req = self.req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                # This is a chunked prefill request and we have more prefill chunks
                # to process. Defer adding draft tokens until the whole prompt is processed.
                self.draft_seq_slot_manager.free_resources(req)
                continue

            # Save tokens to accumulator instead of directly modifying target_model_req.py_draft_tokens
            if req.py_request_id not in self.draft_tokens_accumulator:
                self.draft_tokens_accumulator[
                    req.py_request_id] = [0] * self.max_total_draft_tokens
            self.draft_tokens_accumulator[req.py_request_id][
                draft_position - 1] = req.get_last_tokens(0)

            generation_logits = req.py_result.generation_logits  # forwards Nones
            if generation_logits is not None:
                # generation_logits returns [beam_width, seq_length, vocab_size]
                beam_width = generation_logits.size(0)
                assert beam_width == 1, f"expected beam_width=1, got {beam_width}"
                generation_logits.squeeze_(0)
                # Transfer to CUDA if needed (chunked LogitsStorage stores on CPU)
                if generation_logits.device.type == 'cpu':
                    generation_logits = generation_logits.to('cuda',
                                                             non_blocking=True)
            target_model_req.py_draft_logits = generation_logits

            if req.state != LlmRequestState.GENERATION_COMPLETE and draft_position < target_model_req.py_draft_pages_allocated:
                new_requests.append(req)
            else:
                if cleanup_resources:
                    self.draft_seq_slot_manager.free_resources(req)

        return new_requests

    def should_forward_draft_model(self,
                                   scheduled_batch: ScheduledRequests) -> bool:
        """
        Determine if the draft model should be forwarded for the given batch.

        Args:
            scheduled_batch: The scheduled requests to check

        Returns:
            bool: True if draft model should be forwarded, False otherwise
        """
        all_disable_speculative_decoding = True
        for request in scheduled_batch.all_requests():
            if not request.py_disable_speculative_decoding:
                all_disable_speculative_decoding = False
                break
        if all_disable_speculative_decoding:
            return False
        for request in scheduled_batch.context_requests:
            if request.is_first_context_chunk:
                continue
            return True

        for request in scheduled_batch.generation_requests:
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                continue

            if request.max_beam_num_tokens - 1 >= self.draft_model_engine.max_seq_len:
                continue
            return True

        return False

    def _initialize_draft_tokens_for_target_inputs(
        self,
        scheduled_batch: ScheduledRequests,
        target_inputs: Optional[SampleStateTensorsMTP] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None
    ) -> Optional[SampleStateTensorsMTP]:
        """
        Convert tensors for draft model processing.

        Args:
            scheduled_batch: The scheduled requests
            new_tensors_device: The device tensors to convert

        Returns:
            SampleStateTensorsMTP: Converted tensors or None
        """
        if target_inputs is None:
            return None
        # Get device from the new_tokens tensor
        device = target_inputs.new_tokens.device

        new_tokens_lens = None
        next_draft_tokens = None
        has_draft_tokens = False
        batch_size = target_inputs.new_tokens.shape[1]
        # Iterate through generation requests and copy tokens based on accepted draft tokens
        for request in scheduled_batch.all_requests():
            if request.state == LlmRequestState.GENERATION_IN_PROGRESS:
                has_draft_tokens = True

        if has_draft_tokens:
            # We already updated the target state, so the new_tokens_lens should be all ones.
            new_tokens_lens = torch.ones(batch_size,
                                         dtype=torch.int,
                                         device=device)
            new_tokens_lens += num_accepted_tokens_device
            next_draft_tokens = torch.zeros(batch_size,
                                            self.max_draft_len,
                                            dtype=torch.int,
                                            device=device)
        target_inputs.new_tokens_lens = new_tokens_lens
        target_inputs.next_draft_tokens = next_draft_tokens
        return target_inputs

    @nvtx_range("_update_draft_tokens_for_target_inputs")
    def _update_draft_tokens_for_target_inputs(
            self, target_inputs: SampleStateTensorsMTP,
            draft_tensors: Optional[torch.Tensor], draft_position: int,
            draft_length: int, draft_batch: ScheduledRequests) -> None:
        """
        Update target inputs with new draft tokens from sample state.
        """
        if target_inputs.next_draft_tokens is None:
            return

        draft_indices = []
        target_indices = []
        for req_idx, request in enumerate(draft_batch.all_requests()):
            target_req = self.req_id_to_old_request[request.py_request_id]
            if target_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                # Skip prefill requests
                continue
            # Get the index of the draft/target tokens in the device tensor
            draft_idx = req_idx if self.use_static_draft_loop else request.py_seq_slot
            target_idx = target_req.py_seq_slot
            draft_indices.append(draft_idx)
            target_indices.append(target_idx)

        if len(draft_indices) == 0:
            return

        device = draft_tensors.device

        # Create index tensors
        draft_indices_tensor = torch.tensor(draft_indices,
                                            dtype=torch.long,
                                            pin_memory=True).to(
                                                device, non_blocking=True)
        target_indices_tensor = torch.tensor(target_indices,
                                             dtype=torch.long,
                                             pin_memory=True).to(
                                                 device, non_blocking=True)

        # Pre-slice draft tensors: [draft_length, batch_size]
        draft_slice = draft_tensors[0:draft_length]

        # Gather all source data at once using single index_select kernel
        # Result shape: [draft_length, num_requests]
        gathered = draft_slice.index_select(1, draft_indices_tensor).to(
            torch.int32)

        # Scatter to new_tokens using advanced indexing (single kernel)
        # Shape: [draft_length, num_requests] -> [seq_len, batch_size, beam_width]
        target_inputs.new_tokens[draft_position + 1:draft_position +
                                 draft_length + 1, target_indices_tensor,
                                 0] = gathered

        # Scatter to next_draft_tokens using advanced indexing (single kernel)
        # Shape: [num_requests, draft_length] -> [batch_size, max_draft_len]
        target_inputs.next_draft_tokens[target_indices_tensor,
                                        draft_position:draft_position +
                                        draft_length] = gathered.t()

    def _setup_draft_batch_and_resources(
            self,
            scheduled_batch: ScheduledRequests) -> Optional[ScheduledRequests]:
        """
        Setup draft batch and prepare resources.

        Args:
            scheduled_batch: The scheduled requests

        Returns:
            draft_batch or None if no batch
        """

        draft_batch = self._prepare_draft_batch(scheduled_batch)
        if draft_batch.batch_size == 0:
            return None

        self.req_id_to_old_request = {
            req.py_request_id: req
            for req in scheduled_batch.all_requests()
        }

        for request in draft_batch.all_requests():
            target_model_req = self.req_id_to_old_request[request.py_request_id]
            if target_model_req.is_context_init_state:
                continue
            target_model_req.py_draft_tokens = [0] * self.max_draft_len

        self.draft_seq_slot_manager.prepare_resources(draft_batch)
        return draft_batch

    def process_static_draft_outputs(self, outputs: dict[str, torch.Tensor]
                                     | tuple[torch.Tensor, SampleState],
                                     draft_batch: ScheduledRequests) -> None:
        """
        Process outputs from static draft loop, update target requests, and clean up resources.

        Args:
            outputs: The outputs from the draft model
            draft_batch: The draft batch that was processed
        """

        if isinstance(outputs, dict):
            draft_tokens_host = outputs["new_draft_tokens"].cpu()
            draft_logits = outputs["draft_logits"]
        else:
            draft_logits = outputs[0]
            draft_tokens_host = outputs[1].host.new_tokens
            outputs[1].sampler_event.synchronize()

        for req_idx, req in enumerate(draft_batch.all_requests()):
            target_model_req = self.req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                # Chunked prefill request in progress; no need to append draft tokens
                continue
            target_model_req.py_draft_tokens = []
            py_draft_logits = []
            for token_idx in range(self.max_total_draft_tokens):
                target_model_req.py_draft_tokens.append(
                    draft_tokens_host[token_idx][req_idx])
                py_draft_logits.append(draft_logits[token_idx][req_idx])

            # The overlap scheduler doesn't support rejection sampling yet, so we don't update the py_draft_logits to get it fallback to greedy sampling.
            if self.disable_overlap_scheduler:
                target_model_req.py_draft_logits = torch.stack(py_draft_logits)

    def process_dynamic_draft_outputs(
            self,
            outputs: Any,
            resource_manager: Optional[ResourceManager] = None,
            cleanup_resources: bool = True) -> None:
        """
        Process outputs from dynamic draft loop, update target requests, and clean up resources.
        """
        self.update_requests(outputs, resource_manager)

        # Create accumulator for draft tokens and process them
        self.process_decoded_tokens(outputs.scheduled_requests,
                                    self.max_draft_len, cleanup_resources)

        # Update py_draft_tokens after processing
        for req_id, tokens in self.draft_tokens_accumulator.items():
            target_model_req = self.req_id_to_old_request[req_id]
            target_model_req.py_draft_tokens = tokens

    @nvtx_range("_execute_draft_iteration")
    def _execute_draft_iteration(
        self,
        draft_batch: ScheduledRequests,
        resource_manager: ResourceManager,
        previous_draft_state: Optional[SampleState],
        cur_draft_layer_idx: int,
        num_accepted_tokens_device: Optional[torch.Tensor] = None
    ) -> Tuple[Any, Optional[SampleState]]:
        self.update_cur_draft_layer_idx(
            cur_draft_layer_idx, resource_manager
        )  # Update the current draft layer index in the resource manager.
        """Forward pass through the draft model."""
        outputs = self.forward_draft_model(
            draft_batch,
            resource_manager,
            is_first_draft_token=False,
            previous_tensors=previous_draft_state.device
            if previous_draft_state else None,
            num_accepted_tokens_device=num_accepted_tokens_device)

        if previous_draft_state is not None:
            self.update_requests(previous_draft_state, resource_manager)

        if self.guided_decoder is not None:
            self.guided_decoder.add_batch(draft_batch)
            self.guided_decoder.execute(outputs['logits'],
                                        d2t=outputs.get('d2t'))

        sample_state = self.sample_async(draft_batch, outputs, resource_manager)
        self.update_request_states(draft_batch)

        return outputs, sample_state

    @nvtx_range("_process_previous_draft_results")
    def _process_previous_draft_results(self) -> None:
        """
        Process the previous draft batch results.
        This should be called after the current draft forward to enable overlap scheduling.
        """
        if (self.previous_draft_batch is None
                or self.previous_draft_outputs is None
                or self.previous_scheduled_batch is None):
            return

        # Save current req_id_to_old_request temporarily
        current_req_id_to_old_request = self.req_id_to_old_request

        # Set req_id_to_old_request for the previous batch,
        # this will be used in process_static_draft_outputs and process_dynamic_draft_outputs
        self.req_id_to_old_request = {
            req.py_request_id: req
            for req in self.previous_scheduled_batch.all_requests()
        }

        if self.use_static_draft_loop:
            self.process_static_draft_outputs(self.previous_draft_outputs,
                                              self.previous_draft_batch)
        elif self.previous_draft_outputs is not None:
            self.process_dynamic_draft_outputs(self.previous_draft_outputs,
                                               cleanup_resources=False)

        self.req_id_to_old_request = current_req_id_to_old_request

        # Pad draft tokens to the max draft length for CUDA graph compatibility
        self.pad_draft_tokens_for_cuda_graph(self.previous_scheduled_batch)

    def cleanup_previous_draft_resources(self) -> None:
        if self.previous_draft_batch is None:
            return

        # Free draft_seq_slot_manager resources for all requests in the previous draft batch
        for req in self.previous_draft_batch.all_requests():
            self.draft_seq_slot_manager.free_resources(req)

    def _execute_draft_loop(
        self,
        draft_batch: ScheduledRequests,
        resource_manager: ResourceManager,
        target_inputs: Optional[SampleStateTensorsMTP] = None,
        num_draft_reqs: Optional[int] = None,
        initial_draft_state: Optional[SampleState] = None
    ) -> Optional[SampleState]:
        """
        Execute the iterative draft loop.

        Args:
            draft_batch: The draft batch to process
            resource_manager: The resource manager
            target_inputs: Optional target inputs to update (for overlap mode)
            num_draft_reqs: Number of draft requests (for overlap mode)
            initial_draft_state: The initial draft state from the first forward pass

        Returns:
            The final sample state
        """
        # Convert context requests to generation requests
        for req in draft_batch.generation_requests:
            req.py_is_first_draft = False
        draft_batch.generation_requests = draft_batch.context_requests + draft_batch.generation_requests
        draft_batch.context_requests = []

        previous_draft_state = initial_draft_state
        # reset draft tokens accumulator
        self.draft_tokens_accumulator = {}
        # Generate remaining draft tokens iteratively
        for i in range(self.max_draft_len - 1):
            if len(draft_batch.generation_requests) == 0:
                break

            _, sample_state = self._execute_draft_iteration(
                draft_batch, resource_manager, previous_draft_state, i + 1)

            # Update target inputs if provided (for overlap mode)
            if target_inputs is not None and num_draft_reqs is not None:
                draft_tensors = sample_state and sample_state.device and sample_state.device.new_tokens
                self._update_draft_tokens_for_target_inputs(
                    target_inputs,
                    draft_tensors,
                    draft_position=i + 1,
                    draft_length=1,
                    draft_batch=draft_batch)

            if sample_state is not None and previous_draft_state is not None:
                new_requests = self.process_decoded_tokens(
                    previous_draft_state.scheduled_requests,
                    draft_position=i + 1)
            else:
                new_requests = []

            draft_batch.generation_requests = new_requests
            previous_draft_state = sample_state

        return previous_draft_state

    @nvtx_range("generate_draft_tokens_with_overlap")
    def generate_draft_tokens_with_overlap(
            self,
            scheduled_batch: ScheduledRequests,
            resource_manager: ResourceManager,
            previous_tensors: Optional[SampleStateTensors],
            target_inputs: Optional[SampleStateTensorsMTP],
            num_accepted_tokens_device: Optional[torch.Tensor] = None) -> None:
        """
        Generate draft tokens with overlap scheduling support.

        Args:
            scheduled_batch: The scheduled requests
            resource_manager: The resource manager
            previous_tensors: Previous iteration tensors
            guided_decoder: The guided decoder

        Returns:
            Tuple[Optional[SampleStateTensorsMTP], Optional[SampleState]]:
                - Updated target inputs or None
                - Draft sample state or None
        """
        import time
        draft_start_time = time.time()

        self.disable_overlap_scheduler = False
        if target_inputs is None:
            return

        draft_batch = self._setup_draft_batch_and_resources(scheduled_batch)
        if draft_batch is None:
            return

        self._initialize_draft_tokens_for_target_inputs(
            scheduled_batch, target_inputs, num_accepted_tokens_device)

        # Initial forward pass
        self.update_cur_draft_layer_idx(
            0, resource_manager
        )  # Update the current draft layer index in the resource manager.
        outputs = self.forward_draft_model(
            draft_batch,
            resource_manager,
            is_first_draft_token=True,
            previous_tensors=previous_tensors,
            num_accepted_tokens_device=num_accepted_tokens_device)

        # Process previous draft results after current forward pass
        # This enables overlap scheduling: process old batch while new batch is prepared
        self._process_previous_draft_results()

        num_draft_reqs = len(draft_batch.all_requests())
        if self.use_static_draft_loop:
            # Only update target inputs, cleanup will be done in executor loop
            self._update_draft_tokens_for_target_inputs(
                target_inputs,
                outputs["new_draft_tokens"],
                draft_position=0,
                draft_length=self.max_draft_len,
                draft_batch=draft_batch)

            new_tokens_host = outputs["new_draft_tokens"].to(device='cpu',
                                                             non_blocking=True)
            sampler_event = torch.cuda.Event()
            sampler_event.record()

            sample_state = SampleState(
                scheduled_requests=draft_batch,
                device=SampleStateTensors(
                    new_tokens=outputs["new_draft_tokens"]),
                host=SampleStateTensors(new_tokens=new_tokens_host),
                sampler_event=sampler_event)

            # Store current batch for processing in next iteration
            self.previous_draft_batch = draft_batch
            self.previous_draft_outputs = (outputs["draft_logits"],
                                           sample_state)
            self.previous_scheduled_batch = scheduled_batch

            return

        # Handle guided decoder and sampling for non-static loop
        if self.guided_decoder is not None:
            self.guided_decoder.add_batch(draft_batch)
            self.guided_decoder.execute(outputs['logits'],
                                        d2t=outputs.get('d2t'))
        draft_sample_state = self.sample_async(draft_batch, outputs,
                                               resource_manager)

        # Update target inputs with first iteration results
        draft_tensors = draft_sample_state and draft_sample_state.device and draft_sample_state.device.new_tokens
        self._update_draft_tokens_for_target_inputs(target_inputs,
                                                    draft_tensors,
                                                    draft_position=0,
                                                    draft_length=1,
                                                    draft_batch=draft_batch)

        self.update_request_states(draft_batch)

        # Execute the iterative draft loop
        previous_draft_state = self._execute_draft_loop(draft_batch,
                                                        resource_manager,
                                                        target_inputs,
                                                        num_draft_reqs,
                                                        draft_sample_state)

        # Store current batch for processing in next iteration
        self.previous_draft_batch = draft_batch
        self.previous_draft_outputs = previous_draft_state
        self.previous_scheduled_batch = scheduled_batch

        # Record draft latency for stats
        draft_end_time = time.time()
        self.last_draft_latency_ms = (draft_end_time - draft_start_time) * 1e3

    @nvtx_range("prepare_draft_tokens")
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare draft tokens for the scheduled requests.

        Args:
            scheduled_requests: The scheduled requests for this iteration
            resource_manager: The resource manager for this iteration
        """
        import time
        draft_start_time = time.time()

        self.disable_overlap_scheduler = True
        if not self.draft_model_engine:
            raise ValueError("Draft model engine is not set")

        if resource_manager is None:
            raise ValueError("Resource manager is required")

        try:
            draft_batch = self._setup_draft_batch_and_resources(
                scheduled_requests)
            if draft_batch is None:
                return

            self.update_cur_draft_layer_idx(
                0, resource_manager
            )  # Update the current draft layer index in the resource manager.
            # Initial forward pass. May do the complete drafting loop
            # if use_static_draft_loop is set.
            outputs = self.forward_draft_model(draft_batch,
                                               resource_manager,
                                               is_first_draft_token=True)

            if self.use_static_draft_loop:
                self.process_static_draft_outputs(outputs, draft_batch)
                # Clean up draft_seq_slot_manager resources
                for req in draft_batch.all_requests():
                    self.draft_seq_slot_manager.free_resources(req)
                # Record draft latency before returning
                draft_end_time = time.time()
                self.last_draft_latency_ms = (draft_end_time -
                                              draft_start_time) * 1e3
                return

            if self.guided_decoder is not None:
                self.guided_decoder.add_batch(draft_batch)
                self.guided_decoder.execute(outputs['logits'],
                                            d2t=outputs.get('d2t'))
            sample_state = self.sample_async(draft_batch, outputs,
                                             resource_manager)
            self.update_request_states(draft_batch)

            # Execute the iterative draft loop
            previous_draft_state = self._execute_draft_loop(
                draft_batch, resource_manager, None, None, sample_state)

            # Final cleanup
            if previous_draft_state is not None:
                self.process_dynamic_draft_outputs(previous_draft_state)

            # Update py_draft_tokens after the loop completes
            for req_id, tokens in self.draft_tokens_accumulator.items():
                target_model_req = self.req_id_to_old_request[req_id]
                target_model_req.py_draft_tokens = tokens

            # Record draft latency for stats
            draft_end_time = time.time()
            self.last_draft_latency_ms = (draft_end_time -
                                          draft_start_time) * 1e3
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            raise e
