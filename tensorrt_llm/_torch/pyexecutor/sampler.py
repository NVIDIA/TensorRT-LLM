from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.pyexecutor.handle_context_logits import \
    HandleContextLogits
from tensorrt_llm._torch.pyexecutor.handle_generation_logits import \
    HandleGenerationLogits
from tensorrt_llm._torch.pyexecutor.make_decoding_batch_input_output import \
    MakeDecodingBatchInputOutput
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import (CudaStream, DataType, ModelConfig,
                                   WorldConfig, make_sampling_config)
from tensorrt_llm.bindings.executor import (DecodingConfig, DecodingMode,
                                            ExecutorConfig, FinishReason)
from tensorrt_llm.bindings.internal.algorithms import CreateNewDecoderRequests
from tensorrt_llm.bindings.internal.batch_manager import (DecoderBuffers,
                                                          DecoderInputBuffers)
from tensorrt_llm.bindings.internal.runtime import (BufferManager, DecoderState,
                                                    GptDecoderBatched)
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.logger import logger

from .llm_request import LlmRequest, LlmRequestState
from .scheduler import ScheduledRequests


@dataclass(frozen=True, kw_only=True)
class SampleStateTensors:
    new_tokens: torch.Tensor

    def values(self):
        return vars(self).values()


@dataclass(kw_only=True)
class SampleState:
    scheduled_requests: ScheduledRequests

    logits: Optional[torch.Tensor] = None
    "Starts off as None, then set to logits once outputs are computed"
    logits_host: Optional[torch.Tensor] = None
    "Logits on the host, when applicable"

    # Set when decode_async() has evaluated these to avoid computing again in update_requests()
    # log_probs[request_idx][token_idx]
    log_probs: Optional[list[list[float] | None]] = None

    device: Optional[SampleStateTensors] = None
    host: Optional[SampleStateTensors] = None

    sampler_event: Optional[torch.cuda.Event] = None


class Sampler(ABC):

    SampleState = SampleState

    def setup_sampler_step(self, scheduled_requests: ScheduledRequests):
        pass

    @abstractmethod
    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleState:
        raise NotImplementedError

    @abstractmethod
    def update_requests(self, state: SampleState) -> None:
        raise NotImplementedError


class EarlyStopSampler(Sampler):
    """
    Use for skipping decoding step for non generation model,
    such as encoder-only model (e.g., BERT) or reward models that only need context phase.
    """

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleState:
        return SampleState(scheduled_requests=scheduled_requests,
                           logits=model_outputs['logits'])

    def update_requests(self, state: SampleState) -> None:
        scheduled_requests = state.scheduled_requests
        assert (not scheduled_requests.generation_requests)
        for idx, request in enumerate(scheduled_requests.context_requests):
            request.state = LlmRequestState.GENERATION_COMPLETE
            # NOTE: This is a hack: set finish reason manually and set the beam 0
            request.set_finished_reason(FinishReason.LENGTH, 0)
            if request.py_return_context_logits:
                logits = state.logits[idx]
                if logits.ndim == 1:
                    # For BERT: Add axis to be compatible with LogitsStorage
                    # (LogitsStorage will interpret this dim as the prompt_len which
                    # is not relevant for outputting logits of encoder only model).
                    logits = logits.unsqueeze(0)
                request.py_result.append_context_logits(logits)


def top_k_sampling_batch(logits, top_k=50):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    raw_probs = torch.softmax(logits, dim=-1)

    # get first top_k logits of each sample and their indices
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

    # set the logits who is less than first top_k logits to -inf
    logits = torch.where(logits < min_values,
                         torch.full_like(logits, float('-inf')), logits)

    # compute probability distribution
    probs = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def top_p_sampling_batch(logits, top_p=0.9):
    logits_dim = logits.dim()
    if logits_dim == 1:
        logits = logits.unsqueeze(0)
    # logits should be 2D ：[batch_size, vocab_size]
    batch_size, vocab_size = logits.size()

    raw_probs = torch.softmax(logits, dim=-1)

    # sort the logits of each sample in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # compute  cumulative probability distribution of each sample
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                    dim=-1)

    # get the location of top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # set the logits to -inf whose is outside top_p
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # compute probability distribution
    probs = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def greedy_search_sampling_batch(logits):
    raw_probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.argmax(logits, dim=-1)
    token_probs = torch.gather(raw_probs, dim=1,
                               index=next_tokens.unsqueeze(1)).squeeze(-1)
    log_probs = torch.log(token_probs)
    return next_tokens, log_probs


def decode_single_request(request: LlmRequest, logits):
    assert logits.dim(
    ) == 2 and logits.shape[0] == 1, "logits should have shape [1, vocab_size]"
    if request.sampling_config.top_p is not None and len(
            request.sampling_config.top_p) > 0:
        next_tokens, log_probs = top_p_sampling_batch(
            logits, request.sampling_config.top_p[0])
    elif request.sampling_config.top_k is not None and len(
            request.sampling_config.top_k) > 0:
        next_tokens, log_probs = top_k_sampling_batch(
            logits, request.sampling_config.top_k[0])
    else:
        next_tokens, log_probs = greedy_search_sampling_batch(logits)
    return next_tokens, log_probs


class TorchSampler(Sampler):

    def __init__(self, max_seq_len: int, mixed_sampler: bool = False):
        self.max_seq_len = max_seq_len
        self.mixed_sampler = mixed_sampler

    def _meet_max_token_stop_criteria(self, request: LlmRequest,
                                      num_tokens: int):
        return (num_tokens - request.py_orig_prompt_len
                >= request.py_max_new_tokens) or (num_tokens
                                                  >= self.max_seq_len)

    def _meet_stop_token_criteria(self, request: LlmRequest):
        if request.py_stop_words_list:
            assert isinstance(
                request.py_stop_words_list,
                list), "request.py_stop_words_list should be a list"
            stop_words_list, prefix_sum = request.py_stop_words_list
            tokens = request.get_tokens(0)
            offset = 0
            for i, offset_end in enumerate(prefix_sum):
                if i > 0:
                    offset = prefix_sum[i - 1]
                stop_word = stop_words_list[offset:offset_end]
                if len(stop_word) > len(tokens):
                    continue
                if tokens[-len(stop_word):] == stop_word:
                    return True
        return False

    def _handle_stop_criteria(self, request: LlmRequest, new_token: int,
                              num_tokens: int, beam_idx: int) -> bool:
        """Handle stop criteria and set appropriate finish reasons and state.
        Returns True if generation should stop."""
        if new_token == request.py_end_id:
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(FinishReason.END_ID, beam_idx)
            return True

        if self._meet_max_token_stop_criteria(request, num_tokens):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(FinishReason.LENGTH, beam_idx)
            return True

        if self._meet_stop_token_criteria(request):
            request.state = LlmRequestState.GENERATION_COMPLETE
            request.set_finished_reason(FinishReason.STOP_WORDS, beam_idx)
            return True

        return False

    def update_requests(self, state: SampleState) -> None:
        """
        Uses the new SampleState to update each request.
        After updates, we should have
        """
        if state.sampler_event:
            state.sampler_event.synchronize()

        # When we've sampled, we should have new_tokens under the host entry.
        new_tokens_list = state.host.new_tokens.tolist()
        scheduled_requests = state.scheduled_requests

        request_idx = 0
        token_idx = 0
        beam_idx = 0

        def advance_idx(num_tokens=1):
            """Advance the request and token indices after handling"""
            nonlocal request_idx, token_idx
            request_idx += 1
            token_idx += num_tokens

        def handle_logits(request: LlmRequest, tokens: list[int], count=1):
            """
            For the logits currently being processed, append them to the requests generation results.
            Also append the log probs of each token (generate them from logits if not provided).
            """
            if state.logits is None:
                return
            if not request.py_return_generation_logits and not request.py_return_log_probs:
                return

            current_slice = slice(token_idx, token_idx + count)
            current_logits = state.logits[current_slice]

            # Add logits to the request results (if requested)
            request.py_result.append_generation_logits(current_logits)

            if not request.py_return_log_probs:
                return

            if state.log_probs:
                log_probs = state.log_probs[request_idx]
            else:
                _, log_probs = greedy_search_sampling_batch(current_logits)

            token_log_probs = [{
                token: Logprob(logprob=logprob, rank=1)
            } for token, logprob in zip(tokens, log_probs.tolist())]

            # Add log_probs to the request results (also only if requested)
            request.py_result.append_log_probs([token_log_probs])

        # TODO(marcelroed): When is this set?
        if hasattr(scheduled_requests, 'chunked_requests'):
            request_idx += len(scheduled_requests.chunked_requests)
            token_idx += len(scheduled_requests.chunked_requests)

        ## CONTEXT REQUESTS
        for request in scheduled_requests.context_requests:
            if request.context_remaining_length != 0:
                advance_idx()
                continue
        
            # request.context_remaining_length is 0

            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[token_idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)
                handle_logits(request, [new_token])
                request.py_decoding_iter += 1
            advance_idx()

        extend_requests: list[LlmRequest] = []
        generation_requests: list[LlmRequest] = []
        for request in scheduled_requests.generation_requests:
            if len(request.py_draft_tokens) > 0:
                extend_requests.append(request)
            else:
                generation_requests.append(request)

        # Continued context (if broken into pieces)
        for request in extend_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[token_idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                if self._handle_stop_criteria(request, new_token, num_tokens,
                                              beam_idx):
                    continue

                # Accept draft tokens (if we have any) if and only if they match the new
                # token exactly.
                num_accepted = 0
                new_tokens = [new_token]
                for draft_token in request.py_draft_tokens:
                    if draft_token != new_token:
                        # Reject.
                        break
                    num_accepted += 1
                    new_token = new_tokens_list[token_idx + num_accepted]
                    num_tokens = request.add_new_token(new_token, beam_idx)
                    new_tokens.append(num_tokens)  # `num_tokens`->`new_token`

                    if self._handle_stop_criteria(request, new_token,
                                                  num_tokens, beam_idx):
                        break
                handle_logits(request, new_tokens, num_accepted)
                request.py_decoding_iter += 1
                request.py_num_accepted_draft_tokens = num_accepted
                request.py_rewind_len = request.py_draft_pages_allocated - num_accepted
            advance_idx(len(request.py_draft_tokens) + 1)

        # Generation requests
        for request in generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                new_token = new_tokens_list[token_idx]
                num_tokens = request.add_new_token(new_token, beam_idx)
                self._handle_stop_criteria(request, new_token, num_tokens,
                                           beam_idx)
                handle_logits(request, [new_token])
                request.py_decoding_iter += 1
            advance_idx()

    def _mixed_sample(self, scheduled_requests: ScheduledRequests,
                      model_outputs) -> SampleState:
        """
        Mixed refers to the fact that each request may have different sampling parameters (I think?).
        TODO(marcelroed): Verify this
        """
        logits = model_outputs["logits"]
        log_probs = []
        new_tokens_device_array = []

        idx = 0

        for request in scheduled_requests.context_requests:
            assert not request.py_return_context_logits, "Return context logits not supported"
            token_logits = logits[idx:idx + 1, :]
            new_token, probs = decode_single_request(request, token_logits)
            new_tokens_device_array.append(new_token)
            probs = [probs.tolist()] if request.py_return_log_probs else None
            log_probs.append(probs)  # Currently always beam_width=1
            idx += 1

        for request in scheduled_requests.generation_requests:
            if request.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            assert len(
                request.py_draft_tokens
            ) == 0, "Speculative decoding not supported in SeparateDecoder."
            token_logits = logits[idx:idx + 1, :]
            new_token, probs = decode_single_request(request, token_logits)
            new_tokens_device_array.append(new_token)
            probs = [probs.tolist()] if request.py_return_log_probs else None
            log_probs.append(probs)  # Currently always beam_width=1
            idx += 1

        new_tokens_device = torch.cat(new_tokens_device_array)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        sampler_event = torch.cuda.Event()
        sampler_event.record()

        return SampleState(
            scheduled_requests=scheduled_requests,
            logits=logits,
            device=SampleStateTensors(new_tokens=new_tokens_device),
            host=SampleStateTensors(new_tokens=new_tokens_host),
            sampler_event=sampler_event,
            log_probs=log_probs)

    def _batch_sample(self, scheduled_requests: ScheduledRequests,
                      model_outputs) -> SampleState:
        """
        Simply gets the argmax of the logits to produce new tokens.
        Stores these for both the host and device.
        """
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleState(
            scheduled_requests=scheduled_requests,
            logits=logits,
            device=SampleStateTensors(new_tokens=new_tokens_device),
            host=SampleStateTensors(new_tokens=new_tokens_host),
            sampler_event=sampler_event,
        )

    def sample_async(
            self,
            scheduled_requests: ScheduledRequests,
            model_outputs,
        ) -> SampleState:
        """
        Calls _batch_sample to produce a SampleState for the given scheduled requests given model outputs.
        """
        if self.mixed_sampler:
            return self._mixed_sample(scheduled_requests, model_outputs)
        else:
            return self._batch_sample(scheduled_requests, model_outputs)


class TorchStarAttentionSampler(TorchSampler):

    def update_one_request(self, request: LlmRequest,
                           new_tokens_list: list[int], logits: torch.Tensor):
        beam_idx = 0

        output_token_idx = request.output_token_idx
        new_token = new_tokens_list[output_token_idx]
        num_tokens = request.add_new_token(new_token, beam_idx)

        current_logits = logits[output_token_idx].unsqueeze(0)
        if request.py_return_generation_logits:
            request.py_result.append_generation_logits(current_logits)
        if request.py_return_log_probs:
            _, log_probs = greedy_search_sampling_batch(current_logits)
            request.py_result.append_log_probs([[{
                new_token:
                Logprob(logprob=log_probs.item(), rank=1)
            }]])

        self._handle_stop_criteria(request, new_token, num_tokens, beam_idx)
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            request.py_decoding_iter += 1

    def update_requests(self, state: SampleState):
        if state.sampler_event:
            state.sampler_event.synchronize()
        new_tokens_list = state.host.new_tokens.tolist()
        logits = state.logits

        for request in state.scheduled_requests.context_requests:
            if request.state == LlmRequestState.GENERATION_IN_PROGRESS:
                self.update_one_request(request, new_tokens_list, logits)

        for request in state.scheduled_requests.generation_requests:
            self.update_one_request(request, new_tokens_list, logits)

@dataclass
class SampleStateBlockPrediction(SampleState):
    """Sample state for block prediction with masked chunks."""
    masked_chunks: Optional[torch.Tensor] = None  # [batch_size, block_size] tensor of masked tokens
    block_probs: Optional[torch.Tensor] = None  # [batch_size, block_size] tensor of probabilities
    block_tokens: Optional[torch.Tensor] = None  # [batch_size, block_size] tensor of predicted tokens
    iteration_count: Optional[int] = None  # Number of iterations performed


class BlockPredictionSampler(TorchSampler):
    """
    Sampler for block prediction where we allocate a block of N tokens, all starting as masked tokens,
    then run forward passes with no causal mask, unmasking tokens with softmax probabilities greater than a threshold.
    """
    
    def __init__(self, max_seq_len: int, block_size: int = 8, keep_threshold: float = 0.8, 
                 mask_token_id: int = 151666, max_iterations: int = 10):
        super().__init__(max_seq_len)
        self.block_size = block_size
        self.keep_threshold = keep_threshold
        self.mask_token_id = mask_token_id
        self.max_iterations = max_iterations
    
    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleStateBlockPrediction:
        """
        Sample tokens using block prediction with iterative unmasking.
        """
        
        # Extract logits from model outputs
        logits = model_outputs["logits"]
        # print(f"[BLOCK_PREDICTION] Logits shape: {logits.shape}")
        
        batch_size = scheduled_requests.batch_size
        
        # Initialize or retrieve existing masked chunks
        if hasattr(scheduled_requests, '_block_prediction_state'):
            # Continue from previous iteration
            masked_chunks = scheduled_requests._block_prediction_state['masked_chunks']
            iteration_count = scheduled_requests._block_prediction_state['iteration_count'] + 1
            # print(f"[BLOCK_PREDICTION] Continuing from iteration {iteration_count}")
        else:
            # First iteration - initialize masked chunks
            masked_chunks = torch.full((batch_size, self.block_size), 
                                      self.mask_token_id, 
                                      dtype=torch.int64, 
                                      device=logits.device)
            iteration_count = 1
        
        
        # Track probabilities and predicted tokens for this iteration
        block_probs = torch.zeros((batch_size, self.block_size), 
                                 dtype=logits.dtype, 
                                 device=logits.device)
        block_tokens = torch.zeros((batch_size, self.block_size), 
                                  dtype=torch.int64, 
                                  device=logits.device)
        
        # Extract logits for the block positions
        if logits.dim() == 3:
            # Standard case: [batch_size, seq_len, vocab_size]
            if logits.size(1) >= self.block_size:
                block_logits = logits[:, -self.block_size:, :]
            else:
                # If sequence length is less than block size, pad or truncate
                block_logits = torch.zeros((batch_size, self.block_size, logits.size(-1)),
                                          device=logits.device, dtype=logits.dtype)
                block_logits[:, :logits.size(1), :] = logits
        elif logits.dim() == 2:
            # Missing batch dimension, add it back in
            block_logits = logits[-self.block_size:]
            block_logits = block_logits.unsqueeze(0)
        else:
            raise ValueError(f"Not implemented for shape {logits.shape}")
        
        # Compute probabilities
        probs = torch.softmax(block_logits, dim=-1)
        
        # Get predicted tokens (argmax)
        pred_tokens = torch.argmax(block_logits, dim=-1)
        
        # Get confidence scores (max probability for each position)
        confidence_scores = torch.max(probs, dim=-1)[0]
        
        # Update masked chunks based on confidence threshold
        # Always unmask at least one token (the one with highest confidence)
        tokens_unmasked_this_iteration = 0
        
        for batch_idx in range(batch_size):
            # Find positions that are still masked
            masked_positions = (masked_chunks[batch_idx] == self.mask_token_id)
            
            if not torch.any(masked_positions):
                continue
            
            # Get confidence scores for masked positions
            masked_confidences = confidence_scores[batch_idx][masked_positions]
            masked_pred_tokens = pred_tokens[batch_idx][masked_positions]
            
            # Find positions above threshold
            above_threshold = masked_confidences > self.keep_threshold
            
            # Always unmask at least one token (highest confidence)
            if not torch.any(above_threshold) and torch.any(masked_positions):
                # Find the position with highest confidence
                max_conf_idx = torch.argmax(masked_confidences)
                above_threshold[max_conf_idx] = True
            
            # Update the masked chunks
            masked_indices = torch.where(masked_positions)[0]
            update_indices = masked_indices[above_threshold]
            
            if len(update_indices) > 0:
                masked_chunks[batch_idx, update_indices] = masked_pred_tokens[above_threshold]
                block_probs[batch_idx, update_indices] = masked_confidences[above_threshold]
                block_tokens[batch_idx, update_indices] = masked_pred_tokens[above_threshold]
                tokens_unmasked_this_iteration += len(update_indices)
                
                # Print the details of newly unmasked tokens for this batch
                # newly_unmasked_tokens = masked_pred_tokens[above_threshold].cpu().tolist()
                # print(f"[BLOCK_PREDICTION] Iter {iteration_count} - batch {batch_idx}: "
                #       f"unmasked {len(update_indices)} tokens -> {newly_unmasked_tokens}")
        
        # print(f"[BLOCK_PREDICTION] Iteration {iteration_count} completed: "
        #       f"{tokens_unmasked_this_iteration} tokens unmasked")
        
        # Check if all tokens are unmasked
        all_unmasked = torch.all(masked_chunks != self.mask_token_id)
        
        if all_unmasked:
            # print(f"[BLOCK_PREDICTION] All tokens unmasked after {iteration_count} iterations")
            # Block is complete - return the first token for each batch
            final_tokens = masked_chunks[:, 0]  # Take first token from each block
            
            # Create a standard SampleState with the final tokens
            new_tokens_device = final_tokens.to('cuda', non_blocking=True)
            new_tokens_host = final_tokens.to('cpu', non_blocking=True)
            sampler_event = torch.cuda.Event()
            sampler_event.record()
            
            # Clear the block prediction state
            if hasattr(scheduled_requests, '_block_prediction_state'):
                delattr(scheduled_requests, '_block_prediction_state')
            
            return SampleStateBlockPrediction(
                scheduled_requests=scheduled_requests,
                logits=logits,
                device=SampleStateTensors(new_tokens=new_tokens_device),
                host=SampleStateTensors(new_tokens=new_tokens_host),
                sampler_event=sampler_event,
                masked_chunks=masked_chunks,
                block_probs=block_probs,
                block_tokens=block_tokens,
                iteration_count=iteration_count
            )
        else:
            # print(f"[BLOCK_PREDICTION] Block not complete, {torch.sum(masked_chunks == self.mask_token_id)} tokens still masked")
            
            # Block is not complete - store state for next iteration
            scheduled_requests._block_prediction_state = {
                'masked_chunks': masked_chunks,
                'iteration_count': iteration_count
            }
            
            # Return a special state indicating the block needs more iterations
            # The executor should detect this and continue the block prediction loop
            return SampleStateBlockPrediction(
                scheduled_requests=scheduled_requests,
                logits=logits,
                device=None,  # No new tokens to add yet
                host=None,
                sampler_event=None,
                masked_chunks=masked_chunks,
                block_probs=block_probs,
                block_tokens=block_tokens,
                iteration_count=iteration_count
            )
    
    def update_requests(self, state: SampleStateBlockPrediction) -> None:
        """
        Update requests with block prediction results.
        
        Args:
            state: The block prediction sample state
        """
        if state.sampler_event:
            state.sampler_event.synchronize()
        
        scheduled_requests = state.scheduled_requests

        # TODO(marcelroed): state.host is None when the block is not complete
        assert state.host is not None
        new_tokens_host = state.host.new_tokens
        
        for batch_idx, request in enumerate(scheduled_requests.all_requests):
            if request.is_context_init_state:
                continue
            
            # Add the first new token to the request
            new_token = new_tokens_host[batch_idx]
            request.add_new_token(new_token, 0)
            
            # Store block prediction results in the request for potential future use
            if not hasattr(request, 'py_block_prediction_results'):
                request.py_block_prediction_results = {}
            
            request.py_block_prediction_results.update({
                'masked_chunks': state.masked_chunks[batch_idx] if state.masked_chunks is not None else None,
                'block_probs': state.block_probs[batch_idx] if state.block_probs is not None else None,
                'block_tokens': state.block_tokens[batch_idx] if state.block_tokens is not None else None,
                'iteration_count': state.iteration_count,
                'block_size': self.block_size,
                'keep_threshold': self.keep_threshold,
            })
            
            # Increment the decoding iteration counter
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.py_decoding_iter += 1


class Algorithms:

    def defined_algorithms(self):
        return [attr for attr in dir(self) if not attr.startswith("__")]

    def __repr__(self):
        algs = self.defined_algorithms()
        return f"Algs({', '.join(algs)})"


@dataclass(frozen=True, kw_only=True)
class SampleStateTensorsHostTRTLLM(SampleStateTensors):
    finished_sum: torch.Tensor
    finish_reasons: torch.Tensor
    sequence_lengths: torch.Tensor
    log_probs: torch.Tensor
    cum_log_probs: torch.Tensor


@dataclass(kw_only=True)
class SampleStateTRTLLM(SampleState):
    host: SampleStateTensorsHostTRTLLM
    device: SampleStateTensors


class TRTLLMSampler(Sampler):
    MAX_DECODING_TOKENS = 1  # It must be 1 when not in speculative decoding
    SampleState = SampleStateTRTLLM

    def __init__(
        self,
        executor_config: ExecutorConfig,
        model,
        model_dtype,
        mapping: Mapping,
        decoding_mode: DecodingMode,
        disable_overlap_scheduler: bool,
    ):

        vocab_size = model.config.vocab_size
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads

        self.model_datatype = torch_dtype_to_binding(model_dtype)
        self.logits_datatype = DataType.FLOAT
        self.decoding_mode = decoding_mode
        self.executor_config = executor_config
        self.decoding_config = self.executor_config.decoding_config if self.executor_config.decoding_config else DecodingConfig(
            decoding_mode)
        max_attn_window = self.executor_config.kv_cache_config.max_attention_window
        self.max_attention_window = max_attn_window if max_attn_window is not None else executor_config.max_seq_len
        self.max_num_sequences = mapping.pp_size * self.executor_config.max_batch_size
        self.max_seq_idle_microseconds = 180 * 1000 * 1000
        self.is_trt_overlap = not disable_overlap_scheduler

        self.world_config = WorldConfig.mpi(mapping.gpus_per_node,
                                            mapping.tp_size, mapping.pp_size)
        self.model_config = ModelConfig(vocab_size, num_hidden_layers,
                                        num_hidden_layers, 0, num_heads,
                                        hidden_size, self.model_datatype)

        self._initialize_store()
        self._instantiate_algorithms()

    def _initialize_store(self):
        torch_stream = torch.cuda.current_stream().cuda_stream
        cuda_stream = CudaStream(torch_stream)
        buffer_manager = BufferManager(stream=torch_stream)

        self.store = {
            "torch_stream":
            torch_stream,
            "cuda_stream":
            cuda_stream,
            "buffer_manager":
            buffer_manager,
            "decoder_buffers":
            DecoderBuffers(self.max_num_sequences,
                           self.executor_config.max_beam_width,
                           self.max_attention_window, self.MAX_DECODING_TOKENS,
                           buffer_manager, self.model_config,
                           self.world_config),
            "decoder_input_buffers":
            DecoderInputBuffers(self.executor_config.max_batch_size,
                                self.MAX_DECODING_TOKENS, buffer_manager),
            "new_tokens_device_tensor":
            torch.empty((
                self.executor_config.max_batch_size,
                self.executor_config.max_beam_width,
            ),
                        dtype=torch.int,
                        device='cuda'),
            "sequence_lengths_host":
            torch.empty((
                self.executor_config.max_batch_size,
                self.executor_config.max_beam_width,
            ),
                        dtype=torch.int)
        }

    def _instantiate_algorithms(self):
        self.algs = Algorithms()
        self.algs.decoder = GptDecoderBatched(stream=self.store["torch_stream"])
        self.algs.decoder.setup(
            mode=self.decoding_mode,
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            dtype=self.logits_datatype,
            model_config=self.model_config,
            world_config=self.world_config)
        self.algs.decoder_state = DecoderState(
            dtype=self.logits_datatype,
            buffer_manager=self.store["buffer_manager"])
        self.algs.decoder_state.setup(
            max_batch_size=self.executor_config.max_batch_size,
            max_beam_width=self.executor_config.max_beam_width,
            max_attention_window=self.max_attention_window,
            sink_token_length=0,
            max_sequence_length=self.executor_config.max_seq_len,
            model_config=self.model_config,
            world_config=self.world_config,
            buffer_manager=self.store["buffer_manager"])
        self.algs.create_new_decoder_requests = CreateNewDecoderRequests(
            speculative_decoding_fast_logits=False,
            is_leader_in_orch_mode=False,
            is_normalize_log_probs=False)
        self.algs.handle_context_logits = HandleContextLogits()
        self.algs.handle_generation_logits = HandleGenerationLogits()
        self.algs.make_decoding_batch_input_output = MakeDecodingBatchInputOutput(
        )

    def setup_sampler_step(self, requests):
        batch_slots, sampling_configs, lookahead_prompt, lookahead_algo_configs = self.algs.create_new_decoder_requests(
            self.model_config, self.world_config, self.decoding_config,
            requests, self.store["buffer_manager"], self.logits_datatype,
            self.store["decoder_input_buffers"], self.algs.decoder_state,
            self.store["cuda_stream"], self.algs.decoder.decoder_stream,
            self.executor_config.max_seq_len, self.beam_width(requests))

        local_batch_size = len(batch_slots)
        if local_batch_size > 0:
            sampling_config = make_sampling_config(sampling_configs)
            self.algs.decoder.underlying_decoder().setup(
                sampling_config, local_batch_size, batch_slots,
                self.algs.decoder_state.joint_decoding_output,
                self.model_config.data_type, lookahead_prompt,
                lookahead_algo_configs)

    @staticmethod
    def beam_width(scheduled_requests: Iterable[LlmRequest]) -> int:
        for req in scheduled_requests:
            return req.sampling_config.beam_width
        return 0

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs) -> SampleStateTRTLLM:
        batch_size = scheduled_requests.batch_size
        beam_width = self.beam_width(scheduled_requests.all_requests)

        self.setup_sampler_step(scheduled_requests.context_requests)

        num_context_logits = [1] * batch_size
        for batch_index, request in enumerate(
                scheduled_requests.context_requests):
            num_context_logits[
                batch_index] = request.context_chunk_size if request.py_return_context_logits else 1

        logits_index = self.algs.handle_context_logits(
            scheduled_requests.context_requests, num_context_logits,
            model_outputs["logits"], self.store["decoder_buffers"])

        self.algs.handle_generation_logits(
            logits_index, scheduled_requests.generation_requests,
            self.store["decoder_buffers"], model_outputs["logits"])

        decoding_input, self.decoding_output = self.algs.make_decoding_batch_input_output(
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
            self.store["decoder_buffers"], self.store["decoder_input_buffers"],
            self.algs.decoder_state, self.model_config, self.max_num_sequences)

        self.algs.decoder.forward_async(self.algs.decoder_state,
                                        self.decoding_output, decoding_input)

        # NOTE: The following code prepares a new_tokens_device_tensor in accordance with the
        #       current implementation of model_engine.
        # TODO: When we support speculative decoding:
        # new_tokens_device_tensor should be, for speculative decoding cases: [batch, 1 + draft_len], others: [batch]
        new_tokens_device_tensor = self.store[
            "new_tokens_device_tensor"][:batch_size, :beam_width]
        seq_slots = [
            request.seq_slot for request in scheduled_requests.all_requests
        ]
        new_tokens_device_tensor.copy_(
            self.algs.decoder_state.all_new_tokens[0][seq_slots],
            non_blocking=True)
        new_tokens_device_tensor = new_tokens_device_tensor.view(-1)

        new_output_tokens = self.algs.decoder_state.all_new_tokens.to(
            'cpu', non_blocking=True)
        finished_sum = self.algs.decoder_state.finished_sum.to(
            'cpu', non_blocking=True)
        finish_reasons = self.algs.decoder_state.finish_reasons.to(
            'cpu', non_blocking=True)
        sequence_lengths = self.algs.decoder_state.sequence_lengths.to(
            'cpu', non_blocking=True)

        log_probs = torch.empty([0], dtype=torch.float, device='cpu')
        cum_log_probs = torch.empty([0], dtype=torch.float, device='cpu')
        if any(request.py_return_log_probs
               for request in scheduled_requests.all_requests):
            log_probs = self.algs.decoder_state.log_probs.to('cpu',
                                                             non_blocking=True)
            cum_log_probs = self.algs.decoder_state.cum_log_probs.to(
                'cpu', non_blocking=True)

        device = SampleStateTensors(new_tokens=new_tokens_device_tensor)

        host = SampleStateTensorsHostTRTLLM(new_tokens=new_output_tokens,
                                            finished_sum=finished_sum,
                                            finish_reasons=finish_reasons,
                                            sequence_lengths=sequence_lengths,
                                            log_probs=log_probs,
                                            cum_log_probs=cum_log_probs)

        sampler_event = torch.cuda.Event()
        sampler_event.record()

        return SampleStateTRTLLM(scheduled_requests=scheduled_requests,
                                 logits=model_outputs["logits"],
                                 device=device,
                                 host=host,
                                 sampler_event=sampler_event)

    def update_requests(self, state: SampleStateTRTLLM):
        assert isinstance(state, SampleStateTRTLLM)

        scheduled_requests = state.scheduled_requests
        assert scheduled_requests.batch_size > 0
        beam_width = self.beam_width(scheduled_requests.all_requests)
        sampler_event = state.sampler_event

        if sampler_event:
            sampler_event.synchronize()

        new_tokens_host = state.host.new_tokens
        finished_sum_host = state.host.finished_sum
        finish_reasons_host = state.host.finish_reasons
        sequence_lengths_host_data = state.host.sequence_lengths

        for request in scheduled_requests.all_requests:
            if request.is_context_init_state:
                continue

            seq_slot = request.seq_slot
            num_generated_tokens = request.num_draft_tokens + 1
            current_num_of_tokens = request.max_beam_num_tokens
            num_new_tokens = [0] * beam_width

            log_probs = []
            cum_log_probs = []

            for beam in range(beam_width):
                seq_len = sequence_lengths_host_data[seq_slot * beam_width +
                                                     beam].item()
                num_new_tokens[beam] = min(
                    num_generated_tokens,
                    seq_len - request.get_num_tokens(beam))

                for step in range(num_new_tokens[beam]):
                    new_token = new_tokens_host[step][seq_slot][beam]
                    request.add_new_token(new_token, beam)

                    if request.py_return_log_probs:
                        # NOTE: Log probs with drafting has not been tested yet.
                        begin_log_probs_offset = request.prompt_len if request.sampling_config.beam_width == 1 else 0
                        current_token = seq_len - request.prompt_len - num_new_tokens[
                            beam] + step

                        log_probs.append({
                            new_token.item():
                            Logprob(logprob=state.host.log_probs[seq_slot][beam]
                                    [begin_log_probs_offset +
                                     current_token].item(),
                                    rank=1)
                        })

                if request.py_return_log_probs:
                    cum_log_probs.append(
                        state.host.cum_log_probs[seq_slot * beam_width +
                                                 beam].item())

                finish_reason = finish_reasons_host[seq_slot * beam_width +
                                                    beam].item()
                request.set_finished_reason(FinishReason(finish_reason), beam)

            if request.py_return_log_probs:
                request.py_result.append_log_probs([log_probs], cum_log_probs)

            # Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
            request.update_num_tokens_per_iteration(
                request.max_beam_num_tokens - current_num_of_tokens,
                self.model_config)

            # Increment the decoding iteration counter
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.py_decoding_iter += 1

            if finished_sum_host[seq_slot] == beam_width:
                request.state = LlmRequestState.GENERATION_COMPLETE

