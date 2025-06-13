from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings.internal.runtime import (DecoderBatchInput,
                                                    DecoderBatchOutput)


@dataclass
class MakeDecodingBatchInputOutput:
    """Python implementation of MakeDecodingBatchInputOutput algorithm.

    This class is responsible for creating decoder batch inputs and outputs for the decoding process.
    It handles both context and generation requests, managing their logits and batch slots.
    """

    @staticmethod
    def create_decoder_batch_inputs(
        active_slots: List[int],
        decoder_state,
        logits: List[torch.Tensor],
        max_num_sequences: int,
        batch_slots: List[torch.Tensor],
        cache_indirection_input: Optional[torch.Tensor] = None
    ) -> DecoderBatchInput:
        """Create decoder batch inputs from active slots and logits.

        Args:
            active_slots: List of active sequence slots
            decoder_state: Current decoder state
            logits: List of logit tensors for each slot
            max_num_sequences: Maximum number of sequences to process
            batch_slots: List of batch slot tensors for each decoding step
            cache_indirection_input: Optional cache indirection input tensor

        Returns:
            DecoderBatchInput containing the prepared inputs
        """
        num_decoding_engine_tokens = decoder_state.num_decoding_engine_tokens
        max_decoding_engine_tokens = decoder_state.max_decoding_engine_tokens
        max_decoding_decoder_tokens = decoder_state.max_decoding_decoder_tokens
        max_decoder_steps = (max_decoding_engine_tokens +
                             max_decoding_decoder_tokens -
                             1) // max_decoding_decoder_tokens

        # Resize batch slots for each step
        for step in range(max_decoder_steps):
            batch_slots[step].resize_(max_num_sequences)

        # Track batch indices and find max active decoder steps
        batch_idx = [0] * max_decoder_steps
        max_active_decoder_steps = 1

        for slot in active_slots:
            num_decoder_steps = (num_decoding_engine_tokens[slot] +
                                 max_decoding_decoder_tokens -
                                 1) // max_decoding_decoder_tokens
            max_active_decoder_steps = max(max_active_decoder_steps,
                                           num_decoder_steps)

            for step in range(num_decoder_steps):
                batch_slots[step][batch_idx[step]] = slot
                batch_idx[step] += 1

        # Resize batch slots to actual size used
        for step in range(max_decoder_steps):
            batch_slots[step].resize_(batch_idx[step])

        # Create logits vector for each step
        single_request = 1
        logits_vec = [[] for _ in range(max_active_decoder_steps)]

        for step in range(max_active_decoder_steps):
            batch_slots_range = batch_slots[step]

            for slot in batch_slots_range:
                target_logits = logits[slot]
                logits_slice = target_logits[step:step + single_request]
                logits_vec[step].append(logits_slice)

        # Create decoder batch input
        decoding_input = DecoderBatchInput(logits_vec, max_active_decoder_steps)
        decoding_input.batch_slots = batch_slots

        return decoding_input

    def __call__(
        self, context_requests: List[LlmRequest],
        generation_requests: List[LlmRequest],
        decoder_buffer_logits: list[torch.Tensor], decoder_input_buffers,
        decoder_state, max_num_sequences: int,
        cache_indirections: List[torch.Tensor]
    ) -> Tuple[DecoderBatchInput, DecoderBatchOutput]:
        """Create decoder batch inputs and outputs for the given requests.

        Args:
            context_requests: List of context requests
            generation_requests: List of generation requests
            decoder_buffer_logits: Decoder buffer logits
            decoder_input_buffers: Decoder input buffers
            decoder_state: Current decoder state
            max_num_sequences: Maximum number of sequences to process
            cache_indirections: Cache indirections

        Returns:
            Tuple of (DecoderBatchInput, DecoderBatchOutput)
        """
        # Get active slots and generation steps
        active_slots = []
        generation_steps = []

        for requests in [context_requests, generation_requests]:
            for request in requests:
                if request.is_generation_in_progress_state or request.is_last_context_chunk:
                    active_slots.append(request.seq_slot)
                    generation_steps.append(request.decoding_iter)

        # Sort by slot number
        sorted_indices = sorted(range(len(active_slots)),
                                key=lambda i: active_slots[i])
        active_slots = [active_slots[i] for i in sorted_indices]
        generation_steps = [generation_steps[i] for i in sorted_indices]

        # Create decoder batch inputs
        decoding_input = self.create_decoder_batch_inputs(
            active_slots, decoder_state, decoder_buffer_logits,
            max_num_sequences, decoder_input_buffers.forward_batch_slots)
        decoding_input.generation_steps = generation_steps
        decoding_input.cache_indirection = cache_indirections[0]

        # TODO: Handle speculative decoding modes
        # if model_config.speculative_decoding_mode.has_draft_logits:
        #     decoding_input.predicted_draft_logits = decoder_buffers.draft_buffers.predicted_draft_logits

        # TODO: fused_runtime_buffers is not created in the pytorch framework.
        # if model_config.speculative_decoding_mode.is_explicit_draft_tokens:
        #     if fused_runtime_buffers is None:
        #         raise RuntimeError("Fused runtime buffers required for explicit draft tokens")
        #     decoding_input.batch_slots_request_order = fused_runtime_buffers.seq_slots
        #     decoding_input.explicit_draft_tokens_inputs = fused_runtime_buffers.explicit_draft_tokens_buffers.engine_outputs
        #     decoding_input.explicit_draft_tokens_last_inputs = fused_runtime_buffers.explicit_draft_tokens_buffers.engine_inputs
        # elif model_config.speculative_decoding_mode.is_eagle:
        #     if fused_runtime_buffers is None:
        #         raise RuntimeError("Fused runtime buffers required for eagle mode")
        #     decoding_input.batch_slots_request_order = fused_runtime_buffers.seq_slots
        #     decoding_input.eagle_inputs = fused_runtime_buffers.eagle_buffers.engine_outputs
        #     decoding_input.eagle_last_inputs = fused_runtime_buffers.eagle_buffers.engine_inputs

        # Create decoder batch output
        decoding_output = DecoderBatchOutput()
        decoding_output.cache_indirection = cache_indirections[1]

        return decoding_input, decoding_output
