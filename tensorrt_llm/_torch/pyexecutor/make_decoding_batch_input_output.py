from dataclasses import dataclass
from typing import List

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings.internal.runtime import DecoderBatchInput
from tensorrt_llm._utils import nvtx_range

@dataclass
class MakeDecodingBatchInputOutput:
    """Python implementation of MakeDecodingBatchInputOutput algorithm.

    This class is responsible for creating decoder batch inputs and outputs for the decoding process.
    It handles both context and generation requests, managing their logits and batch slots.
    """

    @torch.inference_mode()
    @nvtx_range("make_decoding_batch_input_output")
    def __call__(
        self,
        scheduled_requests,
        logits: torch.Tensor,
        decoder_input_buffers,
        decoder_state,
        model_config,
        max_num_sequences: int,
        beam_width: int,
        num_context_logits_prefix_sum: List[int],
    ) -> DecoderBatchInput:
        """Create decoder batch inputs and outputs for the given requests.

        Args:
            context_requests: List of context requests
            generation_requests: List of generation requests
            decoder_input_buffers: Decoder input buffers
            decoder_state: Current decoder state
            model_config: Model configuration
            max_num_sequences: Maximum number of sequences to process
            fused_runtime_buffers: Optional fused runtime buffers

        Returns:
            DecoderBatchInput
        """
        # In order to make a decoding_input assuming no drafting, we need:
        # 1. logits_vec = [[logits_slice of each active slot]]
        # 2. batch_slots = [[active_slots]]
        # 3. generation_steps = [decoding_iters]

        active_slots = [[]]
        generation_steps = []
        logits_vec = [[]]
        for i, r in enumerate(scheduled_requests.context_requests):
            if r.is_last_context_chunk:
                active_slots[0].append(r.py_seq_slot)
                generation_steps.append(r.decoding_iter)
                logits_vec[0].append(logits[num_context_logits_prefix_sum[i]:num_context_logits_prefix_sum[i+1]].unsqueeze(0))
        
        logits_index = num_context_logits_prefix_sum[-1]
        for i, r in enumerate(scheduled_requests.generation_requests):
            if r.is_generation_in_progress_state:
                active_slots[0].append(r.py_seq_slot)
                generation_steps.append(r.decoding_iter)
                logits_vec[0].append(logits.narrow(dim=0, start=logits_index + i * beam_width, length=beam_width).unsqueeze(0))

        decoding_input = DecoderBatchInput(logits_vec, 1)
        decoding_input.generation_steps = generation_steps
        decoding_input.batch_slots = [torch.tensor(active_slots[0], dtype=torch.int32)]

        return decoding_input
