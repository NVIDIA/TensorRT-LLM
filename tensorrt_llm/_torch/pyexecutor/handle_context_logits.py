from typing import List

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings.internal.batch_manager import DecoderBuffers
from tensorrt_llm.logger import logger


class HandleContextLogits:

    def __call__(self, context_requests: List[LlmRequest],
                 num_context_logits_vec: List[int], logits: torch.Tensor,
                 decoder_buffers: DecoderBuffers) -> int:
        """Handle context logits for a batch of requests.

        Args:
            context_requests: List of context requests to process
            num_context_logits_vec: Number of context logits for each request
            logits: Input logits tensor
            decoder_buffers: Decoder buffers for storing intermediate results

        Returns:
            int: Index into logits tensor after processing all requests
        """
        logits_index = 0

        # Copy logits into decoderBuffers.logits
        decoder_buffer_logits = [torch.empty(0)] * len(decoder_buffers.logits)
        for batch_index, llm_req in enumerate(context_requests):
            num_context_logits = num_context_logits_vec[batch_index]
            draft_length = llm_req.num_draft_tokens if llm_req.is_last_context_chunk(
            ) else 0

            if llm_req.py_return_context_logits:
                if llm_req.prepopulated_prompt_len > 0:
                    logger.warning(
                        f"Because of KV cache reuse, not all context logits could be produced for request {llm_req.request_id}."
                    )

                context_logits_device_view = logits[logits_index:logits_index +
                                                    num_context_logits]
                llm_req.py_result.append_context_logits(
                    context_logits_device_view)

            logits_index += num_context_logits + draft_length

            # Get the logits from the last context token and draft tokens
            num_decoder_logits = 1 + draft_length
            seq_slot = llm_req.seq_slot
            logits_view = logits[logits_index - num_decoder_logits:logits_index]

            # Create a view of logits_view with shape (logits_view.shape[0], 1, logits_view.shape[1])
            # This creates a new tensor that shares the same underlying data
            decoder_buffer_logits[seq_slot] = logits_view.reshape(
                logits_view.shape[0], 1, logits_view.shape[1])

            # Save the last context token logits in generation logits storage
            if llm_req.py_return_generation_logits and llm_req.is_last_context_chunk:
                llm_req.py_result.append_generation_logits(logits_view)

            # TODO: Implement this once we have beam width support
            # Scatter the output logits to the decoderLogits
            # req_beam_width = llm_req.get_beam_width_by_iter()
            # if req_beam_width > 1:
            #     # Tile logits of context requests
            #     logits_shape = logits_view.shape
            #     logits_type = logits_view.dtype
            #     # decoder_logits = buffer_manager.gpu((req_beam_width, logits_shape[1]), logits_type)
            #     # tensorrt_llm.runtime.kernels.tile_tensor(decoder_logits, logits_view, req_beam_width, stream)
            #     decoder_logits = decoder_logits.unsqueeze(0)
            # else:
            #     decoder_buffer_logits[seq_slot] = logits_view[:logits_view.shape[0], :1, :logits_view.shape[1]]

        # Needs to be done in bulk for the copy to work
        decoder_buffers.logits = decoder_buffer_logits

        return logits_index
