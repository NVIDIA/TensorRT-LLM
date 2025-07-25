from typing import List

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger


class HandleLogits:

    @torch.inference_mode()
    @nvtx_range("handle_logits")
    def __call__(
        self,
        context_requests: List[LlmRequest],
        generation_requests: List[LlmRequest],
        logits: torch.Tensor,
        num_context_logits_prefix_sum: List[int],
        max_num_sequences: int,
        beam_width: int,
    ):
        """Handles context and generation logits for a batch of requests.

        Args:
            context_requests: List of context requests to process
            generation_requests: List of generation requests to process
            logits: Input logits tensor
            num_context_logits_prefix_sum: Prefix sum of context logits for each request
            max_num_sequences: Maximum number of sequences to process
            beam_width: Beam width for the generation requests
        """
        # Copy logits into decoderBuffers.logits
        for batch_index, llm_req in enumerate(context_requests):
            logits_begin = num_context_logits_prefix_sum[batch_index]
            logits_end = num_context_logits_prefix_sum[batch_index + 1]

            if llm_req.py_return_context_logits:
                if llm_req.prepopulated_prompt_len > 0:
                    logger.warning(
                        f"Because of KV cache reuse, not all context logits could be produced for request {llm_req.request_id}."
                    )
                context_logits_device_view = logits[logits_begin:logits_end]
                llm_req.py_result.append_context_logits(
                    context_logits_device_view)

            if llm_req.py_return_generation_logits and llm_req.is_last_context_chunk:
                # Get the logits from the last context token and draft tokens
                logits_view = logits[logits_end - 1:logits_end]
                if beam_width > 1:
                    # Replicate logits across all beams
                    llm_req.py_result.append_generation_logits(
                        torch.tile(logits_view, (1, beam_width, 1)))
                else:
                    llm_req.py_result.append_generation_logits(
                        logits_view.unsqueeze(1))

        total_context_logits = num_context_logits_prefix_sum[-1]
        for batch_index, llm_req in enumerate(generation_requests):
            logits_begin = total_context_logits + batch_index * beam_width
            logits_end = logits_begin + beam_width

            if llm_req.py_return_generation_logits:
                logits_view = logits[logits_begin:logits_end].reshape(
                    1, beam_width, -1)
                llm_req.py_result.append_generation_logits(logits_view)
