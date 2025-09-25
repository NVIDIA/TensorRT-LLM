from itertools import chain
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
        beam_width: int,
        num_context_logits_prefix_sum: list[int],
        is_generation_model: bool,
    ):
        """Handles context and generation logits for a batch of requests.

        Args:
            context_requests: List of context requests to process
            generation_requests: List of generation requests to process
            logits: Input logits tensor
            beam_width: Beam width for the generation requests
            num_context_logits_prefix_sum: Prefix sum of the logits
            is_generation_model: Bool containing whether the model is generation or not
        """
        if not any(r.py_return_context_logits or r.py_return_generation_logits
                   for r in chain(context_requests, generation_requests)):
            return

        if not is_generation_model:
            for llm_req, logits_temp in zip(context_requests, logits):
                if logits_temp.ndim == 1:
                    # For BERT: Add axis to be compatible with LogitsStorage
                    # (LogitsStorage will interpret this dim as the prompt_len which
                    # is not relevant for outputting logits of encoder only model).
                    logits_temp = logits_temp.unsqueeze(0)
                llm_req.py_result.append_context_logits(logits_temp)
            return

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

        # Finalize any remaining logits transfers for all requests in chunked mode
        for llm_req in chain(context_requests, generation_requests):
            if llm_req.py_use_chunked_generation_logits and llm_req.py_return_generation_logits:
                llm_req.py_result.transfer_remaining_device_logits()
