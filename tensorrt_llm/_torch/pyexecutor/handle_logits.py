# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import chain
from typing import List

import torch

from tensorrt_llm._torch.pyexecutor.context_logits_plan import \
    store_context_logits
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._utils import nvtx_range


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
        context_logits_handled: bool = False,
    ):
        """Handles context and generation logits for a batch of requests.

        Args:
            context_requests: List of context requests to process
            generation_requests: List of generation requests to process
            logits: Input logits tensor
            beam_width: Beam width for the generation requests
            num_context_logits_prefix_sum: Prefix sum of the logits
        """
        if not any(r.py_return_context_logits or r.py_return_generation_logits
                   for r in chain(context_requests, generation_requests)):
            return

        # Copy logits into the per-request logits buffers
        for batch_index, llm_req in enumerate(context_requests):
            logits_begin = num_context_logits_prefix_sum[batch_index]
            logits_end = num_context_logits_prefix_sum[batch_index + 1]

            if llm_req.py_return_context_logits and not context_logits_handled:
                context_logits_device_view = logits[logits_begin:logits_end]
                store_context_logits(llm_req, context_logits_device_view)

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
            if llm_req.state == LlmRequestState.GENERATION_COMPLETE:
                continue

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
