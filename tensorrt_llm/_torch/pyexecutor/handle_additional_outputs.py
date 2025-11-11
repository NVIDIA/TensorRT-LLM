from itertools import chain
from typing import Dict, List

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger


class HandleAdditionalOutputs:

    @torch.inference_mode()
    @nvtx_range("handle_additional_outputs")
    def __call__(
        self,
        context_requests: List[LlmRequest],
        generation_requests: List[LlmRequest],
        outputs: Dict[str, torch.Tensor],
        beam_width: int,
        num_context_tokens: int,
    ):
        """Handles context and generation logits for a batch of requests.

        Args:
            context_requests: List of context requests to process
            generation_requests: List of generation requests to process
            outputs: Additional outputs tensors
            beam_width: Beam width for the generation requests
            num_context_tokens: Number of context tokens in the batch
        """

        additional_outputs = set()
        for r in chain(context_requests, generation_requests):
            if r.py_additional_outputs is not None:
                additional_outputs.update(r.py_additional_outputs)

        if not additional_outputs:
            return

        output_length_with_context = num_context_tokens + beam_width * len(
            generation_requests)
        output_length_without_context = len(
            context_requests) + beam_width * len(generation_requests)

        gather_context = {}
        for name in additional_outputs:
            if outputs[name].shape[0] == output_length_with_context:
                gather_context[name] = True
            else:
                gather_context[name] = False

        output_index_with_context = 0
        output_index_without_context = 0

        # Copy additional outputs into decoderBuffers.additional_outputs
        for llm_req in context_requests:
            context_output_length = llm_req.context_chunk_size

            outputs_begin = output_index_with_context
            outputs_end = output_index_with_context + context_output_length

            additional_outputs = llm_req.py_additional_outputs
            req_context_output = False
            for name in additional_outputs:
                if gather_context[name]:
                    output_device_view = outputs[name][
                        outputs_begin:outputs_end]
                    llm_req.py_result.append_additional_context_outputs(
                        name, output_device_view)
                    req_context_output = True

            if req_context_output and llm_req.prepopulated_prompt_len > 0:
                logger.warning(
                    f"Because of KV cache reuse, not all additional context outputs could be produced for request {llm_req.request_id}."
                )

            output_index_with_context += context_output_length
            output_index_without_context += 1

            if llm_req.is_last_context_chunk:
                for name in additional_outputs:
                    outputs_begin = (output_index_with_context
                                     if gather_context[name] else
                                     output_index_without_context) - 1
                    outputs_end = outputs_begin + 1

                    output_device_view = outputs[name][
                        outputs_begin:outputs_end]
                    llm_req.py_result.append_additional_generation_outputs(
                        name, torch.tile(output_device_view,
                                         (1, beam_width, 1)))

        for llm_req in generation_requests:
            additional_outputs = llm_req.py_additional_outputs

            for name in additional_outputs:
                outputs_begin = (output_index_with_context
                                 if gather_context[name] else
                                 output_index_without_context)
                outputs_end = outputs_begin + beam_width

                output_device_view = outputs[name][
                    outputs_begin:outputs_end].reshape(1, beam_width, -1)
                llm_req.py_result.append_additional_generation_outputs(
                    name, output_device_view)

            output_index_with_context += beam_width
            output_index_without_context += beam_width

        assert output_index_with_context == output_length_with_context, f"output_index_with_context: {output_index_with_context}, output_length_with_context: {output_length_with_context}"
        assert output_index_without_context == output_length_without_context, f"output_index_without_context: {output_index_without_context}, output_length_without_context: {output_length_without_context}"
