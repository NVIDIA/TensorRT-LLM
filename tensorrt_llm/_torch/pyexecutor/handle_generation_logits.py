from typing import List

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


class HandleGenerationLogits:

    def __call__(
        self,
        decoder_buffer_logits: List[torch.Tensor],
        generation_requests: list[LlmRequest],
        logits: torch.Tensor,
        logits_index: int,
    ):
        for llm_req in generation_requests:
            beam_width = llm_req.get_beam_width_by_iter()
            seq_slot = llm_req.seq_slot
            draft_length = llm_req.num_draft_tokens
            num_logits = beam_width + draft_length
            num_tokens = 1 + draft_length

            # logits_view shape: [num_tokens, beam_width, vocab_size]
            logits_view = logits[logits_index:logits_index + beam_width +
                                 draft_length].reshape(num_tokens, beam_width,
                                                       -1)

            decoder_buffer_logits[seq_slot] = logits_view

            if llm_req.py_return_generation_logits:
                llm_req.py_result.append_generation_logits(
                    decoder_buffer_logits[seq_slot])

            logits_index += num_logits

        return decoder_buffer_logits
