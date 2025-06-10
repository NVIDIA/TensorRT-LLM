import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings.internal.batch_manager import DecoderBuffers


class HandleGenerationLogits:

    def __call__(
        self,
        logits_index: int,
        generation_requests: list[LlmRequest],
        decoder_buffers: DecoderBuffers,
        logits: torch.Tensor,
    ):
        decoder_buffer_logits = decoder_buffers.logits
        for llm_req in generation_requests:
            beam_width = llm_req.get_beam_width_by_iter()
            seq_slot = llm_req.seq_slot

            # logits_view shape: [beamWidth, vocabSize]
            logits_view = logits[logits_index:logits_index + beam_width]

            if beam_width > 1:
                decoder_buffer_logits[seq_slot] = logits_view.unsqueeze(0)
            else:
                decoder_buffer_logits[seq_slot] = logits_view.unsqueeze(1)

            if llm_req.py_return_generation_logits:
                llm_req.py_result.append_generation_logits(logits_view)

            logits_index += beam_width

        # Needs to be done in bulk for the copy to work
        decoder_buffers.logits = decoder_buffer_logits
