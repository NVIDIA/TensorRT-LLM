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

            # logits_view shape: [beamWidth, 1, vocabSize] - that 1 in the middle is added from unsqueeze in sample_async
            logits_view = logits[logits_index:logits_index + beam_width]

            # if beam_width > 1:
            #    decoder_buffer_logits[seq_slot] = logits_view
            #    decoder_buffer_logits[seq_slot].unsqueeze(0)
            # else:
            #    print(f"{seq_slot=}")
            #    decoder_buffer_logits[seq_slot] = logits_view[:logits_view.shape[0], :1, :logits_view.shape[1]]

            decoder_buffer_logits[seq_slot] = logits_view

            if beam_width > 1:
                # TODO: Why is this necessary?
                decoder_buffer_logits.unsqueeze(0)

            if llm_req.py_return_generation_logits:
                llm_req.py_result.append_generation_logits(logits_view)

            logits_index += beam_width

        # Needs to be done in bulk for the copy to work
        decoder_buffers.logits = decoder_buffer_logits
