import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import ModelConfig
from tensorrt_llm.bindings.internal.batch_manager import DecoderBuffers


class HandleGenerationLogits:

    def __call__(
        self,
        logits_index: int,
        generation_requests: list[LlmRequest],
        decoder_buffers: DecoderBuffers,
        model_config: ModelConfig,
        logits: torch.Tensor,
    ):
        # TODO: Should we check that there's no speculative decoding? can be determined with info from model_config
        # NOTE: Currently doesn't support speculative decoding, which is supported in handleContextLogits.cpp
        for i, llm_req in enumerate(generation_requests):
            beam_width = llm_req.get_beam_width_by_iter()
            seq_slot = llm_req.seq_slot

            # logits_view shape: [beamWidth, vocabSize]
            logits_view = logits[logits_index:logits_index + beam_width]

            # TODO: I don't understand why that slicing is done when beam_width <= 1, it seems to work well without it.
            # if beam_width > 1:
            #    decoder_buffer_logits[seq_slot] = logits_view
            #    decoder_buffer_logits[seq_slot].unsqueeze(0)
            # else:
            #    print(f"{seq_slot=}")
            #    decoder_buffer_logits[seq_slot] = logits_view[:logits_view.shape[0], :1, :logits_view.shape[1]]

            # TODO: Seems like setting an element in the logits vector doesn't work, so I (hopefully temporarily) added the set_logits_at method
            # decoder_buffers.logits[seq_slot] = logits_view
            decoder_buffers.set_logits_at(seq_slot, logits_view)

            if beam_width > 1:
                # TODO: Why is this necessary?
                decoder_buffers.logits[seq_slot].unsqueeze(0)

            if llm_req.py_return_generation_logits:
                llm_req.py_result.append_generation_logits(logits_view)

            logits_index += beam_width
