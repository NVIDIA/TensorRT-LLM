from itertools import count

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import convert_wordlist
from tensorrt_llm._torch.pyexecutor.sampler import LlmRequest, TorchSampler
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason


class Request:
    _counter = count()

    def __init__(self,
                 input_tokens: list[int],
                 *,
                 new_tokens: list[int],
                 finish_reasons: list[FinishReason],
                 max_new_tokens: int = 10,
                 end_id: int = None,
                 stop_words_list: list[list[int]] = None):
        idx = next(self._counter)
        self.input_tokens = input_tokens
        self.request = LlmRequest(
            request_id=idx,
            seq_slot=idx,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            stop_words_list=convert_wordlist(stop_words_list)
            if stop_words_list is not None else None,
            end_id=end_id,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert len(new_tokens) == len(finish_reasons)
        self.new_tokens = new_tokens
        self.finish_reasons = finish_reasons


def test_write_finish_reasons():
    sampler_args = TorchSampler.Args(max_seq_len=20,
                                     max_draft_len=1,
                                     max_num_sequences=4,
                                     max_beam_width=1,
                                     enable_mixed_sampler=False)
    sampler = TorchSampler(args=sampler_args)
    assert sampler.max_tokens == 2

    STOP_SEQUENCE = [12, 13]
    END_ID = 99

    requests = [
        Request([1, 2, 3, 4],
                end_id=END_ID,
                new_tokens=[55, END_ID],
                finish_reasons=[
                    FinishReason.NOT_FINISHED,
                    FinishReason.END_ID,
                ]),
        Request([4, 5, 6],
                max_new_tokens=2,
                new_tokens=[56, 57],
                finish_reasons=[
                    FinishReason.NOT_FINISHED,
                    FinishReason.LENGTH,
                ]),
        Request([7, 8, 9],
                stop_words_list=[STOP_SEQUENCE],
                new_tokens=[STOP_SEQUENCE[0], STOP_SEQUENCE[1]],
                finish_reasons=[
                    FinishReason.NOT_FINISHED,
                    FinishReason.STOP_WORDS,
                ]),
        Request([13, 14],
                new_tokens=[60, 61],
                finish_reasons=[
                    FinishReason.NOT_FINISHED,
                    FinishReason.NOT_FINISHED,
                ])
    ]

    new_tokens = torch.tensor([req.new_tokens for req in requests],
                              dtype=torch.int32,
                              device="cuda").T.unsqueeze(-1)
    assert new_tokens.shape == sampler.store.new_tokens.shape

    seq_slots = torch.tensor([req.request.py_seq_slot for req in requests],
                             device="cuda",
                             dtype=torch.int32)
    sampler.write_finish_reasons([req.request for req in requests],
                                 new_tokens=new_tokens,
                                 seq_slots=seq_slots)

    actual_finish_reasons = sampler.store.finish_reasons[:, seq_slots,
                                                         sampler.BEAM].T.tolist(
                                                         )

    for actual, request in zip(actual_finish_reasons, requests, strict=True):
        expected = request.finish_reasons
        msg = f"""\
actual={[FinishReason(reason) for reason in actual]} != expected={expected}
For request: {request.request.request_id=}, {request.input_tokens=}, {request.new_tokens=}
"""
        assert actual == [reason.value for reason in expected], msg
