import random

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import convert_wordlist
from tensorrt_llm._torch.pyexecutor.sampler import (BEAM_0, LlmRequest,
                                                    TorchSampler)
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason

MAX_NUM_SEQUENCES = 128


class RequestCase:
    MAX_NEW_TOKENS = 10
    seq_slots = random.sample(range(MAX_NUM_SEQUENCES), MAX_NUM_SEQUENCES)

    def __init__(self,
                 *,
                 prompt: list[int],
                 new_tokens: list[int],
                 finish_reasons: list[FinishReason],
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 end_id: int = None,
                 stop_words_list: list[list[int]] = None):
        seq_slot = self.seq_slots.pop()  # random seq slot in MAX_NUM_SEQUENCES
        self.prompt = prompt
        self.request = LlmRequest(
            request_id=seq_slot,
            seq_slot=seq_slot,
            input_tokens=prompt,
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

    def __repr__(self):
        return f"RequestCase({self.prompt=}, {self.new_tokens=}, {self.finish_reasons=}, {self.request.max_new_tokens=}, {self.request.end_id=}, {self.request.stop_words_list=})"


def test_write_finish_reasons():
    NOT_FINISHED = FinishReason.NOT_FINISHED
    STOP_WORDS = FinishReason.STOP_WORDS
    END_ID = FinishReason.END_ID
    LENGTH = FinishReason.LENGTH
    """We don't really care about the finish reason past the first infraction, because we're not going to use it, although in some instance it is written anyway."""

    MAX_DRAFT_LEN = 2
    sampler_args = TorchSampler.Args(
        max_seq_len=20,
        max_draft_len=MAX_DRAFT_LEN,
        # Fill with many more max requests than below, so we can test that write_finish_reasons uses seq_slots correctly
        max_num_sequences=MAX_NUM_SEQUENCES,
        max_beam_width=1,
        enable_mixed_sampler=False)
    sampler = TorchSampler(args=sampler_args)
    assert sampler.max_tokens == MAX_DRAFT_LEN + 1

    # fill with garbage value so we can observe that finish reasons are filled with NOT_FINISHED before we write to them.
    sampler.store.finish_reasons.fill_(205)

    requests = [
        RequestCase(
            prompt=[13, 14],
            new_tokens=[60, 61, 62],
            # We pre-fill the finish reasons with NOT_FINISHED.
            finish_reasons=[NOT_FINISHED, NOT_FINISHED, NOT_FINISHED],
        ),
        RequestCase(
            prompt=[7, 8, 6],
            stop_words_list=[[12, 13]],
            new_tokens=[12, 13, 60],
            finish_reasons=[NOT_FINISHED, STOP_WORDS, NOT_FINISHED],
        ),
        RequestCase(
            prompt=[1, 2, 3, 4],
            end_id=99,
            new_tokens=[55, 99, 58],
            finish_reasons=[NOT_FINISHED, END_ID, NOT_FINISHED],
        ),
        RequestCase(
            prompt=[4, 5, 6],
            max_new_tokens=2,
            new_tokens=[56, 57, 59],
            # The LENGTH check happens to not have an early exit
            finish_reasons=[NOT_FINISHED, LENGTH, LENGTH]),
        RequestCase(
            prompt=[1, 12],
            stop_words_list=[[12, 13], [14, 15]],
            new_tokens=[13, 14, 15],
            # We have an early exit specifically for stop words
            finish_reasons=[STOP_WORDS, NOT_FINISHED, NOT_FINISHED],
        ),
        RequestCase(
            prompt=[1],
            max_new_tokens=2,
            end_id=99,
            stop_words_list=[[1, 12]],
            new_tokens=[12, 99, 63],
            # Different infractions are written to different places as we don't have an early exit between infractions
            finish_reasons=[STOP_WORDS, END_ID, LENGTH],
        ),
        RequestCase(
            prompt=[1, 12, 56, 67, 68, 234, 678],
            stop_words_list=[[12, 56, 67, 68, 234, 678, 129, 182]],
            new_tokens=[129, 182, 600],
            # Notice the offending stop sequence is concatenated, as we lookback
            finish_reasons=[NOT_FINISHED, STOP_WORDS, NOT_FINISHED],
        ),
        RequestCase(
            prompt=[1, 12],
            end_id=99,
            max_new_tokens=1,
            stop_words_list=[[1, 12, 99]],
            new_tokens=[99, 100, 101],
            # The latest infraction check overrides the earlier infraction checks, hence the first finish_reason is END_ID
            finish_reasons=[END_ID, LENGTH, LENGTH],
        ),
    ]

    assert sampler.max_tokens == 3

    seq_slots = torch.tensor([req.request.py_seq_slot for req in requests],
                             device="cuda",
                             dtype=torch.int64)  # for index_fill_

    new_tokens = torch.tensor([req.new_tokens for req in requests],
                              dtype=torch.int32,
                              device="cuda").T

    sampler.store.new_tokens[:, seq_slots, BEAM_0] = new_tokens
    sampler._write_finish_reasons([req.request for req in requests],
                                  finish_reasons=sampler.store.finish_reasons,
                                  new_tokens=sampler.store.new_tokens,
                                  seq_slots=seq_slots)

    actual_finish_reasons = sampler.store.finish_reasons[:, seq_slots,
                                                         BEAM_0].T.tolist()

    for actual, request in zip(actual_finish_reasons, requests, strict=True):
        expected = request.finish_reasons
        msg = f"""\
actual={[FinishReason(reason) for reason in actual]} != expected={expected}
For {request}
"""
        assert actual == [reason.value for reason in expected], msg
