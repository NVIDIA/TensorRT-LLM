# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact PP beam-state transfers preserve each request's cache history."""

import pickle

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTorch, TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.beam_search import BeamSearchRelayState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.sampling_params import SamplingParams


def _make_sampler(speculative_d2h: bool = False) -> TorchSampler:
    return TorchSampler(
        TorchSampler.Args(
            max_seq_len=9000,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=4,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=True,
            share_beam_search_state=True,
            enable_speculative_beam_history_d2h=speculative_d2h,
        )
    )


def _make_requests(prompt_lens: list[int], round_id: int = 0) -> list[LlmRequest]:
    return [
        LlmRequest(
            request_id=request_id + 1000 * round_id,
            seq_slot=slot,
            input_tokens=[1] * prompt_len,
            max_new_tokens=4,
            end_id=-1,
            sampling_config=SamplingConfig(
                SamplingParams(
                    n=width,
                    best_of=width,
                    use_beam_search=True,
                    beam_width_array=[2, 3, 4] if row == 0 else None,
                )._get_sampling_config()
            ),
            return_log_probs=True,
            num_logprobs=0,
            is_streaming=False,
        )
        for row, (request_id, slot, prompt_len, width) in enumerate(
            zip([71, 29], [3, 1], prompt_lens, [4, 3])
        )
    ]


@pytest.mark.threadleak(enabled=False)
def test_handoff_replays_parent_history_and_reinitializes_reused_slots() -> None:
    """Parent maps restore known paths and discard the previous slot occupant's history."""
    sampler = _make_sampler()
    store = sampler.store.beam_search_store
    assert store is not None
    store.cache_indirection.fill_(-1)
    requests = _make_requests([128, 8192])

    def apply(
        parents: list[list[int]], lengths: list[int], widths: list[int], contexts: int
    ) -> None:
        sampler._beam_search.apply_relay(
            requests,
            BeamSearchRelayState(
                request_ids=[request.py_request_id for request in requests],
                num_context_requests=contexts,
                predecessor_beams=torch.tensor(parents, dtype=torch.int32),
                seq_lens=lengths,
                beam_widths=widths,
                cum_log_probs=torch.zeros((2, 4)),
            ),
        )

    apply([[0, 0, 0], [0, 0, 0]], [128, 8192], [2, 3], 2)
    assert store.cache_indirection[3, :2, 128].tolist() == [0, 1]
    assert store.cache_indirection[1, :3, 8192].tolist() == [0, 1, 2]
    apply([[1, 0, 1], [2, 0, 1]], [129, 8193], [3, 3], 0)
    assert store.cache_indirection[3, :3, 128:130].tolist() == [[1, 0], [0, 1], [1, 2]]
    assert store.cache_indirection[1, :3, 8192:8194].tolist() == [[2, 0], [0, 1], [1, 2]]
    apply([[2, 0, 2, 1], [1, 1, 2, 0]], [130, 8194], [4, 3], 0)
    assert store.cache_indirection[3, :4, 128:131].tolist() == [
        [1, 2, 0],
        [1, 0, 1],
        [1, 2, 2],
        [0, 1, 3],
    ]
    assert store.cache_indirection[1, :3, 8192:8195].tolist() == [[0, 1, 0], [0, 1, 1], [1, 2, 2]]

    requests = _make_requests([8192, 128], round_id=1)
    apply([[0, 0, 0], [0, 0, 0]], [8192, 128], [2, 3], 2)
    assert torch.count_nonzero(store.cache_indirection[3, :2, :8192]).item() == 0
    assert torch.count_nonzero(store.cache_indirection[1, :3, :128]).item() == 0
    assert store.cache_indirection[3, :2, 8192].tolist() == [0, 1]
    assert store.cache_indirection[1, :3, 128].tolist() == [0, 1, 2]


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("speculative_d2h", [False, True])
def test_handoff_replays_variable_width_steps_with_bounded_payload(speculative_d2h: bool) -> None:
    """Unequal prompts and changing beam widths keep the receiver in step."""
    sender, receiver = _make_sampler(speculative_d2h), _make_sampler(speculative_d2h)
    sender_store, receiver_store = sender.store.beam_search_store, receiver.store.beam_search_store
    assert sender_store is not None and receiver_store is not None
    receiver_store.cache_indirection.fill_(-1)
    generator = torch.Generator(device="cuda").manual_seed(47)
    for round_id, prompt_lens in enumerate(([128, 8192], [8192, 128])):
        sampled_requests = _make_requests(prompt_lens, round_id)
        received_requests = _make_requests(prompt_lens, round_id)
        for step in range(4):
            scheduled = ScheduledRequests()
            if step == 0:
                scheduled.context_requests_last_chunk = sampled_requests
            else:
                scheduled.generation_requests = sampled_requests
            state = sender.sample_async(
                scheduled,
                model_outputs={
                    "logits": torch.randn(
                        (2 if step == 0 else 7, 32), device="cuda", generator=generator
                    )
                },
                num_context_logits_prefix_sum=[0, 1, 2] if step == 0 else [0],
            )
            assert state.sampler_event is not None
            state.sampler_event.synchronize()
            sender.prepare_pp_relay(state)
            assert state.host is not None and state.host.beam_relay is not None
            relay = state.host.beam_relay
            assert relay.seq_lens == [length + step for length in prompt_lens]
            assert relay.beam_widths == [min(step + 2, 4), 3]
            assert (
                relay.predecessor_beams.numel() * relay.predecessor_beams.element_size() <= 32
            )
            relayed = SampleStateTorch(
                requests=received_requests, host=pickle.loads(pickle.dumps(state.host))
            )
            sender.update_requests(state)
            receiver.update_requests(relayed)
            for row, (sampled, received) in enumerate(zip(sampled_requests, received_requests)):
                slot = sampled.py_seq_slot
                width = relay.beam_widths[row]
                positions = prompt_lens[row] + step + 1
                assert torch.equal(
                    sender_store.cache_indirection[slot, :width, :positions],
                    receiver_store.cache_indirection[slot, :width, :positions],
                )
                assert [sampled.get_tokens(beam) for beam in range(sampled.py_beam_width)] == [
                    received.get_tokens(beam) for beam in range(received.py_beam_width)
                ]
                assert received.is_finished == sampled.is_finished == (step == 3)
                for request in (sampled, received):
                    if not request.is_finished:
                        request.state = LlmRequestState.GENERATION_IN_PROGRESS
                        request.decoding_iter = request.py_decoding_iter
        for sampled, received in zip(sampled_requests, received_requests):
            assert received.py_result.log_probs == sampled.py_result.log_probs
            assert received.py_result.cum_log_probs == sampled.py_result.cum_log_probs


@pytest.mark.threadleak(enabled=False)
def test_handoff_preserves_reused_slots_of_completed_requests() -> None:
    """A late state for a cancelled request must not overwrite the next slot owner."""
    sender, receiver = _make_sampler(), _make_sampler()
    sampled_requests = _make_requests([128, 8192])
    received_requests = _make_requests([128, 8192])
    scheduled = ScheduledRequests()
    scheduled.context_requests_last_chunk = sampled_requests
    state = sender.sample_async(
        scheduled,
        model_outputs={
            "logits": torch.randn(
                (2, 32), device="cuda", generator=torch.Generator(device="cuda").manual_seed(19)
            )
        },
        num_context_logits_prefix_sum=[0, 1, 2],
    )
    assert state.sampler_event is not None
    state.sampler_event.synchronize()
    sampled_requests[0].finish_by_reason(FinishReason.CANCELLED)
    received_requests[0].finish_by_reason(FinishReason.CANCELLED)
    sender.prepare_pp_relay(state)
    relayed = SampleStateTorch(
        requests=received_requests, host=pickle.loads(pickle.dumps(state.host))
    )
    sender.update_requests(state)

    store = receiver.store.beam_search_store
    sender_store = sender.store.beam_search_store
    assert store is not None and sender_store is not None
    slot = received_requests[0].py_seq_slot
    # Generation-only admission seeds its first token before its first forward.
    store.cache_indirection[slot].zero_()
    store.cache_indirection[slot, :, 2] = torch.arange(4, device="cuda", dtype=torch.int32)
    next_occupant = store.cache_indirection[slot].clone()
    receiver.update_requests(relayed)

    assert torch.equal(store.cache_indirection[slot], next_occupant)
    assert torch.equal(
        store.cache_indirection[1, :3, :8193], sender_store.cache_indirection[1, :3, :8193]
    )
    assert [received_requests[1].get_tokens(beam) for beam in range(3)] == [
        sampled_requests[1].get_tokens(beam) for beam in range(3)
    ]
