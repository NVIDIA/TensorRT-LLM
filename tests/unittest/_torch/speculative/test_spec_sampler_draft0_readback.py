# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the draft-0 host-readback shrink in SpecSampler.

In the draft-0 speculative band (runtime_draft_len == 0) speculation is disabled
for the batch: every request accepts exactly its single base token
(new_tokens_lens == 1) and produces no next-iteration draft tokens.
``update_requests`` therefore reads back only the first accepted-token row and
skips the ``next_draft_tokens`` host copy entirely, and
``_request_common_handling`` assigns an empty ``py_draft_tokens`` directly
(equivalent to the old ``[:0]`` slice). The draft>0 path is unchanged.

These exercise the pure host-side logic with lightweight stand-ins
(``types.SimpleNamespace`` + a recording tensor shim), so they construct no real
GPU tensors and run no model forward.
"""

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateSpec, SpecSampler


def _make_request(seq_slot: int = 0):
    return SimpleNamespace(
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=seq_slot,
        py_draft_tokens=None,
        py_decoding_iter=0,
        py_num_accepted_draft_tokens=0,
        py_num_draft_tokens_verified=0,
        py_rewind_len=0,
    )


# ---- _request_common_handling ----


def test_common_handling_draft0_none_yields_empty():
    req = _make_request()
    SpecSampler._request_common_handling(object(), req, None, 0)
    assert req.py_draft_tokens == []
    assert req.py_decoding_iter == 1


def test_common_handling_draft0_with_buffer_yields_empty():
    # Even when a next_draft_tokens buffer is present, runtime_draft_len == 0
    # slices it to empty (the [:0] slice), matching the None fast path.
    req = _make_request(seq_slot=1)
    SpecSampler._request_common_handling(object(), req, [[7, 8], [9, 10]], 0)
    assert req.py_draft_tokens == []


def test_common_handling_draft_positive_slices_per_slot():
    req = _make_request(seq_slot=1)
    SpecSampler._request_common_handling(object(), req, [[7, 8, 9], [10, 11, 12]], 2)
    assert req.py_draft_tokens == [10, 11]
    assert req.py_decoding_iter == 1


# ---- update_requests readback branch ----


class _RecordTensor:
    """Minimal tensor stand-in recording tolist()/slice access."""

    def __init__(self, data):
        self.data = data
        self.tolist_calls = 0
        self.slices = []

    def tolist(self):
        self.tolist_calls += 1
        return self.data

    def __getitem__(self, idx):
        self.slices.append(idx)
        return _RecordTensor(self.data[idx])


def _make_state(runtime_draft_len: int):
    # new_tokens is [max_accepted_path_len, seq_slots, beam]; one slot, one beam.
    new_tokens = _RecordTensor([[[101]], [[102]], [[103]]])
    new_tokens_lens = _RecordTensor([1])
    next_draft_tokens = _RecordTensor([[0, 0]])
    state = SampleStateSpec.__new__(SampleStateSpec)
    state.sampler_event = SimpleNamespace(synchronize=lambda: None)
    state.host = SimpleNamespace(
        new_tokens=new_tokens,
        new_tokens_lens=new_tokens_lens,
        next_draft_tokens=next_draft_tokens,
    )
    state.runtime_draft_len = runtime_draft_len
    state.requests = [_make_request()]
    state.draft_lens = None
    return state, new_tokens, next_draft_tokens


def _make_sampler():
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.draft_len = 2
    sampler.max_seq_len = 4096
    sampler.max_accepted_path_len = 3
    return sampler


def _patch_token_helpers(monkeypatch):
    import tensorrt_llm._torch.speculative.spec_sampler_base as ssb

    monkeypatch.setattr(ssb, "add_token", lambda *a, **k: 0)
    monkeypatch.setattr(ssb, "handle_stop_criteria", lambda *a, **k: False)


def test_update_requests_draft0_skips_next_draft_readback(monkeypatch):
    _patch_token_helpers(monkeypatch)
    state, new_tokens, next_draft_tokens = _make_state(runtime_draft_len=0)

    _make_sampler().update_requests(state)

    # Draft-0: next_draft_tokens host copy is skipped, new_tokens sliced to [:1].
    assert next_draft_tokens.tolist_calls == 0
    assert slice(None, 1, None) in new_tokens.slices
    assert state.requests[0].py_draft_tokens == []


def test_update_requests_draft_positive_reads_next_draft(monkeypatch):
    _patch_token_helpers(monkeypatch)
    state, new_tokens, next_draft_tokens = _make_state(runtime_draft_len=2)

    _make_sampler().update_requests(state)

    # Draft>0: full readback of both buffers, no slice of new_tokens.
    assert next_draft_tokens.tolist_calls == 1
    assert new_tokens.tolist_calls == 1
    assert new_tokens.slices == []
