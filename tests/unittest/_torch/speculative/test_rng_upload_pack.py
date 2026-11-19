# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Philox seed/offset uploads of a sampled one-model step, staged into the step-input pack."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and hasattr(torch.ops.trtllm, "step_inputs_scatter")),
        reason="torch.ops.trtllm.step_inputs_scatter is not registered",
    ),
]

MAX_DRAFT_LEN = 3
ROWS_PER_REQUEST = MAX_DRAFT_LEN + 1


def _meta(pack_rng_uploads: bool) -> SpecMetadata:
    meta = SpecMetadata(
        max_num_requests=4, max_draft_len=MAX_DRAFT_LEN, max_total_draft_tokens=MAX_DRAFT_LEN
    )
    meta.pack_rng_uploads = pack_rng_uploads
    return meta


def _request(slot: int, seed: Optional[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        py_seq_slot=slot,
        py_decoding_iter=0,
        py_request_id=slot,
        sampling_config=types.SimpleNamespace(seed=seed),
    )


def _populate(meta: SpecMetadata) -> None:
    requests = [_request(2, seed=1234), _request(0, seed=None)]
    normalized = [(0.7, 0, 0.9, 0.0, ROWS_PER_REQUEST) for _ in requests]
    meta._populate_request_rng_state(requests, normalized)


def _rng_state(meta: SpecMetadata) -> dict[str, torch.Tensor]:
    torch.cuda.synchronize()
    return {
        name: getattr(meta, name).clone()
        for name in ("request_seeds", "request_offsets", "seeds", "offsets")
    }


def test_packed_uploads_land_at_the_flush_with_the_plain_values():
    plain = _meta(pack_rng_uploads=False)
    _populate(plain)
    expected = _rng_state(plain)

    packed = _meta(pack_rng_uploads=True)
    pack = StepInputPack(torch.device("cuda", torch.cuda.current_device()))
    with step_inputs.active(pack):
        step_inputs.begin()
        _populate(packed)
        before_flush = _rng_state(packed)
        step_inputs.flush()
    after_flush = _rng_state(packed)

    for name, tensor in expected.items():
        assert torch.equal(after_flush[name], tensor), name
    # Staged, not copied: the buffers still hold their zero init until the flush.
    assert not torch.equal(before_flush["seeds"], expected["seeds"])


def test_without_an_open_pack_the_uploads_are_plain_copies():
    plain = _meta(pack_rng_uploads=False)
    _populate(plain)
    expected = _rng_state(plain)

    packed = _meta(pack_rng_uploads=True)
    _populate(packed)

    for name, tensor in _rng_state(packed).items():
        assert torch.equal(tensor, expected[name]), name


def test_packing_is_off_by_default():
    assert not SpecMetadata(
        max_num_requests=1, max_draft_len=MAX_DRAFT_LEN, max_total_draft_tokens=MAX_DRAFT_LEN
    ).pack_rng_uploads
