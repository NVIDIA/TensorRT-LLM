# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Philox seed/offset uploads of a sampled one-model step, merged into the step's pack by a copy plan."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.device_copy import CopyPlan, CopySite, DeviceCopier
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and hasattr(torch.ops.trtllm, "device_copy_scatter")),
        reason="torch.ops.trtllm.device_copy_scatter is not registered",
    ),
]

MAX_DRAFT_LEN = 3
ROWS_PER_REQUEST = MAX_DRAFT_LEN + 1


_RNG_SITES = {
    CopySite.SPEC_RNG_REQUEST_SEEDS,
    CopySite.SPEC_RNG_REQUEST_OFFSETS,
    CopySite.SPEC_RNG_SEEDS,
    CopySite.SPEC_RNG_OFFSETS,
}


@pytest.fixture(autouse=True)
def _no_check_mode(monkeypatch):
    # The test reads the merged destinations before the flush; check mode (on for unit tests) would poison them.
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


def _meta() -> SpecMetadata:
    return SpecMetadata(
        max_num_requests=4, max_draft_len=MAX_DRAFT_LEN, max_total_draft_tokens=MAX_DRAFT_LEN
    )


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


def test_merged_uploads_land_at_the_flush_with_the_plain_values():
    plain = _meta()
    _populate(plain)
    expected = _rng_state(plain)

    packed = _meta()
    copier = DeviceCopier(
        CopyPlan(merge=_RNG_SITES), torch.device("cuda", torch.cuda.current_device())
    )
    with device_copy.active(copier):
        device_copy.begin()
        _populate(packed)
        before_flush = _rng_state(packed)
        device_copy.flush()
    after_flush = _rng_state(packed)

    for name, tensor in expected.items():
        assert torch.equal(after_flush[name], tensor), name
    # Staged, not copied: the buffers still hold their zero init until the flush.
    assert not torch.equal(before_flush["seeds"], expected["seeds"])


def test_without_an_open_pack_the_uploads_are_plain_copies():
    plain = _meta()
    _populate(plain)
    expected = _rng_state(plain)

    packed = _meta()
    copier = DeviceCopier(
        CopyPlan(merge=_RNG_SITES), torch.device("cuda", torch.cuda.current_device())
    )
    with device_copy.active(copier):
        _populate(packed)

    for name, tensor in _rng_state(packed).items():
        assert torch.equal(tensor, expected[name]), name


def test_a_plan_without_the_rng_sites_copies_them_plainly():
    """Merging is the plan's choice per site: a plan that merges other sites leaves these uploads plain."""
    plain = _meta()
    _populate(plain)
    expected = _rng_state(plain)

    other = _meta()
    copier = DeviceCopier(
        CopyPlan(merge={CopySite.ATTN_KV_LENS}), torch.device("cuda", torch.cuda.current_device())
    )
    with device_copy.active(copier):
        device_copy.begin()
        _populate(other)
        before_flush = _rng_state(other)
        device_copy.flush()
    for name, tensor in expected.items():
        assert torch.equal(before_flush[name], tensor), name
