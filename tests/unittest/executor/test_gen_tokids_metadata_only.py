# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the disagg-gen prompt_token_ids host-cost optimizations.

* Phase 1 -- ``GenerationRequest`` int32-ndarray lazy path: constructing with an
int32 ndarray must stash a lazy ``_prompt_token_ids_i32`` buffer (no O(ISL)
``.tolist()`` on the GIL-held submit thread) while still yielding the REAL
token values on read and across a pickle (RPC) round-trip.

* Phase 0 -- ``gen_tokids_metadata_only`` guardrail: requests that read the
prompt token VALUES (sampling penalties / echo) must be detected so the
orchestrator relays real tokens instead of the length-only placeholder.
"""

import pickle

import numpy as np
import pytest

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams

REAL = list(range(4096))


def _make(prompt_token_ids):
    return GenerationRequest(prompt_token_ids, sampling_params=SamplingParams(max_tokens=8))


def test_int32_ndarray_is_lazy():
    req = _make(np.asarray(REAL, dtype=np.int32))
    # Lazy: the list is NOT built at construction; the int32 buffer is stashed.
    assert req.__dict__["_prompt_token_ids"] is None
    assert req.__dict__.get("_prompt_token_ids_i32") is not None
    # Reading the property materializes the REAL values (not a placeholder).
    assert req.prompt_token_ids == REAL
    # Cached after first read.
    assert req.__dict__["_prompt_token_ids"] == REAL


def test_int32_ndarray_pickle_roundtrip_preserves_real_values():
    req = _make(np.asarray(REAL, dtype=np.int32))
    state = pickle.loads(pickle.dumps(req.__getstate__()))
    # Wire form is int32 bytes (no per-token PyLong frame).
    assert isinstance(state["_prompt_token_ids"], tuple)
    recv = GenerationRequest.__new__(GenerationRequest)
    recv.__setstate__(state)
    # Receiver stays lazy but yields the REAL values on read.
    assert recv.__dict__["_prompt_token_ids"] is None
    assert recv.prompt_token_ids == REAL


def test_list_path_unchanged():
    req = _make(list(REAL))
    assert req.prompt_token_ids == REAL


def test_non_int32_ndarray_eager_tolist():
    req = _make(np.asarray(REAL, dtype=np.int64))
    assert req.prompt_token_ids == REAL


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({}, False),
        ({"frequency_penalty": 0.0}, False),
        ({"presence_penalty": 0.0}, False),
        ({"repetition_penalty": 1.0}, False),
        ({"logprobs": True}, False),  # output logprobs -> fast path OK
        ({"frequency_penalty": 0.5}, True),
        ({"presence_penalty": 0.3}, True),
        ({"repetition_penalty": 1.1}, True),
        ({"echo": True}, True),
    ],
)
def test_metadata_only_guardrail(attrs, expected):
    from tensorrt_llm.serve.openai_disagg_service import _chat_request_reads_prompt_token_values

    class _Req:
        pass

    req = _Req()
    for k, v in attrs.items():
        setattr(req, k, v)
    assert _chat_request_reads_prompt_token_values(req) is expected


def test_prompt_inputs_ndarray_preserved():
    """An int32 ndarray survives prompt_inputs as a TokensPrompt (no .tolist())."""
    from tensorrt_llm.inputs.data import prompt_inputs

    arr = np.asarray(REAL, dtype=np.int32)
    out = prompt_inputs(arr)
    assert out["prompt_token_ids"] is arr  # same object, not copied to a list
