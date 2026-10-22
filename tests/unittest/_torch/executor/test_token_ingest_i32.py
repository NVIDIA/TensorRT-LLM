# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""int32 token ingest: ``Request.input_token_ids_i32`` (owned copy) and the ``LlmRequest`` ctor's
memcpy of a 1-D contiguous int32 array, with the sequence-cast fallback for any other input."""

import gc

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest as CppLlmRequest

pytestmark = pytest.mark.cpu_only

_PROMPT = [7, 0, 2**31 - 1, 3, 151_643, 42, 1, 99_999]


def _make_cpp_request(input_tokens) -> CppLlmRequest:
    return CppLlmRequest(
        request_id=1,
        max_new_tokens=4,
        input_tokens=input_tokens,
        sampling_config=trtllm.SamplingConfig(),
        is_streaming=False,
    )


def test_input_token_ids_i32_matches_input_token_ids():
    request = trtllm.Request(input_token_ids=_PROMPT, max_tokens=4)
    arr = request.input_token_ids_i32

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    assert arr.ndim == 1
    assert arr.flags.c_contiguous
    assert arr.tolist() == request.input_token_ids == _PROMPT
    assert request.num_input_tokens == arr.shape[0]


def test_input_token_ids_i32_is_owned_copy():
    request = trtllm.Request(input_token_ids=_PROMPT, max_tokens=4)
    first = request.input_token_ids_i32
    second = request.input_token_ids_i32

    assert not np.shares_memory(first, second)
    first[0] = -5
    assert request.input_token_ids == _PROMPT
    assert second.tolist() == _PROMPT

    # The array owns its buffer and outlives the request it was copied from.
    del request
    gc.collect()
    assert second.tolist() == _PROMPT


@pytest.mark.parametrize(
    "make_tokens",
    [
        pytest.param(lambda: list(_PROMPT), id="list"),
        pytest.param(lambda: np.asarray(_PROMPT, dtype=np.int32), id="ndarray_i32"),
        pytest.param(lambda: np.asarray(_PROMPT, dtype=np.int64), id="ndarray_i64_fallback"),
        pytest.param(
            lambda: np.repeat(np.asarray(_PROMPT, dtype=np.int32), 2)[::2],
            id="ndarray_i32_strided_fallback",
        ),
    ],
)
def test_llm_request_ctor_round_trips_tokens(make_tokens):
    tokens = make_tokens()
    request = _make_cpp_request(tokens)

    assert request.get_tokens(0) == _PROMPT
    assert request.orig_prompt_len == len(_PROMPT)


def test_llm_request_ctor_copies_ndarray():
    tokens = np.asarray(_PROMPT, dtype=np.int32)
    request = _make_cpp_request(tokens)
    tokens[:] = 0

    assert request.get_tokens(0) == _PROMPT


def test_llm_request_ctor_rejects_non_integer_tokens():
    # The fallback nb::cast raises std::bad_cast, which nanobind maps to RuntimeError.
    with pytest.raises((TypeError, RuntimeError)):
        _make_cpp_request(["a", "b"])


def test_executor_request_to_llm_request_round_trips_tokens():
    executor_request = trtllm.Request(input_token_ids=_PROMPT, max_tokens=4)
    llm_request = executor_request_to_llm_request(
        3, executor_request, child_req_ids=[], exclude_last_generation_logits=False
    )

    assert llm_request.get_tokens(0) == _PROMPT
    assert llm_request.orig_prompt_len == len(_PROMPT)
