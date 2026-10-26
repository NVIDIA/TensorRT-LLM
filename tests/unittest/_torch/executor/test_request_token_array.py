# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompt tokens handed to LlmRequest as the executor request's int32 array: which requests take it (context-only by
default, every type with the opt-in, none with the kill switch), and the same request as the list path."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor import llm_request as lr_mod
from tensorrt_llm.bindings import executor as tllm

pytestmark = pytest.mark.cpu_only

PROMPT = [7, 0, 151643, 2**31 - 1, 42, 3]


def _executor_request(tokens=PROMPT):
    return tllm.Request(input_token_ids=list(tokens), max_tokens=4)


def test_input_token_ids_np_is_an_owned_int32_copy():
    req = _executor_request()
    arr = req.input_token_ids_np
    assert arr.dtype == np.int32 and arr.ndim == 1 and arr.flags["C_CONTIGUOUS"]
    assert arr.tolist() == PROMPT == req.input_token_ids
    arr[0] = -5  # the copy is the caller's: the request keeps its tokens
    assert req.input_token_ids == PROMPT
    del req
    gc.collect()
    assert arr.tolist() == [-5] + PROMPT[1:]  # and it outlives the request


def test_input_token_ids_np_reads_again():
    # Each read returns its own owned copy; a borrowed (reference_internal) property would raise on the second read.
    req = _executor_request()
    first, second = req.input_token_ids_np, req.input_token_ids_np
    assert first.tolist() == second.tolist() == PROMPT
    assert not np.shares_memory(first, second)


def test_empty_prompt_is_rejected():
    # The executor request needs a prompt, so the array accessor never sees an empty one.
    with pytest.raises(RuntimeError):
        tllm.Request(input_token_ids=[], max_tokens=4)


RT = tllm.RequestType


def _typed_request(request_type, with_array=True):
    req = SimpleNamespace(request_type=request_type, input_token_ids=list(PROMPT))
    if with_array:
        req.input_token_ids_np = np.asarray(PROMPT, dtype=np.int32)
    return req


def _scope(monkeypatch, on=True, all_requests=False):
    monkeypatch.setattr(lr_mod, "_L1_TOKEN_ARRAY", on)
    monkeypatch.setattr(lr_mod, "_L1_ALL_REQUESTS", all_requests)


@pytest.mark.parametrize(
    "request_type,array",
    [
        (RT.REQUEST_TYPE_CONTEXT_ONLY, True),
        (RT.REQUEST_TYPE_CONTEXT_AND_GENERATION, False),
        (RT.REQUEST_TYPE_GENERATION_ONLY, False),
    ],
)
def test_default_scope_is_context_only(monkeypatch, request_type, array):
    _scope(monkeypatch)
    out = lr_mod._l1_input_tokens(_typed_request(request_type))
    assert isinstance(out, np.ndarray) is array
    assert list(out) == PROMPT


def test_opt_in_covers_every_request_type(monkeypatch):
    _scope(monkeypatch, all_requests=True)
    for request_type in (RT.REQUEST_TYPE_CONTEXT_AND_GENERATION, RT.REQUEST_TYPE_GENERATION_ONLY):
        assert isinstance(lr_mod._l1_input_tokens(_typed_request(request_type)), np.ndarray)


def test_kill_switch_wins_over_the_opt_in(monkeypatch):
    _scope(monkeypatch, on=False, all_requests=True)
    assert lr_mod._l1_input_tokens(_typed_request(RT.REQUEST_TYPE_CONTEXT_ONLY)) == PROMPT


def test_request_without_the_array_uses_the_list(monkeypatch):
    _scope(monkeypatch, all_requests=True)
    out = lr_mod._l1_input_tokens(_typed_request(RT.REQUEST_TYPE_CONTEXT_ONLY, with_array=False))
    assert out == PROMPT


@pytest.mark.parametrize("all_requests", [True, False])
def test_llm_request_tokens_match_the_list_path(monkeypatch, all_requests):
    # A context-and-generation request: the array path with the opt-in, the list path without it.
    _scope(monkeypatch, all_requests=all_requests)
    llm_req = lr_mod.executor_request_to_llm_request(
        req_id=1,
        executor_request=_executor_request(),
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )
    assert llm_req.get_tokens(0) == PROMPT
    assert llm_req.orig_prompt_len == len(PROMPT)


def test_explicit_input_token_ids_win(monkeypatch):
    _scope(monkeypatch, all_requests=True)
    llm_req = lr_mod.executor_request_to_llm_request(
        req_id=2,
        executor_request=_executor_request(),
        child_req_ids=[],
        exclude_last_generation_logits=False,
        input_token_ids=[9, 8, 7],
    )
    assert llm_req.get_tokens(0) == [9, 8, 7]


def test_llm_request_accepts_an_int32_array_and_a_list():
    def make(tokens):
        return lr_mod.LlmRequest(
            request_id=3,
            max_new_tokens=4,
            input_tokens=tokens,
            sampling_config=tllm.SamplingConfig(1),
            is_streaming=False,
        )

    from_array = make(np.asarray(PROMPT, dtype=np.int32))
    from_list = make(list(PROMPT))
    assert from_array.get_tokens(0) == from_list.get_tokens(0) == PROMPT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA tensor")
def test_llm_request_rejects_a_device_array():
    # Only a host int32 array is copied as raw memory; a CUDA tensor takes the sequence cast, which rejects it (it is
    # never read through a device pointer).
    tokens = torch.tensor(PROMPT, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        lr_mod.LlmRequest(
            request_id=4,
            max_new_tokens=4,
            input_tokens=tokens,
            sampling_config=tllm.SamplingConfig(1),
            is_streaming=False,
        )
