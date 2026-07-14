# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the fp8 context-MLA attention-workspace reservation and its admission cap.

The estimator reserves KV-cache headroom for the total_kv_len-scaled context-MLA workspace, and the
scheduler caps summed context attended-KV length at the resulting pool capacity so that workspace can
never exceed the reservation (which would OOM mid-forward). These tests cover the pure-Python halves:
the per-request attended-KV computation, the admission trim, and the non-MLA no-op gate.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import get_mla_context_workspace_bytes_per_token
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _ctx_req(
    context_current_position,
    is_first_context_chunk,
    estimated_reusable_tokens,
    context_chunk_size,
    orig_prompt_len,
):
    return SimpleNamespace(
        context_current_position=context_current_position,
        is_first_context_chunk=is_first_context_chunk,
        estimated_reusable_tokens=estimated_reusable_tokens,
        context_chunk_size=context_chunk_size,
        orig_prompt_len=orig_prompt_len,
    )


@pytest.mark.parametrize(
    "req,expected",
    [
        # V1: reuse not yet applied at schedule time; credit is in estimated_reusable_tokens.
        (_ctx_req(0, True, 100, 50, 150), 150),
        # V2: context_current_position already advanced past the reused prefix.
        (_ctx_req(100, True, 100, 50, 150), 150),
        # Fresh prefill, no reuse.
        (_ctx_req(0, True, 0, 80, 80), 80),
        # Middle (non-first) chunk: reuse credit does not apply, position already advanced.
        (_ctx_req(200, False, 0, 100, 500), 300),
        # Reuse estimate above the prompt length is clamped to the prompt length.
        (_ctx_req(0, True, 999, 50, 150), 150),
    ],
)
def test_context_attended_kv_len(req, expected):
    assert PyExecutor._context_attended_kv_len(req) == expected


def _make_executor(cap):
    # Bypass __init__; the trim only reads the cached cap and the two helper methods.
    exe = object.__new__(PyExecutor)
    exe._ctx_mla_kv_len_cap = cap  # pre-set so the cap getter skips the C++ binding
    return exe


def test_cap_trims_tail_by_total_kv_len():
    exe = _make_executor(cap=120)
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(3)]  # cumulative 50, 100, 150
    kept = exe._cap_context_by_total_kv_len(reqs)
    assert len(kept) == 2  # the third would push cumulative to 150 > 120


def test_cap_keeps_all_when_within_budget():
    exe = _make_executor(cap=200)
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(3)]
    assert len(exe._cap_context_by_total_kv_len(reqs)) == 3


def test_cap_always_keeps_first_even_if_it_alone_exceeds():
    exe = _make_executor(cap=120)
    reqs = [_ctx_req(0, True, 0, 200, 200), _ctx_req(0, True, 0, 50, 50)]
    kept = exe._cap_context_by_total_kv_len(reqs)
    assert len(kept) == 1  # first kept despite exceeding cap (forward-progress guard)


def test_cap_single_request_never_trimmed():
    exe = _make_executor(cap=1)
    reqs = [_ctx_req(0, True, 0, 500, 500)]
    assert exe._cap_context_by_total_kv_len(reqs) == reqs


def test_no_cap_returns_untouched():
    exe = _make_executor(cap=None)  # non-fp8-MLA model: no reservation, no trimming
    reqs = [_ctx_req(0, True, 0, 50, 50) for _ in range(5)]
    assert exe._cap_context_by_total_kv_len(reqs) is reqs


def test_workspace_bytes_zero_for_non_mla_model():
    # No kv_lora_rank on the config -> not MLA -> 0 (early return, no binding call needed).
    model_config = SimpleNamespace(pretrained_config=SimpleNamespace(), quant_config=None)
    assert get_mla_context_workspace_bytes_per_token(model_config, Mock()) == 0
