# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Boundary validation for trtllm-serve request models (issue #6329).

Invalid sampling parameters must be rejected at the API boundary with a
pydantic ValidationError (surfaced to clients as HTTP 422) instead of crashing
the server downstream.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)


def _completion(**kw):
    return CompletionRequest(model="m", prompt="hi", **kw)


def _chat(**kw):
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **kw)


def _responses(**kw):
    return ResponsesRequest(model="m", input="hi", **kw)


# top_k only exists on completion / chat requests.
BUILDERS_WITH_TOP_K = [_completion, _chat]
ALL_BUILDERS = [_completion, _chat, _responses]


@pytest.mark.parametrize("build", ALL_BUILDERS)
def test_valid_sampling_params_accepted(build):
    # Boundary-valid values must pass unchanged.
    req = build(temperature=0.0, top_p=1.0)
    assert req.temperature == 0.0
    assert req.top_p == 1.0


@pytest.mark.parametrize("build", ALL_BUILDERS)
def test_negative_temperature_rejected(build):
    with pytest.raises(ValidationError):
        build(temperature=-0.1)


@pytest.mark.parametrize("build", ALL_BUILDERS)
@pytest.mark.parametrize("bad_top_p", [-0.1, 1.1, 2.0])
def test_out_of_range_top_p_rejected(build, bad_top_p):
    with pytest.raises(ValidationError):
        build(top_p=bad_top_p)


@pytest.mark.parametrize("build", BUILDERS_WITH_TOP_K)
def test_negative_top_k_rejected(build):
    # top_k = -1 is vLLM's default; TRT-LLM's backend requires top_k >= 0, so
    # reject it cleanly at the boundary rather than crashing.
    with pytest.raises(ValidationError):
        build(top_k=-1)


@pytest.mark.parametrize("build", BUILDERS_WITH_TOP_K)
def test_top_k_zero_accepted(build):
    assert build(top_k=0).top_k == 0
