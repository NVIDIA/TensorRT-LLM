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
import pytest

from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS, SamplingParams, check_logprobs_limit
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs", "top_logprobs"])
def test_check_logprobs_limit(field):
    check_logprobs_limit(field, None)
    check_logprobs_limit(field, 0)
    check_logprobs_limit(field, MAX_TOP_LOGPROBS)

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        check_logprobs_limit(field, -1)

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        check_logprobs_limit(field, MAX_TOP_LOGPROBS + 1)


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_logprobs_request_limit(field):
    SamplingParams(**{field: MAX_TOP_LOGPROBS})

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        SamplingParams(**{field: MAX_TOP_LOGPROBS + 1})


def test_chat_top_logprobs_request_limit():
    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=MAX_TOP_LOGPROBS + 1,
        )


def test_completion_logprobs_assignment_revalidates():
    request = CompletionRequest(model="test", prompt="hi", logprobs=MAX_TOP_LOGPROBS + 1)

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        request.to_sampling_params(backend="pytorch")
