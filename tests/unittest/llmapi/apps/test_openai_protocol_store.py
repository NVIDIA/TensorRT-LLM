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
"""A chat request may state OpenAI's store field, as long as it asks for no storage."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only

_REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("store", [{}, {"store": False}, {"store": None}])
def test_chat_request_accepts_no_storage(store):
    assert not ChatCompletionRequest(**_REQUEST, **store).store


def test_chat_request_refuses_storage():
    with pytest.raises(ValidationError, match="store"):
        ChatCompletionRequest(**_REQUEST, store=True)
