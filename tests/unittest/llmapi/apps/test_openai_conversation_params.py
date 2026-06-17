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
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ConversationParams


def test_openai_conversation_params_requires_conversation_id():
    with pytest.raises(ValidationError):
        ConversationParams()


def test_openai_conversation_params_normalizes_conversation_id():
    params = ConversationParams(conversation_id=" body-id ")

    assert params.conversation_id == "body-id"

    params.conversation_id = " next-id "

    assert params.conversation_id == "next-id"


def test_openai_conversation_params_rejects_empty_conversation_id():
    with pytest.raises(ValidationError):
        ConversationParams(conversation_id=" ")
