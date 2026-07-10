# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``TokenizeRequest`` schema.

The ``/_internal/tokenize`` endpoint only tokenizes a plain-text ``prompt``;
it is the required field of the request. ``messages`` is intentionally
unsupported (see ``TokenizeRequest`` docstring), so it must be rejected.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import TokenizeRequest


def test_missing_prompt_rejected():
    with pytest.raises(ValidationError):
        TokenizeRequest()


def test_messages_rejected():
    with pytest.raises(ValidationError):
        TokenizeRequest(messages=[{"role": "user", "content": "hello"}])
