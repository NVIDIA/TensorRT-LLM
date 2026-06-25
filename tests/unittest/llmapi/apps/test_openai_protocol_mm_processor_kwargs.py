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
"""Unit tests for ChatCompletionRequest.mm_processor_kwargs.

This field carries per-request kwargs to the multimodal HF processor
(e.g. ``num_frames`` for video models). The server reads it via
``request.mm_processor_kwargs`` and attaches it to the prompt before
dispatch, so the request schema is the authoritative contract.
"""

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest


def _base_request(**extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


class TestChatCompletionMmProcessorKwargs:
    def test_default_is_none(self):
        """Field is optional; absent → None (so the server's truthiness check skips it)."""
        req = ChatCompletionRequest(**_base_request())
        assert req.mm_processor_kwargs is None

    def test_accepts_arbitrary_dict(self):
        """Field is Dict[str, Any]; a populated dict round-trips unchanged."""
        kwargs = {"num_frames": 8, "fps": None, "max_pixels": 1234}
        req = ChatCompletionRequest(**_base_request(mm_processor_kwargs=kwargs))
        assert req.mm_processor_kwargs == kwargs
