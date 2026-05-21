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

import json

from tensorrt_llm.executor.result import GenerationResultBase
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.serve.postprocess_handlers import (ChatPostprocArgs,
                                                     chat_stream_post_processor)


def _parse_sse_data(chunk: str) -> dict:
    assert chunk.startswith("data: ")
    return json.loads(chunk[len("data: "):])


def test_chat_stream_post_processor_keeps_empty_text_token_chunk():
    result = GenerationResultBase(id=1, sampling_params=SamplingParams())
    output = result.outputs[0]
    output.token_ids = [123]
    output.text = ""

    args = ChatPostprocArgs(tokenizer=None,
                            role="assistant",
                            model="test-model",
                            first_iteration=False)

    chunks = chat_stream_post_processor(result, args)

    assert len(chunks) == 1
    data = _parse_sse_data(chunks[0])
    choice = data["choices"][0]
    assert choice["delta"]["content"] == ""
    assert choice["finish_reason"] is None


def test_chat_stream_post_processor_skips_empty_non_token_chunk():
    result = GenerationResultBase(id=1, sampling_params=SamplingParams())
    output = result.outputs[0]
    output.token_ids = []
    output.text = ""

    args = ChatPostprocArgs(tokenizer=None,
                            role="assistant",
                            model="test-model",
                            first_iteration=False)

    assert chat_stream_post_processor(result, args) == []
