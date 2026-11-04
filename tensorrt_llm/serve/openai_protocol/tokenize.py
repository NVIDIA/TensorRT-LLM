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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

from typing import List, Optional

from tensorrt_llm.serve.openai_protocol.common import OpenAIBaseModel


class TokenizeRequest(OpenAIBaseModel):
    """Request body for the ``POST /_internal/tokenize`` endpoint.

    Asks the server to encode some text into the token-id sequence its
    loaded tokenizer would produce, without running any generation.

    Only plain-text ``prompt`` tokenization is supported. A ``messages``
    path was intentionally left out: to be meaningful it must reuse the full
    ``/v1/chat/completions`` preprocessing (server chat template, tools,
    documents, template kwargs, model-type resolution, multimodal handling),
    otherwise the returned token ids would silently diverge from the actual
    inference prompt. It can be added once that path is shared.

    Fields:
        model: Target model name. Accepted for compatibility with
            multi-model routers but ignored -- the server always tokenizes
            against its own loaded model.
        prompt: Plain text to tokenize directly via ``tokenizer.encode``.
    """

    model: Optional[str] = None
    prompt: str


class TokenizeResponse(OpenAIBaseModel):
    """Response body for the ``POST /_internal/tokenize`` endpoint.

    Fields:
        count: Number of tokens in the encoded sequence. Always populated;
            clients needing only the length can ignore ``tokens``.
        tokens: The encoded token ids. Kept ``Optional`` so future
            count-only variants can omit it without breaking the schema.
    """

    count: int
    tokens: Optional[List[int]] = None
