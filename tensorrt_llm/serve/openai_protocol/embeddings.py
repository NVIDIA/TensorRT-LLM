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

import time
import uuid
from typing import List, Literal, Optional, Union

from pydantic import Field, PositiveInt

from tensorrt_llm.serve.openai_protocol.common import OpenAIBaseModel


class EmbeddingRequest(OpenAIBaseModel):
    # OpenAI-compatible embeddings request.
    # https://platform.openai.com/docs/api-reference/embeddings/create
    model: str = Field(description="The model to use for the embedding.")
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        description="Text(s) or pre-tokenized token-id list(s) to embed. Accepts "
        "the four OpenAI forms: str, list[str], list[int], list[list[int]].")
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the returned embedding: a JSON array of floats "
        "(\"float\") or a base64-encoded string of packed little-endian float32 "
        "values (\"base64\").")
    dimensions: Optional[PositiveInt] = Field(
        default=None,
        description="Number of dimensions the resulting embedding should have. "
        "Only supported by Matryoshka-trained text-embedding models; rejected "
        "otherwise.")
    user: Optional[str] = Field(
        default=None,
        description="A stable identifier for the end-user (OpenAI-compatible; "
        "accepted but unused).")
    add_special_tokens: bool = Field(
        default=True,
        description=
        "TRT-LLM extension: whether to add the model's special tokens "
        "(e.g. [CLS]/[SEP]) when tokenizing string inputs. Default matches "
        "llm.encode(); encoder models such as BERT generally need them.")


class EmbeddingResponseData(OpenAIBaseModel):
    index: int = Field(
        description="Index of this embedding in the request's input list.")
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], str] = Field(
        description=
        "The embedding: a list of floats for encoding_format=\"float\", "
        "or a base64 string of packed little-endian float32 values for "
        "encoding_format=\"base64\".")


class EmbeddingUsageInfo(OpenAIBaseModel):
    """Token usage for embeddings.

    OpenAI's embeddings `usage` object contains only `prompt_tokens` and
    `total_tokens` (no `completion_tokens`), unlike chat/completions.
    """
    prompt_tokens: int = Field(default=0,
                               description="Number of tokens in the input.")
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed (equal to prompt_tokens for "
        "embeddings).")


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{str(uuid.uuid4().hex)}")
    object: Literal["list"] = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(description="The model used for the embedding.")
    data: List[EmbeddingResponseData] = Field(
        description="The list of embeddings, one per input item.")
    usage: EmbeddingUsageInfo = Field(
        description="Token usage for the request.")
