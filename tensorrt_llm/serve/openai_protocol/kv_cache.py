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

from tensorrt_llm.serve.openai_protocol.chat import ChatCompletionMessageParam
from tensorrt_llm.serve.openai_protocol.common import OpenAIBaseModel
from tensorrt_llm.serve.openai_protocol.tools import ChatCompletionToolsParam


class KVCacheTruncateRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam] = []
    messages_to_retain: List[ChatCompletionMessageParam] = []
    tools: Optional[List[ChatCompletionToolsParam]] = None
    add_generation_prompt: Optional[bool] = True
    documents: Optional[list] = None
    chat_template: Optional[str] = None
    chat_template_kwargs: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    tool_choice: Optional[str] = None


class KVCacheTruncateTokensRequest(OpenAIBaseModel):
    """Token-level analog of :class:`KVCacheTruncateRequest`.

    Posted to the ``/_control/kv_cache/truncate_tokens`` endpoint. Bypasses
    the chat-template tokenization path entirely: callers (e.g.
    trace replay) that already operate on raw token ids hand the server a
    batch of full token-id sequences plus the prefix length to retain for
    each, and the server trims the corresponding KV-cache blocks in the
    radix tree. Each ``prefixes[i]`` becomes one
    :class:`tensorrt_llm.executor.request.TruncateKVCacheRequest` on the
    executor's KV-cache control queue, so one batched HTTP request fans out
    into N atomic per-prefix radix-tree mutations while paying a single
    network round-trip.

    Fields:
        model: Target model name. Accepted for router compatibility but
            ignored -- the server uses its own loaded model.
        prefixes: Batch of full token-id sequences. Each entry is one
            cached prefix whose KV-cache blocks should be truncated.
        num_tokens_to_keep: Per-prefix number of leading tokens to retain,
            parallel to ``prefixes`` (``num_tokens_to_keep[i]`` applies to
            ``prefixes[i]``). Blocks beyond this length are dropped from the
            radix tree.
    """

    model: Optional[str] = None
    prefixes: List[List[int]]
    num_tokens_to_keep: List[int]
