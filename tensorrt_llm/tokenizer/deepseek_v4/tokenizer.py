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

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..tokenizer import TransformersTokenizer

BOS_TOKEN = "<｜begin▁of▁sentence｜>"  # nosec B105
EOS_TOKEN = "<｜end▁of▁sentence｜>"  # nosec B105
USER_TOKEN = "<｜User｜>"  # nosec B105
ASSISTANT_TOKEN = "<｜Assistant｜>"  # nosec B105
THINKING_END_TOKEN = "</think>"  # nosec B105


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n\n".join(parts)
    return str(content)


class DeepseekV4Tokenizer(TransformersTokenizer):
    """DeepSeek-V4 tokenizer with the checkpoint reference chat format."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "DeepseekV4Tokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        return cls(tokenizer)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        if tools:
            raise NotImplementedError("DeepSeek-V4 tool-call chat formatting is not supported yet.")

        add_generation_prompt = kwargs.get("add_generation_prompt", True)
        tokenize = kwargs.get("tokenize", False)

        rendered = BOS_TOKEN
        for idx, message in enumerate(messages):
            role = message.get("role")
            content = _message_content_to_text(message.get("content"))
            next_role = messages[idx + 1].get("role") if idx + 1 < len(messages) else None

            if role == "system":
                rendered += content
            elif role in ("user", "developer"):
                rendered += USER_TOKEN + content
                if next_role == "assistant" or (next_role is None and add_generation_prompt):
                    rendered += ASSISTANT_TOKEN + THINKING_END_TOKEN
            elif role == "assistant":
                rendered += content + EOS_TOKEN
            else:
                raise NotImplementedError(f"Unsupported DeepSeek-V4 message role: {role}")

        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered
