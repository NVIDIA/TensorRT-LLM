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
"""Serving extension for Nemotron checkpoints using required-tool grammars."""

from tensorrt_llm.serve.serving_extensions import (
    RequiredToolChoiceMode,
    ServingExtension,
    register_serving_extension,
)


@register_serving_extension(
    model_types=("nemotron_h", "nemotron_h_omni", "nemotron4"),
    reasoning_parsers=("nemotron-v3", "nano-v3"),
)
class NemotronServingExtension(ServingExtension):
    """Use mandatory tool-call grammars for native Nemotron chat requests."""

    def required_tool_choice_mode(self) -> RequiredToolChoiceMode:
        return "grammar"

    def structured_output_format(self, content: dict, chat_template_kwargs: dict | None):
        """Place the constraint after the reasoning the chat template has opened.

        The template already emits <think> (enabled) or <think></think> (disabled).
        """
        if not (chat_template_kwargs or {}).get("enable_thinking", True):
            return None
        return {
            "type": "sequence",
            "elements": [
                {"type": "tag", "begin": "", "content": {"type": "any_text"}, "end": "</think>"},
                content,
            ],
        }
