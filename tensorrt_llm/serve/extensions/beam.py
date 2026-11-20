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
"""Request constraints and structured-output placement for Beam's text format."""

from typing import Any

from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

_FINAL_CHANNEL_START = "<|channel|>final<|message|>"


@register_serving_extension(model_types=("beam",), reasoning_parsers=("beam_harmony_text",))
class BeamServingExtension(ServingExtension):
    """Constrain the final answer while retaining Beam's text parser and chat template."""

    def validate_tools(self, tools: list[Any] | None) -> None:
        if tools:
            raise ValueError(
                "Tool calling is not supported with reasoning_parser='beam_harmony_text'. "
                "Omit tools or pass an empty tools list."
            )

    def structured_output_format(
        self, content: dict[str, Any], chat_template_kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        # The assistant prefix may already be in the prompt; trigger on the
        # final channel itself, whether or not an analysis message precedes it.
        return {
            "type": "triggered_tags",
            "triggers": [_FINAL_CHANNEL_START],
            "tags": [{"begin": _FINAL_CHANNEL_START, "content": content, "end": ""}],
            "stop_after_first": True,
        }
