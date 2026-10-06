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
"""Serving extension for gpt-oss (Harmony) checkpoints."""

from typing import Any, Dict, Optional

from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

_FINAL_CHANNEL_START = "<|start|>assistant<|channel|>final<|message|>"


@register_serving_extension(reasoning_parsers=("gpt_oss",))
class GptOssServingExtension(ServingExtension):
    """Structured output for the Harmony ``gpt_oss`` reasoning parser.

    Chat-request preprocessing is the generic path; only the placement of the
    guided-decoding constraint is model specific.
    """

    def structured_output_format(
        self, content: Dict[str, Any], chat_template_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Trigger the user constraint once the model opens its final channel.

        Analysis/commentary channels stay unconstrained; the constraint binds
        the message body of the first ``final`` channel and nothing after it.
        """
        return {
            "type": "triggered_tags",
            "triggers": [_FINAL_CHANNEL_START],
            "tags": [
                {
                    "begin": _FINAL_CHANNEL_START,
                    "content": content,
                    "end": "",
                },
            ],
            "stop_after_first": True,
        }
