# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/glm47_moe_detector.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple

from tensorrt_llm.serve.tool_parser.glm4_parser import Glm4ToolParser


class Glm47ToolParser(Glm4ToolParser):
    r"""Tool parser for GLM-4.7 and GLM-5 models.

    GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
      - The function name may appear on the same line as ``<tool_call>`` without
        a newline separator before the first ``<arg_key>``.
      - Tool calls may have zero arguments
        (e.g. ``<tool_call>func</tool_call>``).

    Example format::

        <tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>
        <arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>

    Or zero-argument::

        <tool_call>get_time</tool_call>
    """

    def _split_call(self, body: str) -> Tuple[str, str, str]:
        """Split a call's body into (name, markup after the name, argument text).

        The name runs to the first ``<`` and the arguments from the first ``<arg_key>``.
        """
        name_end = body.find("<")
        if name_end == -1:
            return body, "", ""
        args_start = body.find("<arg_key>", name_end)
        if args_start == -1:
            args_start = len(body)
        return body[:name_end], body[name_end:args_start], body[args_start:]
