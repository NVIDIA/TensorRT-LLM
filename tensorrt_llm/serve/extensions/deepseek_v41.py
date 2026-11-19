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
"""Serving extension for DeepSeek-V4.1 checkpoints.

The checkpoint's prompt renderer (``tensorrt_llm/tokenizer/deepseek_v41``)
reads tool namespaces itself, so a tool declared inside a namespace is handed
to it in that form rather than flattened.
"""

from typing import Any, Dict

from tensorrt_llm.serve.openai_protocol import NamespacedFunctionToolParam
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension


@register_serving_extension(model_types=("deepseek_v41", "deepseek_v41_text"))
class DeepseekV41ServingExtension(ServingExtension):
    """Tool namespaces in the form the DeepSeek-V4.1 checkpoint takes them."""

    def serialize_tool(self, tool) -> Dict[str, Any]:
        """Dump a namespaced tool with its namespace, any other tool as usual.

        The checkpoint's encoder reads a ``namespace`` (name and description)
        beside ``function``. It shows the tool as ``<namespace>::<name>``, the
        name the model was trained to call it by, and puts the namespace
        description in front of the tool's own. A tool without a namespace is
        dumped exactly as on the generic path.
        """
        dumped = tool.model_dump()
        if not isinstance(tool, NamespacedFunctionToolParam):
            return dumped
        if "::" in tool.namespace or "::" in tool.declared_name:
            # The encoder rejects such a name; the flattened one renders.
            return dumped
        function = dumped["function"]
        function["name"] = tool.declared_name
        function["description"] = tool.declared_description
        return {
            "type": dumped["type"],
            "namespace": {
                "name": tool.namespace,
                "description": tool.namespace_description,
            },
            "function": function,
        }
