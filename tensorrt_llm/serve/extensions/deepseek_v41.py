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
"""Serving extension for DeepSeek-V4.1 checkpoints (``model_type == "deepseek_v41"``).

Preserve namespaces for the checkpoint prompt renderer and declare token runs
that mark a generation that has broken down.
"""

from typing import Any, Dict, Optional

from tensorrt_llm.serve.openai_protocol import NamespacedFunctionToolParam
from tensorrt_llm.serve.serving_extensions import (
    DegenerateTokens,
    ServingExtension,
    last_vocabulary_id,
    register_serving_extension,
)


@register_serving_extension(model_types=("deepseek_v41", "deepseek_v41_text"))
class DeepseekV41ServingExtension(ServingExtension):
    """DeepSeek-V4.1 serving behavior."""

    def degenerate_tokens(self, hf_config: Any) -> Optional[DegenerateTokens]:
        """``<｜▁pad▁｜>`` and the last vocabulary id, ``<|place_holder_mm_span_0442|>``.

        Neither belongs in generated text: the pad token fills batches, and the
        last vocabulary id is a multimodal placeholder that no prompt uses. The
        pad id comes from the model config, not the tokenizer, whose pad token
        is ``<｜end▁of▁sentence｜>``.
        """
        pad_token_id = getattr(hf_config, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            pad_token_id = None
        candidates = (pad_token_id, last_vocabulary_id(hf_config))
        token_ids = tuple(dict.fromkeys(i for i in candidates if i is not None))
        return DegenerateTokens(token_ids=token_ids) if token_ids else None

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
