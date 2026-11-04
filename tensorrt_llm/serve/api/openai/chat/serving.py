# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/chat/completions``: chat, and gpt-oss through ``chat_harmony``."""

from typing import Any, Optional

from tensorrt_llm.serve._base.serving import GenerateBaseServing
from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter
from tensorrt_llm.serve.render import RenderResources


class ChatServing(GenerateBaseServing):
    """``POST /v1/chat/completions``: chat, and gpt-oss through ``chat_harmony``.

    No harmony subclass: ``chat_harmony`` is a method here until the temporary
    switches are removed, chosen by the router exactly as the server chose it
    before this class existed. The class is also the chat backend the Anthropic
    Messages serving class runs on (``stream_model`` and ``error_message``).
    """

    def __init__(self, *, tool_parser: Optional[str], tool_call_id_type: str,
                 multimodal_server_config: Any,
                 allow_request_chat_template: bool, use_harmony: bool,
                 harmony_adapter: Optional[HarmonyAdapter],
                 render_resources: RenderResources, **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self.tool_parser = tool_parser
        self.tool_call_id_type = tool_call_id_type
        self.multimodal_server_config = multimodal_server_config
        self.allow_request_chat_template = allow_request_chat_template
        self.use_harmony = use_harmony
        # The gpt-oss adapter, resolved lazily by chat_harmony on first use,
        # as the server did.
        self.harmony_adapter = harmony_adapter
        # Built once by the composition root; never RenderResources.from_server
        # on this object, which would drop the server chat template.
        self._render_resources = render_resources
