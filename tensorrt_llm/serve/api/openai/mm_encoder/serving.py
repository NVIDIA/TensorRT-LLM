# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/chat/completions`` on the multimodal-encoder role."""

from typing import Any

from tensorrt_llm.serve._base.serving import GenerateBaseServing
from tensorrt_llm.serve.render import RenderResources


class MMEncoderServing(GenerateBaseServing):
    """``POST /v1/chat/completions`` on the multimodal-encoder role.

    Returns the multimodal embedding handles of the request instead of
    generated text. Rendering uses the ``RenderResources`` the composition
    root built once, never resources read back from this object.
    """

    def __init__(self, *, render_resources: RenderResources,
                 multimodal_server_config: Any,
                 allow_request_chat_template: bool, **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self._render_resources = render_resources
        self.multimodal_server_config = multimodal_server_config
        self.allow_request_chat_template = allow_request_chat_template
