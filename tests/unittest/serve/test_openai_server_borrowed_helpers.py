# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OpenAIServer`` borrows the helpers that moved to the serving bases; it keeps no copies.

A helper that moved into a serving base while handlers still on ``OpenAIServer``
call it is reachable on the class as the moved object itself. A copy would
drift from the moved one, so identity is what this file pins. Each move that
takes the last ``OpenAIServer`` caller of a helper removes its row here.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve._base.serving import BaseServing, GenerateBaseServing
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

# Helper name -> the base class it moved to.
BORROWED = {
    "create_error_response": BaseServing,
    "await_disconnected": GenerateBaseServing,
    "_attach_multimodal_data": GenerateBaseServing,
}


@pytest.mark.parametrize("name", sorted(BORROWED))
def test_openai_server_borrows_the_moved_helper(name):
    base = BORROWED[name]
    assert getattr(OpenAIServer, name) is getattr(base, name), (
        f"OpenAIServer.{name} must be the helper moved to {base.__name__} "
        f"(staticmethod({base.__name__}.{name}) for a static method, {base.__name__}.{name} "
        "otherwise); never copy a moved helper back: a copy drifts from the moved one"
    )


def test_static_helpers_survived_aliasing():
    # A bare `Base.helper` alias would bind the server as the first argument.
    assert OpenAIServer.create_error_response("x").status_code == 400
    prompt = {}
    OpenAIServer._attach_multimodal_data(prompt, SimpleNamespace(prompt_token_ids=None), "media")
    assert prompt == {"multi_modal_data": "media", "chat_template_applied": True}
