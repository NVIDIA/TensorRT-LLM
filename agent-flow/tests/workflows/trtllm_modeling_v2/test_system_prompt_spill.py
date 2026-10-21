# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A role's system prompt reaches the Claude CLI through the spill.

The Claude backend writes a system prompt above a threshold to a file under
the session cwd and passes a pointer on argv instead, because Linux caps one
argv string at 128 KiB. Role prompts are composed per run from the role
doctrine, the launch environment, the embargo and the decision-file blocks, so
their size is not fixed. These tests follow a role layer's own client options
into the backend, so a change to the layer's cwd or to how it builds its client
would show up here.
"""

from __future__ import annotations

from typing import Any

from agent_flow.backends import claude_code as cc_mod
from agent_flow.backends import create_backend

from .test_workflow import _workflow


async def _launched_options(monkeypatch, layer) -> Any:
    captured: dict[str, Any] = {}

    class FakeSdkClient:
        def __init__(self, options):
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(cc_mod, "ClaudeSDKClient", FakeSdkClient)
    monkeypatch.setattr(cc_mod, "create_sdk_mcp_server", lambda **_: object())
    backend = create_backend(layer.config.backend)
    async with backend.create_client(**layer._client_options(layer.config.system_prompt)):
        pass
    return captured["options"]


async def test_an_oversized_role_prompt_is_spilled_into_the_run_workspace(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path)
    prompt = workflow.prompts.assembler + "\n" + "x" * (cc_mod.SYSTEM_PROMPT_FILE_THRESHOLD + 1)
    layer = workflow._make_agent("assembler", prompt)

    options = await _launched_options(monkeypatch, layer)

    spilled = list((workflow.workspace / ".agent-flow").glob("system-prompt-*.md"))
    assert len(spilled) == 1
    assert spilled[0].read_text(encoding="utf-8") == prompt
    pointer = options.system_prompt["append"]
    assert str(spilled[0]) in pointer
    assert len(pointer.encode("utf-8")) < 4096


async def test_a_role_prompt_under_the_threshold_is_passed_inline(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path)
    layer = workflow._make_agent("reviewer", "a short role prompt")

    options = await _launched_options(monkeypatch, layer)

    assert options.system_prompt["append"] == "a short role prompt"
    assert not (workflow.workspace / ".agent-flow").exists()
