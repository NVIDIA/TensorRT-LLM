# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starting a session as a fork of an earlier one, and learning a session's id."""

from __future__ import annotations

from unittest.mock import patch

import anyio

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig

from .helpers import FakeBackend


def _layer(fork_from: str | None = None) -> AgentLayer:
    return AgentLayer(
        AgentLayerConfig(
            name="fork-layer",
            system_prompt="Do the thing.",
            backend=BackendConfig(kind="claude-code", model="test-model"),
            session=SessionConfig(mode="persistent", fork_from=fork_from),
        )
    )


def test_the_layer_records_the_session_id_and_forks_when_asked():
    backend = FakeBackend([{"text": "ok", "session_id": "s-1"}])
    primer = _layer()
    with patch("agent_flow.layers.create_backend", return_value=backend):
        with primer:
            primer("read")
    assert primer.session_id == "s-1"
    assert backend.clients[-1].fork_from is None

    fork = _layer(fork_from="s-1")
    with patch("agent_flow.layers.create_backend", return_value=backend):
        with fork:
            fork("work")
    assert backend.clients[-1].fork_from == "s-1"


class _RecordingSDKClient:
    captured: object | None = None

    def __init__(self, options):
        type(self).captured = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def test_the_claude_backend_resumes_the_session_it_forks(tmp_path):
    from agent_flow.backends.claude_code import ClaudeCodeBackend

    async def _open(**kwargs) -> None:
        with patch("agent_flow.backends.claude_code.ClaudeSDKClient", _RecordingSDKClient):
            async with ClaudeCodeBackend().create_client(
                system_prompt="sys", model="m", cwd=tmp_path, **kwargs
            ):
                pass

    anyio.run(lambda: _open(fork_from="s-1"))
    options = _RecordingSDKClient.captured
    assert (options.resume, options.fork_session) == ("s-1", True)
    anyio.run(_open)
    options = _RecordingSDKClient.captured
    assert (options.resume, options.fork_session) == (None, False)


def test_the_codex_layer_keeps_the_primer_thread_and_forks_its_history(monkeypatch):
    from agent_flow.backends import codex

    from .test_codex_backend import FakeTransport
    from .test_codex_events import completed

    transports = []

    class Transport(FakeTransport):
        def __init__(self, config):
            super().__init__(config)
            transports.append(self)

        async def request(self, method, params):
            result = await super().request(method, params)
            if method == "thread/fork":
                result["thread"]["id"] = f"fork-{params['threadId']}"
            return result

        async def stream_turn(self, thread_id, message):
            yield completed(thread=thread_id)

    monkeypatch.setattr(codex, "CodexTransport", Transport)
    monkeypatch.setattr(codex, "_resolve_codex_bin", lambda: "/unused/codex")

    def layer(fork_from=None):
        return AgentLayer(
            AgentLayerConfig(
                backend=BackendConfig(kind="codex", model="m"),
                system_prompt="worker",
                session=SessionConfig(mode="stateless", fork_from=fork_from),
                print_activity=False,
            )
        )

    primer = layer()
    with primer:
        primer("read")
    assert primer.session_id == "thread-1"
    worker = layer(primer.session_id)
    with worker:
        worker("implement")
    assert worker.session_id == "fork-thread-1"
    fork = next(params for method, params in transports[-1].requests if method == "thread/fork")
    assert fork["threadId"] == primer.session_id
    assert ("thread/unsubscribe", {"threadId": worker.session_id}) in transports[-1].requests
