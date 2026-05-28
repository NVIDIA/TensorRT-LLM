from __future__ import annotations

import gc
from unittest.mock import patch

import pytest

from agent_flow import (AgentLayer, AgentLayerConfig, AgentRequest,
                        BackendConfig, SessionConfig)

from .helpers import FakeBackend


def _config() -> AgentLayerConfig:
    return AgentLayerConfig(
        name="session-layer",
        system_prompt="Keep context.",
        backend=BackendConfig(kind="claude-code", model="test-model"),
        session=SessionConfig(mode="persistent"),
    )


def test_context_manager_releases_persistent_client_and_backend():
    backend = FakeBackend([{"text": "ok"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config()) as layer:
            layer("hello")

    assert backend.create_client_calls == 1
    assert backend.client_exit_count == 1
    assert backend.exit_count == 1
    assert backend.clients[0].closed is True


def test_failed_persistent_send_recreates_client_on_next_call():
    backend = FakeBackend([
        {
            "error": RuntimeError("boom")
        },
        {
            "text": "recovered"
        },
    ])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config()) as layer:
            with pytest.raises(RuntimeError, match="boom"):
                layer("first")

            result = layer("second")

    assert backend.create_client_calls == 2
    assert result == "recovered"
    assert backend.clients[0].closed is True
    assert backend.clients[1].closed is True


def test_persistent_sessions_require_stable_system_prompt():
    backend = FakeBackend([{"text": "ok"}])

    prompts = iter(["one", "two"])

    def prompt_builder(content):
        return AgentRequest(
            content=content,
            system_prompt=next(prompts),
        )

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config(), prompt_builder=prompt_builder) as layer:
            layer("first")
            with pytest.raises(ValueError, match="stable system prompt"):
                layer("second")


def test_persistent_layer_cleans_up_when_garbage_collected():
    backend = FakeBackend([{"text": "ok"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        layer = AgentLayer(_config())
        layer("hello")
        del layer
        gc.collect()

    assert backend.create_client_calls == 1
    assert backend.client_exit_count == 1
    assert backend.exit_count == 1
    assert backend.clients[0].closed is True


async def test_async_context_manager_releases_persistent_client_and_backend():
    backend = FakeBackend([{"text": "ok"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        async with AgentLayer(_config()) as layer:
            result = await layer.aforward("hello")

    assert result == "ok"
    assert backend.create_client_calls == 1
    assert backend.client_exit_count == 1
    assert backend.exit_count == 1
