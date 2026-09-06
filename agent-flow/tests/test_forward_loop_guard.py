from __future__ import annotations

from unittest.mock import patch

import anyio
import pytest

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig

from .helpers import FakeBackend


def _config(mode: str) -> AgentLayerConfig:
    return AgentLayerConfig(
        name="loop-guard-layer",
        system_prompt="Guard the loop.",
        backend=BackendConfig(kind="claude-code", model="test-model"),
        session=SessionConfig(mode=mode),
    )


@pytest.mark.parametrize("mode", ["stateless", "persistent"])
def test_forward_inside_running_loop_raises(mode):
    """Sync ``forward()`` inside a running loop must fail loudly.

    The error must point at ``aforward``, for both the stateless
    (``anyio.run``) and persistent (``PortalRunner``) paths.
    """
    backend = FakeBackend([{"text": "ok"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config(mode)) as layer:

            async def inner() -> None:
                layer.forward("hi")

            with pytest.raises(RuntimeError, match="aforward"):
                anyio.run(inner)


@pytest.mark.parametrize("mode", ["stateless", "persistent"])
def test_forward_without_running_loop_ok(mode):
    """With no running loop, ``forward()`` keeps working normally."""
    backend = FakeBackend([{"text": "ok"}])

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with AgentLayer(_config(mode)) as layer:
            result = layer.forward("hi")

    assert isinstance(result, str)
    assert result == "ok"
