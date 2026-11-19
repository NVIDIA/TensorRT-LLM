from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import anyio

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig

from .helpers import FakeBackend


def _make_layer(backend: FakeBackend, mode: str, cwd=None) -> AgentLayer:
    """Build a layer whose session runs in ``cwd``.

    This repo carries ``cwd`` on :class:`BackendConfig` rather than on
    :class:`SessionConfig`, so that is where the per-node worktree is threaded
    through; the behaviour under test — does it reach ``create_client`` — is the
    same either way.
    """
    config = AgentLayerConfig(
        name="cwd-layer",
        system_prompt="Do the thing.",
        backend=BackendConfig(kind="claude-code", model="test-model", cwd=cwd),
        session=SessionConfig(mode=mode),
    )
    return AgentLayer(config)


def test_cwd_from_backend_config_reaches_create_client(tmp_path):
    backend = FakeBackend([{"text": "ok"}])
    layer = _make_layer(backend, "stateless", cwd=tmp_path)

    with patch("agent_flow.layers.create_backend", return_value=backend):
        anyio.run(layer.aforward, "hi")

    assert backend.clients[-1].cwd == tmp_path


def test_cwd_defaults_to_none():
    backend = FakeBackend([{"text": "ok"}])
    layer = _make_layer(backend, "stateless")

    with patch("agent_flow.layers.create_backend", return_value=backend):
        anyio.run(layer.aforward, "hi")

    assert backend.clients[-1].cwd is None


def test_cwd_reaches_persistent_client(tmp_path):
    backend = FakeBackend([{"text": "ok"}])
    layer = _make_layer(backend, "persistent", cwd=tmp_path)

    with patch("agent_flow.layers.create_backend", return_value=backend):
        with layer:
            layer("hi")

    assert backend.clients[-1].cwd == tmp_path


class _RecordingSDKClient:
    """Async-context-manager stand-in that records the options it is built with."""

    captured: object | None = None

    def __init__(self, options):
        type(self).captured = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def test_claude_backend_forwards_cwd_into_options(tmp_path):
    """Real-behavior: a set cwd reaches ``ClaudeAgentOptions(cwd=...)``."""
    from agent_flow.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    _RecordingSDKClient.captured = None

    async def _drive() -> None:
        with patch("agent_flow.backends.claude_code.ClaudeSDKClient", _RecordingSDKClient):
            async with backend.create_client(
                system_prompt="sys",
                model="test-model",
                cwd=tmp_path,
            ):
                pass

    anyio.run(_drive)

    assert _RecordingSDKClient.captured is not None
    # This repo forwards the Path itself where upstream stringifies it; the SDK
    # accepts either, so compare on the resolved path rather than the type.
    assert str(_RecordingSDKClient.captured.cwd) == str(tmp_path)


def test_claude_backend_cwd_none_is_unchanged_from_today(tmp_path):
    """Real-behavior: ``cwd=None`` leaves options identical to today (``Path.cwd()``)."""
    from agent_flow.backends.claude_code import ClaudeCodeBackend

    backend = ClaudeCodeBackend()
    _RecordingSDKClient.captured = None

    async def _drive() -> None:
        with patch("agent_flow.backends.claude_code.ClaudeSDKClient", _RecordingSDKClient):
            async with backend.create_client(
                system_prompt="sys",
                model="test-model",
            ):
                pass

    anyio.run(_drive)

    assert _RecordingSDKClient.captured is not None
    assert _RecordingSDKClient.captured.cwd == Path.cwd()
