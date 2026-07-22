from __future__ import annotations

from ..config import BackendConfig
from . import claude_code, codex
from .base import Backend, BackendClient, ResultEvent

__all__ = [
    "Backend",
    "BackendClient",
    "ResultEvent",
    "create_backend",
]


def create_backend(config: BackendConfig | str) -> Backend:
    kind = config.kind if isinstance(config, BackendConfig) else config

    if kind == "claude-code":
        return claude_code.ClaudeCodeBackend()

    if kind == "codex":
        return codex.CodexBackend()

    raise ValueError(f"Unknown backend: {kind!r}")
