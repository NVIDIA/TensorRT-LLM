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
    effort = config.reasoning_effort if isinstance(config, BackendConfig) else None

    if kind == "claude-code":
        return claude_code.ClaudeCodeBackend(reasoning_effort=effort)

    if kind == "codex":
        return codex.CodexBackend(reasoning_effort=effort)

    raise ValueError(f"Unknown backend: {kind!r}")
