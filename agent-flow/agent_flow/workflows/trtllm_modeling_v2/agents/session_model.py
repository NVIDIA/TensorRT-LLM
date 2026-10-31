# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the launching session's model without starting an agent turn."""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Mapping
from uuid import UUID

from agent_flow.config import BackendKind

CLAUDE_SESSION_ID_ENV = "CLAUDE_CODE_SESSION_ID"
CODEX_SESSION_ENV = "CODEX_THREAD_ID"
MODEL_LOOKUP_TIMEOUT_S = 5.0


class ModelSelectionError(ValueError):
    """The launching model is unavailable and no explicit model was configured."""


def session_model(backend: BackendKind, environ: Mapping[str, str]) -> str:
    """Resolve an exact session identity; never choose another session or a global default."""
    key = CODEX_SESSION_ENV if backend == "codex" else CLAUDE_SESSION_ID_ENV
    session_id = environ.get(key, "").strip()
    try:
        UUID(session_id)
    except ValueError:
        raise ModelSelectionError(f"{key} does not identify a readable {backend} session") from None
    if backend == "codex":
        return _codex_model(session_id)
    return _claude_model(session_id, environ)


def _claude_model(session_id: str, environ: Mapping[str, str]) -> str:
    from claude_agent_sdk import get_session_messages

    config = Path(environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    try:
        matches = list((config / "projects").glob(f"*/{session_id}.jsonl"))
        if len(matches) != 1:
            raise ModelSelectionError("the launching session transcript is missing or ambiguous")
        messages = get_session_messages(session_id)
    except (OSError, ValueError) as exc:
        raise ModelSelectionError("cannot read the launching Claude session's model") from exc
    for message in reversed(messages):
        if message.type != "assistant" or message.session_id != session_id:
            continue
        if message.parent_tool_use_id or message.parent_agent_id:
            continue
        model = message.message.get("model") if isinstance(message.message, dict) else None
        if isinstance(model, str) and model.strip() and not model.startswith("<"):
            warnings.warn(
                f"Using {model} from the launching Claude session's last recorded assistant turn. "
                "Session writes can lag a model switch; set agents.defaults.model explicitly "
                "if that record does not represent the intended model.",
                UserWarning,
                stacklevel=3,
            )
            return model.strip()
        raise ModelSelectionError("the launching Claude session's last assistant has no model")
    raise ModelSelectionError("the launching Claude session has no recorded assistant model")


def _codex_model(session_id: str) -> str:
    from openai_codex.client import CodexClient, CodexConfig
    from openai_codex.errors import CodexError

    from agent_flow.backends.codex import _resolve_codex_bin

    client = CodexClient(
        CodexConfig(
            launch_args_override=(_resolve_codex_bin(), "app-server", "proxy"),
            client_name="modeling_v2_model_lookup",
        )
    )
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        client.close()

    timer = threading.Timer(MODEL_LOOKUP_TIMEOUT_S, expire)
    timer.daemon = True
    try:
        client.start()
        timer.start()
        client.initialize()
        thread = client.thread_read(session_id, include_turns=False).thread
        if expired.is_set():
            raise ModelSelectionError("timed out reading the launching Codex thread")
        if thread.id != session_id or thread.status.root.type == "notLoaded":
            raise ModelSelectionError("the launching Codex thread is not loaded in this server")
        if not isinstance(thread.model, str) or not thread.model.strip():
            raise ModelSelectionError("the launching Codex thread has no configured model")
        return thread.model.strip()
    except (CodexError, OSError, ValueError) as exc:
        raise ModelSelectionError("cannot read the launching Codex thread's live model") from exc
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        client.close()


__all__ = ["CLAUDE_SESSION_ID_ENV", "CODEX_SESSION_ENV", "ModelSelectionError", "session_model"]
