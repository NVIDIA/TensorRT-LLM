# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent-model lookup uses an exact session and never starts or resumes a turn."""

from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents import session_model as selection

SESSION = "12345678-1234-1234-1234-123456789abc"


def test_claude_reads_last_recorded_root_model_and_rejects_ambiguity(tmp_path, monkeypatch):
    import claude_agent_sdk

    config = tmp_path / "claude"
    project = config / "projects" / "project"
    project.mkdir(parents=True)
    (project / f"{SESSION}.jsonl").write_text("placeholder")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))

    def message(model, **kwargs):
        return SimpleNamespace(
            type="assistant",
            session_id=SESSION,
            message={"model": model},
            parent_tool_use_id=kwargs.get("parent_tool_use_id"),
            parent_agent_id=None,
        )

    seen = []

    def messages(session_id):
        seen.append(session_id)
        return [
            message("older-model"),
            message("selected-model"),
            message("child-model", parent_tool_use_id="spawn"),
        ]

    monkeypatch.setattr(claude_agent_sdk, "get_session_messages", messages)
    env = {"CLAUDE_CONFIG_DIR": str(config), "CLAUDE_CODE_SESSION_ID": SESSION}
    with pytest.warns(UserWarning, match="last recorded"):
        assert selection.session_model("claude-code", env) == "selected-model"
    assert seen == [SESSION]
    other = config / "projects" / "duplicate"
    other.mkdir()
    (other / f"{SESSION}.jsonl").write_text("duplicate")
    with pytest.raises(selection.ModelSelectionError):
        selection.session_model("claude-code", env)
    assert seen == [SESSION]


@pytest.mark.parametrize(
    "status,returned_id,model,ok",
    [
        ("active", SESSION, "current-codex", True),
        ("notLoaded", SESSION, "persisted-model", False),
        ("active", "another-thread", "other-model", False),
        ("active", SESSION, None, False),
    ],
)
def test_codex_reads_only_the_live_matching_thread(monkeypatch, status, returned_id, model, ok):
    from openai_codex import client as sdk

    from agent_flow.backends import codex

    calls = []
    monkeypatch.setattr(codex, "_resolve_codex_bin", lambda: "/bin/codex")

    class Client:
        def __init__(self, config):
            assert config.launch_args_override == ("/bin/codex", "app-server", "proxy")

        def start(self):
            calls.append("start-proxy")

        def initialize(self):
            calls.append("initialize")

        def thread_read(self, thread_id, include_turns=False):
            calls.append((thread_id, include_turns))
            return SimpleNamespace(
                thread=SimpleNamespace(
                    id=returned_id,
                    model=model,
                    status=SimpleNamespace(root=SimpleNamespace(type=status)),
                )
            )

        def close(self):
            calls.append("close-proxy")

    monkeypatch.setattr(sdk, "CodexClient", Client)
    env = {"CODEX_THREAD_ID": SESSION}
    if ok:
        assert selection.session_model("codex", env) == model
    else:
        with pytest.raises(selection.ModelSelectionError):
            selection.session_model("codex", env)
    assert calls == ["start-proxy", "initialize", (SESSION, False), "close-proxy"]


def test_codex_lookup_timeout_closes_the_proxy(monkeypatch):
    import threading

    from openai_codex import client as sdk
    from openai_codex.errors import TransportClosedError

    from agent_flow.backends import codex

    closed = threading.Event()
    monkeypatch.setattr(codex, "_resolve_codex_bin", lambda: "/bin/codex")
    monkeypatch.setattr(selection, "MODEL_LOOKUP_TIMEOUT_S", 0.01)

    class Client:
        def __init__(self, config):
            pass

        def start(self):
            pass

        def initialize(self):
            assert closed.wait(1)
            raise TransportClosedError("closed")

        def close(self):
            closed.set()

    monkeypatch.setattr(sdk, "CodexClient", Client)
    with pytest.raises(selection.ModelSelectionError):
        selection.session_model("codex", {"CODEX_THREAD_ID": SESSION})
    assert closed.is_set()
