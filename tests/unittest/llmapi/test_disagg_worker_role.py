# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-serve forwards an explicit disaggregated server role to MPI workers."""

import importlib
from unittest.mock import Mock

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole

serve = importlib.import_module("tensorrt_llm.commands.serve")
pytestmark = pytest.mark.cpu_only

_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


def _launch_until_configured(monkeypatch: pytest.MonkeyPatch, llm_args: dict, role) -> None:
    # Stop before binding a socket or constructing a model.
    monkeypatch.setattr(
        serve, "_init_multi_frontend_mode", Mock(side_effect=RuntimeError("configured"))
    )
    with pytest.raises(RuntimeError, match="configured"):
        serve.launch_server("localhost", 8000, llm_args, server_role=role)


@pytest.mark.parametrize(
    "role,expected",
    [(ServerRole.CONTEXT, "context"), (ServerRole.GENERATION, "generation")],
)
def test_launch_forwards_role_to_worker_env_overrides(
    monkeypatch: pytest.MonkeyPatch, role: ServerRole, expected: str
) -> None:
    monkeypatch.setenv(_ROLE_ENV, "context")
    llm_args = {
        "backend": "pytorch",
        "model": "dummy",
        "env_overrides": {"KEEP_ME": "1", _ROLE_ENV: "context"},
    }
    _launch_until_configured(monkeypatch, llm_args, role)
    assert llm_args["env_overrides"] == {"KEEP_ME": "1", _ROLE_ENV: expected}


def test_launch_without_role_leaves_worker_env_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_args = {"backend": "pytorch", "model": "dummy"}
    _launch_until_configured(monkeypatch, llm_args, None)
    assert "env_overrides" not in llm_args
