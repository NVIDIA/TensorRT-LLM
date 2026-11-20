# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-serve forwards an explicit disaggregated server role to MPI workers."""

import copy
import importlib
from unittest.mock import Mock

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole

serve = importlib.import_module("tensorrt_llm.commands.serve")
pytestmark = pytest.mark.cpu_only

_ROLE_ENV = "TRTLLM_DISAGG_ROLE"


def _launch_until_configured(monkeypatch: pytest.MonkeyPatch, llm_args: dict, role) -> dict:
    # Stop before binding a socket or constructing a model.
    configure = Mock(side_effect=RuntimeError("configured"))
    monkeypatch.setattr(serve, "_init_multi_frontend_mode", configure)
    with pytest.raises(RuntimeError, match="configured"):
        serve.launch_server("localhost", 8000, llm_args, server_role=role)
    configure.assert_called_once()
    return configure.call_args.args[0]


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
    snapshot = copy.deepcopy(llm_args)
    effective = _launch_until_configured(monkeypatch, llm_args, role)
    assert effective["env_overrides"] == {"KEEP_ME": "1", _ROLE_ENV: expected}
    assert llm_args == snapshot
    assert effective is not llm_args
    assert effective["env_overrides"] is not llm_args["env_overrides"]


def test_launch_without_role_leaves_worker_env_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_args = {"backend": "pytorch", "model": "dummy"}
    effective = _launch_until_configured(monkeypatch, llm_args, None)
    assert effective == llm_args == {"backend": "pytorch", "model": "dummy"}
    assert effective is not llm_args
