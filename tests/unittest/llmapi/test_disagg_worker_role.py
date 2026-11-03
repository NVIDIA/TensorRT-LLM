# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Propagate the launcher role to workers without a cache feature opt-in."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole

serve = importlib.import_module("tensorrt_llm.commands.serve")
pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "role,expected",
    [
        (ServerRole.CONTEXT, "context"),
        (ServerRole.GENERATION, "generation"),
        ("CONTEXT", "context"),
        ("GENERATION", "generation"),
        (None, ""),
    ],
)
def test_launch_propagates_role_and_overrides_stale_context(monkeypatch, role, expected):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    monkeypatch.delenv("TRTLLM_FP4_MLA_CONTEXT_ONLY_CACHE", raising=False)
    # Stop before binding a socket or constructing a model.
    monkeypatch.setattr(
        serve, "_init_multi_frontend_mode", Mock(side_effect=RuntimeError("configured"))
    )
    args = {
        "backend": "pytorch",
        "model": "dummy",
        "env_overrides": {"KEEP_ME": "1", "TRTLLM_DISAGG_ROLE": "context"},
    }
    with pytest.raises(RuntimeError, match="configured"):
        serve.launch_server("localhost", 8000, args, server_role=role)
    assert args["env_overrides"] == {"KEEP_ME": "1", "TRTLLM_DISAGG_ROLE": expected}


@pytest.mark.parametrize(
    "worker_type,expected", [("ctx", ServerRole.CONTEXT), ("gen", ServerRole.GENERATION)]
)
def test_disagg_launcher_derives_role_from_server_type(monkeypatch, worker_type, expected):
    monkeypatch.setenv("TLLM_DISAGG_INSTANCE_IDX", "0")
    config = SimpleNamespace(
        server_configs=[SimpleNamespace(type=worker_type, hostname="localhost", port=8000)],
        allow_request_chat_template=False,
    )
    monkeypatch.setattr(serve, "parse_disagg_config_file", lambda _: config)
    monkeypatch.setattr(serve, "mpi_rank", lambda: 0)
    launch = Mock()
    monkeypatch.setattr(serve, "launch_server", launch)
    serve._launch_disaggregated_server("unused.yaml", {})
    assert launch.call_args.kwargs["server_role"] == expected
