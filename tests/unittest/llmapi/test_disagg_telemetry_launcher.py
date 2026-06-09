# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke tests for disaggregated launcher telemetry environment propagation."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.commands import serve


def test_disaggregated_command_sets_shared_deployment_id(monkeypatch) -> None:
    """The top-level disagg launcher assigns one deployment id for child workers."""
    monkeypatch.delenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID,
        raising=False,
    )

    disagg_config = SimpleNamespace(hostname="127.0.0.1", port=0, schedule_style=None)
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value = fake_socket
    deployment_id = SimpleNamespace(hex="deploy123")

    with (
        mock.patch.object(serve.uuid, "uuid4", return_value=deployment_id),
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(serve.socket, "socket", return_value=fake_socket),
        mock.patch.object(serve, "parse_metadata_server_config_file", return_value=None),
        mock.patch.object(serve, "OpenAIDisaggServer"),
        mock.patch.object(serve.asyncio, "run"),
    ):
        serve.disaggregated.callback(
            config_file="disagg.yaml",
            metadata_server_config_file=None,
            server_start_timeout=180,
            request_timeout=180,
            log_level="info",
            metrics_log_interval=0,
            schedule_style=None,
        )

    assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID] == "deploy123"
    fake_socket.bind.assert_called_once_with(("127.0.0.1", 0))


def test_launch_disaggregated_leader_propagates_deployment_id(monkeypatch) -> None:
    """Leader subprocess env keeps the shared telemetry deployment id."""
    observed = {}

    class _FakeComm:
        def Get_rank(self):
            return 0

    class _FakePopen:
        pid = 12345

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["env"] = kwargs["env"]
            self._status = None

        def poll(self):
            return self._status

        def terminate(self):
            self._status = -15

        def wait(self, timeout=None):
            self._status = -15
            return self._status

        def kill(self):
            self._status = -9

    def _fake_split_mpi_env():
        return dict(os.environ), {}

    monkeypatch.setenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID,
        "deploy123",
    )
    monkeypatch.setattr(serve, "find_free_ipc_addr", lambda: "ipc://fake-proxy")
    monkeypatch.setattr(serve.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(serve.sys, "argv", ["trtllm-serve"])
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.llmapi.mgmn_leader_node",
        SimpleNamespace(launch_server_main=lambda sub_comm: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.llmapi.mpi_session",
        SimpleNamespace(split_mpi_env=_fake_split_mpi_env),
    )

    serve._launch_disaggregated_leader(_FakeComm(), 2, "disagg.yaml", "info")

    assert observed["env"][serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID] == "deploy123"
    assert observed["env"][serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] == "2"
    assert (
        observed["env"][serve.DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT] == "1"
    )
    assert observed["command"] == [
        "python3",
        "trtllm-serve",
        "disaggregated_mpi_worker",
        "-c",
        "disagg.yaml",
        "--log_level",
        "info",
    ]


@pytest.mark.parametrize(
    ("server_type", "expected_role"),
    [
        ("ctx", "context"),
        ("gen", "generation"),
    ],
)
def test_launch_disaggregated_server_sets_worker_role(
    monkeypatch,
    server_type,
    expected_role,
) -> None:
    """Worker launch maps disagg server type to telemetry role env."""
    monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
    monkeypatch.delenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE, raising=False)

    llm_args = {"model": "dummy/model"}
    server_config = SimpleNamespace(type=server_type, hostname="127.0.0.1", port=8000)
    disagg_config = SimpleNamespace(server_configs=[server_config])

    with (
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(serve, "mpi_rank", return_value=0),
        mock.patch.object(serve, "launch_server") as mock_launch_server,
    ):
        serve._launch_disaggregated_server("disagg.yaml", llm_args)

    assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE] == expected_role
    mock_launch_server.assert_called_once_with(
        host="127.0.0.1",
        port=8000,
        llm_args=llm_args,
    )
