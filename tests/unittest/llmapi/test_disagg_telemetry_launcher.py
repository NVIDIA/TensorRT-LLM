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
import signal
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner

from tensorrt_llm.commands import serve

pytestmark = pytest.mark.cpu_only


def _mock_llmapi_modules(
    monkeypatch: pytest.MonkeyPatch,
    split_mpi_env: Callable[[], tuple[dict[str, str], dict[str, str]]],
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.llmapi.mgmn_leader_node",
        SimpleNamespace(launch_server_main=lambda sub_comm: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.llmapi.mpi_session",
        SimpleNamespace(split_mpi_env=split_mpi_env),
    )


def test_disaggregated_command_sets_shared_deployment_id(monkeypatch) -> None:
    """The top-level disagg launcher assigns one deployment id for child workers."""
    monkeypatch.delenv(
        serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID,
        raising=False,
    )

    disagg_config = SimpleNamespace(
        hostname="127.0.0.1",
        port=0,
        schedule_style=None,
        num_workers=1,
        disagg_coordinator_url=None,
    )
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value = fake_socket
    deployment_id = SimpleNamespace(hex="deploy123")

    with (
        mock.patch.object(serve.uuid, "uuid4", return_value=deployment_id),
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(serve._command_telemetry, "apply_disaggregated_telemetry_config"),
        mock.patch.object(serve.socket, "socket", return_value=fake_socket),
        mock.patch.object(serve, "parse_metadata_server_config_file", return_value=None),
        mock.patch.object(serve, "OpenAIDisaggServer"),
        mock.patch.object(serve.uvloop, "run"),
    ):
        serve.disaggregated.callback(
            config_file="disagg.yaml",
            metadata_server_config_file=None,
            server_start_timeout=180,
            request_timeout=180,
            log_level="info",
            metrics_log_interval=0,
            schedule_style=None,
            telemetry=True,
        )

    assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID] == "deploy123"
    assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE] == "server_coordinator"
    fake_socket.bind.assert_called_once_with(("127.0.0.1", 0))


def test_disaggregated_command_identifies_coordinator(monkeypatch) -> None:
    """A multi-frontend deployment labels its coordinator process explicitly."""
    disagg_config = SimpleNamespace(
        hostname="127.0.0.1",
        port=8000,
        schedule_style=None,
        num_workers=2,
        disagg_coordinator_url=None,
    )

    with (
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(serve._command_telemetry, "apply_disaggregated_telemetry_config"),
        mock.patch.object(serve, "parse_metadata_server_config_file", return_value=None),
        mock.patch.object(serve, "_serve_coordinator_and_fleet") as serve_coordinator,
    ):
        serve.disaggregated.callback(
            config_file="disagg.yaml",
            metadata_server_config_file=None,
            server_start_timeout=180,
            request_timeout=180,
            log_level="info",
            metrics_log_interval=0,
            schedule_style=None,
            telemetry=True,
        )

    assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE] == "coordinator"
    serve_coordinator.assert_called_once()


def test_disaggregated_command_accepts_no_telemetry() -> None:
    """The public disaggregated command has an effective CLI opt-out."""
    disagg_config = SimpleNamespace(
        hostname="127.0.0.1",
        port=0,
        schedule_style=None,
        num_workers=1,
        disagg_coordinator_url=None,
    )
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value = fake_socket

    with (
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(
            serve._command_telemetry, "apply_disaggregated_telemetry_config"
        ) as apply_config,
        mock.patch.object(serve.socket, "socket", return_value=fake_socket),
        mock.patch.object(serve, "parse_metadata_server_config_file", return_value=None),
        mock.patch.object(serve, "OpenAIDisaggServer"),
        mock.patch.object(serve.uvloop, "run"),
    ):
        result = CliRunner().invoke(
            serve.disaggregated,
            ["--no-telemetry", "--config", "disagg.yaml"],
        )

    assert result.exit_code == 0, result.output
    apply_config.assert_called_once_with(
        "disagg.yaml",
        telemetry=False,
    )


@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
def test_launch_disagg_fleet_propagates_resolved_schedule_style(
    monkeypatch,
    schedule_style,
) -> None:
    """Fleet children receive the resolved CLI-or-config schedule style."""
    real_popen = subprocess.Popen
    observed_envs = []
    fleet_processes = []
    registered_cleanups = []
    signal_handlers = {}

    class _FakePopen:
        pid = 12345

        def __init__(self, _command, **kwargs):
            observed_envs.append(kwargs["env"])
            fleet_processes.append(self)
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -signal.SIGTERM

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -signal.SIGKILL

    disagg_config = SimpleNamespace(
        hostname="127.0.0.1",
        port=8000,
        schedule_style=schedule_style,
    )
    monkeypatch.setattr(serve.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(serve.atexit, "register", registered_cleanups.append)
    monkeypatch.setattr(
        serve.signal,
        "signal",
        lambda signal_number, handler: signal_handlers.__setitem__(signal_number, handler),
    )
    monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
    monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE, "coordinator")

    serve._launch_disagg_fleet(
        disagg_config,
        "disagg.yaml",
        None,
        180,
        180,
        1,
        "http://coordinator:8001",
    )

    assert observed_envs[0][serve.DisaggWorkerEnvs.TLLM_DISAGG_SCHEDULE_STYLE] == schedule_style
    assert observed_envs[0]["TRTLLM_NO_USAGE_STATS"] == "1"
    assert serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE not in observed_envs[0]
    assert signal_handlers[signal.SIGTERM] is serve._command_telemetry.raise_signal_exit
    with pytest.raises(serve._command_telemetry.SignalExit):
        signal_handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert fleet_processes[0].poll() is None
    registered_cleanups[0]()
    assert fleet_processes[0].poll() == -signal.SIGTERM

    child = real_popen(
        [
            sys.executable,
            "-c",
            "import os,sys; sys.exit(0 if os.environ.get('TRTLLM_NO_USAGE_STATS') == '1' else 1)",
        ],
        env=observed_envs[0],
    )
    assert child.wait(timeout=10) == 0


@pytest.mark.parametrize(
    ("config_style", "resolved_style"),
    [
        ("context_first", "generation_first"),
        ("generation_first", "context_first"),
    ],
)
def test_build_fleet_worker_applies_resolved_schedule_style(
    monkeypatch,
    config_style,
    resolved_style,
) -> None:
    """A fleet worker overrides its reparsed YAML with the parent resolution."""
    disagg_config = SimpleNamespace(schedule_style=config_style)
    monkeypatch.setenv(serve.DisaggWorkerEnvs.TLLM_DISAGG_CONFIG_FILE, "disagg.yaml")
    monkeypatch.setenv(
        serve.DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_URL,
        "http://coordinator:8001",
    )
    monkeypatch.setenv(
        serve.DisaggWorkerEnvs.TLLM_DISAGG_SCHEDULE_STYLE,
        resolved_style,
    )

    with (
        mock.patch.object(serve, "parse_disagg_config_file", return_value=disagg_config),
        mock.patch.object(serve, "parse_metadata_server_config_file", return_value=None),
        mock.patch.object(serve, "OpenAIDisaggServer") as mock_server,
    ):
        serve._build_disagg_server_from_env()

    assert disagg_config.schedule_style == resolved_style
    assert mock_server.call_args.kwargs["config"] is disagg_config


def test_launch_disaggregated_leader_propagates_deployment_id(monkeypatch) -> None:
    """Leader subprocess env keeps the shared telemetry deployment id."""
    observed = {}
    installed_handlers = []

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
    monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
    monkeypatch.setattr(serve, "find_free_ipc_addr", lambda: "ipc://fake-proxy")
    monkeypatch.setattr(serve.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        serve.signal,
        "signal",
        lambda signal_number, handler: installed_handlers.append((signal_number, handler)),
    )
    monkeypatch.setattr(serve.sys, "argv", ["trtllm-serve"])
    _mock_llmapi_modules(monkeypatch, _fake_split_mpi_env)

    serve._launch_disaggregated_leader(_FakeComm(), 2, "disagg.yaml", "info")

    assert observed["env"][serve.DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID] == "deploy123"
    assert observed["env"][serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] == "2"
    assert observed["env"]["TRTLLM_NO_USAGE_STATS"] == "1"
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
    assert installed_handlers[:2] == [
        (signal.SIGTERM, serve._command_telemetry.raise_signal_exit),
        (signal.SIGINT, serve._command_telemetry.raise_signal_exit),
    ]


def test_fleet_worker_uses_process_wide_telemetry_setting() -> None:
    """The direct fleet boundary does not construct an enabled override."""
    with (
        mock.patch.object(serve, "apply_usage_session_config") as apply_config,
        mock.patch.object(serve, "_run_fleet_worker_impl"),
        mock.patch.object(
            serve._command_telemetry.usage,
            "get_observed_signal",
            return_value=0,
        ),
        mock.patch.object(serve._command_telemetry.usage, "set_lifecycle_phase"),
        mock.patch.object(
            serve._command_telemetry.usage,
            "report_exit",
        ) as report_exit,
    ):
        serve._run_fleet_worker()

    assert apply_config.call_args.args == ()
    assert report_exit.call_args.kwargs["telemetry_config"] is None
    assert report_exit.call_args.kwargs["default_usage_context"] == "disaggregated"


def test_disaggregated_mpi_worker_exposes_telemetry_option() -> None:
    """Independent MPI deployment roots can apply the same CLI opt-out."""
    parameter_names = {parameter.name for parameter in serve.disaggregated_mpi_worker.params}
    assert "telemetry" in parameter_names


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
    disagg_config = SimpleNamespace(
        server_configs=[server_config],
        allow_request_chat_template=False,
    )

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
        allow_request_chat_template=False,
        multi_frontend_enabled=False,
    )
