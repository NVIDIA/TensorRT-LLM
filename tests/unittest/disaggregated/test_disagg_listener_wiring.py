# Copyright (c) 2026, NVIDIA CORPORATION.
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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from tensorrt_llm.commands import serve
from tensorrt_llm.commands.serve import DisaggLauncherEnvs, DisaggWorkerEnvs

pytestmark = pytest.mark.cpu_only


def _config(**overrides):
    values = {
        "hostname": "advertised.example",
        "bind_host": "127.0.0.9",
        "port": 9000,
        "num_workers": 1,
        "disagg_coordinator_url": None,
        "schedule_style": "context_first",
        "server_keep_alive_timeout": 10,
        "internal_request_auth_key": None,
    }
    values.update(overrides)
    config = SimpleNamespace(**values)
    config.effective_bind_host = config.bind_host or config.hostname
    return config


@pytest.mark.parametrize(
    ("bind_host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8999"),
        ("192.0.2.10", "http://192.0.2.10:8999"),
    ],
)
def test_local_coordinator_url_is_reachable(bind_host, expected):
    assert serve._local_coordinator_url(bind_host, 8999) == expected


def test_single_server_binds_bind_host_and_advertises_hostname(monkeypatch):
    config = _config()
    socket_obj = MagicMock()
    socket_obj.__enter__.return_value = socket_obj
    socket_factory = Mock(return_value=socket_obj)
    server = Mock()
    server.return_value = object()
    publish = Mock()

    monkeypatch.setenv(DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID, "previous-deployment")
    monkeypatch.setenv(DisaggLauncherEnvs.TLLM_DISAGG_ROLE, "previous-role")
    monkeypatch.setattr(serve.logger, "set_level", Mock())
    monkeypatch.setattr(serve, "set_usage_context", Mock())
    monkeypatch.setattr(serve, "set_prometheus_multiproc_dir", Mock())
    monkeypatch.setattr(
        serve._command_telemetry,
        "apply_disaggregated_telemetry_config",
        Mock(),
    )
    monkeypatch.setattr(serve, "parse_disagg_config_file", Mock(return_value=config))
    monkeypatch.setattr(serve, "parse_metadata_server_config_file", Mock(return_value=None))
    monkeypatch.setattr(serve, "_create_disagg_socket", socket_factory)
    monkeypatch.setattr(serve, "_publish_bound_address", publish)
    monkeypatch.setattr(serve, "OpenAIDisaggServer", Mock(return_value=server))
    monkeypatch.setattr(serve, "set_lifecycle_phase", Mock())
    monkeypatch.setattr(serve.gc, "disable", Mock())
    run = Mock()
    monkeypatch.setattr(serve.uvloop, "run", run)

    serve.disaggregated.callback(
        config_file="disagg.yaml",
        metadata_server_config_file=None,
        server_start_timeout=180,
        request_timeout=180,
        log_level="info",
        metrics_log_interval=0,
        schedule_style=None,
        telemetry=False,
        report_addr="bound-address.txt",
    )

    socket_factory.assert_called_once_with()
    socket_obj.bind.assert_called_once_with(("127.0.0.9", 9000))
    publish.assert_called_once_with("bound-address.txt", "advertised.example", 9000)
    server.assert_called_once_with("127.0.0.9", 9000, sockets=[socket_obj])
    run.assert_called_once_with(server.return_value)


def test_fleet_worker_binds_bind_host(monkeypatch):
    config = _config()
    server = Mock()
    server._config = config
    server.return_value = object()
    socket_obj = MagicMock()

    monkeypatch.setattr(serve, "_init_fleet_worker_process", Mock())
    monkeypatch.setattr(serve, "_build_disagg_server_from_env", Mock(return_value=server))
    monkeypatch.setattr(
        serve,
        "_create_disagg_socket",
        Mock(return_value=socket_obj),
    )
    monkeypatch.setattr(serve, "set_lifecycle_phase", Mock())
    run = Mock()
    monkeypatch.setattr(serve.asyncio, "run", run)

    serve._run_fleet_worker_impl()

    socket_obj.bind.assert_called_once_with(("127.0.0.9", 9000))
    server.assert_called_once_with("127.0.0.9", 9000, sockets=[socket_obj])
    run.assert_called_once_with(server.return_value)
    assert server._config.hostname == "advertised.example"


def test_tcp_coordinator_uses_loopback_url_and_bind_host(monkeypatch):
    config = _config(bind_host="0.0.0.0", num_workers=4)
    launch_fleet = Mock(return_value=[])
    coordinator = object()
    coordinator_service = Mock(return_value=coordinator)
    coordinator_serve = AsyncMock()
    coordinator_server = Mock(return_value=coordinator_serve)

    monkeypatch.setenv(DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_UDS, "0")
    monkeypatch.setattr(serve, "_launch_disagg_fleet", launch_fleet)
    monkeypatch.setattr(serve, "set_lifecycle_phase", Mock())

    from tensorrt_llm.serve import coordinator_server as coordinator_server_module
    from tensorrt_llm.serve import disagg_coordinator

    monkeypatch.setattr(coordinator_server_module, "CoordinatorServer", coordinator_server)
    monkeypatch.setattr(disagg_coordinator, "DisaggCoordinatorService", coordinator_service)

    serve._serve_coordinator_and_fleet(
        config,
        "disagg.yaml",
        None,
        None,
        request_timeout=180,
        server_start_timeout=180,
        num_workers=4,
    )

    assert launch_fleet.call_args.args[-1] == "http://127.0.0.1:8999"
    assert launch_fleet.call_args.args[0].hostname == "advertised.example"
    coordinator_serve.assert_awaited_once_with("0.0.0.0", 8999, uds=None, keep_alive_timeout=10)
