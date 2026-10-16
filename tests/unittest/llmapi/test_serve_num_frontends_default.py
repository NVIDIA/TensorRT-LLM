# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit coverage for trtllm-serve's multi-frontend default.

``--num_serve_frontends`` defaults to several HTTP frontend processes while
the LlmArgs field keeps default 1. What makes the default reach the executor
-- surviving the CLI-vs-LlmArgs default filter, yielding to the YAML and
--set, and falling back to a single frontend where more cannot run -- is
covered here down to the arguments ``launch_server`` receives; spawning the
frontends needs a GPU engine.
"""

from typing import Optional
from unittest import mock

import click
import pytest

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.commands.serve import (
    DEFAULT_NUM_SERVE_FRONTENDS,
    _resolve_default_num_serve_frontends,
    get_llm_args,
)
from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.executor.utils import MAX_NUM_FRONTENDS
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs
from tensorrt_llm.serve._frontend_processes import _init_multi_frontend_mode

pytestmark = pytest.mark.cpu_only


def _resolve(
    llm_args: dict,
    *,
    requested: bool = False,
    port: int = 8000,
    grpc: bool = False,
    report_addr: Optional[str] = None,
    enable_sleep_mode: bool = False,
    metadata_server: bool = False,
    disagg_cluster: bool = False,
) -> int:
    _resolve_default_num_serve_frontends(
        llm_args,
        requested=requested,
        port=port,
        grpc=grpc,
        report_addr=report_addr,
        enable_sleep_mode=enable_sleep_mode,
        metadata_server=metadata_server,
        disagg_cluster=disagg_cluster,
    )
    return llm_args.get("num_serve_frontends", 1)


def test_cli_default_is_multi_frontend() -> None:
    option = next(p for p in serve_cmd.serve.params if p.name == "num_serve_frontends")
    assert option.default == DEFAULT_NUM_SERVE_FRONTENDS
    assert 1 < DEFAULT_NUM_SERVE_FRONTENDS <= MAX_NUM_FRONTENDS


def test_llm_args_default_stays_single_frontend() -> None:
    # A bare LLM() has no HTTP frontends to fan out to; only trtllm-serve
    # spawns them, so the executor-side default must stay at one lane.
    assert BaseLlmArgs.model_fields["num_serve_frontends"].default == 1


def test_cli_default_survives_the_llm_args_default_filter() -> None:
    # get_llm_args drops CLI values equal to the LlmArgs default unless the
    # flag was typed. The serve default differs from the LlmArgs default, so
    # it must reach the LLM constructor without being typed on the CLI ...
    llm_args, _ = get_llm_args(
        model="m",
        backend="pytorch",
        gpus_per_node=1,
        num_serve_frontends=DEFAULT_NUM_SERVE_FRONTENDS,
    )
    assert llm_args["num_serve_frontends"] == DEFAULT_NUM_SERVE_FRONTENDS
    # ... while an untyped 1 collapses onto the LlmArgs default.
    llm_args, _ = get_llm_args(model="m", backend="pytorch", gpus_per_node=1, num_serve_frontends=1)
    assert "num_serve_frontends" not in llm_args


def test_plain_config_keeps_the_default() -> None:
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS}
    assert _resolve(llm_args) == DEFAULT_NUM_SERVE_FRONTENDS
    assert _init_multi_frontend_mode(llm_args, enabled=True).is_launcher


_FALLBACKS = [
    pytest.param({}, {"grpc": True}, "--grpc", id="grpc"),
    pytest.param({}, {"port": 0}, "--port 0", id="port0"),
    pytest.param({}, {"report_addr": "/tmp/bound.addr"}, "--report_addr", id="report_addr"),
    pytest.param({"orchestrator_type": "rpc"}, {}, "orchestrator_type='rpc'", id="rpc"),
    pytest.param({"orchestrator_type": "ray"}, {}, "orchestrator_type='ray'", id="ray"),
    pytest.param({"enable_resource_governor": True}, {}, "enable_resource_governor", id="governor"),
    pytest.param({}, {"enable_sleep_mode": True}, "--enable_sleep_mode", id="sleep_mode"),
    pytest.param(
        {}, {"metadata_server": True}, "--metadata_server_config_file", id="metadata_server"
    ),
    pytest.param({}, {"disagg_cluster": True}, "disagg_cluster", id="disagg_cluster"),
]


@pytest.mark.parametrize("extra_llm_args,kwargs,reason", _FALLBACKS)
def test_default_falls_back_to_one_frontend(
    monkeypatch: pytest.MonkeyPatch, extra_llm_args: dict, kwargs: dict, reason: str
) -> None:
    messages = []
    monkeypatch.setattr(serve_cmd.logger, "info", lambda msg, *a, **k: messages.append(msg))
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, **extra_llm_args}
    assert _resolve(llm_args, **kwargs) == 1
    # The fallback says why, and how to override it.
    assert len(messages) == 1
    assert reason in messages[0] and "--num_serve_frontends" in messages[0]
    # It leaves a consistent single-frontend configuration behind: the mode
    # resolver no longer sees a launcher and does not raise.
    assert not _init_multi_frontend_mode(llm_args, enabled=True).is_launcher


@pytest.mark.parametrize("extra_llm_args,kwargs,reason", _FALLBACKS)
def test_explicit_request_is_not_silently_downgraded(
    extra_llm_args: dict, kwargs: dict, reason: str
) -> None:
    # The value must survive so that the guard owning the incompatibility
    # still raises (or, for OpenEngine gRPC, honors it).
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, **extra_llm_args}
    assert _resolve(llm_args, requested=True, **kwargs) == DEFAULT_NUM_SERVE_FRONTENDS


def test_single_frontend_is_left_alone() -> None:
    llm_args = {"num_serve_frontends": 1, "orchestrator_type": "rpc"}
    assert _resolve(llm_args) == 1
    assert _resolve({"orchestrator_type": "rpc"}) == 1


def _serve(args: list, monkeypatch: pytest.MonkeyPatch, tmp_path) -> mock.MagicMock:
    """Run the CLI down to launch_server and return the mocked launcher."""
    with (
        mock.patch(
            "tensorrt_llm.commands.serve.get_is_diffusion_only_model",
            return_value=False,
        ),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.launch_server") as launch,
    ):
        serve_main(args=[str(tmp_path), *args], standalone_mode=False)
    assert launch.call_count == 1
    return launch


def _launched_frontends(launch: mock.MagicMock) -> int:
    return launch.call_args.args[2].get("num_serve_frontends", 1)


def _yaml(tmp_path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_cli_default_reaches_launch_server(monkeypatch, tmp_path) -> None:
    assert _launched_frontends(_serve([], monkeypatch, tmp_path)) == DEFAULT_NUM_SERVE_FRONTENDS


@pytest.mark.parametrize(
    "args_factory,expected",
    [
        pytest.param(lambda p: ["--num_serve_frontends", "1"], 1, id="cli_1"),
        pytest.param(lambda p: ["--num_serve_frontends", "3"], 3, id="cli_3"),
        pytest.param(lambda p: ["--config", _yaml(p, "num_serve_frontends: 1\n")], 1, id="yaml_1"),
        pytest.param(lambda p: ["--set", "num_serve_frontends=2"], 2, id="set_2"),
        # An explicit count survives a configuration it conflicts with, so
        # launch_server's own guard rejects it rather than it shrinking to 1.
        pytest.param(
            lambda p: ["--num_serve_frontends", "8", "--port", "0"], 8, id="explicit_port0"
        ),
    ],
)
def test_requested_counts_reach_launch_server(
    monkeypatch, tmp_path, args_factory, expected: int
) -> None:
    launch = _serve(args_factory(tmp_path), monkeypatch, tmp_path)
    assert _launched_frontends(launch) == expected


@pytest.mark.parametrize(
    "args_factory",
    [
        pytest.param(lambda p: ["--port", "0"], id="port0"),
        pytest.param(lambda p: ["--report_addr", str(p / "bound.addr")], id="report_addr"),
        pytest.param(lambda p: ["--config", _yaml(p, "orchestrator_type: rpc\n")], id="rpc"),
        pytest.param(
            lambda p: [
                "--server_role",
                "context",
                "--metadata_server_config_file",
                _yaml(p, "server_type: etcd\n"),
            ],
            id="metadata_server",
        ),
        pytest.param(
            lambda p: ["--server_role", "context", "--disagg_cluster_uri", "etcd://localhost:2379"],
            id="disagg_cluster",
        ),
    ],
)
def test_default_falls_back_before_launch_server(monkeypatch, tmp_path, args_factory) -> None:
    launch = _serve(args_factory(tmp_path), monkeypatch, tmp_path)
    assert _launched_frontends(launch) == 1


def test_sleep_mode_default_falls_back_before_launch_server(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")
    launch = _serve(["--enable_sleep_mode"], monkeypatch, tmp_path)
    assert _launched_frontends(launch) == 1
    assert launch.call_args.kwargs["enable_runtime_control_endpoints"]


def test_yaml_count_with_grpc_is_rejected(monkeypatch, tmp_path) -> None:
    # An explicit YAML count is honored, so the gRPC guard sees it and rejects
    # several frontends on the SMG protocol instead of silently running one.
    with (
        mock.patch(
            "tensorrt_llm.commands.serve.get_is_diffusion_only_model",
            return_value=False,
        ),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        pytest.raises(click.UsageError, match="Multiple gRPC frontends"),
    ):
        serve_main(
            args=[
                str(tmp_path),
                "--grpc",
                "--config",
                _yaml(tmp_path, "num_serve_frontends: 8\n"),
            ],
            standalone_mode=False,
        )
