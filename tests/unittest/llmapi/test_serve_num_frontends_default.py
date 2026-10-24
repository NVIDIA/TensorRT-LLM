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
"""CPU tests for trtllm-serve's multi-frontend default and its fallback to one."""

from unittest import mock

import pytest

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.commands.serve import (
    DEFAULT_NUM_SERVE_FRONTENDS,
    _resolve_default_num_serve_frontends,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "llm_args,conflict,reason",
    [
        pytest.param({}, {"grpc": True}, "--grpc", id="grpc"),
        pytest.param({}, {"port": 0}, "--port 0", id="port0"),
        pytest.param({}, {"report_addr": "/tmp/bound.addr"}, "--report_addr", id="report_addr"),
        pytest.param({"orchestrator_type": "rpc"}, {}, "orchestrator_type='rpc'", id="rpc"),
        pytest.param({"enable_resource_governor": True}, {}, "enable_resource_governor", id="rg"),
        pytest.param({}, {"enable_sleep_mode": True}, "--enable_sleep_mode", id="sleep_mode"),
        pytest.param({}, {"metadata_server": True}, "--metadata_server_config_file", id="metadata"),
        pytest.param({}, {"disagg_cluster": True}, "disagg_cluster", id="disagg_cluster"),
    ],
)
def test_default_falls_back_to_one_frontend(monkeypatch, llm_args, conflict, reason) -> None:
    messages = []
    monkeypatch.setattr(serve_cmd.logger, "info", lambda msg, *a, **k: messages.append(msg))
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, **llm_args}
    no_conflict = dict(
        requested=False,
        port=8000,
        grpc=False,
        report_addr=None,
        enable_sleep_mode=False,
        metadata_server=False,
        disagg_cluster=False,
    )

    _resolve_default_num_serve_frontends(llm_args, **{**no_conflict, **conflict})

    assert llm_args["num_serve_frontends"] == 1
    assert len(messages) == 1 and reason in messages[0] and "--num_serve_frontends" in messages[0]


@pytest.mark.parametrize(
    "args,config,expected",
    [
        pytest.param([], None, DEFAULT_NUM_SERVE_FRONTENDS, id="default"),
        pytest.param([], "num_serve_frontends: 1", 1, id="yaml_opt_out"),
        pytest.param(["--port", "0"], None, 1, id="port0"),
        pytest.param(["--enable_sleep_mode"], None, 1, id="sleep_mode"),
        pytest.param(
            ["--server_role", "context", "--disagg_cluster_uri", "etcd://localhost:2379"],
            None,
            1,
            id="disagg_cluster",
        ),
        # An explicit count is honored; launch_server's own guard rejects the conflict.
        pytest.param(["--port", "0", "--num_serve_frontends", "8"], None, 8, id="explicit"),
    ],
)
def test_frontend_count_reaching_launch_server(
    monkeypatch, tmp_path, args, config, expected
) -> None:
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "secret")  # for --enable_sleep_mode
    if config is not None:
        (tmp_path / "config.yaml").write_text(config)
        args = [*args, "--config", str(tmp_path / "config.yaml")]
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=False),
        mock.patch.object(serve_cmd, "device_count", return_value=1),
        mock.patch.object(serve_cmd, "launch_server") as launch,
    ):
        serve_cmd.main(args=[str(tmp_path), *args], standalone_mode=False)

    assert launch.call_args.args[2].get("num_serve_frontends", 1) == expected
