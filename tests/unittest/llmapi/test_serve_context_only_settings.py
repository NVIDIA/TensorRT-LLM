# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""trtllm-serve checks the settings meant for context servers against the server's role.

Dense DWDP (``dwdp_config.enable_dense``) and ``drop_inflight_requests_on_shutdown`` are meant for
context servers. The role is the one the server's ranks see in ``TRTLLM_DISAGG_ROLE``:
``launch_server`` publishes it from ``--server_role``, a disaggregated MPI worker from its server's
type, and without either the server keeps whatever its environment holds. A generation server is
refused before the LLM is built. Without the context role, dense DWDP gets one warning, and the
drop setting gets one on a disaggregated server without a role; both name ``--server_role CONTEXT``.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import click
import pytest

from tensorrt_llm.commands import serve
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.llmapi.llm_args import DwdpConfig

pytestmark = pytest.mark.cpu_only

_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
_DROP = {"drop_inflight_requests_on_shutdown": True}
_DENSE = {"dwdp_config": DwdpConfig(enable_dense=True, dwdp_size=4)}
_TRANSCEIVER = {"cache_transceiver_config": {"backend": "DEFAULT"}}


class _LlmBuilt(Exception):
    """Raised by the stand-in LLM: the launch got past the role check."""


def _clear_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the role for this test, and restore it afterwards.

    ``launch_server`` and the disaggregated MPI worker publish the role into ``os.environ``
    directly. ``monkeypatch.delenv`` records nothing for an unset variable, so it is set first:
    monkeypatch then restores it as it was, unset included, and a role a test published does not
    reach the tests that follow.
    """
    monkeypatch.setenv(_ROLE_ENV, "")
    monkeypatch.delenv(_ROLE_ENV)


@pytest.fixture(autouse=True, scope="module")
def _role_is_restored() -> Iterator[None]:
    """The tests leave the role as they found it, for the test modules that run after them."""
    before = os.environ.get(_ROLE_ENV)
    yield
    assert os.environ.get(_ROLE_ENV) == before


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """A stand-in LLM that stops the launch when it is built, with no role published yet.

    Returns the stand-in, so a test can tell whether the launch reached it.
    """
    _clear_role(monkeypatch)
    llm = Mock(side_effect=_LlmBuilt)
    monkeypatch.setattr(serve, "PyTorchLLM", llm)
    return llm


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The warnings trtllm-serve logs."""
    warning = Mock()
    monkeypatch.setattr(serve.logger, "warning", warning)
    return warning


def _launch_server(tmp_path: Path, settings: dict, **kwargs: object) -> None:
    """Run launch_server on a kernel-assigned port, with one frontend, up to the LLM."""
    llm_args = {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 1, **settings}
    serve.launch_server("127.0.0.1", 0, llm_args, report_addr=str(tmp_path / "addr"), **kwargs)


def _warnings_about(warnings: Mock, setting: str) -> list[str]:
    return [call.args[0] for call in warnings.call_args_list if setting in call.args[0]]


def _drop_warnings(warnings: Mock) -> list[str]:
    return _warnings_about(warnings, "drop_inflight_requests_on_shutdown")


def _dense_warnings(warnings: Mock) -> list[str]:
    return _warnings_about(warnings, "dwdp_config.enable_dense")


@pytest.mark.parametrize("value", [True, "true", 1], ids=["bool", "string", "int"])
def test_a_generation_server_refuses_drop_inflight_before_the_llm_is_built(
    launch: Mock, tmp_path: Path, value: object
) -> None:
    """--server_role GENERATION refuses the setting, in any form TorchLlmArgs accepts."""
    with pytest.raises(click.BadParameter, match="is for context servers only"):
        _launch_server(
            tmp_path,
            {"drop_inflight_requests_on_shutdown": value},
            server_role=ServerRole.GENERATION,
        )
    launch.assert_not_called()


@pytest.mark.parametrize(
    "dwdp_config",
    [DwdpConfig(enable_dense=True, dwdp_size=4), {"enable_dense": True, "dwdp_size": 4}],
    ids=["config", "mapping"],
)
def test_a_generation_server_refuses_dense_dwdp_before_the_llm_is_built(
    launch: Mock, tmp_path: Path, dwdp_config: object
) -> None:
    """--server_role GENERATION refuses enable_dense, parsed or still a YAML mapping."""
    with pytest.raises(click.BadParameter, match="dwdp_config.enable_dense is for context servers"):
        _launch_server(tmp_path, {"dwdp_config": dwdp_config}, server_role=ServerRole.GENERATION)
    launch.assert_not_called()


@pytest.mark.parametrize(
    "role,dwdp_config,warned",
    [
        (None, DwdpConfig(enable_dense=True, dwdp_size=4), True),
        ("", {"enable_dense": True}, True),
        ("context", DwdpConfig(enable_dense=True, dwdp_size=4), False),
        (None, DwdpConfig(dwdp_size=4), False),
        (None, None, False),
    ],
    ids=["no-role", "unknown-role", "context", "expert-only", "no-dwdp"],
)
def test_dense_dwdp_without_the_context_role_is_warned_once(
    monkeypatch: pytest.MonkeyPatch,
    warnings: Mock,
    role: str | None,
    dwdp_config: object,
    warned: bool,
) -> None:
    """Any server without the context role is warned about dense DWDP.

    Dense DWDP gathers every covered layer in every forward, with or without a cache transceiver.
    """
    if role is None:
        _clear_role(monkeypatch)
    else:
        monkeypatch.setenv(_ROLE_ENV, role)

    serve._check_context_only_settings({"dwdp_config": dwdp_config}, warn=True)

    found = _dense_warnings(warnings)
    assert len(found) == int(warned)
    if warned:
        assert "--server_role CONTEXT" in found[0]


@pytest.mark.parametrize(
    "role,settings,warned",
    [
        (None, {**_DROP, **_TRANSCEIVER}, True),
        ("", {**_DROP, **_TRANSCEIVER}, True),
        (None, _DROP, False),
        ("context", {**_DROP, **_TRANSCEIVER}, False),
        (None, {"drop_inflight_requests_on_shutdown": False, **_TRANSCEIVER}, False),
        (None, {"drop_inflight_requests_on_shutdown": "no-such-bool", **_TRANSCEIVER}, False),
    ],
    ids=["no-role", "unknown-role", "aggregated", "context", "off", "malformed"],
)
def test_without_a_role_a_disaggregated_server_is_warned_once(
    monkeypatch: pytest.MonkeyPatch, warnings: Mock, role: str | None, settings: dict, warned: bool
) -> None:
    """Without a role only a server with a cache transceiver is warned, naming the remedy.

    A context server and an aggregated one are silent, and a value that does not parse is left
    to the LLM arguments' validation.
    """
    if role is None:
        _clear_role(monkeypatch)
    else:
        monkeypatch.setenv(_ROLE_ENV, role)

    serve._check_context_only_settings(dict(settings), warn=True)

    found = _drop_warnings(warnings)
    assert len(found) == int(warned)
    if warned:
        assert "--server_role CONTEXT" in found[0]


def test_an_attached_frontend_does_not_repeat_the_warning(
    monkeypatch: pytest.MonkeyPatch, warnings: Mock
) -> None:
    """The launcher warns; an attached frontend re-runs the check silently."""
    _clear_role(monkeypatch)

    serve._check_context_only_settings({**_DROP, **_DENSE, **_TRANSCEIVER}, warn=False)

    assert warnings.call_args_list == []


def test_an_inherited_role_is_checked_like_a_published_one(
    monkeypatch: pytest.MonkeyPatch, launch: Mock, tmp_path: Path
) -> None:
    """Without --server_role, launch_server keeps the role its environment holds."""
    monkeypatch.setenv(_ROLE_ENV, "generation")

    with pytest.raises(click.BadParameter, match="is for context servers only"):
        _launch_server(tmp_path, _DROP)
    launch.assert_not_called()


@pytest.mark.parametrize(
    "server_type,refused,warned",
    [("gen", True, False), ("ctx", False, False), ("mm_encoder", False, True)],
    ids=["gen", "ctx", "unknown"],
)
def test_a_disaggregated_mpi_worker_takes_its_role_from_the_server_type(
    monkeypatch: pytest.MonkeyPatch,
    launch: Mock,
    warnings: Mock,
    tmp_path: Path,
    server_type: str,
    refused: bool,
    warned: bool,
) -> None:
    """A gen server is refused, a ctx server passes, and an unknown type has no role.

    With both settings set, the refusal names dense DWDP, which the check reads first.
    """
    server = SimpleNamespace(type=server_type, hostname="127.0.0.1", port=0)
    config = SimpleNamespace(
        server_configs=[server], allow_request_chat_template=False, internal_request_auth_key=None
    )
    monkeypatch.setattr(serve, "parse_disagg_config_file", lambda path: config)
    monkeypatch.setenv("TLLM_DISAGG_INSTANCE_IDX", "0")
    real_launch_server = serve.launch_server
    monkeypatch.setattr(
        serve,
        "launch_server",
        lambda **kwargs: real_launch_server(**kwargs, report_addr=str(tmp_path / "addr")),
    )
    llm_args = {"backend": "pytorch", "model": "dummy", **_DROP, **_DENSE, **_TRANSCEIVER}

    if refused:
        with pytest.raises(click.BadParameter, match="dwdp_config.enable_dense is for context"):
            serve._launch_disaggregated_server("unused.yaml", llm_args)
        launch.assert_not_called()
    else:
        with pytest.raises(_LlmBuilt):
            serve._launch_disaggregated_server("unused.yaml", llm_args)
    assert len(_drop_warnings(warnings)) == int(warned)
    assert len(_dense_warnings(warnings)) == int(warned)
