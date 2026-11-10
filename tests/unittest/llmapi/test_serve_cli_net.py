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
"""The listening-socket helpers of trtllm-serve and the policy of each listener.

Every listener takes its address family from _listener_address_family(), and
the ones that share the bind helper go through _bind_listener(). The policy
and the SO_REUSEPORT choice of each listener are pinned here, so that changing
one is a decision of its own rather than a side effect of another change.
"""

import contextlib
import errno
import inspect
import socket
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, NamedTuple, Optional

import pytest
import yaml

import tensorrt_llm.commands.serve as serve_module
import tensorrt_llm.commands.serve_cli._net as net_module
from tensorrt_llm.commands.serve import DisaggLauncherEnvs, DisaggWorkerEnvs
from tensorrt_llm.commands.serve_cli._net import _bind_listener, _listener_address_family
from tensorrt_llm.logger import Logger
from tensorrt_llm.serve.multi_frontend import LAUNCHER_UDS_ENV

pytestmark = pytest.mark.cpu_only

_HOST = "h.test"
_PORT = 12345


def _addr_info(*families):
    """getaddrinfo() results of the given families, in that order."""
    return [
        (
            family,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("::1", _PORT, 0, 0) if family == socket.AF_INET6 else ("127.0.0.1", _PORT),
        )
        for family in families
    ]


# --- _listener_address_family -------------------------------------------------


@pytest.mark.parametrize(
    "families,expected",
    [
        ((socket.AF_INET6, socket.AF_INET), socket.AF_INET),
        ((socket.AF_INET6, socket.AF_INET6), socket.AF_INET6),
        ((socket.AF_INET,), socket.AF_INET),
    ],
    ids=["mixed", "ipv6-only", "ipv4-only"],
)
def test_auto_policy_binds_ipv4_unless_every_address_is_ipv6(
    monkeypatch: pytest.MonkeyPatch, families, expected
) -> None:
    lookups = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        lookups.append((host, port, args, kwargs))
        return _addr_info(*families)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert _listener_address_family("server", _HOST, _PORT) == expected
    assert _listener_address_family("server", _HOST, _PORT, policy="auto") == expected
    assert lookups == [(_HOST, _PORT, (socket.AF_UNSPEC, socket.SOCK_STREAM), {})] * 2


def test_auto_policy_reports_a_failed_lookup_as_a_bind_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup failure is reported like a bind failure, without a port-holder diagnosis."""
    error = socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    def unresolvable(host, *args, **kwargs):
        raise error

    diagnosed_ports = []
    monkeypatch.setattr(socket, "getaddrinfo", unresolvable)
    monkeypatch.setattr(net_module, "_diagnose_port_in_use", diagnosed_ports.append)

    with pytest.raises(RuntimeError) as raised:
        _listener_address_family("server", _HOST, _PORT)

    assert str(raised.value) == (
        f"Failed to bind socket to {_HOST}:{_PORT}: address lookup failed for host {_HOST!r}: {error}"
    )
    assert raised.value.__cause__ is error
    assert diagnosed_ports == []


def test_ipv4_policy_never_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_lookup(*args, **kwargs):
        pytest.fail(f"the ipv4 policy must not resolve the host, but called getaddrinfo{args}")

    monkeypatch.setattr(socket, "getaddrinfo", no_lookup)

    assert _listener_address_family("coordinator", _HOST, _PORT, policy="ipv4") == socket.AF_INET


def test_first_policy_takes_the_first_result_and_propagates_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups = []

    def ipv6_first(host, port, *args, **kwargs):
        lookups.append((host, port, args, kwargs))
        return _addr_info(socket.AF_INET6, socket.AF_INET)

    monkeypatch.setattr(socket, "getaddrinfo", ipv6_first)
    family = _listener_address_family("OpenEngine server", _HOST, _PORT, policy="first")
    assert family == socket.AF_INET6
    assert lookups == [(_HOST, _PORT, (), {"type": socket.SOCK_STREAM})]

    error = socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    def unresolvable(host, *args, **kwargs):
        raise error

    monkeypatch.setattr(socket, "getaddrinfo", unresolvable)
    with pytest.raises(socket.gaierror) as raised:
        _listener_address_family("OpenEngine server", _HOST, _PORT, policy="first")
    assert raised.value is error


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown address-family policy 'dual'"):
        _listener_address_family("server", _HOST, _PORT, policy="dual")


# --- _bind_listener -----------------------------------------------------------


class _FakeSocket:
    """Records what _bind_listener does to the socket it creates."""

    def __init__(
        self,
        family,
        type_,
        *,
        setsockopt_error: Optional[OSError] = None,
        bind_error: Optional[OSError] = None,
    ) -> None:
        self.family = family
        self.type = type_
        self.options = []
        self.bound = []
        self.closed = False
        self._setsockopt_error = setsockopt_error
        self._bind_error = bind_error

    def setsockopt(self, level, option, value) -> None:
        if self._setsockopt_error is not None:
            raise self._setsockopt_error
        self.options.append((level, option, value))

    def bind(self, address) -> None:
        if self._bind_error is not None:
            raise self._bind_error
        self.bound.append(address)

    def close(self) -> None:
        self.closed = True


def _fake_sockets(monkeypatch: pytest.MonkeyPatch, **errors) -> list:
    """Make socket.socket() hand out recording fakes; returns the fakes created."""
    created = []

    def factory(family, type_):
        fake = _FakeSocket(family, type_, **errors)
        created.append(fake)
        return fake

    monkeypatch.setattr(socket, "socket", factory)
    return created


@pytest.mark.parametrize(
    "reuse_port,expected_options",
    [
        (False, [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]),
        (
            True,
            [
                (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
                (socket.SOL_SOCKET, socket.SO_REUSEPORT, 1),
            ],
        ),
    ],
    ids=["one-process", "shared-port"],
)
def test_bind_listener_sets_reuse_options(
    monkeypatch: pytest.MonkeyPatch, reuse_port: bool, expected_options
) -> None:
    created = _fake_sockets(monkeypatch)

    sock = _bind_listener("server", _HOST, _PORT, family=socket.AF_INET6, reuse_port=reuse_port)

    assert created == [sock]
    assert (sock.family, sock.type) == (socket.AF_INET6, socket.SOCK_STREAM)
    assert sock.options == expected_options
    assert sock.bound == [(_HOST, _PORT)]
    assert not sock.closed


def test_bind_listener_closes_and_reports_a_bind_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    error = OSError(errno.EADDRINUSE, "Address already in use")
    created = _fake_sockets(monkeypatch, bind_error=error)
    monkeypatch.setattr(net_module, "_diagnose_port_in_use", lambda port: f"pid=1 holds {port}")

    with pytest.raises(RuntimeError) as raised:
        _bind_listener("server", _HOST, _PORT, family=socket.AF_INET)

    assert str(raised.value) == (
        f"Failed to bind socket to {_HOST}:{_PORT}: {error}. Port holder(s): pid=1 holds {_PORT}"
    )
    assert raised.value.__cause__ is error
    assert [fake.closed for fake in created] == [True]


def test_bind_listener_closes_when_an_option_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    error = OSError(errno.ENOPROTOOPT, "Protocol not available")
    created = _fake_sockets(monkeypatch, setsockopt_error=error)

    with pytest.raises(OSError) as raised:
        _bind_listener("server", _HOST, _PORT, family=socket.AF_INET)

    assert raised.value is error
    assert [fake.closed for fake in created] == [True]
    assert created[0].bound == []


# --- the policy of every listener ---------------------------------------------


class _Stop(BaseException):
    """Ends a listener as soon as the probe has recorded the call under test."""


def _probe(monkeypatch: pytest.MonkeyPatch, module: ModuleType, *, family_result) -> dict:
    """Replace the two helpers where the listener looks them up with recorders.

    The family recorder returns ``family_result``, or ends the listener when it
    is None (listeners that do not reach the bind helper). The bind recorder
    always ends the listener.
    """
    calls = {"family": [], "bind": []}

    def listener_address_family(kind, host, port, policy="auto"):
        calls["family"].append((kind, host, port, policy))
        if family_result is None:
            raise _Stop()
        return family_result

    def bind_listener(kind, host, port, *, family, reuse_port=False):
        calls["bind"].append((kind, host, port, family, reuse_port))
        raise _Stop()

    monkeypatch.setattr(module, "_listener_address_family", listener_address_family)
    monkeypatch.setattr(module, "_bind_listener", bind_listener, raising=False)
    return calls


def _launch_server(num_frontends: int, *, attached: bool):
    def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType) -> None:
        if attached:
            monkeypatch.setenv("TLLM_EXECUTOR_ATTACH_INFO", "set by the test")
        # The launcher reserves the port with a real probe socket before it
        # binds; the probe is not the listener under test.
        monkeypatch.setattr(
            module,
            "_reserve_multi_frontend_port",
            lambda host, port, family: contextlib.nullcontext(),
        )
        llm_args = {"backend": "pytorch", "model": "dummy"}
        if num_frontends > 1:
            llm_args["num_serve_frontends"] = num_frontends
        module.launch_server(_HOST, _PORT, llm_args)

    return run


def _run_visual_gen_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType
) -> None:
    # Under a multi-rank launcher the VisualGen entry point would act on that
    # launch before reaching the listener.
    import tensorrt_llm._torch.visual_gen.executor as visual_gen_executor

    monkeypatch.setattr(visual_gen_executor, "_detect_external_launch", lambda: None)
    module.launch_visual_gen_server(_HOST, _PORT, "dummy")


def _run_disaggregated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType) -> None:
    # set_prometheus_multiproc_dir() deletes the directory its own variable
    # points at on its second call in a process; a directory that outlives the
    # call keeps this case independent of the others.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    config = tmp_path / "disagg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "hostname": _HOST,
                "port": _PORT,
                "num_workers": 1,
                "context_servers": {},
                "generation_servers": {},
            }
        )
    )
    module.disaggregated.callback(
        config_file=str(config),
        metadata_server_config_file=None,
        server_start_timeout=180,
        request_timeout=180,
        log_level="info",
        metrics_log_interval=0,
        schedule_style=None,
        telemetry=True,
        report_addr=None,
    )


def _run_coordinator_and_fleet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType
) -> None:
    disagg_cfg = SimpleNamespace(hostname=_HOST, port=_PORT)
    module._serve_coordinator_and_fleet(disagg_cfg, "disagg.yaml", None, None, 180, 180, 2)


def _run_fleet_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType) -> None:
    # The worker's process setup disables the garbage collector of the test
    # process; its server builder reads the launcher's environment.
    monkeypatch.setattr(module, "_init_fleet_worker_process", lambda: None)
    monkeypatch.setattr(
        module,
        "_build_disagg_server_from_env",
        lambda: SimpleNamespace(_config=SimpleNamespace(hostname=_HOST, port=_PORT)),
    )
    module._run_fleet_worker_impl()


def _run_openengine_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: ModuleType
) -> None:
    module.launch_server(
        _HOST, _PORT, {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 2}
    )


def _openengine_launch_server():
    from tensorrt_llm.grpc.openengine.server import launch_server

    return launch_server


class _Site(NamedTuple):
    name: str
    function: Callable[[], Callable]  # the listener; resolved when the test runs
    run: Callable[[pytest.MonkeyPatch, Path, ModuleType], None]
    family_call: tuple  # (kind, host, port, policy)
    bind_call: Optional[tuple]  # (kind, host, port, family, reuse_port); None: not reached


_SITES = [
    _Site(
        "launch_server with one frontend",
        lambda: serve_module.launch_server,
        _launch_server(1, attached=False),
        ("server", _HOST, _PORT, "auto"),
        ("server", _HOST, _PORT, socket.AF_INET, False),
    ),
    _Site(
        "launch_server as the launcher of two frontends",
        lambda: serve_module.launch_server,
        _launch_server(2, attached=False),
        ("server", _HOST, _PORT, "auto"),
        ("server", _HOST, _PORT, socket.AF_INET, True),
    ),
    _Site(
        "launch_server as an attached frontend",
        lambda: serve_module.launch_server,
        _launch_server(2, attached=True),
        ("server", _HOST, _PORT, "auto"),
        ("server", _HOST, _PORT, socket.AF_INET, True),
    ),
    _Site(
        "launch_visual_gen_server",
        lambda: serve_module.launch_visual_gen_server,
        _run_visual_gen_server,
        ("VisualGen server", _HOST, _PORT, "auto"),
        ("VisualGen server", _HOST, _PORT, socket.AF_INET, False),
    ),
    _Site(
        "disaggregated with one worker",
        lambda: serve_module.disaggregated.callback,
        _run_disaggregated,
        ("disaggregated server", _HOST, _PORT, "ipv4"),
        ("disaggregated server", _HOST, _PORT, socket.AF_INET, False),
    ),
    _Site(
        "coordinator of a worker fleet",
        lambda: serve_module._serve_coordinator_and_fleet,
        _run_coordinator_and_fleet,
        ("coordinator", _HOST, _PORT - 1, "ipv4"),
        ("coordinator", _HOST, _PORT - 1, socket.AF_INET, False),
    ),
    _Site(
        "fleet worker",
        lambda: serve_module._run_fleet_worker_impl,
        _run_fleet_worker,
        ("fleet worker", _HOST, _PORT, "ipv4"),
        None,
    ),
    _Site(
        "OpenEngine launcher trial bind",
        _openengine_launch_server,
        _run_openengine_launcher,
        ("OpenEngine server", _HOST, _PORT, "first"),
        None,
    ),
]


@pytest.fixture
def restored_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the process state the listeners change out of later tests.

    Every environment variable the listeners read or write is restored. The
    disaggregated server also sets the level of the process-wide logger and the
    context of a live usage session; neither belongs to a listener's behavior,
    so both calls do nothing here.
    """
    for name in (
        "TRTLLM_RESPONSES_API_DISABLE_STORE",
        "TLLM_EXECUTOR_ATTACH_INFO",
        "PROMETHEUS_MULTIPROC_DIR",
        LAUNCHER_UDS_ENV,
        DisaggLauncherEnvs.TLLM_DISAGG_ROLE,
        DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID,
        DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_PORT,
        DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_UDS,
    ):
        # Recording the variable makes monkeypatch put its old value back,
        # also when the listener sets it later in the test.
        monkeypatch.delenv(name, raising=False)
    # Patched on the class, so undoing it leaves no attribute on the logger instance.
    monkeypatch.setattr(Logger, "set_level", lambda self, min_severity: None)
    monkeypatch.setattr(
        inspect.getmodule(serve_module.disaggregated.callback),
        "set_usage_context",
        lambda usage_context: None,
    )


@pytest.mark.parametrize("site", _SITES, ids=lambda site: site.name)
def test_every_listener_keeps_its_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restored_environment, site: _Site
) -> None:
    module = inspect.getmodule(site.function())
    calls = _probe(monkeypatch, module, family_result=socket.AF_INET if site.bind_call else None)

    with pytest.raises(_Stop):
        site.run(monkeypatch, tmp_path, module)

    expected = ([site.family_call], [site.bind_call] if site.bind_call else [])
    actual = (calls["family"], calls["bind"])
    assert actual == expected, (
        f"Listener {site.name} changed its address-family policy or SO_REUSEPORT choice: "
        f"expected {expected}, got {actual}. Each listener keeps its policy (see "
        "_listener_address_family in tensorrt_llm/commands/serve_cli/_net.py); changing "
        "one is a behavior change for its own PR."
    )


# --- the kernel-assigned port and the launcher's port lock --------------------

_KERNEL_PORT = 45678


class _BoundSocket:
    """What the bind recorder hands back: a bound socket with a known kernel-assigned port."""

    def __init__(self, port: int) -> None:
        self._port = port

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def getsockname(self):
        return (_HOST, self._port)


def _bind_to_kernel_port(monkeypatch: pytest.MonkeyPatch, module: ModuleType, calls: list) -> None:
    """A port-0 bind answers with _KERNEL_PORT; publishing records its arguments and stops."""
    monkeypatch.setattr(
        module, "_listener_address_family", lambda kind, host, port, policy="auto": socket.AF_INET
    )

    def bind_listener(kind, host, port, *, family, reuse_port=False):
        calls.append(("bind", host, port))
        return _BoundSocket(_KERNEL_PORT if port == 0 else port)

    def publish_bound_address(report_addr, host, port):
        calls.append(("publish", report_addr, host, port))
        raise _Stop()

    monkeypatch.setattr(module, "_bind_listener", bind_listener)
    monkeypatch.setattr(module, "_publish_bound_address", publish_bound_address)


def test_launch_server_publishes_the_kernel_assigned_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restored_environment
) -> None:
    module = inspect.getmodule(serve_module.launch_server)
    calls = []
    _bind_to_kernel_port(monkeypatch, module, calls)
    report_addr = str(tmp_path / "server.addr")

    with pytest.raises(_Stop):
        module.launch_server(
            _HOST, 0, {"backend": "pytorch", "model": "dummy"}, report_addr=report_addr
        )

    assert calls == [("bind", _HOST, 0), ("publish", report_addr, _HOST, _KERNEL_PORT)], (
        "launch_server must publish the port the kernel assigned to a port-0 bind, read back "
        f"from the bound socket, not the requested port 0; recorded {calls}"
    )


def test_disaggregated_publishes_the_kernel_assigned_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restored_environment
) -> None:
    module = inspect.getmodule(serve_module.disaggregated.callback)
    calls = []
    _bind_to_kernel_port(monkeypatch, module, calls)
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    config = tmp_path / "disagg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "hostname": _HOST,
                "port": 0,
                "num_workers": 1,
                "context_servers": {},
                "generation_servers": {},
            }
        )
    )
    report_addr = str(tmp_path / "disagg.addr")

    with pytest.raises(_Stop):
        module.disaggregated.callback(
            config_file=str(config),
            metadata_server_config_file=None,
            server_start_timeout=180,
            request_timeout=180,
            log_level="info",
            metrics_log_interval=0,
            schedule_style=None,
            telemetry=True,
            report_addr=report_addr,
        )

    # The published port is disagg_cfg.port: the readback must have been written into it.
    assert calls == [("bind", _HOST, 0), ("publish", report_addr, _HOST, _KERNEL_PORT)], (
        "the disaggregated server must write the kernel-assigned port of a port-0 bind back "
        f"into its config and publish that port, not the requested port 0; recorded {calls}"
    )


class _RecordingLock:
    """Stands in for the launcher's port lock; records when it is entered and left."""

    def __init__(self, events: list) -> None:
        self._events = events

    def __enter__(self):
        self._events.append("lock entered")
        return self

    def __exit__(self, *exc) -> bool:
        self._events.append("lock left")
        return False


def test_launch_server_holds_the_port_lock_across_the_bind(
    monkeypatch: pytest.MonkeyPatch, restored_environment
) -> None:
    module = inspect.getmodule(serve_module.launch_server)
    events = []

    def reserve_multi_frontend_port(host, port, family):
        events.append(("reserved", host, port, family))
        return _RecordingLock(events)

    def bind_listener(kind, host, port, *, family, reuse_port=False):
        events.append(("bind", host, port))
        raise _Stop()

    monkeypatch.setattr(module, "_reserve_multi_frontend_port", reserve_multi_frontend_port)
    monkeypatch.setattr(
        module, "_listener_address_family", lambda kind, host, port, policy="auto": socket.AF_INET
    )
    monkeypatch.setattr(module, "_bind_listener", bind_listener)

    with pytest.raises(_Stop):
        module.launch_server(
            _HOST, _PORT, {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 2}
        )

    assert events == [
        ("reserved", _HOST, _PORT, socket.AF_INET),
        "lock entered",
        ("bind", _HOST, _PORT),
        "lock left",
    ], (
        "the launcher must hold the port lock while it binds the listener: the lock is "
        "reserved and entered before the bind and left after it, also when the bind fails; "
        f"recorded {events}"
    )
