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
"""Unit coverage for trtllm-serve's --report_addr publish/read contract.

The integration migrations that consume this only run on GPU stages, so the
publisher, the reader and the guards that reject --report_addr where it cannot
work are covered here instead.
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import click
import pytest
import yaml
from click.testing import CliRunner

from tensorrt_llm.commands.serve import _publish_bound_address, disaggregated, launch_server

# The reader half lives with the integration helpers, so both halves of the
# round trip are exercised together and a change to either side fails here.
_INTEGRATION_TESTS_DIR = Path(__file__).resolve().parents[2] / "integration"
if str(_INTEGRATION_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_TESTS_DIR))
from defs.common import wait_for_reported_addr  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _FakeProcess:
    """Stands in for a Popen handle; returncode None means still running."""

    def __init__(self, returncode: Optional[int] = None) -> None:
        self.returncode = returncode

    def poll(self) -> Optional[int]:
        return self.returncode


@pytest.mark.parametrize(
    "host,expected_host",
    [
        ("localhost", "localhost"),
        ("nvl72d066-T01", "nvl72d066-T01"),
        ("10.67.24.211", "10.67.24.211"),
        # Bracketed so the value is a usable URL authority.
        ("::1", "[::1]"),
        ("fe80::1", "[fe80::1]"),
    ],
)
def test_publish_read_round_trip(tmp_path: Path, host: str, expected_host: str) -> None:
    addr_path = str(tmp_path / "server.addr")
    _publish_bound_address(addr_path, host, 22183)

    assert open(addr_path).read() == f"{expected_host}:22183\n"
    assert wait_for_reported_addr(addr_path, timeout=5) == (expected_host, 22183)


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
def test_publish_replaces_wildcard_with_hostname(tmp_path: Path, wildcard: str) -> None:
    """A reader cannot dial a wildcard bind address."""
    addr_path = str(tmp_path / "server.addr")
    _publish_bound_address(addr_path, wildcard, 8000)

    reported_host = open(addr_path).read().strip().rsplit(":", 1)[0]
    assert reported_host.strip("[]") == socket.gethostname()


def test_publish_creates_missing_parent_directories(tmp_path: Path) -> None:
    addr_path = str(tmp_path / "deep" / "nested" / "server.addr")
    _publish_bound_address(addr_path, "localhost", 8000)
    assert open(addr_path).read() == "localhost:8000\n"


def test_publish_is_noop_without_a_path(tmp_path: Path) -> None:
    """None/empty means the caller did not ask for the address."""
    before = set(os.listdir(tmp_path))
    _publish_bound_address(None, "localhost", 8000)
    _publish_bound_address("", "localhost", 8000)
    assert set(os.listdir(tmp_path)) == before


def test_publish_overwrites_without_leaking_temp_files(tmp_path: Path) -> None:
    addr_path = str(tmp_path / "server.addr")
    for port in (8000, 8001, 8002):
        _publish_bound_address(addr_path, "localhost", port)

    assert open(addr_path).read() == "localhost:8002\n"
    assert os.listdir(tmp_path) == ["server.addr"]


def test_concurrent_reader_never_sees_a_partial_line(tmp_path: Path) -> None:
    """The rename must be atomic: readers poll this file while it is rewritten."""
    addr_path = str(tmp_path / "server.addr")
    _publish_bound_address(addr_path, "localhost", 8000)

    # Every value the file is ever given. A reader must observe one of these
    # exactly: a truncated write such as "localhost:\n" or "localhost:9\n" is
    # what a non-atomic publisher would expose, and must fail this test.
    published = {"localhost:8000\n"} | {f"localhost:{9000 + i}\n" for i in range(3)}
    seen = set()
    stop = threading.Event()
    # Without this the publish loop can finish and set stop before the reader
    # is ever scheduled, leaving seen empty and failing the test with no
    # atomicity defect present.
    reading = threading.Event()

    def read_loop() -> None:
        while not stop.is_set():
            try:
                seen.add(open(addr_path).read())
            except FileNotFoundError:
                seen.add("<missing>")
            reading.set()

    reader = threading.Thread(target=read_loop)
    reader.start()
    try:
        assert reading.wait(timeout=30), "reader thread never ran"
        for i in range(300):
            _publish_bound_address(addr_path, "localhost", 9000 + (i % 3))
    finally:
        stop.set()
        reader.join()

    assert seen, "reader observed nothing"
    assert seen <= published, f"observed partial or unexpected values: {seen - published}"


def test_reader_fails_fast_when_the_server_dies(tmp_path: Path) -> None:
    """A crashed server must not burn the whole timeout."""
    addr_path = str(tmp_path / "server.addr")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        wait_for_reported_addr(addr_path, timeout=30, process=_FakeProcess(1))
    assert time.monotonic() - started < 10


def test_reader_times_out_when_nothing_is_published(tmp_path: Path) -> None:
    addr_path = str(tmp_path / "server.addr")
    with pytest.raises(TimeoutError, match="did not report its address"):
        wait_for_reported_addr(addr_path, timeout=1)


def test_reader_waits_for_a_late_write(tmp_path: Path) -> None:
    addr_path = str(tmp_path / "server.addr")
    timer = threading.Timer(1.0, lambda: _publish_bound_address(addr_path, "localhost", 12345))
    timer.start()
    try:
        assert wait_for_reported_addr(addr_path, timeout=20, process=_FakeProcess()) == (
            "localhost",
            12345,
        )
    finally:
        timer.join()


@pytest.mark.parametrize("port,report_addr", [(0, "/tmp/a.addr"), (8000, "/tmp/a.addr"), (0, None)])
def test_launch_server_rejects_multiple_frontends(port: int, report_addr: Optional[str]) -> None:
    """Each frontend re-execs and would bind (and publish) its own port."""
    llm_args = {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 2}
    with pytest.raises(click.BadParameter, match="single serving frontend"):
        launch_server("localhost", port, llm_args, report_addr=report_addr)


def test_launch_server_requires_a_way_to_learn_the_port() -> None:
    """Port 0 with neither service discovery nor --report_addr is unusable."""
    llm_args = {"backend": "pytorch", "model": "dummy"}
    with pytest.raises(AssertionError, match="Port must be specified"):
        launch_server("localhost", 0, llm_args)


@pytest.mark.parametrize(
    "port,extra_args",
    [
        (0, []),
        (0, ["--report_addr", "/tmp/a.addr"]),
        (8000, ["--report_addr", "/tmp/a.addr"]),
    ],
)
def test_disaggregated_rejects_a_worker_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: int, extra_args: list[str]
) -> None:
    """A fleet spreads the port over N workers, so one published address is wrong.

    Only the rejected combinations are exercised: anything the guard lets
    through goes on to bind a socket and serve.
    """
    # set_prometheus_multiproc_dir, which disaggregated() calls before reaching
    # the guard, deletes the directory its own environment variable points at on
    # the second call in a process, so the third call raises FileNotFoundError.
    # Pointing the variable at a directory that outlives every invocation keeps
    # these parametrised cases independent of one another.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))

    config = tmp_path / "disagg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "hostname": "localhost",
                "port": port,
                "num_workers": 2,
                "context_servers": {},
                "generation_servers": {},
            }
        )
    )

    result = CliRunner().invoke(disaggregated, ["-c", str(config)] + extra_args)

    assert result.exit_code != 0
    assert "single self-contained disaggregated server" in result.output
