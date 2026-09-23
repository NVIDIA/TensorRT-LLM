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
"""Unit tests for the two Mooncake pool commands as `trtllm-serve` runs them.

Runs without a Mooncake installation and without a GPU. The resource each
command holds is a context manager that records its own release, and the wait
the command would otherwise spend idling is where the signal under test is
delivered to this process.
"""

import contextlib
import json
import os
import signal
import time
from types import SimpleNamespace

import click
import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import CONFIG_PATH_ENV
from tensorrt_llm.commands import _telemetry
from tensorrt_llm.commands import mooncake as mooncake_commands
from tensorrt_llm.usage.config import UsageContext

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM]
COMMANDS = ["mooncake_master", "mooncake_donor"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool config: these tests drive the commands from their flags."""
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Both commands install handlers process-wide, so put pytest's back."""
    installed = {number: signal.getsignal(number) for number in HANDLED_SIGNALS}
    yield
    for number, handler in installed.items():
        signal.signal(number, handler)


@pytest.fixture(autouse=True)
def terminal_outcomes(monkeypatch):
    """Collect what the shared boundary reports, without a usage session.

    Autouse because a command reaching the real session would apply this
    process's opt-out for the rest of the run.
    """
    outcomes = []
    monkeypatch.setattr(
        _telemetry.usage, "report_exit", lambda outcome, **_kwargs: outcomes.append(outcome)
    )
    monkeypatch.setattr(_telemetry.usage, "apply_usage_session_config", lambda *a, **k: True)
    monkeypatch.setattr(_telemetry.usage, "set_lifecycle_phase", lambda *a, **k: None)
    monkeypatch.setattr(_telemetry.usage, "get_observed_signal", lambda: 0)
    monkeypatch.setattr(_telemetry.usage, "record_observed_signal", lambda *a, **k: None)
    return outcomes


class Resource:
    """Stands in for the master process or the mounted segment.

    Records its release, so a test can tell a signal that unwound the command
    from one that ended it with the `finally` never reached, and what it was
    asked for, so a test can check what the command resolved before asking.
    """

    def __init__(self, held):
        self.held = held
        self.released = False
        #: `None` until the resource is taken, which distinguishes a command
        #: that gave up beforehand from one that took it.
        self.taken_with = None

    @contextlib.contextmanager
    def holding(self, *args, **_kwargs):
        self.taken_with = args
        try:
            yield self.held
        finally:
            self.released = True


class CommandUnderTest:
    """One command, its stubbed resource, and the arguments that reach it."""

    def __init__(self, resource: Resource, args: list[str]):
        self.resource = resource
        self.args = args

    def run(self, *extra_args: str) -> None:
        group = _telemetry.TelemetryGroup(
            name="trtllm-serve",
            telemetry_usage_context=UsageContext.CLI_SERVE,
            telemetry_component="server",
            commands={
                "mooncake_master": mooncake_commands.mooncake_master,
                "mooncake_donor": mooncake_commands.mooncake_donor,
            },
        )
        group.main(
            args=[*self.args, *extra_args],
            prog_name="trtllm-serve",
            standalone_mode=False,
        )


@pytest.fixture
def master(monkeypatch, tmp_path) -> CommandUnderTest:
    resource = Resource(
        SimpleNamespace(
            address="127.0.0.1:50051",
            log_path=str(tmp_path / "master.log"),
            # Alive, so the command idles rather than reporting a dead master.
            process=SimpleNamespace(poll=lambda: None),
        )
    )
    monkeypatch.setattr(mooncake_store, "running_master", resource.holding)
    return CommandUnderTest(resource, ["mooncake_master", "--run_dir", str(tmp_path)])


@pytest.fixture
def donor(monkeypatch) -> CommandUnderTest:
    resource = Resource("10.0.0.1:12345")
    monkeypatch.setattr(mooncake_store, "donate_segment", resource.holding)
    monkeypatch.setattr(mooncake_store, "resolve_master_address", lambda address, _timeout: address)
    monkeypatch.setattr(mooncake_store, "wait_for_master", lambda _address: None)
    return CommandUnderTest(
        resource,
        ["mooncake_donor", "--master_server_address", "127.0.0.1:50051", "--segment_size", "1GiB"],
    )


@pytest.fixture
def command(request, master, donor) -> CommandUnderTest:
    return {"mooncake_master": master, "mooncake_donor": donor}[request.param]


def signal_on_idle(monkeypatch, number: int) -> None:
    """Deliver `number` to this process the first time a command idles.

    The handler runs at the next bytecode boundary in the main thread, so the
    loop here only has to give the interpreter one. It bounds the wait rather
    than blocking, so a handler that never fires fails the test rather than
    hanging it.
    """

    def sleep(_seconds):
        os.kill(os.getpid(), number)
        for _ in range(100):
            time.sleep(0.01)

    monkeypatch.setattr(
        mooncake_commands, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic)
    )


@pytest.mark.parametrize("number", HANDLED_SIGNALS)
@pytest.mark.parametrize("command", COMMANDS, indirect=True)
def test_a_signal_is_reported_once_and_still_releases_what_was_held(
    monkeypatch, terminal_outcomes, command, number
):
    """The signal reaches the boundary, and the resource is still given up.

    A command that returned normally instead would be reported as a clean
    exit before any model was loaded.
    """
    signal_on_idle(monkeypatch, number)

    with pytest.raises(_telemetry.SignalExit) as stopped:
        command.run()

    assert stopped.value.signal_number == number
    assert command.resource.released

    assert len(terminal_outcomes) == 1
    outcome = terminal_outcomes[0]
    assert outcome.termination_kind == "signal"
    assert outcome.signal_number == number
    assert outcome.exit_code == 128 + number


@pytest.mark.parametrize("flag", ["--telemetry", "--no-telemetry"])
@pytest.mark.parametrize("command", COMMANDS, indirect=True)
def test_the_opt_out_the_group_documents_is_accepted(monkeypatch, command, flag):
    """Without the option Click rejects the flag as a usage error instead."""
    signal_on_idle(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        command.run(flag)

    assert command.resource.released


# ---- what the donor resolves before it offers anything ----


@pytest.fixture
def segment(monkeypatch) -> Resource:
    """Stub only the segment, so the command's address handling still runs."""
    resource = Resource("10.0.0.1:12345")
    monkeypatch.setattr(mooncake_store, "donate_segment", resource.holding)
    return resource


def donor_running(resource: Resource, *args: str) -> CommandUnderTest:
    return CommandUnderTest(resource, ["mooncake_donor", *args])


def test_a_donor_resolves_a_published_master_address_before_joining(monkeypatch, segment, tmp_path):
    """What lets a generation node name a path instead of a scheduler's choice."""
    address_file = tmp_path / "master.addr"
    address_file.write_text("10.0.0.9:50051\n")
    probed = []
    monkeypatch.setattr(mooncake_store, "wait_for_master", probed.append)
    signal_on_idle(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        donor_running(
            segment,
            "--master_server_address",
            f"file://{address_file}",
            "--segment_size",
            "2GiB",
        ).run()

    # The resolved address is both what was probed and what Mooncake is given:
    # Mooncake cannot dial a file:// URL.
    assert probed == ["10.0.0.9:50051"]
    assert segment.taken_with[0] == "10.0.0.9:50051"
    # A size string reaching Mooncake unparsed would be a segment of nothing.
    assert segment.taken_with[1] == 2 * 1024**3


def test_a_donor_reports_an_unreachable_master_before_offering_the_segment(monkeypatch, segment):
    """Otherwise this is a status code from setup, with no address in it."""

    def refuse(address):
        raise TimeoutError(f"The Mooncake master at {address} did not accept connections")

    monkeypatch.setattr(mooncake_store, "resolve_master_address", lambda address, _t: address)
    monkeypatch.setattr(mooncake_store, "wait_for_master", refuse)

    with pytest.raises(TimeoutError, match="10.0.0.1:50051"):
        donor_running(segment, "--master_server_address", "10.0.0.1:50051").run()

    assert segment.taken_with is None


@pytest.mark.parametrize("size", ["", "16 GB!"])
def test_a_donor_rejects_a_size_it_cannot_parse(monkeypatch, segment, size):
    """Caught as a usage error rather than as a segment of some other size."""
    monkeypatch.setattr(mooncake_store, "resolve_master_address", lambda address, _t: address)
    monkeypatch.setattr(mooncake_store, "wait_for_master", lambda _address: None)

    with pytest.raises(click.UsageError, match="--segment_size"):
        donor_running(
            segment, "--master_server_address", "10.0.0.1:50051", "--segment_size", size
        ).run()

    assert segment.taken_with is None


# ---- describing the pool from the master's own run directory ----


def test_the_master_writes_the_client_config_it_was_given(monkeypatch, master, tmp_path):
    """Lets workers be pointed at the run directory instead of a hand-written file."""
    config = tmp_path / "pool.json"
    config.write_text(
        json.dumps({"protocol": "tcp", "global_segment_size": "8GiB", "model_key": "m"})
    )
    written = {}
    monkeypatch.setattr(
        mooncake_store,
        "write_client_config",
        lambda pool, address, run_dir: written.update(pool=pool, address=address),
    )
    signal_on_idle(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        master.run("--config", str(config))

    # Written only once the master answers, and naming that master rather than
    # whichever one the config was copied from.
    assert written["address"] == "127.0.0.1:50051"
    assert written["pool"].protocol == "tcp"
    assert written["pool"].global_segment_size == "8GiB"
    assert written["pool"].launch_master is True
    assert written["pool"].master_server_address is None


def test_the_master_writes_no_client_config_unless_asked(monkeypatch, master):
    """A deployment that renders its own must not have it overwritten."""
    calls = []
    monkeypatch.setattr(mooncake_store, "write_client_config", lambda *args: calls.append(args))
    signal_on_idle(monkeypatch, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        master.run()

    assert calls == []
