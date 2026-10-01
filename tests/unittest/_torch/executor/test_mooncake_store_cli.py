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
"""Unit tests for the Mooncake pool commands as `trtllm-serve` runs them.

Runs without a Mooncake installation and without a GPU. The master process the
command holds is a context manager that records its own release, and the signal
under test is delivered from the poll that precedes the command's idle wait,
which is the point at which everything it holds is held.
"""

import contextlib
import os
import signal
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import CONFIG_PATH_ENV
from tensorrt_llm.commands import _telemetry
from tensorrt_llm.commands import mooncake as mooncake_commands
from tensorrt_llm.usage.config import UsageContext

pytestmark = pytest.mark.cpu_only

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool config: these tests drive the commands from their flags."""
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """The command installs handlers process-wide, so put pytest's back."""
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


class IdleMaster:
    """A master that stays alive, with a hook for the first time it is polled.

    The command idles by waiting on its stop event rather than by sleeping, so
    a test interrupts it from the poll that precedes that wait.
    """

    def __init__(self, address: str, log_path: str):
        self.address = address
        self.log_path = log_path
        self.on_poll = None
        self.process = SimpleNamespace(poll=self._poll)

    def _poll(self):
        if self.on_poll is not None:
            hook, self.on_poll = self.on_poll, None
            hook()
        # Alive, so the command idles rather than reporting a dead master.
        return None


class Resource:
    """Stands in for the master process the command holds.

    Records its release, so a test can tell a signal that unwound the command
    from one that ended it with the `finally` never reached.
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
                "mooncake_pool_report": mooncake_commands.mooncake_pool_report,
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
        IdleMaster(address="127.0.0.1:50051", log_path=str(tmp_path / "master.log"))
    )
    monkeypatch.setattr(mooncake_store, "running_master", resource.holding)
    return CommandUnderTest(resource, ["mooncake_master", "--run_dir", str(tmp_path)])


def signal_on_idle(command: CommandUnderTest, number: int, on_idle=None) -> None:
    """Deliver `number` to this process the first time the command idles.

    The handler runs at the next bytecode boundary in the main thread, so the
    command's own wait is what gives the interpreter one.

    `on_idle` runs before the signal, which is the only point at which the
    command's resources are all held. Anything released on the way out has to
    be observed from there rather than after `run` returns.
    """

    def interrupt():
        if on_idle is not None:
            on_idle()
        os.kill(os.getpid(), number)

    command.resource.held.on_poll = interrupt


@pytest.mark.parametrize("number", HANDLED_SIGNALS)
def test_a_signal_is_reported_once_and_still_releases_what_was_held(
    terminal_outcomes, master, number
):
    """The signal reaches the boundary, and the resource is still given up.

    A command that returned normally instead would be reported as a clean
    exit before any model was loaded.
    """
    signal_on_idle(master, number)

    with pytest.raises(_telemetry.SignalExit) as stopped:
        master.run()

    assert stopped.value.signal_number == number
    assert master.resource.released

    assert len(terminal_outcomes) == 1
    outcome = terminal_outcomes[0]
    assert outcome.termination_kind == "signal"
    assert outcome.signal_number == number
    assert outcome.exit_code == 128 + number


def test_the_signal_is_reported_only_after_the_master_is_reaped(master):
    """Reporting from the handler would leave the child unreaped.

    The release is in a `finally`, so the exit has to come from after the
    context manager rather than from inside the handler that observed it.
    """
    released_when_signalled = []
    signal_on_idle(
        master,
        signal.SIGTERM,
        on_idle=lambda: released_when_signalled.append(master.resource.released),
    )

    with pytest.raises(_telemetry.SignalExit):
        master.run()

    assert released_when_signalled == [False]
    assert master.resource.released


@pytest.mark.parametrize("flag", ["--telemetry", "--no-telemetry"])
def test_the_opt_out_the_group_documents_is_accepted(master, flag):
    """Without the option Click rejects the flag as a usage error instead."""
    signal_on_idle(master, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        master.run(flag)

    assert master.resource.released


def test_the_pool_report_accepts_the_same_opt_out(monkeypatch, tmp_path):
    """It sits in the same group, so it has to offer the same flag."""
    monkeypatch.setattr(mooncake_store, "format_pool_report", lambda *_args: "no segments")

    command = CommandUnderTest(Resource(None), ["mooncake_pool_report", "--run_dir", str(tmp_path)])
    command.run("--no-telemetry")
