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
"""Unit tests for the pool lifecycle `trtllm-serve` owns.

Two seams: the signal that ends `mooncake_master`, which has to release the
master on the way out rather than leave it unreaped, and the provisioning a
server is wrapped in, which has to describe the pool its ranks will actually
join.

Runs without a Mooncake installation and without a GPU. The master process the
command holds is a context manager that records its own release, and the signal
under test arrives from the poll before the command's idle wait, the point at
which everything it holds is held.
"""

import contextlib
import json
import os
import signal
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import CONFIG_PATH_ENV
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import provision_pool
from tensorrt_llm.commands import mooncake as mooncake_commands
from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig

pytestmark = pytest.mark.cpu_only

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool config: these tests set what they mean to test."""
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """The command installs handlers process-wide, so put pytest's back."""
    installed = {number: signal.getsignal(number) for number in HANDLED_SIGNALS}
    yield
    for number, handler in installed.items():
        signal.signal(number, handler)


class IdleMaster:
    """A master that stays alive, with a hook for the first time it is polled.

    The command idles between polls, so a test interrupts it from the poll that
    precedes one.
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
        command = {
            "mooncake_master": mooncake_commands.mooncake_master,
            "mooncake_pool_report": mooncake_commands.mooncake_pool_report,
        }[self.args[0]]
        command.main(
            args=[*self.args[1:], *extra_args],
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
    command's own wait is what gives the interpreter one. `on_idle` runs just
    before, which is the only point at which the command's resources are all
    held, so anything released on the way out has to be observed from there.
    """

    def interrupt():
        if on_idle is not None:
            on_idle()
        os.kill(os.getpid(), number)

    command.resource.held.on_poll = interrupt


@pytest.mark.parametrize("number", HANDLED_SIGNALS)
def test_a_signal_ends_the_command_and_releases_what_it_held(master, number):
    """The loop returns and the resource is given up.

    A handler that took a lock would instead deadlock against the wait it
    interrupted, leaving the master unreaped and its manifest on disk.
    """
    signal_on_idle(master, number)

    master.run()

    assert master.resource.released


def test_the_master_is_reaped_before_the_command_reports_the_signal(master):
    """The release is in a `finally`, so it cannot come from the handler.

    Observed from the point the signal arrives, where the command still holds
    everything it took.
    """
    released_when_signalled = []
    signal_on_idle(
        master,
        signal.SIGTERM,
        on_idle=lambda: released_when_signalled.append(master.resource.released),
    )

    master.run()

    assert released_when_signalled == [False]
    assert master.resource.released


def test_an_inherited_config_restates_the_settings_it_supersedes(monkeypatch, tmp_path):
    """What `mooncake_store` asked for is not what an inherited pool gives.

    `MOONCAKE_CONFIG_PATH` wins, so the ranks join as that file says. The block
    is read afterwards by the usage report the LLM constructor sends, which
    would otherwise record a server lending 16 GiB as `both`.
    """
    inherited = tmp_path / "external.json"
    inherited.write_text(
        json.dumps(
            {
                "master_server_address": "10.0.0.7:50051",
                "role": "capacity",
                "global_segment_size": "32GiB",
                "namespace": "external",
                "model_key": "their-checkpoint",
                "transfer_batch_size": 128,
                "stage_through_host": True,
            }
        )
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))

    pool = MooncakeStoreConfig(
        pool="file:///shared/pool.json",
        model_key="our-checkpoint",
        role="both",
        segment_size="16GiB",
    )
    with provision_pool(pool) as rendered:
        # Nothing was rendered: the inherited config is left in charge.
        assert rendered is None
        assert pool.role == "capacity"
        assert pool.segment_size == 32 * (1 << 30)
        assert pool.namespace == "external"
        assert pool.model_key == "their-checkpoint"
        assert pool.transfer_batch_size == 128
        assert pool.stage_through_host is True

    # How to reach the pool and where this run's files go are the server's
    # own, so an inherited config has nothing to say about them.
    assert pool.pool == "file:///shared/pool.json"


def test_an_unreadable_inherited_config_leaves_the_block_alone(monkeypatch, tmp_path):
    """The workers report a bad config; this path only describes one."""
    inherited = tmp_path / "truncated.json"
    inherited.write_text("{not json")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))

    pool = MooncakeStoreConfig(
        pool="file:///shared/pool.json", model_key="our-checkpoint", segment_size="16GiB"
    )
    with provision_pool(pool) as rendered:
        assert rendered is None
    assert pool.role == "both"
    assert pool.segment_size == "16GiB"
