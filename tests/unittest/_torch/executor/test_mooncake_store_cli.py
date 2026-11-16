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

Three seams: the signal that ends `mooncake_master` or `mooncake_donor`, which
has to release what the command holds before the telemetry boundary reports
it; the provisioning a server is wrapped in, which has to leave an externally
managed pool in charge and hand its ranks the RDMA devices the deployment
pinned; and the donor's reading of the pool it lends to.

Runs without a Mooncake installation and without a GPU. The master process the
command holds is a context manager that records its own release, and the signal
under test arrives from the poll before the command's idle wait, the point at
which everything it holds is held. The Mooncake binding, where a test opens a
store, records the setup it was given.
"""

import contextlib
import json
import os
import signal
import sys
import time
import types
from types import SimpleNamespace

import click
import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import donor as donor_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import ledger as ledger_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import master as master_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import validation as validation_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.ledger import (
    DONOR_ROLE,
    SegmentRecord,
    read_segments,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import provision_pool
from tensorrt_llm.commands import _telemetry
from tensorrt_llm.commands import mooncake as mooncake_commands
from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig
from tensorrt_llm.usage.config import UsageContext

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
        group = _telemetry.TelemetryGroup(
            name="trtllm-serve",
            telemetry_usage_context=UsageContext.CLI_SERVE,
            telemetry_component="server",
            commands={
                "mooncake_master": mooncake_commands.mooncake_master,
                "mooncake_donor": mooncake_commands.mooncake_donor,
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
def test_a_signal_is_reported_once_and_still_releases_what_was_held(
    terminal_outcomes, master, number
):
    """The signal reaches the boundary, and the resource is still given up.

    A command that returned normally instead would be reported as a clean exit
    before any model was loaded, and one whose handler took a lock would
    deadlock against the wait it interrupted and report nothing at all.
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


def test_the_opt_out_the_group_documents_is_accepted(master):
    """Without the option Click rejects the flag as a usage error instead.

    One form is enough: both come from the same paired declaration.
    """
    signal_on_idle(master, signal.SIGTERM)

    with pytest.raises(_telemetry.SignalExit):
        master.run("--no-telemetry")

    assert master.resource.released


def test_the_pool_report_accepts_the_same_opt_out(monkeypatch, tmp_path):
    """It sits in the same group, so it has to offer the same flag."""
    monkeypatch.setattr(mooncake_store, "format_pool_report", lambda *_args: "no segments")

    command = CommandUnderTest(Resource(None), ["mooncake_pool_report", "--run_dir", str(tmp_path)])
    command.run("--no-telemetry")


def test_an_inherited_config_is_left_in_charge(monkeypatch, tmp_path):
    """`MOONCAKE_CONFIG_PATH` names a pool the deployment provisioned itself.

    Nothing is rendered over it, and the settings its ranks then join as are
    restated by `settings.apply_effective_settings`.
    """
    inherited = tmp_path / "external.json"
    inherited.write_text(json.dumps({"master_server_address": "10.0.0.7:50051"}))
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))

    pool = MooncakeStoreConfig(pool="file:///shared/pool.json", model_key="our-checkpoint")
    with provision_pool(pool) as rendered:
        assert rendered is None
    assert os.environ[CONFIG_PATH_ENV] == str(inherited)


@pytest.fixture
def mooncake_binding(monkeypatch):
    """The Mooncake Python binding, replaced by one that records each store opened."""
    opened = []

    class DistributedStore:
        def __init__(self):
            self.setup_args = None
            self.closed = False
            opened.append(self)

        def setup(self, *args):
            self.setup_args = args
            return 0

        def close(self):
            self.closed = True

    package = types.ModuleType("mooncake")
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = DistributedStore
    package.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    return opened


@pytest.fixture
def answering_master(monkeypatch):
    """A master already up, on a host where InfiniBand detection finds one device."""
    monkeypatch.setattr(master_module, "wait_for_master", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(master_module, "_highest_rate_ib_devices", lambda *_args: ["mlx5_4"])


def join_pool(tmp_path, **settings) -> MooncakeStoreConnectorConfig:
    """Provision a server against a bare master address; return what its ranks read."""
    pool = MooncakeStoreConfig(
        pool="10.0.0.7:50051",
        model_key="our-checkpoint",
        run_dir=str(tmp_path / "server"),
        **settings,
    )
    with provision_pool(pool) as rendered:
        return MooncakeStoreConnectorConfig.from_file(rendered)


def test_a_pinned_device_list_wins_over_detection_and_reaches_the_store(
    tmp_path, monkeypatch, answering_master, mooncake_binding
):
    """Detection would pick a device here; the deployment's pin is what Mooncake gets.

    On a RoCE fabric detection finds nothing and Mooncake's own discovery takes
    NICs on planes the peers cannot reach, so the pin is the only way to steer it.
    """
    config = join_pool(tmp_path, device_name="mlx5_0,mlx5_2", local_hostname="10.0.0.9")
    assert config.device_name == "mlx5_0,mlx5_2"

    monkeypatch.setattr(worker_module, "select_registration_path", lambda *_a, **_k: False)
    worker_module._open_store(config)
    (store,) = mooncake_binding
    host, _metadata, _segment, _buffer, protocol, devices, master = store.setup_args
    assert (host, protocol, devices, master) == (
        "10.0.0.9",
        "rdma",
        "mlx5_0,mlx5_2",
        "10.0.0.7:50051",
    )


def test_an_unpinned_device_list_is_still_detected(tmp_path, answering_master):
    assert join_pool(tmp_path).device_name == "mlx5_4"


@pytest.fixture
def ample_host_memory(monkeypatch):
    """A node with far more available memory than any donation here asks for."""
    monkeypatch.setattr(validation_module, "_available_host_memory", lambda: 1 << 50)
    monkeypatch.setattr(validation_module, "_cgroup_room", lambda: None)


@pytest.fixture
def pool_manifest(tmp_path, monkeypatch, answering_master):
    """A manifest naming a master that answers at once, as `mooncake_master` writes it."""
    monkeypatch.setattr(donor_module, "wait_for_master", lambda *_args, **_kwargs: 0.0)
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "master_server_address": "10.0.0.7:50051",
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "rdma",
                "namespace": "trtllm",
            }
        )
    )
    return path


def run_donor(monkeypatch, *args, on_idle=None):
    """Run `mooncake_donor` until its first idle wait, then stop it with SIGTERM.

    `on_idle` runs just before the signal, when everything the command holds
    is held.
    """

    def idle(_seconds):
        if on_idle is not None:
            on_idle()
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(time, "sleep", idle)
    with pytest.raises(_telemetry.SignalExit):
        CommandUnderTest(Resource(None), ["mooncake_donor", *args]).run()


def test_the_donor_lends_a_segment_to_the_pool_its_manifest_names(
    tmp_path, monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    ready = tmp_path / "donor.ready"
    run_dir = tmp_path / "run" / "donor-gen0"
    held = []
    run_donor(
        monkeypatch,
        "--pool_file",
        str(pool_manifest),
        "--segment_size",
        "2GiB",
        "--device_name",
        "mlx5_0,mlx5_2",
        "--local_hostname",
        "10.0.0.9",
        "--ready_file",
        str(ready),
        "--run_dir",
        str(run_dir),
        on_idle=lambda: held.append((ready.read_text(), mooncake_binding[0].closed)),
    )

    (store,) = mooncake_binding
    assert store.setup_args == (
        "10.0.0.9",
        "P2PHANDSHAKE",
        2 * 1024**3,
        donor_module.DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
        "rdma",
        "mlx5_0,mlx5_2",
        "10.0.0.7:50051",
    )
    # Announced while mounted, and neither left behind once stopped.
    assert held == [(f"10.0.0.9 {2 * 1024**3}\n", False)]
    assert store.closed
    assert not ready.exists()
    # Counted in the pool report as memory lent without traffic.
    (record,) = read_segments(str(tmp_path / "run"))
    assert (record.host, record.segment_size, record.role) == ("10.0.0.9", 2 * 1024**3, "donor")


def test_the_pool_report_counts_donors_apart_from_the_ranks():
    """A donor's larger segment neither counts as a rank nor breaks the ranks' uniformity."""
    ranks = [SegmentRecord("10.0.0.1", rank, 240 << 30, "both", "m") for rank in range(2)]
    donors = [SegmentRecord(f"10.0.0.{9 + i}", 0, 768 << 30, DONOR_ROLE, "") for i in range(2)]
    lines = ledger_module._capacity_lines(ranks + donors)
    assert "ranks         : 2 across 1 host(s)" in lines
    assert "donors        : 2 across 2 host(s)" in lines
    assert f"capacity      : {2 * 240 + 2 * 768:.1f} GiB" in lines
    assert "per rank      : 240.0 GiB, uniform" in lines
    # Donors alone leave no rank size to compare.
    assert not any(line.startswith("per rank") for line in ledger_module._capacity_lines(donors))


def test_the_donor_joins_a_bare_master_on_the_pool_defaults(
    monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    """Without a manifest the pool-wide settings are the defaults a server would take."""
    run_donor(
        monkeypatch,
        "--master_server_address",
        "10.0.0.7:50051",
        "--segment_size",
        str(1024**3),
        "--local_hostname",
        "10.0.0.9",
    )

    (store,) = mooncake_binding
    _host, metadata, segment, _buffer, protocol, devices, master = store.setup_args
    # The devices are detected, as a server's provisioning detects them.
    assert (metadata, segment, protocol, devices, master) == (
        "P2PHANDSHAKE",
        1024**3,
        "rdma",
        "mlx5_4",
        "10.0.0.7:50051",
    )
    assert store.closed


def test_the_donor_reads_a_pool_file_given_as_a_url(
    monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    run_donor(
        monkeypatch,
        "--pool_file",
        f"file://{pool_manifest}",
        "--segment_size",
        "1GiB",
        "--local_hostname",
        "10.0.0.9",
    )
    (store,) = mooncake_binding
    assert store.setup_args[-1] == "10.0.0.7:50051"


@pytest.mark.parametrize(
    "pool_args",
    [[], ["--pool_file", "pool.json", "--master_server_address", "10.0.0.7:50051"]],
)
def test_the_donor_needs_exactly_one_way_to_find_the_pool(mooncake_binding, pool_args):
    command = CommandUnderTest(
        Resource(None), ["mooncake_donor", "--segment_size", "1GiB", *pool_args]
    )
    with pytest.raises(click.UsageError, match="exactly one"):
        command.run()
    assert mooncake_binding == []


@pytest.mark.parametrize("size, complaint", [("2GB", "ambiguous"), ("0", "lends nothing")])
def test_the_donor_refuses_a_size_it_cannot_lend(mooncake_binding, size, complaint):
    command = CommandUnderTest(
        Resource(None),
        ["mooncake_donor", "--master_server_address", "10.0.0.7:50051", "--segment_size", size],
    )
    with pytest.raises(click.BadParameter, match=complaint):
        command.run()
    assert mooncake_binding == []


def test_the_donor_refuses_a_donation_the_node_cannot_back(
    tmp_path, monkeypatch, pool_manifest, mooncake_binding
):
    monkeypatch.setattr(validation_module, "_available_host_memory", lambda: 40 * 1024**3)
    monkeypatch.setattr(validation_module, "_cgroup_room", lambda: None)
    ready = tmp_path / "donor.ready"
    command = CommandUnderTest(
        Resource(None),
        [
            "mooncake_donor",
            "--pool_file",
            str(pool_manifest),
            "--segment_size",
            "64GiB",
            "--ready_file",
            str(ready),
        ],
    )
    with pytest.raises(ValueError, match="cannot back a 64.0 GiB donation"):
        command.run()
    # Refused before Mooncake allocated anything.
    assert mooncake_binding == []
    assert not ready.exists()


def test_a_failed_setup_lends_nothing(
    tmp_path, monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    store_class = sys.modules["mooncake.store"].MooncakeDistributedStore
    monkeypatch.setattr(store_class, "setup", lambda self, *args: -1)
    ready = tmp_path / "donor.ready"
    run_dir = tmp_path / "run" / "donor-gen0"
    command = CommandUnderTest(
        Resource(None),
        [
            "mooncake_donor",
            "--pool_file",
            str(pool_manifest),
            "--segment_size",
            "2GiB",
            "--ready_file",
            str(ready),
            "--run_dir",
            str(run_dir),
        ],
    )
    with pytest.raises(RuntimeError, match="setup failed with status -1"):
        command.run()
    (store,) = mooncake_binding
    assert store.closed
    assert not ready.exists()
    assert read_segments(str(tmp_path / "run")) == []


def test_a_container_limit_caps_the_donation_budget(monkeypatch):
    # Plenty free on the host, little left under the container's limit.
    monkeypatch.setattr(validation_module, "_available_host_memory", lambda: 900 * 1024**3)
    monkeypatch.setattr(validation_module, "_cgroup_room", lambda: 50 * 1024**3)
    with pytest.raises(ValueError, match="only 50.0 GiB is available"):
        validation_module.validate_donation_budget(64 * 1024**3, 64 * 1024**2)
    validation_module.validate_donation_budget(16 * 1024**3, 64 * 1024**2)


class RecordingLogger:
    def __init__(self):
        self.lines = []

    def warning(self, message):
        self.lines.append(("warning", message))

    def error(self, message):
        self.lines.append(("error", message))


def test_a_failed_close_is_not_reported_as_a_withdrawal(
    monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    store_class = sys.modules["mooncake.store"].MooncakeDistributedStore

    def broken_close(self):
        raise OSError("close failed")

    monkeypatch.setattr(store_class, "close", broken_close)
    log = RecordingLogger()
    monkeypatch.setattr(donor_module, "logger", log)
    with pytest.raises(OSError, match="close failed"):
        with donor_module.donate_segment(
            f"file://{pool_manifest}", 1024**3, local_hostname="10.0.0.9"
        ):
            pass
    assert not any("withdrew" in line for _, line in log.lines)
    assert any(level == "error" and "may still count" in line for level, line in log.lines)


def test_a_failed_setup_is_reported_even_if_close_fails(
    monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    store_class = sys.modules["mooncake.store"].MooncakeDistributedStore
    monkeypatch.setattr(store_class, "setup", lambda self, *args: -1)

    def broken_close(self):
        raise OSError("close failed")

    monkeypatch.setattr(store_class, "close", broken_close)
    log = RecordingLogger()
    monkeypatch.setattr(donor_module, "logger", log)
    # The setup failure is what surfaces; the failed close is logged beside it.
    with pytest.raises(RuntimeError, match="setup failed with status -1"):
        with donor_module.donate_segment(
            f"file://{pool_manifest}", 1024**3, local_hostname="10.0.0.9"
        ):
            pass
    assert any(level == "error" and "after a failed setup" in line for level, line in log.lines)


def test_the_tightest_cgroup_room_wins(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/job/step\n")
    root = tmp_path / "fs"
    gib = 1024**3
    # The step has the larger limit but the job, nearly full, has less room left.
    for rel, limit, current, file_pages, shmem in (
        ("job", 100 * gib, 90 * gib, 10 * gib, 0),
        ("job/step", 500 * gib, 20 * gib, 0, 0),
    ):
        directory = root / rel
        directory.mkdir(parents=True)
        (directory / "memory.max").write_text(f"{limit}\n")
        (directory / "memory.current").write_text(f"{current}\n")
        (directory / "memory.stat").write_text(f"file {file_pages}\nshmem {shmem}\n")
    # job: 100 - (90 - 10 + 0) = 20 GiB; step: 500 - 20 = 480 GiB.
    assert validation_module._cgroup_room(str(proc), str(root)) == 20 * gib
    (root / "job" / "memory.max").write_text("max\n")
    assert validation_module._cgroup_room(str(proc), str(root)) == 480 * gib
    assert validation_module._cgroup_room(str(tmp_path / "missing"), str(root)) is None


def test_a_raising_setup_is_reported_after_close(
    monkeypatch, pool_manifest, mooncake_binding, ample_host_memory
):
    store_class = sys.modules["mooncake.store"].MooncakeDistributedStore

    def broken_setup(self, *args):
        raise OSError("ibverbs refused")

    monkeypatch.setattr(store_class, "setup", broken_setup)
    with pytest.raises(OSError, match="ibverbs refused"):
        with donor_module.donate_segment(
            f"file://{pool_manifest}", 1024**3, local_hostname="10.0.0.9"
        ):
            pass
    (store,) = mooncake_binding
    assert store.closed


@pytest.mark.parametrize(
    "file_pages, shmem, unevictable, room",
    [
        (10, 8, 0, 12),  # shared memory sits in "file" but reclaim cannot drop it
        (60, 0, 60, 10),  # neither can it drop locked pages
        (5, 8, 0, 10),  # the reclaimable allowance never goes negative
    ],
)
def test_cgroup_room_counts_only_droppable_page_cache(
    tmp_path, file_pages, shmem, unevictable, room
):
    gib = 1024**3
    proc = tmp_path / "cgroup"
    proc.write_text("0::/job\n")
    job = tmp_path / "fs" / "job"
    job.mkdir(parents=True)
    (job / "memory.max").write_text(f"{100 * gib}\n")
    (job / "memory.current").write_text(f"{90 * gib}\n")
    (job / "memory.stat").write_text(
        f"file {file_pages * gib}\nshmem {shmem * gib}\nunevictable {unevictable * gib}\n"
    )
    assert validation_module._cgroup_room(str(proc), str(tmp_path / "fs")) == room * gib


def test_the_donation_budget_keeps_the_reserve_and_counts_the_buffer():
    gib = 1024**3
    buffer = 64 * 1024**2
    available = 100 * gib
    exact = available - validation_module.NODE_BUDGET_RESERVE_BYTES - buffer
    validation_module.validate_donation_budget(exact, buffer, available_bytes=available)
    with pytest.raises(ValueError, match="cannot back"):
        validation_module.validate_donation_budget(exact + 1, buffer, available_bytes=available)


def test_a_finite_cgroup_limit_counts_even_without_statistics(tmp_path):
    gib = 1024**3
    proc = tmp_path / "cgroup"
    proc.write_text("0::/job\n")
    job = tmp_path / "fs" / "job"
    job.mkdir(parents=True)
    (job / "memory.max").write_text(f"{50 * gib}\n")
    (job / "memory.current").write_text(f"{20 * gib}\n")
    # No memory.stat: no page-cache allowance, but the limit still binds.
    assert validation_module._cgroup_room(str(proc), str(tmp_path / "fs")) == 30 * gib
    # No memory.current either: no room is assumed.
    (job / "memory.current").unlink()
    assert validation_module._cgroup_room(str(proc), str(tmp_path / "fs")) == 0
