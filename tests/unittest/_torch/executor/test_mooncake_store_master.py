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
"""Unit tests for provisioning a Mooncake store pool during server bringup.

Runs without a Mooncake installation and without a GPU. A master this process
launches is a fake standing in for `Popen` that opens the RPC port, which is
all the readiness handshake ever observes. A master someone else runs is a
plain socket.
"""

import json
import os
import shutil
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import master as master_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    maybe_provision_pool,
    provision_pool,
    resolve_master_address,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, MooncakeStoreConfig


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


class FakeMasterProcess:
    """The slice of `Popen` that launching a master actually drives.

    `listen_on` makes it answer on that port, which is what a real master does
    last and what the readiness wait keys off. `exit_code` makes it a master
    that failed to start.
    """

    def __init__(self, command, env, listen_on=None, exit_code=None):
        self.command = command
        self.env = env
        self.pid = 4242
        self.terminated = False
        self.killed = False
        self._exit_code = exit_code
        self._listener = None
        if listen_on is not None:
            self._listener = socket.socket()
            self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listener.bind(("", listen_on))
            self._listener.listen(8)

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):
        self.killed = True


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool: these tests are about what provisioning does itself."""
    for name in (
        CONFIG_PATH_ENV,
        master_module.MASTER_BINARY_ENV,
        master_module.MASTER_TIMEOUT_ENV,
        master_module.RUN_DIR_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # Nothing here is slow to start, so a wait that runs long is a failure
    # rather than something that needs more time.
    monkeypatch.setenv(master_module.MASTER_TIMEOUT_ENV, "10")


@pytest.fixture
def fake_master(monkeypatch):
    """Replace the master binary and its process with in-process fakes.

    Returns a callable that arms the fake. Once provisioning has run, the
    launched instance is available as `.process` for inspection.
    """

    class Launcher:
        def __init__(self):
            self.process = None

        def arm(self, listen_on=None, exit_code=None, log_text=None):
            def popen(command, env=None, stdout=None, **_kwargs):
                # A real master writes its log through this handle.
                if log_text is not None and stdout is not None:
                    stdout.write(log_text.encode())
                    stdout.flush()
                self.process = FakeMasterProcess(
                    command, env, listen_on=listen_on, exit_code=exit_code
                )
                return self.process

            # Swap the modules as this module sees them rather than patching
            # attributes on the shared stdlib ones.
            monkeypatch.setattr(
                master_module,
                "shutil",
                SimpleNamespace(which=lambda name: f"/opt/bin/{name}", rmtree=shutil.rmtree),
            )
            monkeypatch.setattr(
                master_module,
                "subprocess",
                SimpleNamespace(
                    Popen=popen,
                    STDOUT=subprocess.STDOUT,
                    TimeoutExpired=subprocess.TimeoutExpired,
                ),
            )

    return Launcher()


@pytest.fixture
def running_master():
    """A socket standing in for a master someone else is running."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        yield f"127.0.0.1:{listener.getsockname()[1]}"
    finally:
        listener.close()


# ---- configuration ----


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(launch_master=True, master_server_address="host:50051"), "not both"),
        (dict(), "needs a master"),
        # master_address_file only writes an address; reading one is
        # master_server_address, so publishing without launching is incoherent.
        (
            dict(master_server_address="host:50051", master_address_file="/shared/master.addr"),
            "needs launch_master",
        ),
    ],
    ids=["two_masters", "no_master", "publishing_without_launching"],
)
def test_pool_needs_exactly_one_master(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MooncakeStoreConfig(**kwargs)


def test_pool_is_rejected_unless_the_connector_is_mooncake_store():
    """The validator keys off the connector, however that was spelled."""
    with pytest.raises(ValueError, match="mooncake_store describes a Mooncake pool"):
        KvCacheConnectorConfig(
            connector="lmcache",
            mooncake_store=MooncakeStoreConfig(launch_master=True),
        )
    # Naming the module rather than the preset selects the same connector.
    KvCacheConnectorConfig(
        connector_module="tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        connector_scheduler_class="MooncakeStoreConnectorScheduler",
        connector_worker_class="MooncakeStoreConnectorWorker",
        mooncake_store=MooncakeStoreConfig(launch_master=True),
    )


# ---- the rendered client config ----


def test_client_config_is_what_the_connector_reads_back(tmp_path):
    """The generated JSON has to survive the connector's own parser."""
    pool = MooncakeStoreConfig(
        master_server_address="10.0.0.1:50051",
        protocol="rdma",
        device_name="mlx5_0",
        global_segment_size="64GiB",
        local_buffer_size="4GiB",
        cache_prefix="trtllm-m3",
        stage_through_host=True,
        transfer_batch_size=32,
    )
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps(master_module._client_config(pool, "10.0.0.1:50051")))

    parsed = MooncakeStoreConnectorConfig.from_file(str(path))
    assert parsed.master_server_address == "10.0.0.1:50051"
    assert parsed.metadata_server == "P2PHANDSHAKE"
    assert parsed.protocol == "rdma"
    assert parsed.device_name == "mlx5_0"
    assert parsed.global_segment_size == 64 * 1024**3
    assert parsed.local_buffer_size == 4 * 1024**3
    assert parsed.cache_prefix == "trtllm-m3"
    assert parsed.stage_through_host is True
    assert parsed.transfer_batch_size == 32


def test_client_config_omits_the_fields_the_pool_left_unset():
    """An absent key leaves the connector its own default; a null would not."""
    pool = MooncakeStoreConfig(master_server_address="host:50051")
    written = master_module._client_config(pool, "host:50051")
    assert "cache_prefix" not in written
    assert "staging_buffer_bytes" not in written


@pytest.mark.parametrize(
    "address, expected",
    [
        ("host:50051", ("host", 50051)),
        ("[::1]:50051", ("::1", 50051)),
        ("unix:///var/run/mooncake", None),
        ("host", None),
    ],
)
def test_master_addresses_are_split_or_declined(address, expected):
    assert master_module._split_address(address) == expected


# ---- provisioning against a master someone else runs ----


def test_provisioning_points_the_workers_at_a_running_master(running_master):
    pool = MooncakeStoreConfig(master_server_address=running_master)

    with provision_pool(pool) as config_path:
        # The workers are spawned inside this window and are told about the
        # pool through the environment, so both have to hold while it is open.
        assert os.environ[CONFIG_PATH_ENV] == config_path
        written = json.loads(open(config_path).read())
        assert written["master_server_address"] == running_master

    assert CONFIG_PATH_ENV not in os.environ
    assert not os.path.exists(config_path)


def test_a_staging_buffer_can_be_sized_where_staging_is_turned_on(running_master):
    """Undersizing it silently shrinks the transfer batch, so it must be settable."""
    pool = MooncakeStoreConfig(
        master_server_address=running_master,
        stage_through_host=True,
        staging_buffer_bytes="4GiB",
    )

    with provision_pool(pool) as config_path:
        written = json.loads(open(config_path).read())
        assert written["stage_through_host"] is True
        assert written["staging_buffer_bytes"] == "4GiB"


def test_provisioning_fails_before_the_model_loads_if_the_master_is_absent(monkeypatch):
    monkeypatch.setenv(master_module.MASTER_TIMEOUT_ENV, "1")
    pool = MooncakeStoreConfig(master_server_address=f"127.0.0.1:{free_port()}")

    with pytest.raises(TimeoutError, match="did not accept connections"):
        with provision_pool(pool):
            pytest.fail("provisioning should not have yielded")
    assert CONFIG_PATH_ENV not in os.environ


def test_an_unparseable_master_address_is_left_to_the_workers():
    """Not every address is host:port, so an unprobeable one passes through."""
    pool = MooncakeStoreConfig(master_server_address="unix:///var/run/mooncake")

    with provision_pool(pool) as config_path:
        written = json.loads(open(config_path).read())
        assert written["master_server_address"] == "unix:///var/run/mooncake"


def test_an_inherited_config_path_wins(monkeypatch, tmp_path):
    """An externally managed pool names itself this way, so provisioning defers."""
    harness_config = tmp_path / "harness.json"
    harness_config.write_text("{}")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(harness_config))
    pool = MooncakeStoreConfig(launch_master=True)

    with provision_pool(pool) as config_path:
        assert config_path is None
        assert os.environ[CONFIG_PATH_ENV] == str(harness_config)

    assert os.environ[CONFIG_PATH_ENV] == str(harness_config)


# ---- provisioning with a master of our own ----


def test_a_launched_master_is_named_in_the_config_and_stopped_on_exit(fake_master):
    port = free_port()
    fake_master.arm(listen_on=port)
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with provision_pool(pool) as config_path:
        written = json.loads(open(config_path).read())
        host, _, named_port = written["master_server_address"].rpartition(":")
        assert int(named_port) == port
        # The address in the config is all a worker on another host gets, so
        # it has to be dialable.
        with socket.create_connection((host, port), timeout=5):
            pass

    assert fake_master.process.terminated
    assert not fake_master.process.killed


def test_a_launched_master_gets_the_flags_and_logging_it_needs(fake_master):
    port = free_port()
    fake_master.arm(listen_on=port)
    pool = MooncakeStoreConfig(
        launch_master=True,
        master_port=port,
        master_metrics_port=free_port(),
        master_eviction_ratio=0.1,
    )

    with provision_pool(pool):
        command = fake_master.process.command
        assert command[0].endswith("mooncake_master")
        assert f"--rpc_port={port}" in command
        assert f"--metrics_port={pool.master_metrics_port}" in command
        assert "--eviction_ratio=0.1" in command
        # Without these the master logs to a file under /tmp and the log the
        # run directory holds stays empty.
        assert fake_master.process.env["GLOG_logtostderr"] == "1"
        assert fake_master.process.env["GLOG_v"] == "1"


def test_a_master_that_dies_during_startup_says_so(fake_master):
    fake_master.arm(exit_code=3)
    pool = MooncakeStoreConfig(launch_master=True, master_port=free_port())

    with pytest.raises(RuntimeError, match="exited with code 3"):
        with provision_pool(pool):
            pytest.fail("provisioning should not have yielded")
    assert CONFIG_PATH_ENV not in os.environ


def test_a_master_that_never_listens_times_out(monkeypatch, fake_master):
    monkeypatch.setenv(master_module.MASTER_TIMEOUT_ENV, "1")
    fake_master.arm()
    pool = MooncakeStoreConfig(launch_master=True, master_port=free_port())

    with pytest.raises(TimeoutError, match="did not accept connections"):
        with provision_pool(pool):
            pytest.fail("provisioning should not have yielded")
    assert fake_master.process.terminated


def test_a_missing_master_binary_names_the_alternatives(monkeypatch):
    monkeypatch.setattr(
        master_module,
        "shutil",
        SimpleNamespace(which=lambda _name: None, rmtree=shutil.rmtree),
    )
    pool = MooncakeStoreConfig(launch_master=True)

    with pytest.raises(FileNotFoundError, match="master_server_address"):
        with provision_pool(pool):
            pytest.fail("provisioning should not have yielded")


def test_a_run_dir_keeps_the_master_log_and_the_config(fake_master, tmp_path):
    port = free_port()
    fake_master.arm(listen_on=port)
    run_dir = tmp_path / "pool"
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with provision_pool(pool, run_dir=str(run_dir)) as config_path:
        assert config_path == str(run_dir / master_module.CLIENT_CONFIG_NAME)

    # An explicit run directory outlives the run that filled it, since the
    # master's log is where pool occupancy and eviction are read from.
    assert (run_dir / master_module.MASTER_LOG_NAME).exists()
    assert (run_dir / master_module.CLIENT_CONFIG_NAME).exists()


# ---- the entry point servers call ----


@pytest.mark.parametrize(
    "config",
    [
        KvCacheConnectorConfig(connector="lmcache"),
        None,
        KvCacheConnectorConfig(connector="mooncake-store"),
    ],
    ids=["another_connector", "no_connector", "pool_left_undescribed"],
)
def test_provisioning_is_a_no_op_unless_a_pool_is_described(config):
    """Without `mooncake_store`, MOONCAKE_CONFIG_PATH is still the only input."""
    with maybe_provision_pool(config):
        assert CONFIG_PATH_ENV not in os.environ


def test_a_described_pool_is_provisioned(running_master):
    config = KvCacheConnectorConfig(
        connector="mooncake-store",
        mooncake_store=MooncakeStoreConfig(master_server_address=running_master),
    )
    with maybe_provision_pool(config):
        written = json.loads(open(os.environ[CONFIG_PATH_ENV]).read())
        assert written["master_server_address"] == running_master
    assert CONFIG_PATH_ENV not in os.environ


# ---- reaching a master whose host nobody knew in advance ----


@pytest.mark.parametrize("address", ["10.0.0.1:50051", "unix:///var/run/mooncake"])
def test_an_address_that_is_not_a_file_passes_through(address):
    assert resolve_master_address(address, timeout=1.0) == address


def test_a_published_address_is_read_from_the_file_that_names_it(tmp_path):
    published = tmp_path / "master.addr"
    published.write_text("10.0.0.7:50051\n")

    assert resolve_master_address(f"file://{published}", timeout=1.0) == "10.0.0.7:50051"


def test_an_address_not_published_yet_is_waited_for(tmp_path):
    """Master and workers are started together; neither one orders the other."""
    published = tmp_path / "master.addr"
    threading.Timer(0.5, published.write_text, ["10.0.0.9:50051\n"]).start()

    assert resolve_master_address(f"file://{published}", timeout=10.0) == "10.0.0.9:50051"


def test_an_empty_address_file_is_not_taken_for_an_address(tmp_path):
    """An existing file is not the same as a published address."""
    published = tmp_path / "master.addr"
    published.write_text("")

    with pytest.raises(TimeoutError, match="No Mooncake master address"):
        resolve_master_address(f"file://{published}", timeout=1.0)


def test_an_unpublished_address_names_the_command_that_publishes_it(tmp_path):
    with pytest.raises(TimeoutError, match="--address-file"):
        resolve_master_address(f"file://{tmp_path / 'absent'}", timeout=1.0)


# ---- a master with a lifetime of its own ----


def test_a_standalone_master_publishes_an_address_that_can_be_dialed(fake_master, tmp_path):
    port = free_port()
    fake_master.arm(listen_on=port)
    address_file = tmp_path / "master.addr"
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with master_module.running_master(
        pool, str(tmp_path / "run"), address_file=str(address_file)
    ) as master:
        assert resolve_master_address(f"file://{address_file}", timeout=5.0) == master.address
        host, _, named_port = master.address.rpartition(":")
        assert int(named_port) == port
        with socket.create_connection((host, port), timeout=5):
            pass


def test_a_stopped_master_leaves_no_address_behind(fake_master, tmp_path):
    """A stale address would send the next run's workers to a dead port."""
    port = free_port()
    fake_master.arm(listen_on=port)
    address_file = tmp_path / "master.addr"
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with master_module.running_master(pool, str(tmp_path / "run"), address_file=str(address_file)):
        assert address_file.exists()

    assert not address_file.exists()
    assert fake_master.process.terminated


def test_a_standalone_master_keeps_its_log(fake_master, tmp_path):
    """A standalone master outlives the servers that used it, so its log is kept."""
    port = free_port()
    fake_master.arm(listen_on=port)
    run_dir = tmp_path / "run"
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with master_module.running_master(pool, str(run_dir)):
        pass

    assert (run_dir / master_module.MASTER_LOG_NAME).exists()


def test_provisioning_joins_a_master_it_was_never_given_the_address_of(fake_master, tmp_path):
    """The address file is how workers reach a master no config names a host for."""
    port = free_port()
    fake_master.arm(listen_on=port)
    address_file = tmp_path / "master.addr"
    standalone = MooncakeStoreConfig(launch_master=True, master_port=port)
    worker = MooncakeStoreConfig(master_server_address=f"file://{address_file}")

    with master_module.running_master(
        standalone, str(tmp_path / "run"), address_file=str(address_file)
    ) as master:
        with provision_pool(worker) as config_path:
            # Mooncake cannot dial a file:// URL, so what reaches the workers
            # has to be the address it resolved to.
            written = json.loads(open(config_path).read())
            assert written["master_server_address"] == master.address


# ---- a master a server launched, made findable ----


def test_a_launched_master_publishes_where_its_run_left_its_logs(fake_master, tmp_path):
    """A finished run's logs still say which pool it used."""
    port = free_port()
    fake_master.arm(listen_on=port)
    run_dir = tmp_path / "run"
    pool = MooncakeStoreConfig(launch_master=True, master_port=port)

    with provision_pool(pool, run_dir=str(run_dir)):
        address = (run_dir / master_module.MASTER_ADDRESS_NAME).read_text().strip()
        assert address.endswith(f":{port}")

    assert not (run_dir / master_module.MASTER_ADDRESS_NAME).exists()


def test_a_launched_master_can_be_published_where_the_donors_look(fake_master, tmp_path):
    """This is what lets a server that launched its own master have donors."""
    port = free_port()
    fake_master.arm(listen_on=port)
    shared = tmp_path / "shared" / "master.addr"
    pool = MooncakeStoreConfig(
        launch_master=True, master_port=port, master_address_file=str(shared)
    )

    with provision_pool(pool, run_dir=str(tmp_path / "run")):
        assert resolve_master_address(f"file://{shared}", timeout=5.0).endswith(f":{port}")

    # Retracted, so the next run's donors wait for a live master rather than
    # joining a pool that no longer exists.
    assert not shared.exists()


def test_a_half_written_address_is_never_read(tmp_path):
    """A reader sees the whole address or nothing, never a prefix of one."""
    target = tmp_path / "master.addr"

    with master_module._published_address("10.0.0.7:50051", [str(target)]):
        assert not (tmp_path / "master.addr.partial").exists()
        assert target.read_text().strip() == "10.0.0.7:50051"


# ---- saying why bringup is stuck ----


def test_an_absent_master_is_named_rather_than_left_to_store_setup(monkeypatch):
    """Otherwise the failure is a bare status code in every rank, after loading."""
    monkeypatch.setenv(master_module.MASTER_TIMEOUT_ENV, "1")
    address = f"127.0.0.1:{free_port()}"

    with pytest.raises(TimeoutError, match=address):
        master_module.wait_for_master(address)


def test_an_address_of_a_shape_we_cannot_probe_is_not_fatal():
    """Mooncake may accept addresses this cannot dial; leave them to it."""
    assert master_module.wait_for_master("unix:///var/run/mooncake") is None


def test_a_master_that_died_starting_is_reported_with_its_last_words(fake_master, tmp_path):
    """The reason is in the master's log, which is only read if the error quotes it."""
    run_dir = tmp_path / "run"
    fake_master.arm(exit_code=1, log_text="E0903 bind(50051) failed: Address already in use\n")
    pool = MooncakeStoreConfig(launch_master=True, master_port=free_port())

    with pytest.raises(RuntimeError, match="Address already in use"):
        with provision_pool(pool, run_dir=str(run_dir)):
            pytest.fail("provisioning should not have yielded")


# ---- choosing the fabric without naming it in a config ----


def fake_hca(root, device, link_layer="InfiniBand", state="4: ACTIVE", rate="800 Gb/sec"):
    port = root / device / "ports" / "1"
    port.mkdir(parents=True)
    (port / "link_layer").write_text(f"{link_layer}\n")
    (port / "state").write_text(f"{state}\n")
    (port / "rate").write_text(f"{rate}\n")


def test_the_compute_fabric_is_picked_over_the_management_adapter(tmp_path):
    """A node's HCAs are not interchangeable: only some are the fast fabric."""
    fake_hca(tmp_path, "mlx5_0")
    fake_hca(tmp_path, "mlx5_1")
    fake_hca(tmp_path, "mlx5_2", rate="400 Gb/sec")
    fake_hca(tmp_path, "mlx5_3", state="1: DOWN")
    fake_hca(tmp_path, "mlx5_4", link_layer="Ethernet")

    assert (
        master_module.resolve_device_name("rdma", "", sysfs_root=str(tmp_path)) == "mlx5_0,mlx5_1"
    )


def test_a_named_device_is_not_second_guessed(tmp_path):
    fake_hca(tmp_path, "mlx5_0")
    assert master_module.resolve_device_name("rdma", "mlx5_7", sysfs_root=str(tmp_path)) == "mlx5_7"


def test_tcp_needs_no_device_and_looks_for_none(tmp_path):
    assert master_module.resolve_device_name("tcp", "", sysfs_root=str(tmp_path)) == ""


def test_a_node_without_infiniband_is_left_to_mooncake_s_own_discovery(tmp_path):
    """Falling back beats failing, since Mooncake may still find a usable device."""
    assert master_module.resolve_device_name("rdma", "", sysfs_root=str(tmp_path / "absent")) == ""


def test_the_detected_device_is_what_the_workers_are_told(fake_master, tmp_path, monkeypatch):
    sysfs = tmp_path / "sysfs"
    fake_hca(sysfs, "mlx5_0")
    monkeypatch.setattr(master_module, "IB_SYSFS_ROOT", str(sysfs))
    port = free_port()
    fake_master.arm(listen_on=port)
    pool = MooncakeStoreConfig(launch_master=True, master_port=port, protocol="rdma")

    with provision_pool(pool, run_dir=str(tmp_path / "run")) as config_path:
        assert json.loads(open(config_path).read())["device_name"] == "mlx5_0"
