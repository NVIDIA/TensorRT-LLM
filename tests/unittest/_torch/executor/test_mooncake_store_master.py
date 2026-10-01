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
"""Unit tests for the Mooncake pool manifest and joining a pool.

Runs without a Mooncake installation and without a GPU. A master this process
launches is a fake standing in for `Popen` that opens the RPC port, which is
all the readiness handshake ever observes. A master someone else runs is a
plain socket.
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors import mooncake_store
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import master as master_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    PoolManifest,
    maybe_provision_pool,
    provision_pool,
    resolve_pool,
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
    """No ambient pool: these tests are about what joining does itself.

    Every wait below is given an explicit timeout, since nothing here is slow
    to start and a wait that runs long is a failure rather than something that
    needs more time.
    """
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)


@pytest.fixture
def fake_master(monkeypatch):
    """Replace the master binary and its process with in-process fakes.

    Returns a callable that arms the fake. Once the master has run, the
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
def live_master():
    """A socket standing in for a master someone else is running."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        yield f"127.0.0.1:{listener.getsockname()[1]}"
    finally:
        listener.close()


def manifest_file(tmp_path, address, **overrides):
    """A published manifest naming `address`, as the master would write it."""
    record = {
        "master_server_address": address,
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "rdma",
        "namespace": "trtllm",
    }
    record.update(overrides)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(record))
    return path


# ---- configuration ----


def test_a_pool_must_be_named():
    """Without it there is no master to reach and nothing to join."""
    with pytest.raises(ValueError):
        MooncakeStoreConfig()


def test_pool_is_rejected_unless_the_connector_is_mooncake_store():
    """The validator keys off the connector, however that was spelled."""
    with pytest.raises(ValueError, match="mooncake_store describes a Mooncake pool"):
        KvCacheConnectorConfig(
            connector="lmcache",
            mooncake_store=MooncakeStoreConfig(pool="host:50051"),
        )
    # Naming the module rather than the preset selects the same connector.
    KvCacheConnectorConfig(
        connector_module="tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        connector_scheduler_class="MooncakeStoreConnectorScheduler",
        connector_worker_class="MooncakeStoreConnectorWorker",
        mooncake_store=MooncakeStoreConfig(pool="host:50051"),
    )


@pytest.mark.parametrize("role", ["both", "producer", "consumer", "capacity"])
def test_every_role_is_configurable(role):
    assert MooncakeStoreConfig(pool="host:50051", role=role).role == role


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        MooncakeStoreConfig(pool="host:50051", role="donor")


# ---- the manifest ----


def test_a_manifest_round_trips():
    original = PoolManifest(
        master_server_address="10.0.0.1:50051",
        metadata_server="http://127.0.0.1:8080/metadata",
        protocol="tcp",
        namespace="experiment-4",
        metrics_port=9100,
        eviction_ratio=0.2,
    )
    assert PoolManifest.from_json(original.to_json()) == original


def test_a_manifest_keeps_keys_it_does_not_understand():
    """A newer master's manifest must survive an older reader unharmed."""
    raw = {"master_server_address": "h:1", "future_setting": "keep me"}
    parsed = PoolManifest.from_json(raw)
    assert parsed.extra == {"future_setting": "keep me"}
    assert parsed.to_json()["future_setting"] == "keep me"


def test_a_manifest_without_a_master_is_refused():
    """Its whole purpose is naming the master, so this is not a usable pool."""
    with pytest.raises(ValueError, match="names no master_server_address"):
        PoolManifest.from_json({"protocol": "rdma"}, "/shared/pool.json")


def test_a_manifest_names_the_command_that_writes_it():
    with pytest.raises(ValueError, match="mooncake_master --pool_file"):
        PoolManifest.from_json({}, "/shared/pool.json")


# ---- the rendered client config ----


def test_client_config_is_what_the_connector_reads_back(tmp_path):
    """The generated JSON has to survive the connector's own parser."""
    pool = MooncakeStoreConfig(
        pool="file:///shared/pool.json",
        role="capacity",
        segment_size="64GiB",
        namespace="trtllm-m3",
        stage_through_host=True,
        transfer_batch_size=32,
    )
    manifest = PoolManifest(
        master_server_address="10.0.0.1:50051",
        metadata_server="P2PHANDSHAKE",
        protocol="rdma",
    )
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps(master_module._client_config(pool, manifest, "mlx5_0")))

    parsed = MooncakeStoreConnectorConfig.from_file(str(path))
    assert parsed.master_server_address == "10.0.0.1:50051"
    assert parsed.metadata_server == "P2PHANDSHAKE"
    assert parsed.protocol == "rdma"
    assert parsed.device_name == "mlx5_0"
    assert parsed.global_segment_size == 64 * 1024**3
    assert parsed.namespace == "trtllm-m3"
    assert parsed.role is StoreRole.CAPACITY
    assert parsed.stage_through_host is True
    assert parsed.transfer_batch_size == 32


def test_the_pool_supplies_what_the_server_does_not(tmp_path):
    """Pool-wide settings come from the manifest, not from each worker config."""
    pool = MooncakeStoreConfig(pool="file:///shared/pool.json")
    manifest = PoolManifest(
        master_server_address="10.9.9.9:50051",
        metadata_server="http://meta:8080/metadata",
        protocol="tcp",
        namespace="from-the-pool",
    )
    written = master_module._client_config(pool, manifest, "")

    assert written["master_server_address"] == "10.9.9.9:50051"
    assert written["metadata_server"] == "http://meta:8080/metadata"
    assert written["protocol"] == "tcp"
    assert written["namespace"] == "from-the-pool"


def test_a_server_may_override_the_pools_namespace():
    """Isolating one deployment's keys is the server's business, not the pool's."""
    pool = MooncakeStoreConfig(pool="h:1", namespace="mine")
    manifest = PoolManifest(master_server_address="h:1", namespace="shared")
    assert master_module._client_config(pool, manifest, "")["namespace"] == "mine"


def test_sizes_reach_the_client_config_as_resolved_integers():
    """vLLM reads this file too, and reads 'GB' as a power of 1024, not 1000."""
    pool = MooncakeStoreConfig(pool="h:1", segment_size="160GiB")
    manifest = PoolManifest(master_server_address="h:1")
    written = master_module._client_config(pool, manifest, "")
    assert written["global_segment_size"] == 160 * 1024**3
    assert isinstance(written["global_segment_size"], int)


def test_an_ambiguous_segment_size_is_refused():
    """It would name two different sizes to the two engines sharing the pool."""
    pool = MooncakeStoreConfig(pool="h:1", segment_size="160GB")
    manifest = PoolManifest(master_server_address="h:1")
    with pytest.raises(ValueError, match="160GiB"):
        master_module._client_config(pool, manifest, "")


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


# ---- joining a pool ----


def test_joining_points_the_workers_at_the_pools_master(live_master, tmp_path):
    pool = MooncakeStoreConfig(pool=f"file://{manifest_file(tmp_path, live_master)}")

    with provision_pool(pool) as config_path:
        # The workers are spawned inside this window and are told about the
        # pool through the environment, so both have to hold while it is open.
        assert os.environ[CONFIG_PATH_ENV] == config_path
        written = json.loads(open(config_path).read())
        assert written["master_server_address"] == live_master

    assert CONFIG_PATH_ENV not in os.environ
    assert not os.path.exists(config_path)


def test_a_bare_master_address_needs_no_manifest(live_master):
    """Joining a master run without this CLI still has to work."""
    pool = MooncakeStoreConfig(pool=live_master)

    with provision_pool(pool) as config_path:
        written = json.loads(open(config_path).read())
        assert written["master_server_address"] == live_master
        assert written["metadata_server"] == "P2PHANDSHAKE"


def test_joining_fails_before_the_model_loads_if_the_master_is_absent(tmp_path):
    absent = f"127.0.0.1:{free_port()}"
    pool = MooncakeStoreConfig(pool=f"file://{manifest_file(tmp_path, absent)}", master_timeout=1)

    with pytest.raises(TimeoutError, match="did not accept connections"):
        with provision_pool(pool):
            pytest.fail("joining should not have yielded")
    assert CONFIG_PATH_ENV not in os.environ


def test_an_unparsable_master_address_is_left_to_the_workers(tmp_path):
    """Not every address is host:port, so an unprobeable one passes through."""
    address = "unix:///var/run/mooncake"
    pool = MooncakeStoreConfig(pool=f"file://{manifest_file(tmp_path, address)}")

    with provision_pool(pool) as config_path:
        assert json.loads(open(config_path).read())["master_server_address"] == address


def test_an_inherited_config_path_wins(monkeypatch, tmp_path):
    """An externally managed pool names itself this way, so joining defers."""
    harness_config = tmp_path / "harness.json"
    harness_config.write_text("{}")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(harness_config))
    pool = MooncakeStoreConfig(pool="file:///nowhere/pool.json")

    with provision_pool(pool) as config_path:
        assert config_path is None
        assert os.environ[CONFIG_PATH_ENV] == str(harness_config)

    assert os.environ[CONFIG_PATH_ENV] == str(harness_config)


def test_the_config_names_where_the_client_config_goes(live_master, tmp_path):
    """The ranks a launcher started read it back from there, having inherited
    nothing, so the path has to be something their own config states.
    """
    run_dir = tmp_path / "run"
    pool = MooncakeStoreConfig(pool=live_master, run_dir=str(run_dir))

    with provision_pool(pool) as config_path:
        assert config_path == str(run_dir / master_module.CLIENT_CONFIG_NAME)

    assert (run_dir / master_module.CLIENT_CONFIG_NAME).exists()


def test_an_unnamed_run_directory_is_temporary(live_master):
    """It keeps the connector working, but loses this run's segment records."""
    pool = MooncakeStoreConfig(pool=live_master)

    with provision_pool(pool) as config_path:
        assert os.path.exists(config_path)

    assert not os.path.exists(config_path)


def test_a_half_written_client_config_is_never_read(live_master, tmp_path):
    """Two servers sharing a run directory must not interleave into a torn one."""
    run_dir = tmp_path / "run"
    pool = MooncakeStoreConfig(pool=live_master, run_dir=str(run_dir))

    with provision_pool(pool) as config_path:
        assert not os.path.exists(f"{config_path}.partial")
        assert json.loads(open(config_path).read())["master_server_address"] == live_master


def test_a_run_dir_keeps_the_client_config(live_master, tmp_path):
    run_dir = tmp_path / "run"
    pool = MooncakeStoreConfig(pool=live_master)

    with provision_pool(pool, run_dir=str(run_dir)) as config_path:
        assert config_path == str(run_dir / master_module.CLIENT_CONFIG_NAME)

    # An explicit run directory outlives the run that filled it, since the
    # segment records in it are what the run's capacity is reported from.
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
def test_joining_is_a_no_op_unless_a_pool_is_described(config):
    """Without `mooncake_store`, MOONCAKE_CONFIG_PATH is still the only input."""
    with maybe_provision_pool(config):
        assert CONFIG_PATH_ENV not in os.environ


def test_a_described_pool_is_joined(live_master):
    config = KvCacheConnectorConfig(
        connector="mooncake-store",
        mooncake_store=MooncakeStoreConfig(pool=live_master),
    )
    with maybe_provision_pool(config):
        written = json.loads(open(os.environ[CONFIG_PATH_ENV]).read())
        assert written["master_server_address"] == live_master
    assert CONFIG_PATH_ENV not in os.environ


# ---- reaching a pool whose host nobody knew in advance ----


@pytest.mark.parametrize("address", ["10.0.0.1:50051", "unix:///var/run/mooncake"])
def test_an_address_that_is_not_a_file_passes_through(address):
    assert resolve_pool(address, timeout=1.0).master_server_address == address


def test_a_published_manifest_is_read_from_the_file_that_names_it(tmp_path):
    path = manifest_file(tmp_path, "10.0.0.7:50051", protocol="tcp")
    resolved = resolve_pool(f"file://{path}", timeout=1.0)
    assert resolved.master_server_address == "10.0.0.7:50051"
    assert resolved.protocol == "tcp"


def test_a_manifest_not_published_yet_is_waited_for(tmp_path):
    """Master and servers are started together; neither one orders the other."""
    path = tmp_path / "pool.json"
    threading.Timer(
        0.5, path.write_text, [json.dumps({"master_server_address": "10.0.0.9:50051"})]
    ).start()

    assert resolve_pool(f"file://{path}", timeout=10.0).master_server_address == "10.0.0.9:50051"


def test_an_empty_manifest_file_is_not_taken_for_a_pool(tmp_path):
    """An existing file is not the same as a published manifest."""
    path = tmp_path / "pool.json"
    path.write_text("")

    with pytest.raises(TimeoutError, match="No Mooncake pool manifest"):
        resolve_pool(f"file://{path}", timeout=1.0)


def test_an_unparsable_manifest_is_treated_as_not_there_yet(tmp_path):
    """What a reader racing a slow filesystem sees, not a reason to fail."""
    path = tmp_path / "pool.json"
    path.write_text("{not json")

    with pytest.raises(TimeoutError, match="No Mooncake pool manifest"):
        resolve_pool(f"file://{path}", timeout=1.0)


def test_an_unpublished_manifest_names_the_command_that_publishes_it(tmp_path):
    with pytest.raises(TimeoutError, match="--pool_file"):
        resolve_pool(f"file://{tmp_path / 'absent'}", timeout=1.0)


def test_a_pool_must_be_named_to_be_resolved():
    with pytest.raises(ValueError, match="pool is required"):
        resolve_pool("", timeout=1.0)


# ---- the master, which owns the pool ----


def test_a_master_publishes_a_manifest_that_can_be_dialed(fake_master, tmp_path):
    port = free_port()
    fake_master.arm(listen_on=port)
    pool_file = tmp_path / "shared" / "pool.json"

    with master_module.running_master(
        str(tmp_path / "run"), pool_file=str(pool_file), rpc_port=port
    ) as master:
        resolved = resolve_pool(f"file://{pool_file}", timeout=5.0)
        assert resolved.master_server_address == master.address
        host, _, named_port = master.address.rpartition(":")
        assert int(named_port) == port
        with socket.create_connection((host, port), timeout=5):
            pass


def test_the_master_states_the_settings_every_participant_shares(fake_master, tmp_path):
    """Stated once here rather than restated in each worker config."""
    port = free_port()
    fake_master.arm(listen_on=port)
    pool_file = tmp_path / "pool.json"

    with master_module.running_master(
        str(tmp_path / "run"),
        pool_file=str(pool_file),
        rpc_port=port,
        metrics_port=9100,
        eviction_ratio=0.25,
        metadata_server="http://meta:8080/metadata",
        protocol="tcp",
        namespace="round5",
    ):
        published = json.loads(pool_file.read_text())

    assert published["protocol"] == "tcp"
    assert published["metadata_server"] == "http://meta:8080/metadata"
    assert published["namespace"] == "round5"
    assert published["eviction_ratio"] == 0.25
    assert published["metrics_port"] == 9100


def test_a_stopped_master_leaves_no_manifest_behind(fake_master, tmp_path):
    """A stale manifest would send the next run's workers to a dead port."""
    port = free_port()
    fake_master.arm(listen_on=port)
    pool_file = tmp_path / "pool.json"

    with master_module.running_master(
        str(tmp_path / "run"), pool_file=str(pool_file), rpc_port=port
    ):
        assert pool_file.exists()

    assert not pool_file.exists()
    assert fake_master.process.terminated


def test_a_master_always_publishes_into_its_run_directory(fake_master, tmp_path):
    """So a finished run's own logs say which pool it used."""
    port = free_port()
    fake_master.arm(listen_on=port)
    run_dir = tmp_path / "run"

    with master_module.running_master(str(run_dir), rpc_port=port):
        assert (run_dir / master_module.POOL_MANIFEST_NAME).exists()

    assert not (run_dir / master_module.POOL_MANIFEST_NAME).exists()


def test_a_master_keeps_its_log(fake_master, tmp_path):
    """It outlives the servers that used it, so its log is kept."""
    port = free_port()
    fake_master.arm(listen_on=port)
    run_dir = tmp_path / "run"

    with master_module.running_master(str(run_dir), rpc_port=port):
        pass

    assert (run_dir / master_module.MASTER_LOG_NAME).exists()


def test_a_master_gets_the_flags_and_logging_it_needs(fake_master, tmp_path):
    port = free_port()
    metrics_port = free_port()
    fake_master.arm(listen_on=port)

    with master_module.running_master(
        str(tmp_path / "run"), rpc_port=port, metrics_port=metrics_port, eviction_ratio=0.1
    ):
        command = fake_master.process.command
        assert command[0].endswith("mooncake_master")
        assert f"--rpc_port={port}" in command
        assert f"--metrics_port={metrics_port}" in command
        assert "--eviction_ratio=0.1" in command
        # Without these the master logs to a file under /tmp and the log the
        # run directory holds stays empty.
        assert fake_master.process.env["GLOG_logtostderr"] == "1"
        assert fake_master.process.env["GLOG_v"] == "1"


def test_a_master_that_dies_during_startup_says_so(fake_master, tmp_path):
    fake_master.arm(exit_code=3)

    with pytest.raises(RuntimeError, match="exited with code 3"):
        with master_module.running_master(str(tmp_path / "run"), rpc_port=free_port()):
            pytest.fail("the master should not have come up")


def test_a_master_that_never_listens_times_out(fake_master, tmp_path):
    fake_master.arm()

    with pytest.raises(TimeoutError, match="did not accept connections"):
        with master_module.running_master(str(tmp_path / "run"), rpc_port=free_port(), timeout=1):
            pytest.fail("the master should not have come up")
    assert fake_master.process.terminated


def test_a_named_binary_is_the_one_that_runs(fake_master, tmp_path):
    """An image may ship the master under another name or path."""
    port = free_port()
    fake_master.arm(listen_on=port)

    with master_module.running_master(
        str(tmp_path / "run"), rpc_port=port, binary="mooncake_master_next"
    ):
        assert fake_master.process.command[0].endswith("mooncake_master_next")


def test_a_missing_master_binary_names_the_alternatives(monkeypatch, tmp_path):
    monkeypatch.setattr(
        master_module,
        "shutil",
        SimpleNamespace(which=lambda _name: None, rmtree=shutil.rmtree),
    )

    with pytest.raises(FileNotFoundError, match="install_mooncake.sh"):
        with master_module.running_master(str(tmp_path / "run")):
            pytest.fail("the master should not have come up")


def test_a_server_joins_a_master_it_was_never_given_the_address_of(fake_master, tmp_path):
    """The manifest is how servers reach a master no config names a host for."""
    port = free_port()
    fake_master.arm(listen_on=port)
    pool_file = tmp_path / "pool.json"

    with master_module.running_master(
        str(tmp_path / "run"), pool_file=str(pool_file), rpc_port=port
    ) as master:
        worker = MooncakeStoreConfig(pool=f"file://{pool_file}")
        with provision_pool(worker) as config_path:
            # Mooncake cannot dial a file:// URL, so what reaches the workers
            # has to be the address it resolved to.
            written = json.loads(open(config_path).read())
            assert written["master_server_address"] == master.address


def test_a_half_written_manifest_is_never_read(tmp_path):
    """A reader sees the whole manifest or nothing, never a prefix of one."""
    target = tmp_path / "pool.json"
    manifest = PoolManifest(master_server_address="10.0.0.7:50051")

    with master_module._published_manifest(manifest, [str(target)]):
        assert not (tmp_path / "pool.json.partial").exists()
        assert json.loads(target.read_text())["master_server_address"] == "10.0.0.7:50051"


# ---- saying why bringup is stuck ----


def test_an_absent_master_is_named_rather_than_left_to_store_setup():
    """Otherwise the failure is a bare status code in every rank, after loading."""
    address = f"127.0.0.1:{free_port()}"

    with pytest.raises(TimeoutError, match=address):
        master_module.wait_for_master(address, timeout=1)


def test_an_address_of_a_shape_we_cannot_probe_is_not_fatal():
    """Mooncake may accept addresses this cannot dial; leave them to it."""
    assert master_module.wait_for_master("unix:///var/run/mooncake") is None


def test_a_master_that_died_starting_is_reported_with_its_last_words(fake_master, tmp_path):
    """The reason is in the master's log, which is only read if the error quotes it."""
    fake_master.arm(exit_code=1, log_text="E0903 bind(50051) failed: Address already in use\n")

    with pytest.raises(RuntimeError, match="Address already in use"):
        with master_module.running_master(str(tmp_path / "run"), rpc_port=free_port()):
            pytest.fail("the master should not have come up")


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


# ---- what a server does with the pool settings around its own lifetime ----


class ProvisioningLog:
    """Entries and exits of the two contexts a serving process holds open.

    Recording only entries would accept a `_provision_kv_cache_pool` that
    closed the pool before yielding, which is the one thing it must not do:
    the pool has to outlive the server it was provisioned for.
    """

    def __init__(self):
        self.events: list = []

    @contextlib.contextmanager
    def holding(self, name, argument):
        self.events.append(("enter", name, argument))
        try:
            yield
        finally:
            self.events.append(("exit", name, argument))

    @property
    def open_contexts(self) -> list:
        entered = [name for event, name, _ in self.events if event == "enter"]
        for event, name, _ in self.events:
            if event == "exit":
                entered.remove(name)
        return entered

    def order(self, event: str) -> list:
        return [name for kind, name, _ in self.events if kind == event]


@pytest.fixture
def provisioning(monkeypatch) -> SimpleNamespace:
    """Drive `_provision_kv_cache_pool` against recorded pool contexts."""
    from tensorrt_llm.commands import serve

    log = ProvisioningLog()
    monkeypatch.setattr(
        mooncake_store, "maybe_provision_pool", lambda config: log.holding("pool", config)
    )
    return SimpleNamespace(serve=serve, log=log)


def test_an_attached_frontend_provisions_nothing(provisioning):
    """It shares the launcher's executor, so a second pool would be its own."""
    llm_args = {"kv_connector_config": {"connector": "mooncake-store"}}

    with provisioning.serve._provision_kv_cache_pool(llm_args, owns_engine=False):
        pass

    assert provisioning.log.events == []


def test_the_pool_stays_open_for_the_server(provisioning):
    """It has to outlive bringup, not just reach the end of provisioning."""
    llm_args = {
        "kv_connector_config": {
            "connector": "mooncake-store",
            "mooncake_store": {"pool": "file:///shared/pool.json"},
        },
    }

    with provisioning.serve._provision_kv_cache_pool(llm_args):
        # Where the server runs. A pool closed by now is capacity the engine
        # would find gone the moment it opened a store handle.
        assert provisioning.log.open_contexts == ["pool"]

    assert provisioning.log.order("exit") == ["pool"]


def test_pool_settings_from_yaml_are_typed_before_the_llm_sees_them(provisioning):
    """A YAML section arrives as a dict, and the pool is described before bringup."""
    llm_args = {
        "kv_connector_config": {
            "connector": "mooncake-store",
            "mooncake_store": {"pool": "file:///shared/pool.json"},
        },
    }

    with provisioning.serve._provision_kv_cache_pool(llm_args):
        pass

    assert isinstance(llm_args["kv_connector_config"], KvCacheConnectorConfig)


def test_the_user_facing_pool_settings_reach_provisioning(monkeypatch):
    """The settings a user wrote are the ones the pool is joined with.

    A field renamed on either side would otherwise surface at serve time, with
    the model already loading.
    """
    provisioned = []

    @contextlib.contextmanager
    def record(pool, **kwargs):
        provisioned.append(pool)
        yield "mooncake.json"

    monkeypatch.setattr(master_module, "provision_pool", record)

    config = KvCacheConnectorConfig(
        connector="mooncake-store",
        mooncake_store=MooncakeStoreConfig(
            pool="file:///shared/pool.json",
            role="capacity",
            segment_size="4GiB",
            namespace="tenant-a",
            transfer_batch_size=8,
        ),
    )
    with master_module.maybe_provision_pool(config):
        pass

    assert len(provisioned) == 1
    pool = provisioned[0]
    assert (pool.pool, pool.role) == ("file:///shared/pool.json", "capacity")
    assert (pool.segment_size, pool.namespace) == ("4GiB", "tenant-a")
    assert pool.transfer_batch_size == 8
