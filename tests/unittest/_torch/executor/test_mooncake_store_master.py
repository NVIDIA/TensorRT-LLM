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
launches is a fake standing in for ``Popen`` that opens the RPC port, which is
all the readiness handshake ever observes; a master someone else runs is a
plain socket.
"""

import json
import os
import shutil
import socket
import subprocess
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
)
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, MooncakeStoreConfig


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


class FakeMasterProcess:
    """The slice of ``Popen`` that launching a master actually drives.

    ``listen_on`` makes it answer on that port, which is what a real master
    does last and what the readiness wait keys off. ``exit_code`` makes it a
    master that failed to start.
    """

    def __init__(self, command, env, listen_on=None, exit_code=None):
        self.command = command
        self.env = env
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

    Returns a callable that arms the fake and, once provisioning has run, the
    launched instance is available as ``.process`` for inspection.
    """

    class Launcher:
        def __init__(self):
            self.process = None

        def arm(self, listen_on=None, exit_code=None):

            def popen(command, env=None, **_kwargs):
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


def test_pool_needs_exactly_one_master():
    with pytest.raises(ValueError, match="not both"):
        MooncakeStoreConfig(launch_master=True, master_server_address="host:50051")
    with pytest.raises(ValueError, match="needs a master"):
        MooncakeStoreConfig()


def test_pool_is_rejected_on_another_connector():
    with pytest.raises(ValueError, match="mooncake_store describes a Mooncake pool"):
        KvCacheConnectorConfig(
            connector="lmcache",
            mooncake_store=MooncakeStoreConfig(launch_master=True),
        )


def test_pool_is_accepted_on_the_module_spelled_out():
    """A config naming the module instead of the preset is still the connector."""
    config = KvCacheConnectorConfig(
        connector_module="tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        connector_scheduler_class="MooncakeStoreConnectorScheduler",
        connector_worker_class="MooncakeStoreConnectorWorker",
        mooncake_store=MooncakeStoreConfig(launch_master=True),
    )
    assert config.mooncake_store.launch_master


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


def test_client_config_leaves_an_unset_prefix_to_the_connector():
    pool = MooncakeStoreConfig(master_server_address="host:50051")
    assert "cache_prefix" not in master_module._client_config(pool, "host:50051")


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


def test_provisioning_fails_before_the_model_loads_if_the_master_is_absent(monkeypatch):
    monkeypatch.setenv(master_module.MASTER_TIMEOUT_ENV, "1")
    pool = MooncakeStoreConfig(master_server_address=f"127.0.0.1:{free_port()}")

    with pytest.raises(TimeoutError, match="did not accept connections"):
        with provision_pool(pool):
            pytest.fail("provisioning should not have yielded")
    assert CONFIG_PATH_ENV not in os.environ


def test_an_unparseable_master_address_is_left_to_the_workers():
    """Not every address is host:port; that is the worker's problem, not ours."""
    pool = MooncakeStoreConfig(master_server_address="unix:///var/run/mooncake")

    with provision_pool(pool) as config_path:
        written = json.loads(open(config_path).read())
        assert written["master_server_address"] == "unix:///var/run/mooncake"


def test_an_inherited_config_path_wins(monkeypatch, tmp_path):
    """The SLURM harness names the pool this way; provisioning must defer."""
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
        # Whatever the config names has to be dialable: it is all a worker on
        # another host is given.
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
        # glog writes to files under /tmp unless told otherwise, which would
        # leave the master's log -- the only view of the pool's own side of
        # the conversation -- empty.
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

    # An explicit run directory has to outlive the run that filled it: the
    # master's log is where pool occupancy and eviction are read from.
    assert (run_dir / master_module.MASTER_LOG_NAME).exists()
    assert (run_dir / master_module.CLIENT_CONFIG_NAME).exists()


# ---- the entry point servers call ----


def test_other_connectors_are_left_alone():
    config = KvCacheConnectorConfig(connector="lmcache")
    with maybe_provision_pool(config):
        assert CONFIG_PATH_ENV not in os.environ


def test_no_connector_at_all_is_left_alone():
    with maybe_provision_pool(None):
        assert CONFIG_PATH_ENV not in os.environ


def test_a_pool_left_undescribed_stays_the_environment_contract():
    """Without ``mooncake_store``, MOONCAKE_CONFIG_PATH is still the only input."""
    config = KvCacheConnectorConfig(connector="mooncake-store")
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
