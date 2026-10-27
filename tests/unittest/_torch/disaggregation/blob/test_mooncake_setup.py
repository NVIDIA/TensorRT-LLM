# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store config, opening the Mooncake SDK and driver, and the blob package's layering.

The SDK is a fake module put into ``sys.modules``, the master probe and the route probe are
replaced, and the InfiniBand sysfs tree is a temporary directory, so nothing here needs a network,
a Mooncake install or a GPU.
"""

from __future__ import annotations

import ast
import pathlib
import socket
import sys
import threading
import types

import pytest
import yaml
from fake_mooncake import FakeMooncakeStore, FakePool

import tensorrt_llm._torch.disaggregation.blob as blob_package
from tensorrt_llm._torch.disaggregation.blob import mooncake_setup
from tensorrt_llm._torch.disaggregation.blob.backend import BlobConfig
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    MooncakeConfig,
    load_store_config,
    open_mooncake_driver,
    open_sdk,
)
from tensorrt_llm._torch.disaggregation.blob.store import RegionLayout
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

MASTER = "10.0.0.1:50051"
DRAIN_S = 5.0  # a drained driver's worker threads have exited
ROUTED_ADDRESS = "10.0.0.7"


def store_config(protocol: str = "tcp", **mooncake) -> dict:
    return {
        "blob": {"key_prefix": "t"},
        "mooncake": {
            "master_server_address": MASTER,
            "global_segment_size": 0,
            "protocol": protocol,
            **mooncake,
        },
    }


class Sdk:
    """The fake ``mooncake.store`` module's state: every store it built, and the masters probed."""

    def __init__(self) -> None:
        self.pool = FakePool()
        self.stores: list[FakeMooncakeStore] = []
        self.masters: list[str] = []
        self.setup_answer = 0

    def build(self) -> FakeMooncakeStore:
        store = FakeMooncakeStore(self.pool)
        store.setup_answer = self.setup_answer
        self.stores.append(store)
        return store


def read_blob(section) -> BlobConfig:
    return BlobConfig(**section)


def built_by(monkeypatch: pytest.MonkeyPatch, *builds: tuple[str, str]) -> None:
    """Makes ``builds`` (name, version) the distributions that list the imported SDK module."""
    monkeypatch.setattr(mooncake_setup, "_builds_listing_file", lambda module: list(builds))


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Sdk:
    sdk = Sdk()
    package = types.ModuleType("mooncake")
    package.__path__ = []
    module = types.ModuleType("mooncake.store")
    module.MooncakeDistributedStore = sdk.build
    package.store = module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", module)
    built_by(monkeypatch, ("mooncake-transfer-engine-cuda13", "0.3.13"))
    monkeypatch.setattr(mooncake_setup, "_wait_for_master", sdk.masters.append)
    monkeypatch.setattr(mooncake_setup, "_local_address", lambda master: ROUTED_ADDRESS)
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(tmp_path / "no-infiniband"))
    for name in ("MC_TCP_PROTO", "MC_STORE_LOCAL_HOT_CACHE_SIZE", "MC_USE_TENT", "MC_USE_TEV1"):
        monkeypatch.delenv(name, raising=False)
    return sdk


def ib_device(root: pathlib.Path, name: str, link_layer: str, state: str, rate: str) -> None:
    port = root / name / "ports" / "1"
    port.mkdir(parents=True)
    (port / "link_layer").write_text(f"{link_layer}\n")
    (port / "state").write_text(f"{state}\n")
    (port / "rate").write_text(f"{rate}\n")


# Store config.


def test_store_config_refuses_missing_and_unknown_sections_and_keys() -> None:
    config = store_config()
    del config["mooncake"]["global_segment_size"]
    with pytest.raises(ValueError, match="'mooncake'.*global_segment_size"):
        load_store_config(config)
    with pytest.raises(ValueError, match="unknown sections"):
        load_store_config({**store_config(), "mooncake_store": {}})
    with pytest.raises(ValueError, match="'mooncake' is a required mapping"):
        load_store_config({"blob": {}})


BAD_VALUES = {
    "blob.key_prefx=t": ("blob", "key_prefx", "t"),  # an unknown key
    # How Mooncake's work runs is set in its own section.
    "blob.workers=2": ("blob", "workers", 2),
    "mooncake.devce_name=mlx5_0": ("mooncake", "devce_name", "mlx5_0"),
    "mooncake.protocol=rdmaa": ("mooncake", "protocol", "rdmaa"),
    "mooncake.global_segment_size=True": ("mooncake", "global_segment_size", True),
    "mooncake.rows_per_call=64.0": ("mooncake", "rows_per_call", 64.0),
    "mooncake.bytes_per_call=1.28e8": ("mooncake", "bytes_per_call", 1.28e8),
    "blob.key_prefix=7": ("blob", "key_prefix", 7),
    "mooncake.workers=2.5": ("mooncake", "workers", 2.5),
    "mooncake.workers=True": ("mooncake", "workers", True),
    "mooncake.workers=0": ("mooncake", "workers", 0),
    "mooncake.global_segment_size=1.0": ("mooncake", "global_segment_size", 1.0),
    "mooncake.local_buffer_size=-1": ("mooncake", "local_buffer_size", -1),
    "mooncake.master_server_address=5": ("mooncake", "master_server_address", 5),
    "mooncake.device_name=None": ("mooncake", "device_name", None),
    # One master's host:port: an HA master list would leave nothing to wait for.
    "mooncake.master_server_address=HA": (
        "mooncake",
        "master_server_address",
        "etcd://10.0.0.1:2379;10.0.0.2:2379",
    ),
}


@pytest.mark.parametrize("case", BAD_VALUES)
def test_unknown_keys_and_values_of_the_wrong_type_are_rejected(sdk: Sdk, case: str) -> None:
    section, name, value = BAD_VALUES[case]
    config = store_config()
    config[section][name] = value
    with pytest.raises(ValueError, match=f"section '{section}'.*{name}"):
        open_mooncake_driver(config, read_blob)
    assert sdk.stores == [] and sdk.masters == []  # refused before the SDK was opened


def test_an_empty_blob_section_is_empty_and_anything_but_a_mapping_is_refused() -> None:
    mooncake = store_config()["mooncake"]
    for empty in ({"blob": None}, {}):  # "blob:" alone, or no blob section
        assert load_store_config({**empty, "mooncake": mooncake})[0] == {}
    for not_a_mapping in ([], False, 0):
        with pytest.raises(ValueError, match="'blob' an optional one"):
            load_store_config({"blob": not_a_mapping, "mooncake": mooncake})
    with pytest.raises(ValueError, match="unknown sections"):  # keys of mixed types
        load_store_config({**store_config(), 1: "x", "extra": "y"})


def test_an_unreadable_store_config_file_is_a_value_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="missing.yaml"):
        load_store_config(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("mooncake: [")
    with pytest.raises(ValueError, match="broken.yaml"):
        load_store_config(broken)


def test_store_config_reads_a_yaml_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "store.yaml"
    path.write_text(yaml.safe_dump(store_config(local_buffer_size=0, workers=2)))
    blob, mooncake = load_store_config(path)
    assert blob == {"key_prefix": "t"}
    assert mooncake == MooncakeConfig(MASTER, 0, protocol="tcp", local_buffer_size=0, workers=2)


# Opening the SDK.


def test_open_passes_setup_arguments_by_name(
    sdk: Sdk, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sysfs = tmp_path / "infiniband"
    ib_device(sysfs, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(sysfs, "mlx5_1", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(sysfs))
    tunings = []

    def driver_recording_its_tuning(store, **tuning) -> MooncakeDriver:
        tunings.append(tuning)
        return MooncakeDriver(store, **tuning)

    monkeypatch.setattr(mooncake_setup, "MooncakeDriver", driver_recording_its_tuning)
    driver, blob = open_mooncake_driver(store_config("rdma", rows_per_call=4), read_blob)
    (store,) = sdk.stores
    (setup,) = store.calls_to("setup")
    assert setup.args == ()
    assert setup.kwargs == {
        "local_hostname": ROUTED_ADDRESS,  # where the route to the master leaves from
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": 0,
        "local_buffer_size": 16 << 20,
        "protocol": "rdma",
        "rdma_devices": "mlx5_0,mlx5_1",  # the fastest active InfiniBand ports
        "master_server_addr": MASTER,
    }
    assert sdk.masters == [MASTER]
    assert isinstance(driver, MooncakeDriver)
    # The config's own values win over the driver's defaults.
    assert tunings == [{"workers": 4, "rows_per_call": 4, "bytes_per_call": 128 << 20}]
    assert blob == BlobConfig(key_prefix="t")
    memory = HostBuffer(64)  # the driver works through the client it was given
    handle = driver.register(memory.address, memory.nbytes, RegionLayout(8, 8))
    assert store.calls_to("register_buffer")
    driver.unregister(handle)
    assert driver.drain(DRAIN_S)  # its worker threads exit
    driver.close()  # the caller owns the driver, and closing it closes the client
    assert len(store.calls_to("close")) == 1

    open_sdk(MooncakeConfig(MASTER, 1 << 30, "tcp", "eth9", "10.9.9.9", "etcd://m:2379", 0))
    assert sdk.stores[1].calls_to("setup")[0].kwargs == {
        "local_hostname": "10.9.9.9",
        "metadata_server": "etcd://m:2379",
        "global_segment_size": 1 << 30,
        "local_buffer_size": 0,
        "protocol": "tcp",
        "rdma_devices": "eth9",
        "master_server_addr": MASTER,
    }


def test_device_detection_picks_the_fastest_active_ib_ports(tmp_path: pathlib.Path) -> None:
    ib_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_1", "InfiniBand", "4: ACTIVE", "200 Gb/sec (4X HDR)")
    ib_device(tmp_path, "mlx5_2", "Ethernet", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_3", "InfiniBand", "1: DOWN", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_4", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path)) == ["mlx5_0", "mlx5_4"]
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path / "missing")) == []


def test_a_configured_device_is_used_as_given(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ib_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(tmp_path))
    configured = MooncakeConfig(MASTER, 0, protocol="rdma", device_name="mlx5_9")
    assert mooncake_setup._rdma_devices(configured) == "mlx5_9"  # not the detected mlx5_0


def test_missing_sdk_reports_install_and_shadow_hint(sdk: Sdk, monkeypatch) -> None:
    shadow = types.ModuleType("mooncake")
    shadow.__file__ = "/opt/nixl/lib/python3/site-packages/mooncake/__init__.py"
    monkeypatch.setitem(sys.modules, "mooncake", shadow)
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    with pytest.raises(ImportError) as missing:
        open_sdk(MooncakeConfig(MASTER, 0, "tcp"))
    message = str(missing.value)
    assert "mooncake-transfer-engine-cuda13==0.3.13" in message
    assert "/opt/nixl/" in message and "shadows" in message
    with pytest.raises(ImportError, match="mooncake-transfer-engine-cuda13"):
        open_mooncake_driver(store_config(), read_blob)


def test_tcp_legacy_framing_is_refused(sdk: Sdk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_TCP_PROTO", "1")
    with pytest.raises(ValueError, match="MC_TCP_PROTO"):
        open_mooncake_driver(store_config("tcp"), read_blob)
    assert sdk.stores == []
    open_sdk(MooncakeConfig(MASTER, 0, "rdma", device_name="mlx5_0"))  # RDMA does not frame
    assert len(sdk.stores) == 1


# The SDK reads the size with stoull, which takes leading spaces and stops at the first non-digit.
@pytest.mark.parametrize("size", ["64", " 64", "64MB", "1"])
def test_local_hot_cache_is_refused(sdk: Sdk, monkeypatch: pytest.MonkeyPatch, size: str) -> None:
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", size)
    with pytest.raises(ValueError, match="MC_STORE_LOCAL_HOT_CACHE_SIZE"):
        open_mooncake_driver(store_config(), read_blob)
    assert sdk.stores == []


@pytest.mark.parametrize("size", ["", "0", " 0"])
def test_local_hot_cache_off_is_accepted(
    sdk: Sdk, monkeypatch: pytest.MonkeyPatch, size: str
) -> None:
    monkeypatch.setenv("MC_STORE_LOCAL_HOT_CACHE_SIZE", size)
    open_sdk(MooncakeConfig(MASTER, 0, "tcp"))
    assert len(sdk.stores) == 1


@pytest.mark.parametrize("variable", ["MC_USE_TENT", "MC_USE_TEV1"])
def test_the_newer_transfer_engine_is_refused(
    sdk: Sdk, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "")  # the SDK only asks whether it is set
    with pytest.raises(ValueError, match=f"{variable} selects a transfer engine"):
        open_mooncake_driver(store_config(), read_blob)
    assert sdk.stores == []


CUDA13 = "mooncake-transfer-engine-cuda13"
SDK_BUILDS = {
    # name: (the distributions that list the imported SDK module, the one refused or None)
    "cuda13": ([(CUDA13, "0.3.13")], None),
    "cuda12": ([("mooncake-transfer-engine", "0.3.13")], None),
    "non_cuda": ([("mooncake-transfer-engine-non-cuda", "0.3.13")], None),
    "cuda12_beside_cuda13": ([(CUDA13, "0.3.13"), ("mooncake-transfer-engine", "0.3.13")], None),
    "newer": ([(CUDA13, "0.3.14")], f"{CUDA13}==0.3.14"),
    "efa_transport": ([("mooncake-transfer-engine-efa", "0.3.13")], "efa==0.3.13"),
    # Installed after the pinned build, the other one's file is the module on disk.
    "older_beside_pinned": (
        [(CUDA13, "0.3.13"), ("mooncake-transfer-engine", "0.3.11")],
        "mooncake-transfer-engine==0.3.11",
    ),
    "no_distribution": ([], "no installed distribution"),  # a source tree or a shadowing package
}


@pytest.mark.parametrize("case", SDK_BUILDS)
def test_only_pinned_sdk_builds_are_opened(
    sdk: Sdk, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    builds, refused = SDK_BUILDS[case]
    built_by(monkeypatch, *builds)
    if refused is None:
        driver, _ = open_mooncake_driver(store_config(), read_blob)
        assert len(sdk.stores) == 1
        assert driver.drain(DRAIN_S)  # its worker threads exit
        driver.close()
        return
    with pytest.raises(ImportError, match=f"comes from .*{refused}.*follows 0.3.13 of"):
        open_mooncake_driver(store_config(), read_blob)
    assert sdk.stores == [] and sdk.masters == []


def install(site: pathlib.Path, dist_info: str, metadata: str, files: list[str]) -> None:
    """Writes an installed distribution's metadata into ``site``, listing ``files``."""
    info = site / f"{dist_info}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(metadata)
    (info / "RECORD").write_text("".join(f"{file},,\n" for file in files))


def test_every_build_listing_the_imported_module_is_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    (real / "mooncake").mkdir(parents=True)
    (real / "mooncake" / "store.so").touch()
    store = "mooncake/store.so"
    install(real, "nixl-0.9", "Name: nixl\nVersion: 0.9\n", ["mooncake/__init__.py"])  # a shadow
    install(
        real, "M_T_E-0.3.13", "Name: Mooncake_Transfer_Engine_CUDA13\nVersion: 0.3.13\n", [store]
    )
    # A reinstall over another version can leave a second dist-info of the same name.
    install(real, "m_t_e-0.3.12", f"Name: {CUDA13}\nVersion: 0.3.12\n", [store])
    install(real, "unnamed-1.0", "Version: 1.0\n", [store])
    site = tmp_path / "site"
    site.symlink_to(real)  # the distributions are found under a symlinked site directory
    monkeypatch.setattr(sys, "path", [str(site)])
    module = types.ModuleType("mooncake.store")
    module.__file__ = str(real / "mooncake" / "store.so")
    assert sorted(mooncake_setup._builds_listing_file(module)) == [
        ("a distribution without a name", "1.0"),
        (CUDA13, "0.3.12"),
        (CUDA13, "0.3.13"),
    ]
    with pytest.raises(ImportError, match=f"{CUDA13}==0.3.12"):
        mooncake_setup._require_pinned_builds(module)
    module.__file__ = str(real / "elsewhere" / "store.so")
    assert mooncake_setup._builds_listing_file(module) == []
    del module.__file__
    assert mooncake_setup._builds_listing_file(module) == []


class RoutedProbe:
    """A datagram socket whose connect to an endpoint in ``routes`` leaves from the address given
    there; any other endpoint has no route."""

    routes: dict[tuple[str, int], str] = {}

    def __init__(self, *args) -> None:
        self.address = ""

    def __enter__(self) -> RoutedProbe:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def connect(self, endpoint: tuple[str, int]) -> None:
        if endpoint not in self.routes:
            raise OSError("no route")
        self.address = self.routes[endpoint]

    def getsockname(self) -> tuple[str, int]:
        return self.address, 0


ROUTE_PROBE = mooncake_setup._ROUTE_PROBE_ENDPOINT
MASTER_ENDPOINT = ("10.0.0.1", 50051)
LOCAL_ADDRESS_CASES = {
    # name: (the routes off this host, the address this host's name resolves to, expected)
    "route_to_the_master": ({MASTER_ENDPOINT: "10.0.0.7"}, None, "10.0.0.7"),
    "loopback_to_the_master": (
        {MASTER_ENDPOINT: "127.0.0.1", ROUTE_PROBE: "10.0.0.8"},
        None,
        "10.0.0.8",
    ),
    "no_route": ({}, "10.0.0.9", "10.0.0.9"),
    "nothing_resolves": ({}, None, "127.0.0.1"),
}


@pytest.mark.parametrize("case", LOCAL_ADDRESS_CASES)
def test_the_local_address_is_where_the_route_to_the_master_leaves(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    routes, resolved, expected = LOCAL_ADDRESS_CASES[case]

    def gethostbyname(name: str) -> str:
        if resolved is None:
            raise socket.gaierror("unknown host")
        return resolved

    probe = type("Probe", (RoutedProbe,), {"routes": routes})
    fake = types.SimpleNamespace(
        socket=probe,
        AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM,
        gethostbyname=gethostbyname,
        gethostname=lambda: "this-host",
    )
    monkeypatch.setattr(mooncake_setup, "socket", fake)
    assert mooncake_setup._local_address(MASTER) == expected


def test_waiting_for_the_master_ends_when_it_accepts_and_names_it_when_it_never_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        address = "127.0.0.1:%d" % server.getsockname()[1]
        mooncake_setup._wait_for_master(address)  # accepts at once
    monkeypatch.setattr(mooncake_setup, "MASTER_WAIT_S", 0.2)
    monkeypatch.setattr(mooncake_setup, "MASTER_RETRY_S", 0.05)
    with pytest.raises(TimeoutError, match=f"master at {address} did not accept"):
        mooncake_setup._wait_for_master(address)  # the port is closed now


def test_waiting_for_the_master_outlasts_one_that_comes_up_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.create_server(("127.0.0.1", 0)) as probe:
        port = probe.getsockname()[1]  # free once the probe closes: nothing listens yet
    servers: list[socket.socket] = []
    master_starts = threading.Timer(
        0.3, lambda: servers.append(socket.create_server(("127.0.0.1", port)))
    )
    monkeypatch.setattr(mooncake_setup, "MASTER_WAIT_S", 5.0)
    monkeypatch.setattr(mooncake_setup, "MASTER_RETRY_S", 0.05)
    master_starts.start()
    try:
        mooncake_setup._wait_for_master(f"127.0.0.1:{port}")
    finally:
        master_starts.join()
        for server in servers:
            server.close()


def test_setup_failure_closes_the_half_open_client(sdk: Sdk) -> None:
    sdk.setup_answer = -1
    with pytest.raises(RuntimeError, match="setup failed with -1"):
        open_mooncake_driver(store_config(), read_blob)
    (store,) = sdk.stores
    assert len(store.calls_to("close")) == 1


def test_a_driver_that_cannot_start_closes_the_client_it_was_given(
    sdk: Sdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = threading.Thread.start

    def refuse_worker_threads(thread: threading.Thread) -> None:
        if thread.name.startswith("mooncake-driver-"):
            raise RuntimeError("can't start new thread")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_worker_threads)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        open_mooncake_driver(store_config(), read_blob)
    (store,) = sdk.stores
    assert store.calls_to("setup") and len(store.calls_to("close")) == 1


# Layering.


BLOB_PACKAGE = blob_package.__name__
DISAGGREGATION = BLOB_PACKAGE.rsplit(".", 1)[0]
THIRD_PARTY_ALLOWED = {"numpy", "yaml"}


def _imports(path: pathlib.Path) -> list[tuple[str, str | None]]:
    """Every module ``path`` imports, absolute, with the function it is imported in (or None)."""
    found: list[tuple[str, str | None]] = []

    class Visitor(ast.NodeVisitor):
        function: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            outer, self.function = self.function, node.name
            self.generic_visit(node)
            self.function = outer

        def visit_Import(self, node: ast.Import) -> None:
            found.extend((alias.name, self.function) for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            if node.level:  # relative to the blob package
                anchor = BLOB_PACKAGE.rsplit(".", node.level - 1)[0]
                if not module:  # "from . import a, b" imports the modules a and b
                    found.extend((f"{anchor}.{alias.name}", self.function) for alias in node.names)
                    return
                module = f"{anchor}.{module}"
            found.append((module, self.function))

    Visitor().visit(ast.parse(path.read_text()))
    return found


def test_blob_modules_import_only_their_layer() -> None:
    root = pathlib.Path(blob_package.__file__).parent
    modules = {path.stem: _imports(path) for path in sorted(root.glob("*.py"))}
    assert {"backend", "mooncake", "mooncake_setup", "store", "keys"} <= set(modules)
    sdk_imports = []
    for name, imports in modules.items():
        for module, function in imports:
            top = module.split(".")[0]
            where = f"blob/{name}.py imports {module}"
            if top == "mooncake":
                sdk_imports.append((name, function))
                continue
            assert top != "torch" and not module.startswith("tensorrt_llm._torch.pyexecutor"), where
            if top == "tensorrt_llm":
                assert module == "tensorrt_llm.logger" or module.startswith(DISAGGREGATION), where
            else:
                assert top in sys.stdlib_module_names or top in THIRD_PARTY_ALLOWED, where
    assert sdk_imports == [("mooncake_setup", "open_sdk")]

    def package_imports(name: str) -> set[str]:
        return {m for m, _ in modules[name] if m.startswith(BLOB_PACKAGE + ".")}

    assert package_imports("backend") == {f"{BLOB_PACKAGE}.keys", f"{BLOB_PACKAGE}.store"}
    assert package_imports("mooncake") == {f"{BLOB_PACKAGE}.store"}
    assert package_imports("mooncake_setup") == {f"{BLOB_PACKAGE}.mooncake"}
    memory_kind = f"{DISAGGREGATION}.base.memory"  # all the boundary shares with the contract
    for name in ("store", "keys"):
        for module, _ in modules[name]:
            top = module.split(".")[0]
            columns = name == "store" and top == "numpy"  # a RowBatch holds its rows as columns
            assert top in sys.stdlib_module_names or columns or module == memory_kind, name
