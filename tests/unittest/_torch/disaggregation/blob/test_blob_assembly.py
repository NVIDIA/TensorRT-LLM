# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store config, opening the Mooncake SDK, assembling a blob backend, and the package's layering.

The SDK is a fake module put into ``sys.modules``, the master probe and the route probe are
replaced, and the InfiniBand sysfs tree is a temporary directory, so nothing here needs a network,
a Mooncake install or a GPU.
"""

from __future__ import annotations

import ast
import gc
import pathlib
import sys
import types

import numpy as np
import pytest
import yaml
from cpu_fixtures import (
    WAIT_S,
    CpuRegion,
    CpuView,
    FakeLender,
    RollbackFailsDriver,
    Rows,
    WakeCounter,
    object_key,
)
from fake_mooncake import FakeMooncakeStore, FakePool
from memory_store import MemoryDriver, MemoryStore

import tensorrt_llm._torch.disaggregation.blob as blob_package
from tensorrt_llm._torch.disaggregation.blob import mooncake_setup
from tensorrt_llm._torch.disaggregation.blob.assembly import BlobAssembly
from tensorrt_llm._torch.disaggregation.blob.backend import ConstructionRejected
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.mooncake_setup import (
    MooncakeConfig,
    load_store_config,
    open_mooncake_store,
    open_sdk,
)

pytestmark = pytest.mark.cpu_only

SLOT = 8
MASTER = "10.0.0.1:50051"
ROUTED_ADDRESS = "10.0.0.7"


def store_config(protocol: str = "tcp", **mooncake) -> dict:
    return {
        "blob": {"key_prefix": "t", "rows_per_batch": 4},
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
    monkeypatch.setattr(mooncake_setup, "_wait_for_master", sdk.masters.append)
    monkeypatch.setattr(mooncake_setup, "_local_address", lambda master: ROUTED_ADDRESS)
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(tmp_path / "no-infiniband"))
    monkeypatch.delenv("MC_TCP_PROTO", raising=False)
    return sdk


def ib_device(root: pathlib.Path, name: str, link_layer: str, state: str, rate: str) -> None:
    port = root / name / "ports" / "1"
    port.mkdir(parents=True)
    (port / "link_layer").write_text(f"{link_layer}\n")
    (port / "state").write_text(f"{state}\n")
    (port / "rate").write_text(f"{rate}\n")


def round_trip(backend, region: CpuRegion, wake: WakeCounter) -> None:
    """Publishes slot 0 and fetches it back into slot 1."""
    region.write(0, b"aaaaaaaa")
    publish = CpuView([Rows(0, object_key("A")[None, :], _slots(region, 0), 0)])
    published = backend.publish(publish, None)
    wake.wait_for(lambda: published.quiesced() and published.poll() is not None, "the publish")
    assert published.poll().delivered_masks[0].tolist() == [True]
    fetch = CpuView([Rows(0, object_key("A")[None, :], _slots(region, 1), 0)])
    fetched = backend.fetch(fetch, None)
    wake.wait_for(lambda: fetched.quiesced() and fetched.poll() is not None, "the fetch")
    assert region.read(1) == b"aaaaaaaa"


def _slots(region: CpuRegion, *indexes: int) -> np.ndarray:
    return np.array([region.slot(i) for i in indexes], dtype=np.int64)


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
    "blob.rows_per_bach=4": ("blob", "rows_per_bach", 4),  # an unknown key
    "mooncake.devce_name=mlx5_0": ("mooncake", "devce_name", "mlx5_0"),
    "mooncake.protocol=rdmaa": ("mooncake", "protocol", "rdmaa"),
    "mooncake.global_segment_size=True": ("mooncake", "global_segment_size", True),
    "blob.rows_per_batch=64.0": ("blob", "rows_per_batch", 64.0),
    "blob.bytes_per_batch=1.28e8": ("blob", "bytes_per_batch", 1.28e8),
    "blob.key_prefix=7": ("blob", "key_prefix", 7),
    "blob.workers=2.5": ("blob", "workers", 2.5),
    "blob.workers=True": ("blob", "workers", True),
    "mooncake.global_segment_size=1.0": ("mooncake", "global_segment_size", 1.0),
    "mooncake.local_buffer_size=-1": ("mooncake", "local_buffer_size", -1),
    "mooncake.master_server_address=5": ("mooncake", "master_server_address", 5),
    "mooncake.device_name=None": ("mooncake", "device_name", None),
}


@pytest.mark.parametrize("case", BAD_VALUES)
def test_unknown_keys_and_values_of_the_wrong_type_are_rejected_before_the_hold(
    sdk: Sdk, case: str
) -> None:
    section, name, value = BAD_VALUES[case]
    config = store_config()
    config[section][name] = value
    lender = FakeLender()
    with pytest.raises(ConstructionRejected, match=name) as rejected:
        BlobAssembly(lender, open_mooncake_store)((CpuRegion(SLOT, 2),), WakeCounter(), config)
    assert lender.holds == []
    if section == "mooncake":
        assert "section 'mooncake'" in str(rejected.value)
        assert sdk.stores == []  # refused before the SDK was set up


def test_store_config_reads_a_yaml_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "store.yaml"
    path.write_text(yaml.safe_dump(store_config(local_buffer_size=0)))
    blob, mooncake = load_store_config(path)
    assert blob == {"key_prefix": "t", "rows_per_batch": 4}
    assert mooncake == MooncakeConfig(MASTER, 0, protocol="tcp", local_buffer_size=0)


# Opening the SDK.


def test_open_passes_setup_arguments_in_order(
    sdk: Sdk, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sysfs = tmp_path / "infiniband"
    ib_device(sysfs, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(sysfs, "mlx5_1", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    monkeypatch.setattr(mooncake_setup, "IB_SYSFS_ROOT", str(sysfs))
    driver, blob = open_mooncake_store(store_config("rdma"))
    (store,) = sdk.stores
    (setup,) = store.calls_to("setup")
    assert setup.args == (
        ROUTED_ADDRESS,
        "P2PHANDSHAKE",
        0,
        16 << 20,
        "rdma",
        "mlx5_0,mlx5_1",
        MASTER,
    )
    assert setup.kwargs == {}
    assert sdk.masters == [MASTER]
    assert isinstance(driver, MooncakeDriver)
    assert blob == {"key_prefix": "t", "rows_per_batch": 4}

    open_sdk(MooncakeConfig(MASTER, 1 << 30, "tcp", "eth9", "10.9.9.9", "etcd://m:2379", 0))
    assert sdk.stores[1].calls_to("setup")[0].args == (
        "10.9.9.9",
        "etcd://m:2379",
        1 << 30,
        0,
        "tcp",
        "eth9",
        MASTER,
    )


def test_device_detection_picks_the_fastest_active_ib_ports(tmp_path: pathlib.Path) -> None:
    ib_device(tmp_path, "mlx5_0", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_1", "InfiniBand", "4: ACTIVE", "200 Gb/sec (4X HDR)")
    ib_device(tmp_path, "mlx5_2", "Ethernet", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_3", "InfiniBand", "1: DOWN", "400 Gb/sec (4X NDR)")
    ib_device(tmp_path, "mlx5_4", "InfiniBand", "4: ACTIVE", "400 Gb/sec (4X NDR)")
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path)) == ["mlx5_0", "mlx5_4"]
    assert mooncake_setup._fastest_active_ib_devices(str(tmp_path / "missing")) == []


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
    lender = FakeLender()
    with pytest.raises(ConstructionRejected, match="mooncake-transfer-engine-cuda13"):
        BlobAssembly(lender, open_mooncake_store)((), WakeCounter(), store_config())
    assert lender.holds == []


def test_tcp_legacy_framing_is_refused(sdk: Sdk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_TCP_PROTO", "1")
    lender = FakeLender()
    with pytest.raises(ConstructionRejected, match="MC_TCP_PROTO"):
        BlobAssembly(lender, open_mooncake_store)((), WakeCounter(), store_config("tcp"))
    assert sdk.stores == [] and lender.holds == []
    open_sdk(MooncakeConfig(MASTER, 0, "rdma", device_name="mlx5_0"))  # RDMA does not frame
    assert len(sdk.stores) == 1


def test_setup_failure_closes_the_half_open_client(sdk: Sdk) -> None:
    sdk.setup_answer = -1
    lender = FakeLender()
    with pytest.raises(ConstructionRejected, match="setup failed with -1"):
        BlobAssembly(lender, open_mooncake_store)((), WakeCounter(), store_config())
    (store,) = sdk.stores
    assert len(store.calls_to("close")) == 1
    assert lender.holds == []


# Assembly.


def memory_assembly(lender: FakeLender, driver: MemoryDriver) -> BlobAssembly:
    return BlobAssembly(lender, lambda config: (driver, {"key_prefix": "t"}))


@pytest.mark.parametrize(
    "failure",
    [
        "open_failed",
        "register_rejected",
        "register_overlaps",
        "register_escaped",
        "rollback_unregister_failed",
    ],
)
def test_construction_failure_releases_the_hold_only_when_nothing_escaped(failure: str) -> None:
    lender = FakeLender()
    regions = (CpuRegion(SLOT, 2), CpuRegion(SLOT, 2))
    if failure == "open_failed":

        def open_driver(config):
            raise RuntimeError("master down")

        with pytest.raises(ConstructionRejected, match="master down"):
            BlobAssembly(lender, open_driver)(regions, WakeCounter(), None)
        assert lender.holds == []
        return

    driver = RollbackFailsDriver(MemoryStore()) if failure.startswith("rollback") else None
    driver = driver or MemoryDriver(MemoryStore())
    if failure == "register_overlaps":  # the driver refuses the second region: DriverRejected
        regions = (regions[0], regions[0])
    elif failure != "rollback_unregister_failed":
        driver.fail_next_register("no memlock", escaped=failure == "register_escaped")
    assembly = memory_assembly(lender, driver)
    with pytest.raises(RuntimeError) as raised:
        assembly(regions, WakeCounter(), None)
    (hold,) = lender.holds
    if failure in ("register_rejected", "register_overlaps"):
        assert isinstance(raised.value, ConstructionRejected)
        assert hold.released and driver.live_registrations() == []
        if failure == "register_overlaps":
            return
        # Nothing was kept, so the same assembly may try again under a new hold.
        backend = assembly(regions, WakeCounter(), None)
        assert len(lender.holds) == 2
        assert backend.shutdown(WAIT_S) == frozenset()
        assembly.finalize()
        assert lender.holds[1].released
        return
    assert not isinstance(raised.value, ConstructionRejected)
    assert not hold.released and driver.live_registrations()
    finalize = assembly.finalize
    del assembly
    gc.collect()
    # Only ``finalize`` still reaches the binding, which still refuses to let the hold go.
    with pytest.raises(RuntimeError, match="region:0"):
        finalize()
    assert not hold.released


def test_finalize_releases_the_hold_only_after_shutdown_and_never_closes_the_sdk(sdk: Sdk) -> None:
    lender = FakeLender()
    assembly = BlobAssembly(lender, open_mooncake_store)
    region, wake = CpuRegion(SLOT, 2), WakeCounter()
    backend = assembly((region,), wake, store_config())
    round_trip(backend, region, wake)
    with pytest.raises(RuntimeError, match="region:0"):
        assembly.finalize()  # still registered: the hold stays
    (hold,) = lender.holds
    assert not hold.released
    assert backend.shutdown(WAIT_S) == frozenset()
    assembly.finalize()
    assert hold.released
    (store,) = sdk.stores
    assert store.calls_to("close") == []
    assert len(store.calls_to("unregister_buffer")) == 1


def test_a_second_call_is_rejected_and_keeps_the_first_binding() -> None:
    lender = FakeLender()
    driver = MemoryDriver(MemoryStore())
    assembly = memory_assembly(lender, driver)
    region, wake = CpuRegion(SLOT, 2), WakeCounter()
    backend = assembly((region,), wake, None)
    with pytest.raises(ConstructionRejected, match="already"):
        assembly((CpuRegion(SLOT, 2),), WakeCounter(), None)
    assert len(lender.holds) == 1 and len(driver.live_registrations()) == 1
    round_trip(backend, region, wake)
    assert backend.shutdown(WAIT_S) == frozenset()
    assembly.finalize()
    assert lender.holds[0].released


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
                module = f"{anchor}.{module}" if module else anchor
            found.append((module, self.function))

    Visitor().visit(ast.parse(path.read_text()))
    return found


def test_blob_modules_import_only_their_layer() -> None:
    root = pathlib.Path(blob_package.__file__).parent
    modules = {path.stem: _imports(path) for path in sorted(root.glob("*.py"))}
    assert {"backend", "mooncake", "mooncake_setup", "assembly", "store", "keys"} <= set(modules)
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

    assert package_imports("mooncake") == {f"{BLOB_PACKAGE}.store"}
    assert package_imports("mooncake_setup") == {f"{BLOB_PACKAGE}.mooncake"}
    for name in ("store", "keys"):
        assert all(m.split(".")[0] in sys.stdlib_module_names for m, _ in modules[name]), name
