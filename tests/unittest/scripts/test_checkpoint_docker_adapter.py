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
"""Exercise the real local host transaction without Docker, a GPU or privilege."""

import copy
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[3] / "examples" / "serve" / "checkpoint"
sys.path.insert(0, str(HERE))
try:
    import deployment
    import docker_adapter as adapter
    from docker_control_probe import verify_listener
    from engine_client import _sign
finally:
    sys.path.remove(str(HERE))

pytestmark = pytest.mark.cpu_only
TEMPLATE, ATTEMPT = "a" * 32, "b" * 32
SOURCE, TARGET, HELPER = "1" * 64, "2" * 64, "3" * 64
IMAGE, DRIVER = "sha256:" + "4" * 64, "sha256:" + "5" * 64
GPU = "GPU-12345678-1234-4321-abcd-0123456789ab"
IDENTITY = {
    "source_session_id": "engine-source",
    "template_id": TEMPLATE,
    "profile_digest": "6" * 64,
    "cohort_digest": "7" * 64,
}
HEADERS = _sign("test-key", "GET", "/checkpoint/status", b"")
LIBRARIES = {
    "libcuinterpose.so": {"sha256": "e" * 64},
    "libcuinterpose_core.so": {"sha256": "f" * 64},
}


def status(serving: bool) -> dict:
    session = ATTEMPT if serving else IDENTITY["source_session_id"]
    binding = {k: v for k, v in IDENTITY.items() if k != "source_session_id"}
    profile = {k: binding[k] for k in ("profile_digest", "cohort_digest")}
    rank = {
        "rank": 0,
        "session_id": session,
        "state_digest": "d" * 64,
        **profile,
        "checks": ["activation_committed"]
        if serving
        else ["admission_closed", "scheduler_drained", "no_external_exposure"],
    }
    return {
        "stage": "serving" if serving else "capture_safe",
        "state_version": 7,
        "operation": {
            **binding,
            "session_id": session,
            "state_version": 7,
            "operation_id": "f" * 32,
            "rank_results": [rank],
            "status": "completed",
            "cleanup_status": "not_requested",
            "readiness_id": "e" * 64 if serving else None,
            "native_prepare_not_dispatched": False,
            "error": None,
        },
        "readiness": {
            **{
                k: serving
                for k in [
                    "admission_open",
                    "memory_ready",
                    "serving_ready",
                    "runtime_ready",
                    "validation_passed",
                ]
            },
            "capture_safe": not serving,
        },
        "remaining_seconds": None if serving else 100,
        "source_session_id": IDENTITY["source_session_id"],
        "restore_session_id": ATTEMPT if serving else None,
        "profile": {
            **profile,
            "ranks": [0],
            "drain_policies": ["natural"],
            "capture_policies": ["resident"],
        },
    }


class ExternalHost:
    """Only the external Docker command, health RPC and kernel boot ID are substituted."""

    def __init__(self, root: Path, patch: pytest.MonkeyPatch) -> None:
        self.cfg = deployment.Deployment(
            root,
            GPU,
            DRIVER,
            Path("/var/lib/docker"),
            "0" * 32,
            {"interpose/" + k: v["sha256"] for k, v in LIBRARIES.items()},
        )
        for d in ["warp-state", "controls", "controls/" + TEMPLATE, "checkpoints", "interpose"]:
            (root / d).mkdir(mode=0o700)
        boot = root / "boot_id"
        boot.write_text("host-boot")
        patch.setattr(adapter, "BOOT_ID", boot)
        patch.setattr(adapter, "geteuid", lambda: 0)
        patch.setattr(adapter, "_run", self.run)
        patch.setattr(adapter, "_docker_exchange", self.exchange)
        patch.setattr(deployment, "check_health", self.health)
        self.calls = []
        self.objects = {SOURCE: self.obj(SOURCE, True)}
        self.live = {SOURCE: status(False), TARGET: status(True)}
        self.helper_updates = {}
        self.terminate = True
        self.top = b"PID\n120"
        self.uncertain = False
        self.ready = True
        self.healthy = True
        self.artifact = root / "checkpoints" / "artifacts" / TEMPLATE / "containers" / "main"

    def obj(self, cid: str, source: bool) -> dict:
        owner = TEMPLATE if source else ATTEMPT
        return {
            "id": cid,
            "image": IMAGE,
            "labels": {adapter.LABEL: owner},
            "networks": {"none": {}},
            "healthcheck": None,
            "user": "",
            "state": {
                "Running": True,
                "Paused": False,
                "Restarting": False,
                "StartedAt": "start",
                "Status": "running",
                "Pid": 110 if source else 120,
            },
            "host": {
                "NetworkMode": "none",
                "PidMode": "",
                "IpcMode": "private",
                "Privileged": False,
                "RestartPolicy": {"Name": "no"},
                "AutoRemove": False,
                "PortBindings": {},
                "PublishAllPorts": False,
                "CapAdd": [],
                "Devices": [],
                "VolumesFrom": [],
                "SecurityOpt": None,
                "NanoCpus": 16_000_000_000,
                "CpuQuota": 0,
                "CpuPeriod": 0,
                "ShmSize": 1 << 30,
                "Memory": 128 << 30,
                "MemorySwap": 128 << 30,
                "DeviceRequests": [{"DeviceIDs": [GPU], "Count": 0, "Capabilities": [["gpu"]]}],
            },
            "mounts": [
                {
                    "Type": "bind",
                    "Source": str(self.cfg.root / src),
                    "Destination": dst,
                    "RW": rw,
                    "Propagation": "rprivate",
                }
                for src, dst, rw in [
                    ("controls/" + owner, "/snapshot-control", True),
                    ("interpose", "/tmp/snapshot-cuda", False),
                ]
            ],
            "path": adapter.LAUNCHER if source else "/bin/sleep",
            "args": ["--library", "/tmp/snapshot-cuda/libcuinterpose.so", "--", "trtllm-serve"]
            if source
            else ["infinity"],
        }

    def health(self, cfg: deployment.Deployment, *, deadline: float) -> dict:
        assert cfg == self.cfg and deadline > time.monotonic()
        if not self.healthy:
            raise ValueError("broker unavailable")
        return {"healthy": True}

    def exchange(
        self,
        container: str,
        port: int,
        path: str,
        deadline: float,
        request: Callable[[], tuple[dict, bytes]],
    ) -> bytes:
        assert port == 8000 and deadline > time.monotonic()
        headers, body = request()
        assert body == b""
        if path == "/ready":
            assert not headers
            if not self.ready:
                raise ValueError("not ready")
            return b""
        assert path == "/checkpoint/status" and headers == HEADERS
        return json.dumps(self.live[container]).encode()

    def run(self, argv: list[str], deadline: float, data: bytes = b"") -> bytes:
        assert tuple(argv[:2]) == deployment.DOCKER and deadline > time.monotonic()
        cmd = argv[2:]
        self.calls.append(cmd)
        if cmd[:2] == ["container", "inspect"]:
            assert cmd[-2] == adapter._INSPECT
            return json.dumps(self.objects[cmd[-1]]).encode()
        if cmd[0] == "exec":
            assert adapter._CONTROL in cmd
            assert cmd[1:3] == ["--env", "TRTLLM_RUNTIME_CONTROL_API_KEY="]
            owner = self.objects[cmd[3]]["labels"][adapter.LABEL]
            st = (self.cfg.root / "controls" / owner).stat()
            return json.dumps([st.st_dev, st.st_ino, "/snapshot-control", GPU]).encode()
        if cmd[0] == "create" and cmd[-1] == DRIVER:
            assert "--privileged" in cmd
            owner = next(x.split("=")[1] for x in cmd if x.startswith(adapter.LABEL + "="))
            self.objects[HELPER] = {
                "id": HELPER,
                "image": DRIVER,
                "labels": {adapter.LABEL: owner},
                "state": {"Status": "created", "Running": False},
            }
            return HELPER.encode()
        if cmd[0] == "create":
            assert cmd[-3:] == ["/bin/sleep", IMAGE, "infinity"] and "--privileged" not in cmd
            assert "--network=none" in cmd and f"{adapter.LABEL}={ATTEMPT}" in cmd
            self.objects[TARGET] = self.obj(TARGET, False)
            return TARGET.encode()
        if cmd == ["start", TARGET]:
            return TARGET.encode()
        if cmd == ["top", TARGET, "-eo", "pid"]:
            return self.top
        if cmd == ["start", "--attach", "--interactive", HELPER]:
            req = json.loads(data)
            assert set(req) == {
                "operation",
                "attempt",
                "artifact",
                "container",
                "image",
                "interpose",
                "containerd_socket",
                "timeout_seconds",
                "deadline_monotonic",
            }
            assert (
                req["containerd_socket"] == deployment.CONTAINERD_SOCKET and req["image"] == IMAGE
            )
            assert req["timeout_seconds"] > 0 and req["interpose"] is True
            assert req["deadline_monotonic"] == deadline
            cid = req["container"]
            op = req["operation"]
            self.objects[HELPER]["state"] = {
                "Status": "exited",
                "Running": False,
                "ExitCode": 0,
                "OOMKilled": False,
            }
            if self.uncertain:
                self.objects[HELPER]["state"].update(Status="running", Running=True)
                raise TimeoutError("lost helper reply")
            result = {
                "snapshot_revision": deployment.SNAPSHOT_REVISION,
                "status": "PASS",
                "operation": op,
                "attempt": req["attempt"],
                "artifact": TEMPLATE,
                "container": cid,
                "host_pid": self.objects[cid]["state"]["Pid"],
            }
            if op == "capture":
                assert (self.cfg.root / "controls" / TEMPLATE / "ready-for-snapshot").is_file()
                if self.terminate:
                    self.objects[SOURCE]["state"].update(Status="exited", Running=False, Pid=0)
                self.artifact.mkdir(parents=True)
                (self.artifact / "inventory.img").write_bytes(b"captured bytes")
                (self.artifact / "manifest.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "artifact": {"contentUID": TEMPLATE, "containerName": "main"},
                            "k8s": {"containerId": SOURCE, "imageId": IMAGE},
                            "cuinterpose": {"pids": [2], "libraries": LIBRARIES},
                            "cudaRestore": {"sourceGpuUuids": [GPU], "pids": [2]},
                        }
                    )
                )
                result.update(
                    source_terminated=True,
                    legacy_artifact_path=f"/checkpoints/artifacts/{TEMPLATE}/containers/main",
                )
            if op == "restore":
                result.update(restored_pid=22, memory_ready=True)
            return json.dumps({**result, **self.helper_updates}).encode()
        if cmd == ["rm", HELPER]:
            del self.objects[HELPER]
            return b""
        if cmd[0] == "kill":
            self.objects[cmd[1]]["state"].update(Status="exited", Running=False, Pid=0)
            return b""
        raise AssertionError(cmd)

    def call(self, action: str, attempt: str | None = None, **fields: object) -> dict:
        attempt = attempt or (TEMPLATE if action in ("bind_source", "capture") else ATTEMPT)
        return adapter.dispatch(
            self.cfg,
            {
                "version": 1,
                "action": action,
                "attempt": attempt,
                "template": TEMPLATE,
                "deadline_monotonic": time.monotonic() + 200,
                "input": fields,
            },
        )["result"]

    def capture(self) -> dict:
        assert self.call("bind_source", source=SOURCE) == {
            "source": SOURCE,
            "control_container": SOURCE,
            "control_port": 8000,
        }
        return self.call("capture", source=SOURCE, identity=IDENTITY, status_headers=HEADERS)

    def restore(self, receipt: dict) -> dict:
        return self.call(
            "restore", source=SOURCE, identity=IDENTITY, artifact_digest=receipt["artifact_digest"]
        )


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExternalHost:
    return ExternalHost(tmp_path, monkeypatch)


def test_private_lifecycle_and_publication(host: ExternalHost) -> None:
    captured = host.capture()
    assert captured["source_terminated"] and len(captured["artifact_digest"]) == 64
    restored = host.restore(captured)
    assert restored["authority_established"] and restored["isolated"]
    assert restored["control_container"] == restored["target"] == TARGET
    assert restored["control_port"] == 8000 and restored["host_result_id"] == ATTEMPT
    with pytest.raises(FileNotFoundError):
        host.call("resolve", target=TARGET)
    host.ready = False
    with pytest.raises(ValueError, match="not ready"):
        host.call("publish", target=TARGET, status_headers=HEADERS)
    host.ready = True
    host.live[TARGET]["operation"]["session_id"] = "other"
    with pytest.raises(RuntimeError, match="stale checkpoint"):
        host.call("publish", target=TARGET, status_headers=HEADERS)
    host.live[TARGET] = status(True)
    assert host.call("publish", target=TARGET, status_headers=HEADERS)["published"]
    assert host.call("resolve", target=TARGET)["control_container"] == TARGET
    with pytest.raises(ValueError, match="published"):
        host.call("discard")
    host.objects[TARGET]["state"]["StartedAt"] = "restarted"
    with pytest.raises(ValueError, match="not bound"):
        host.call("resolve", target=TARGET)


@pytest.mark.parametrize(
    "fault", ["broker", "owner", "network", "port", "control", "mount", "gpu", "helper"]
)
def test_preflight_before_source_quiescence(host: ExternalHost, fault: str) -> None:
    obj = host.objects[SOURCE]
    if fault == "broker":
        host.healthy = False
    elif fault == "owner":
        obj["labels"][adapter.LABEL] = ATTEMPT
    elif fault == "network":
        obj["host"]["NetworkMode"] = "host"
    elif fault == "port":
        obj["host"]["PortBindings"] = {"8000/tcp": [{"HostPort": "8000"}]}
    elif fault == "control":
        (host.cfg.root / "controls" / TEMPLATE / "restore-complete").touch()
    elif fault == "mount":
        obj["mounts"][0]["Source"] = "/etc"
    elif fault == "gpu":
        obj["host"]["DeviceRequests"][0]["DeviceIDs"].append("other")
    elif fault == "helper":
        host.helper_updates["snapshot_revision"] = "stale"
    with pytest.raises((ValueError, RuntimeError)):
        host.call("bind_source", source=SOURCE)
    assert (
        obj["state"]["Running"]
        and not (host.cfg.root / "controls" / TEMPLATE / "ready-for-snapshot").exists()
    )
    assert not (host.cfg.root / "warp-state" / TEMPLATE / "binding.json").exists()


def test_capture_rechecks_live_proof_before_marker(host: ExternalHost) -> None:
    host.call("bind_source", source=SOURCE)
    host.live[SOURCE]["operation"]["template_id"] = ATTEMPT
    with pytest.raises(RuntimeError, match="stale checkpoint"):
        host.call("capture", source=SOURCE, identity=IDENTITY, status_headers=HEADERS)
    assert not (host.cfg.root / "controls" / TEMPLATE / "ready-for-snapshot").exists()
    assert host.call("discard", TEMPLATE)["contained"]
    assert not host.objects[SOURCE]["state"]["Running"]


def test_refuses_unreadable_root_owned_artifacts_before_prepare(
    host: ExternalHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "geteuid", lambda: 1000)
    with pytest.raises(ValueError, match="host-root operator"):
        host.call("bind_source", source=SOURCE)
    assert not host.calls


def test_driver_pass_does_not_prove_termination(host: ExternalHost) -> None:
    host.terminate = False
    with pytest.raises(ValueError, match="process state"):
        host.capture()
    assert not (host.cfg.root / "warp-state" / TEMPLATE / "artifact.json").exists()


@pytest.mark.parametrize("fault", ["bytes", "symlink", "identity", "digest"])
def test_restore_rejects_modified_artifact_before_create(host: ExternalHost, fault: str) -> None:
    receipt = host.capture()
    identity = copy.copy(IDENTITY)
    if fault == "bytes":
        (host.artifact / "inventory.img").write_bytes(b"modified bytes")
    elif fault == "symlink":
        (host.artifact / "escape").symlink_to(host.cfg.root)
    elif fault == "identity":
        identity["profile_digest"] = "a" * 64
    else:
        receipt["artifact_digest"] = "b" * 64
    with pytest.raises(ValueError):
        host.call(
            "restore", source=SOURCE, identity=identity, artifact_digest=receipt["artifact_digest"]
        )
    assert TARGET not in host.objects


def test_reject_non_inert_target_and_contain_only_owned_target(host: ExternalHost) -> None:
    captured = host.capture()
    host.top += b"\n132"
    with pytest.raises(ValueError, match="placeholder"):
        host.restore(captured)
    host.objects[TARGET]["labels"][adapter.LABEL] = TEMPLATE
    with pytest.raises(ValueError, match="owned"):
        host.call("discard")
    assert host.objects[TARGET]["state"]["Running"]
    host.objects[TARGET]["labels"][adapter.LABEL] = ATTEMPT
    assert host.call("discard") == {"contained": True}
    assert host.call("discard") == {"contained": True}
    assert not host.objects[TARGET]["state"]["Running"]
    assert (host.cfg.root / "warp-state" / TEMPLATE / "artifact.json").exists()
    with pytest.raises(ValueError, match="discarded"):
        host.restore(captured)


def test_late_helper_does_not_imply_containment(host: ExternalHost) -> None:
    captured = host.capture()
    host.uncertain = True
    with pytest.raises(ValueError, match="exit unconfirmed"):
        host.restore(captured)
    with pytest.raises(ValueError, match="outstanding helper"):
        host.call("discard")
    assert not (host.cfg.root / "warp-state" / ATTEMPT / "discarded.json").exists()
    host.objects[HELPER]["state"].update(Status="exited", Running=False)
    assert host.call("discard")["contained"]
    assert not host.objects[TARGET]["state"]["Running"] and HELPER not in host.objects


def test_refuses_arbitrary_target_and_input_before_docker(host: ExternalHost) -> None:
    with pytest.raises(ValueError, match="digest"):
        host.call("bind_source", source="tcp://remote:2375")
    with pytest.raises(ValueError, match="fields"):
        host.call("bind_source", source=SOURCE, containerd_socket="/tmp/containerd.sock")
    assert not host.calls


def test_custom_storage_requires_real_device_bytes(host: ExternalHost) -> None:
    host.capture()
    path = host.artifact / "manifest.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["cudaRestore"]["customStorage"] = True
    path.write_text(yaml.safe_dump(manifest))
    device = host.artifact / "device-0.bin"
    device.touch()
    args = (host.cfg, TEMPLATE, SOURCE, IMAGE, time.monotonic() + 10)
    with pytest.raises(ValueError, match="GPU CustomStorage"):
        adapter._artifact(*args)
    device.write_bytes(b"stored GPU bytes")
    original = adapter._artifact(*args)
    device.write_bytes(b"different GPU bytes")
    assert adapter._artifact(*args) != original


def test_socket_must_belong_to_restored_tree(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    row = "0: 0100007F:1F40 00000000:0000 0A 0:0 0:0 0 0 0 231\n"
    (net / "tcp").write_text("header\n" + row)
    for pid, ppid in [(1, 0), (2, 1), (3, 2)]:
        task = tmp_path / str(pid)
        (task / "fd").mkdir(parents=True)
        (task / "stat").write_text(f"{pid} (worker (GPU)) S {ppid} 0 0")
    (tmp_path / "3/fd/5").symlink_to("socket:[231]")
    verify_listener(tmp_path, 8000, 2)
    (tmp_path / "1/fd/6").symlink_to("socket:[231]")
    with pytest.raises(ValueError, match="not owned"):
        verify_listener(tmp_path, 8000, 2)
    (net / "tcp").write_text("header\n" + row.replace("0100007F", "00000000"))
    with pytest.raises(ValueError, match="loopback"):
        verify_listener(tmp_path, 8000, None)
