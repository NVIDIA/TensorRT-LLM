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
"""Operator-only, local Docker/Snapshot backend for one resident TP1 worker."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from os import geteuid
from pathlib import Path

import deployment
import yaml
from coordinator import _check_driver, _private_dir, _remaining, _save, _sync_dir
from engine_client import _FLAGS, _READY, _check_result, _docker_exchange

PORT = 8000
LABEL = "com.nvidia.trtllm.snapshot.attempt"
LAUNCHER = "/tmp/snapshot-cuda/cuinterpose-launch"
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_ID = re.compile("[0-9a-f]{64}")
_ATTEMPT = re.compile("[0-9a-f]{32}")
_IMAGE = re.compile("sha256:[0-9a-f]{64}")
_FIELDS = {
    "bind_source": {"source"},
    "capture": {"source", "identity", "status_headers"},
    "restore": {"source", "identity", "artifact_digest"},
    "publish": {"target", "status_headers"},
    "discard": set(),
    "resolve": {"target"},
}
# Do not export workload environment variables (including its HMAC key).
_INSPECT = (
    '{"id":{{json .Id}},"image":{{json .Image}},"labels":{{json .Config.Labels}},'
    '"host":{{json .HostConfig}},"mounts":{{json .Mounts}},"state":{{json .State}},'
    '"path":{{json .Path}},"args":{{json .Args}},"networks":{{json .NetworkSettings.Networks}},'
    '"user":{{json (index .Config "User")}},"healthcheck":{{json (index .Config "Healthcheck")}}}'
)
_CONTROL = Path(__file__).with_name("docker_control_probe.py").read_text(encoding="ascii")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _matches(pattern: re.Pattern, value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _json(raw: bytes) -> dict:
    def unique(pairs: list) -> dict:
        result = dict(pairs)
        _require(len(result) == len(pairs), "Duplicate JSON key")
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    _require(isinstance(value, dict), "Expected a JSON object")
    return value


def _run(args: list[str], deadline: float, data: bytes = b"") -> bytes:
    _require(tuple(args[:2]) == deployment.DOCKER, "Expected the fixed local Docker daemon")
    with tempfile.TemporaryDirectory() as config:
        result = subprocess.run(
            [*deployment.DOCKER, "--config", config, *args[2:]],
            input=data,
            stdout=subprocess.PIPE,
            check=True,
            timeout=_remaining(deadline),
            env={"PATH": os.defpath, "HOME": config},
        )
    _remaining(deadline)
    return result.stdout


def _artifact(
    config: deployment.Deployment, template: str, source: str, image: str, deadline: float
) -> str:
    """Verify the S2 manifest and hash all on-disk payloads of the local store."""
    tree = config.root / "checkpoints" / "artifacts" / template / "containers" / "main"
    _require(tree.resolve(strict=True) == tree, "Linked artifact path")
    digest = hashlib.sha256()
    seen = set()
    manifest = {}

    def walk(directory: Path) -> None:
        nonlocal manifest
        for path in sorted(directory.iterdir()):
            _remaining(deadline)
            info = path.lstat()
            name = path.relative_to(tree).as_posix()
            meta = [name, info.st_mode, info.st_uid, info.st_gid]
            if stat.S_ISDIR(info.st_mode):
                digest.update(json.dumps(meta).encode() + b"\n")
                walk(path)
                continue
            _require(
                stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Nonregular or linked artifact"
            )
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(fd)
                _require(
                    opened.st_dev == info.st_dev and opened.st_ino == info.st_ino,
                    "Artifact replaced",
                )
                content = hashlib.sha256()
                if name == "manifest.yaml":
                    _require(info.st_size <= 4 << 20, "Oversized Snapshot manifest")
                    block = stream.read((4 << 20) + 1)
                    manifest = yaml.safe_load(block)
                    content.update(block)
                else:
                    while block := stream.read(4 << 20):
                        _remaining(deadline)
                        content.update(block)
                end = os.fstat(stream.fileno())
            _require(
                (info.st_dev, info.st_ino, info.st_ctime_ns)
                == (end.st_dev, end.st_ino, end.st_ctime_ns),
                "Artifact changed during inspection",
            )
            digest.update(json.dumps([*meta, info.st_size, content.hexdigest()]).encode() + b"\n")
            if info.st_size:
                seen.add(name)

    walk(tree)
    _require({"manifest.yaml", "inventory.img"} <= seen, "Incomplete Snapshot artifact")
    custom = manifest["cudaRestore"].get("customStorage", False)
    _require(
        type(custom) is bool
        and (
            not custom or any(re.fullmatch(r"device-[0-9]+\.bin", Path(name).name) for name in seen)
        ),
        "Missing on-disk GPU CustomStorage payload",
    )
    libraries = manifest["cuinterpose"]["libraries"]
    _require(
        manifest["artifact"] == {"contentUID": template, "containerName": "main"}
        and manifest["k8s"]["containerId"] == source
        and manifest["k8s"]["imageId"] == image
        and manifest["cudaRestore"]["sourceGpuUuids"] == [config.gpu]
        and bool(manifest["cudaRestore"]["pids"])
        and all(
            libraries[name]["sha256"] == config.assets[f"interpose/{name}"]
            for name in ("libcuinterpose.so", "libcuinterpose_core.so")
        ),
        "Snapshot manifest does not describe this source, GPU and image",
    )
    return digest.hexdigest()


class Adapter:
    """One request on the local operator's private, serialized deployment."""

    def __init__(
        self, config: deployment.Deployment, attempt: str, template: str, deadline: float
    ) -> None:
        self.config = config
        self.attempt = attempt
        self.template = template
        self.deadline = deadline
        self.state = config.root / "warp-state"
        self.journal = self.state / attempt
        self.control = config.root / "controls" / attempt

    def docker(self, *args: str, data: bytes = b"") -> bytes:
        return _run([*deployment.DOCKER, *args], self.deadline, data)

    def inspect(self, container: str) -> dict:
        _require(_matches(_ID, container), "Expected a full immutable Docker ID")
        obj = _json(self.docker("container", "inspect", "--format", _INSPECT, container))
        _require(
            obj["id"] == container and _matches(_IMAGE, obj["image"]), "Docker identity changed"
        )
        return obj

    def profile(self, container: str, *, source: bool, running: bool = True) -> dict:
        obj = self.inspect(container)
        h, state = obj["host"], obj["state"]
        owner = self.template if source else self.attempt
        _require(obj["labels"].get(LABEL) == owner, "Not owned by this attempt")
        _require(
            h["NetworkMode"] == "none"
            and h["PidMode"] == ""
            and h["IpcMode"] == "private"
            and h["Privileged"] is False
            and h["RestartPolicy"]["Name"] == "no"
            and h["AutoRemove"] is False
            and not h["PortBindings"]
            and not h["PublishAllPorts"]
            and not h["CapAdd"]
            and not h["Devices"]
            and not h["VolumesFrom"]
            and not h["SecurityOpt"]
            and not h.get("Tmpfs")
            and not h.get("DeviceCgroupRules")
            and h.get("UTSMode", "") == ""
            and h.get("UsernsMode", "") == ""
            and set(obj["networks"]) == {"none"}
            and (not obj["healthcheck"] or obj["healthcheck"]["Test"] == ["NONE"]),
            "Unsupported workload isolation",
        )
        _require(
            h["NanoCpus"] == 16_000_000_000
            and h["CpuPeriod"] == h["CpuQuota"] == 0
            and h["ShmSize"] == 1 << 30
            and h["Memory"] == h["MemorySwap"] == 128 << 30,
            "Expected S2 workload resource limits",
        )
        devices = h["DeviceRequests"]
        _require(
            len(devices) == 1
            and devices[0]["DeviceIDs"] == [self.config.gpu]
            and devices[0]["Count"] == 0
            and devices[0]["Capabilities"] == [["gpu"]],
            "Expected one allocated GPU",
        )
        expected = {
            "/snapshot-control": (str(self.config.root / "controls" / owner), True),
            "/tmp/snapshot-cuda": (str(self.config.root / "interpose"), False),
        }
        mounts = obj["mounts"]
        _require(
            len(mounts) == len(expected)
            and all(m["Type"] == "bind" and m["Propagation"] == "rprivate" for m in mounts)
            and {m["Destination"]: (m["Source"], m["RW"]) for m in mounts} == expected,
            "Expected dedicated control and read-only interpose binds only",
        )
        _require(
            state["Running"] is running
            and not state["Restarting"]
            and not state["Paused"]
            and (
                (running and state["Status"] == "running" and state["Pid"] > 1)
                or (not running and state["Status"] == "exited" and state["Pid"] == 0)
            ),
            "Unexpected workload process state",
        )
        if source:
            _require(
                obj["path"] == LAUNCHER
                and obj["args"][:4]
                == ["--library", "/tmp/snapshot-cuda/libcuinterpose.so", "--", "trtllm-serve"],
                "Source must run the pinned interpose launcher",
            )
        else:
            _require(
                obj["path"] == "/bin/sleep" and obj["args"] == ["infinity"], "Not an inert target"
            )
        return obj

    def control_binding(
        self, container: str, pid: int | None = None, *, listener: bool = False
    ) -> list:
        _private_dir(self.control)
        # Corroborate the bind in the actual namespace; a host pathname alone is not identity.
        info = self.control.stat()
        actual = json.loads(
            self.docker(
                "exec",
                "--env",
                "TRTLLM_RUNTIME_CONTROL_API_KEY=",
                container,
                "python3",
                "-IS",
                "-c",
                _CONTROL,
                str(pid) if pid is not None else "",
                str(min(5, _remaining(self.deadline))) if listener else "",
                str(PORT),
            )
        )
        _require(
            actual == [info.st_dev, info.st_ino, "/snapshot-control", self.config.gpu],
            "Control mount or GPU identity mismatch",
        )
        return [info.st_dev, info.st_ino]

    def save(self, name: str, data: dict) -> None:
        _remaining(self.deadline)
        _save(self.journal / name, data)

    def read(self, name: str) -> dict:
        return _json((self.journal / name).read_bytes())

    def stamp(self, obj: dict) -> dict:
        return {
            "container": obj["id"],
            "image": obj["image"],
            "pid": obj["state"]["Pid"],
            "started": obj["state"]["StartedAt"],
            "boot": BOOT_ID.read_text().strip(),
            "control": self.control_binding(obj["id"]),
        }

    def retire_helper(self, helper: str) -> dict:
        info = self.inspect(helper)
        _require(
            info["image"] == self.config.host_image
            and info["labels"].get(LABEL) == self.attempt
            and info["state"]["Status"] == "exited"
            and info["state"]["Running"] is False,
            "Snapshot helper exit unconfirmed; reconcile outstanding helper",
        )
        self.docker("rm", helper)
        (self.journal / "helper-pending").unlink()
        _sync_dir(self.journal)
        return info["state"]

    def driver(self, op: str, obj: dict) -> dict:
        # The intent fences late daemon work even if this adapter is killed mid-RPC.
        token = secrets.token_hex(16)
        self.save("helper-pending", {"op": op, "container": obj["id"], "token": token})
        argv = deployment.host_helper_args(self.config, self.attempt)
        helper = _run(argv, self.deadline).strip().decode()
        _require(_matches(_ID, helper), "No immutable helper ID")
        self.save("helper-" + token, {"helper": helper})
        seconds = int(_remaining(self.deadline))
        _require(seconds >= 1, "Insufficient Snapshot budget")
        request = {
            "operation": op,
            "attempt": self.attempt,
            "artifact": self.template,
            "container": obj["id"],
            "image": obj["image"],
            "interpose": True,
            "containerd_socket": self.config.containerd_socket,
            "timeout_seconds": seconds,
            "deadline_monotonic": self.deadline,
        }
        try:
            raw = self.docker(
                "start", "--attach", "--interactive", helper, data=json.dumps(request).encode()
            )
        finally:
            # A timeout is not an exit. Never remove a running helper or claim containment.
            state = self.retire_helper(helper)
        _require(state["ExitCode"] == 0 and not state["OOMKilled"], "Snapshot helper failed")
        result = _json(raw)
        _check_driver({"snapshot": result}, op, self.attempt, self.template, obj["id"])
        _require(result["host_pid"] == obj["state"]["Pid"], "Snapshot resolved a different task")
        return result

    def status(self, container: str, headers: dict, identity: dict, *, serving: bool) -> str:
        patterns = {
            "auth": r"sha256=[0-9a-f]{64}",
            "timestamp": r"[0-9]{1,12}",
            "nonce": r"[0-9a-f]{32}",
        }
        _require(
            isinstance(headers, dict)
            and len(headers) == len(patterns)
            and all(
                _matches(re.compile(p), headers.get("x-trtllm-runtime-control-" + k))
                for k, p in patterns.items()
            ),
            "Expected signed GET status headers only",
        )
        start = time.monotonic()
        status = _json(self.get(container, "/checkpoint/status", headers))
        session = self.attempt if serving else identity["source_session_id"]
        stage = "serving" if serving else "capture_safe"
        profile, op = status["profile"], status["operation"]
        _require(
            status["stage"] == stage
            and status["source_session_id"] == identity["source_session_id"]
            and status["restore_session_id"] == (self.attempt if serving else None)
            and profile["ranks"] == [0]
            and type(profile["ranks"][0]) is int
            and profile["capture_policies"] == ["resident"]
            and profile["drain_policies"] == ["natural"]
            and all(profile[k] == identity[k] for k in ("profile_digest", "cohort_digest"))
            and type(status["state_version"]) is int
            and status["state_version"] == op["state_version"]
            and _matches(_ATTEMPT, op["operation_id"]),
            "Unexpected live checkpoint status",
        )
        _require(
            all(status["readiness"][k] is (k in _READY[stage]) for k in _FLAGS),
            "Unexpected live admission or readiness",
        )
        binding = {k: identity[k] for k in ("template_id", "profile_digest", "cohort_digest")}
        rank = _check_result(
            op, {**binding, "session_id": session, "operation_id": op["operation_id"]}, stage
        )
        if serving:
            _require(status["remaining_seconds"] is None, "Unexpected serving deadline")
        else:
            left = status["remaining_seconds"]
            _require(type(left) in (float, int) and 0 < left <= 3600, "Expired capture-safe proof")
            self.deadline = min(self.deadline, start + left)
            _remaining(self.deadline)
        return rank["state_digest"]

    def get(self, container: str, path: str, headers: dict) -> bytes:
        return _docker_exchange(container, PORT, path, self.deadline, lambda: (headers, b""))

    def fresh(self) -> None:
        _require(not any(self.control.iterdir()), "Expected fresh empty control mount")

    def bind_source(self, source: str) -> dict:
        # S2 writes root-owned 0600 CRIU files. Refuse before prepare, not after killing the source.
        _require(
            geteuid() == 0, "This local artifact verifier requires an authorized host-root operator"
        )
        deployment.check_health(self.config, deadline=self.deadline)
        _require(
            not any((self.journal / n).exists() for n in ("capture-intent", "helper-pending")),
            "Source was already claimed",
        )
        obj = self.profile(source, source=True)
        stamp = self.stamp(obj)
        self.fresh()
        _require(
            not os.path.lexists(self.config.root / "checkpoints" / "artifacts" / self.template)
            and not os.path.lexists(self.config.root / "checkpoints" / ".attempts" / source),
            "Stale Snapshot source or artifact",
        )
        self.driver("inspect", obj)
        _require(
            stamp == self.stamp(self.profile(source, source=True)),
            "Source changed during preflight",
        )
        self.control_binding(source, listener=True)
        path = self.journal / "binding.json"
        if path.exists():
            _require(self.read("binding.json")["stamp"] == stamp, "Different source already bound")
            path.unlink()
        self.save("binding.json", {"stamp": stamp, "deadline": self.deadline})
        return {"source": source, "control_container": source, "control_port": PORT}

    def capture(self, source: str, identity: dict, status_headers: dict) -> dict:
        binding = self.read("binding.json")
        self.deadline = min(self.deadline, binding["deadline"])
        _remaining(self.deadline)
        obj = self.profile(source, source=True)
        _require(self.stamp(obj) == binding["stamp"], "Expired or changed source binding")
        self.fresh()
        tree = self.config.root / "checkpoints" / "artifacts" / self.template
        _require(not os.path.lexists(tree), "Refusing to capture over an existing artifact")
        state_digest = self.status(source, status_headers, identity, serving=False)
        self.save("capture-intent", {"source": source})
        # Only after a fresh, corroborated capture-safe reply in this exact namespace.
        with (self.control / "ready-for-snapshot").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        _sync_dir(self.control)
        result = self.driver("capture", obj)
        self.profile(source, source=True, running=False)
        _require(
            result.get("legacy_artifact_path")
            == f"/checkpoints/artifacts/{self.template}/containers/main",
            "Unexpected Snapshot artifact location",
        )
        digest = _artifact(self.config, self.template, source, obj["image"], self.deadline)
        record = {"source": source, "identity": identity, "artifact_digest": digest}
        self.save("artifact.json", {**record, "image": obj["image"], "state_digest": state_digest})
        return {**record, "snapshot": result, "source_terminated": True}

    def restore(self, source: str, identity: dict, artifact_digest: str) -> dict:
        deployment.check_health(self.config, deadline=self.deadline)
        record = _json((self.state / self.template / "artifact.json").read_bytes())
        image = record["image"]
        _require(
            all(
                record[k] == v
                for k, v in {
                    "source": source,
                    "identity": identity,
                    "artifact_digest": artifact_digest,
                }.items()
            ),
            "Manifest does not match captured host record",
        )
        original = self.profile(source, source=True, running=False)
        _require(
            original["image"] == image
            and artifact_digest
            == _artifact(self.config, self.template, source, image, self.deadline),
            "Captured artifact changed",
        )
        self.control.mkdir(mode=0o700)
        self.save("create-intent", {"template": self.template})
        cid = (
            self.docker(
                "create",
                "--pull=never",
                "--name",
                f"warp-target-{self.attempt}",
                "--network=none",
                "--restart=no",
                "--no-healthcheck",
                "--ipc=private",
                "--shm-size=1g",
                "--cpus=16",
                "--memory=128g",
                "--memory-swap=128g",
                "--gpus",
                f"device={self.config.gpu}",
                "--env",
                f"NVIDIA_VISIBLE_DEVICES={self.config.gpu}",
                "--env",
                "SNAPSHOT_CONTROL_DIR=/snapshot-control",
                "--label",
                f"{LABEL}={self.attempt}",
                "--mount",
                f"type=bind,src={self.control},dst=/snapshot-control",
                "--mount",
                f"type=bind,src={self.config.root}/interpose,dst=/tmp/snapshot-cuda,readonly",
                "--entrypoint",
                "/bin/sleep",
                image,
                "infinity",
            )
            .strip()
            .decode()
        )
        _require(_matches(_ID, cid) and cid != source, "No distinct immutable destination ID")
        self.save("target.json", {"target": cid})
        self.docker("start", cid)
        obj = self.profile(cid, source=False)
        _require(
            obj["image"] == image
            and obj["user"] == original["user"]
            and all(
                obj["host"].get(key) == original["host"].get(key)
                for key in (
                    "Runtime",
                    "CapDrop",
                    "CgroupnsMode",
                    "ReadonlyRootfs",
                    "MaskedPaths",
                    "ReadonlyPaths",
                    "GroupAdd",
                    "Ulimits",
                    "PidsLimit",
                    "Sysctls",
                    "Init",
                    "CpusetCpus",
                    "CpusetMems",
                    "SecurityOpt",
                )
            ),
            "Destination image, user or runtime constraints changed",
        )
        stamp = self.stamp(obj)
        self.fresh()
        _require(
            self.docker("top", cid, "-eo", "pid").split() == [b"PID", str(stamp["pid"]).encode()],
            "Restore target contains more than the placeholder",
        )
        result = self.driver("restore", obj)
        _require(
            result["restored_pid"] > 1 and self.stamp(self.profile(cid, source=False)) == stamp,
            "Destination identity changed",
        )
        self.control_binding(cid, pid=result["restored_pid"], listener=True)
        self.save(
            "restored.json",
            {
                "stamp": stamp,
                "identity": identity,
                "state_digest": record["state_digest"],
                "restored_pid": result["restored_pid"],
            },
        )
        return {
            "source": source,
            "identity": identity,
            "artifact_digest": artifact_digest,
            "target": cid,
            "isolated": True,
            "authority_established": True,
            "host_result_id": self.attempt,
            "control_container": cid,
            "control_port": PORT,
            "snapshot": result,
        }

    def destination(self, target: str) -> dict:
        record = self.read("restored.json")
        _require(
            record["stamp"]["container"] == target
            and self.stamp(self.profile(target, source=False)) == record["stamp"],
            "Destination not bound to this restore",
        )
        self.control_binding(target, pid=record["restored_pid"], listener=True)
        return record

    def publish(self, target: str, status_headers: dict) -> dict:
        record = self.destination(target)
        state_digest = self.status(target, status_headers, record["identity"], serving=True)
        _require(
            state_digest == record["state_digest"],
            "Activated state differs from captured inventory",
        )
        self.get(target, "/ready", {})
        self.save(
            "published.json",
            {
                "attempt": self.attempt,
                "template": self.template,
                "target": target,
                "control_port": PORT,
            },
        )
        return {"target": target, "published": True}

    def resolve(self, target: str) -> dict:
        _require(
            self.read("published.json")
            == {
                "attempt": self.attempt,
                "template": self.template,
                "target": target,
                "control_port": PORT,
            },
            "Target not published",
        )
        self.destination(target)
        self.get(target, "/ready", {})
        return {
            "target": target,
            "published": True,
            "control_container": target,
            "control_port": PORT,
        }

    def discard(self) -> dict:
        _require(not (self.journal / "published.json").exists(), "Cannot discard published target")
        if (self.journal / "helper-pending").exists():
            token = self.read("helper-pending")["token"]
            self.retire_helper(self.read("helper-" + token)["helper"])
        source = self.attempt == self.template
        if source:
            container = self.read("binding.json")["stamp"]["container"]
        elif (self.journal / "create-intent").exists():
            container = self.read("target.json")["target"]
        else:
            container = None
        if container:
            obj = self.inspect(container)
            if obj["state"]["Running"]:
                current = self.profile(container, source=source)
                if source or (self.journal / "restored.json").exists():
                    record = self.read("binding.json" if source else "restored.json")
                    _require(
                        self.stamp(current) == record["stamp"],
                        "Refusing to kill a replacement process",
                    )
                self.docker("kill", container)
            self.profile(container, source=source, running=False)
        self.save("discarded.json", {"template": self.template})
        return {"contained": True}


def dispatch(config: deployment.Deployment, request: dict) -> dict:
    """Validate and execute one v1 request; never take host settings from its input."""
    _require(
        set(request) == {"version", "action", "attempt", "template", "deadline_monotonic", "input"}
        and type(request["version"]) is int
        and request["version"] == 1,
        "Unknown adapter protocol",
    )
    action, attempt, template = (request[k] for k in ("action", "attempt", "template"))
    fields, deadline = request["input"], request["deadline_monotonic"]
    _require(
        isinstance(action, str)
        and action in _FIELDS
        and isinstance(fields, dict)
        and set(fields) == _FIELDS[action]
        and _matches(_ATTEMPT, attempt)
        and _matches(_ATTEMPT, template),
        "Invalid action, fields or owner",
    )
    _require(
        action == "discard" or (attempt == template) == (action in ("capture", "bind_source")),
        "Invalid source/destination attempt",
    )
    _require(
        type(deadline) in (int, float)
        and math.isfinite(deadline)
        and 0 < _remaining(deadline) <= 3600,
        "Invalid host operation deadline",
    )
    for key in ("source", "target", "artifact_digest"):
        if key in fields:
            _require(_matches(_ID, fields[key]), "Invalid container or artifact digest")
    if "identity" in fields:
        ident = fields["identity"]
        _require(
            isinstance(ident, dict)
            and set(ident)
            == {"source_session_id", "template_id", "profile_digest", "cohort_digest"}
            and ident["template_id"] == template
            and isinstance(ident["source_session_id"], str)
            and 0 < len(ident["source_session_id"].strip()) <= 128
            and all(_matches(_ID, ident[k]) for k in ("profile_digest", "cohort_digest")),
            "Invalid engine identity",
        )
    backend = Adapter(config, attempt, template, deadline)
    _private_dir(backend.state)
    lock = os.open(backend.state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not backend.journal.exists():
            _require(action in ("bind_source", "restore", "discard"), "Unbound attempt")
            backend.journal.mkdir(mode=0o700)
            _sync_dir(backend.state)
            backend.save("owner.json", {"template": template})
        _private_dir(backend.journal)
        _require(backend.read("owner.json") == {"template": template}, "Wrong template for attempt")
        if (backend.journal / "discarded.json").exists():
            _require(action == "discard", "Attempt was discarded")
            result = {"contained": True}
        else:
            operations = {
                "bind_source": backend.bind_source,
                "capture": backend.capture,
                "restore": backend.restore,
                "publish": backend.publish,
                "discard": backend.discard,
                "resolve": backend.resolve,
            }
            result = operations[action](**fields)
        _remaining(backend.deadline)
    finally:
        os.close(lock)
    return {
        "version": 1,
        "action": action,
        "template": template,
        "attempt": attempt,
        "status": "succeeded",
        "result": result,
    }


def main() -> None:
    """Use the operator's explicit local deployment, not credentials or a remote Docker context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(16385)
    _require(len(raw) <= 16384, "Oversized adapter request")
    print(
        json.dumps(
            dispatch(deployment.load_deployment(args.deployment), _json(raw)), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
