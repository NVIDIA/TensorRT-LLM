# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit operator setup and read-only health for the standalone S2 TP1 host.

The Apache-2.0 driver and tests in host_driver/ are adapted from TensorRT-LLM S2
12582e3977c5019c7c361a4d88ff89e58742ca6d, examples/serve/snapshot/host_driver.
The local request requires an absolute host deadline. The bundle also builds a
host-side client for Snapshot's capabilities RPC.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import secrets
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

SNAPSHOT_REVISION = "4b1300a82c6676df020c201fa884cc602251e12e"
S2_REVISION = "12582e3977c5019c7c361a4d88ff89e58742ca6d"
AGENT_DIGEST = "sha256:50a3058893c543dc0e48bfac89dc74a07eb26ff91ad3634c5fef25f37ab9f8c9"
BROKER_IMAGE = (
    "ghcr.io/ai-dynamo/snapshot/pagebroker@sha256:"
    "bf75b039ca231d2fd81cf41291e1a11e26f3b6628788a8d22ce292327505eac6"
)
DOCKER = ("/usr/bin/docker", "--host=unix:///var/run/docker.sock")
CONTAINERD_SOCKET = "/run/docker/containerd/containerd.sock"
DRIVER = Path(__file__).resolve().with_name("host_driver")
_DRIVER_FILES = (
    "main.go",
    "main_test.go",
    "deadline.go",
    "deadline_test.go",
    "Dockerfile",
    ".dockerignore",
    "health/main.go",
)
_ASSETS = {
    "broker-health": "/usr/local/bin/trtllm-broker-health",
    "interpose/cuinterpose-launch": "/usr/local/bin/cuinterpose-launch",
    "interpose/libcuinterpose.so": "/usr/local/lib/snapshot/libcuinterpose.so",
    "interpose/libcuinterpose_core.so": "/usr/local/lib/snapshot/libcuinterpose_core.so",
}
_PRIVATE_DIRS = (
    "warp-state",
    "controls",
    "checkpoints",
    "pagebroker",
    "pagebroker/control",
    "pagebroker/staging",
    "interpose",
)
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}")
_GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_LABEL = "com.nvidia.trtllm.warp.deployment"
_INPUT_LABEL = "com.nvidia.trtllm.warp.build-inputs"
_BROKER_ARGS = [
    "/pagebroker/control/pagebroker.sock",
    "/pagebroker/staging",
    "/checkpoints",
    "--max-concurrent-requests",
    "8",
    "--custom-storage-buffer-count",
    "2",
    "--custom-storage-chunk-bytes",
    "67108864",
    "--custom-storage-max-pinned-bytes",
    "268435456",
]


def _run(argv: list[str], deadline: float, *, log: TextIO | None = None) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Deployment deadline expired")
    # Do not inherit workload credentials, Docker contexts, remote builders or SSH agents.
    result = subprocess.run(
        argv,
        check=False,
        timeout=remaining,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=log or subprocess.PIPE,
        stderr=subprocess.STDOUT if log else subprocess.PIPE,
        env={"HOME": str(Path.home()), "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, argv, output=result.stdout, stderr=result.stderr
        )
    if time.monotonic() >= deadline:
        raise TimeoutError("Deployment deadline expired")
    return (result.stdout or "").strip()


def _docker(*args: str, deadline: float) -> str:
    return _run([*DOCKER, *args], deadline)


def _inspect(kind: str, ref: str, deadline: float) -> dict:
    [value] = json.loads(_docker(kind, "inspect", ref, deadline=deadline))
    return value


def _private(path: Path, *, directory: bool = True) -> None:
    info = path.lstat()
    test = stat.S_ISDIR if directory else stat.S_ISREG
    if not test(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError(
            f"Expected an operator-owned private {'directory' if directory else 'file'}: {path}"
        )


def _absolute(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path == Path("/")
        or path != path.resolve()
        or not re.fullmatch(r"/[a-zA-Z0-9_./-]+", str(path))
    ):
        raise ValueError(
            f"Expected a canonical absolute bind path without links or delimiters: {path}"
        )
    return path


def _private_root(root: Path) -> None:
    _private(_absolute(root))
    for parent in root.parents:
        info = parent.lstat()
        if info.st_uid not in (0, os.geteuid()) or (
            info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
        ):
            raise ValueError(f"Untrusted ancestor of operator deployment: {parent}")


def _hash(path: Path) -> str:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected regular file, not link: {path}")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _inputs() -> dict:
    return {
        "snapshot": SNAPSHOT_REVISION,
        "s2": S2_REVISION,
        "agent": AGENT_DIGEST,
        "broker": BROKER_IMAGE,
        "files": {name: _hash(DRIVER / name) for name in _DRIVER_FILES},
    }


def _input_digest() -> str:
    return hashlib.sha256(json.dumps(_inputs(), sort_keys=True).encode()).hexdigest()


def _write(path: Path, value: dict) -> None:
    # Publish complete receipts without replacing evidence from another invocation.
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temp, path)
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temp.unlink()


def _managed_socket(deadline: float) -> None:
    # Check the actual listening process, not just a conventional socket filename.
    if not stat.S_ISSOCK(os.lstat(CONTAINERD_SOCKET).st_mode):
        raise ValueError("Expected Docker-managed containerd socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(max(0.001, min(3, deadline - time.monotonic())))
        client.connect(CONTAINERD_SOCKET)
        pid, uid, _ = struct.unpack(
            "3i", client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
    proc = Path("/proc") / str(pid)
    status = dict(
        line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line
    )
    parent = Path("/proc") / status["PPid"].strip()
    if (
        uid != 0
        or (proc / "exe").readlink().name != "containerd"
        or (parent / "exe").readlink().name != "dockerd"
    ):
        raise ValueError("Require rootful containerd managed by local dockerd")


def _local_host(deadline: float) -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or any(Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv"))
        or any(k in os.environ for k in ("SLURM_JOB_ID", "KUBERNETES_SERVICE_HOST"))
    ):
        raise ValueError("Require an authorized standalone x86_64 Linux Docker host")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file() or not any(
        line.split()[1:3] == ["/sys/kernel/security", "securityfs"]
        for line in Path("/proc/mounts").read_text().splitlines()
    ):
        raise ValueError("Require host cgroup v2 and mounted securityfs")
    if os.readlink("/proc/self/ns/time") != os.readlink("/proc/1/ns/time"):
        raise ValueError("Require operator and Docker host to share the CLOCK_MONOTONIC namespace")
    _managed_socket(deadline)


def _host(gpu: str, deadline: float) -> Path:
    if not _GPU.fullmatch(gpu):
        raise ValueError("Expected exactly one allocated physical GPU UUID (TP1)")
    _local_host(deadline)
    info = json.loads(_docker("info", "--format", "{{json .}}", deadline=deadline))
    if (
        info["OSType"] != "linux"
        or info["Architecture"] != "x86_64"
        or info["KernelVersion"] != platform.release()
        or info["Driver"] != "overlay2"
        or info["Swarm"]["LocalNodeState"] != "inactive"
        or "nvidia" not in info["Runtimes"]
        or any("rootless" in s or "userns" in s for s in info["SecurityOptions"])
    ):
        raise ValueError("Unsupported Docker host, storage, namespace or GPU runtime")
    gpus = _run(["/usr/bin/nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], deadline)
    if gpu not in gpus.splitlines():
        raise ValueError("Allocated GPU is not present on this host")
    root = _absolute(Path(info["DockerRootDir"]))
    if not root.is_dir():
        raise ValueError("Local Docker storage directory missing")
    return root


@dataclass(frozen=True)
class Deployment:
    """Operator receipt for a single local managed Docker and PageBroker installation."""

    root: Path
    gpu: str
    host_image: str
    docker_root: Path
    deployment_id: str
    assets: dict[str, str]
    containerd_socket = CONTAINERD_SOCKET

    @property
    def broker_name(self) -> str:
        return f"warp-pagebroker-{self.deployment_id}"


def load_deployment(path: Path) -> Deployment:
    """Read the private setup receipt, rejecting missing opt-in, pins or an unsafe layout."""
    root = path.parent
    _private_root(root)
    if path.name != "deployment.json":
        raise ValueError("Expected the operator's deployment.json")
    _private(path, directory=False)
    cfg = json.loads(path.read_bytes())
    if (
        cfg["version"] != 1
        or cfg["allow_privileged_host_driver"] is not True
        or cfg["build_inputs"] != _inputs()
        or cfg["root"] != str(root)
        or not _GPU.fullmatch(cfg["gpu"])
        or not _IMAGE.fullmatch(cfg["host_image"])
        or not re.fullmatch(r"[0-9a-f]{32}", cfg["id"])
        or cfg["tp"] != 1
        or set(cfg["assets"]) != set(_ASSETS)
    ):
        raise ValueError("Deployment authorization, input pins, GPU, or paths do not match")
    for name in _PRIVATE_DIRS:
        _private(root / name)
    for name, digest in cfg["assets"].items():
        if _hash(root / name) != digest:
            raise ValueError(f"Deployment asset mismatch: {name}")
    return Deployment(
        root,
        cfg["gpu"],
        cfg["host_image"],
        _absolute(Path(cfg["docker_root"])),
        cfg["id"],
        cfg["assets"],
    )


def _shared(config: Deployment) -> list[str]:
    return [
        "--restart=no",
        "--privileged",
        "--pid=host",
        "--ipc=host",
        "--cgroupns=host",
        "--network=none",
        "--gpus",
        f"device={config.gpu}",
        "--env",
        f"NVIDIA_VISIBLE_DEVICES={config.gpu}",
        "--mount",
        f"type=bind,source={config.root}/checkpoints,target=/checkpoints",
        "--mount",
        f"type=bind,source={config.root}/pagebroker,target=/pagebroker",
    ]


def host_helper_args(config: Deployment, attempt: str) -> list[str]:
    """Create argv only. The caller owns the helper, deadline, final result and cleanup."""
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("Expected a full attempt ID")
    mounts = [
        ("/proc", "/host/proc", ""),
        ("/sys/fs/cgroup", "/sys/fs/cgroup", ""),
        ("/sys/kernel/security", "/sys/kernel/security", ",readonly"),
        ("/var/run/docker.sock", "/var/run/docker.sock", ""),
        (CONTAINERD_SOCKET, CONTAINERD_SOCKET, ""),
        (str(config.docker_root), str(config.docker_root), ",readonly"),
    ]
    options = [
        v
        for src, dst, mode in mounts
        for v in ("--mount", f"type=bind,source={src},target={dst}{mode}")
    ]
    return [
        *DOCKER,
        "create",
        "--pull=never",
        "--interactive",
        "--label",
        f"com.nvidia.trtllm.snapshot.attempt={attempt}",
        "--label",
        f"{_LABEL}={config.deployment_id}",
        *_shared(config),
        *options,
        config.host_image,
    ]


def _verify_driver(image: str, deadline: float) -> None:
    info = _inspect("image", image, deadline)
    labels = info["Config"]["Labels"]
    if (
        info["Id"] != image
        or info["Architecture"] != "amd64"
        or info["Os"] != "linux"
        or info["Config"]["Entrypoint"] != ["/usr/local/bin/trtllm-snapshot-host"]
        or labels.get("com.nvidia.trtllm.snapshot.revision") != SNAPSHOT_REVISION
        or labels.get("com.nvidia.trtllm.snapshot.agent.digest") != AGENT_DIGEST
        or labels.get(_INPUT_LABEL) != _input_digest()
    ):
        raise ValueError("Host driver differs from pinned build inputs")


def _service(config: Deployment, cid: str, deadline: float) -> dict:
    if not re.fullmatch("[0-9a-f]{64}", cid):
        raise ValueError("Expected full PageBroker container ID")
    image = _inspect("image", BROKER_IMAGE, deadline)
    container = _inspect("container", cid, deadline)
    hc = container["HostConfig"]
    cc = container["Config"]
    requests = hc.get("DeviceRequests") or []
    mounts = {(m["Source"], m["Destination"], m["Type"], m["RW"]) for m in container["Mounts"]}
    expected = {
        (str(config.root / name), f"/{name}", "bind", True)
        for name in ("pagebroker", "checkpoints")
    }
    env = dict(e.split("=", 1) for e in image["Config"]["Env"] or [])
    env["NVIDIA_VISIBLE_DEVICES"] = config.gpu
    if (
        image["Architecture"] != "amd64"
        or image["Os"] != "linux"
        or container["Id"] != cid
        or container["Name"] != "/" + config.broker_name
        or container["Image"] != image["Id"]
        or cc["Labels"].get(_LABEL) != config.deployment_id
        or cc["Entrypoint"] != image["Config"]["Entrypoint"]
        or cc["Cmd"] != _BROKER_ARGS
        or cc["User"] != image["Config"]["User"]
        or cc.get("Healthcheck") != image["Config"].get("Healthcheck")
        or cc.get("WorkingDir") != image["Config"].get("WorkingDir")
        or sorted(cc["Env"]) != sorted(f"{k}={v}" for k, v in env.items())
        or hc["Privileged"] is not True
        or hc["PidMode"] != "host"
        or hc["IpcMode"] != "host"
        or hc["CgroupnsMode"] != "host"
        or hc["NetworkMode"] != "none"
        or hc["PortBindings"]
        or hc["RestartPolicy"]["Name"] != "no"
        or hc.get("VolumesFrom")
        or hc.get("Devices")
        or hc.get("AutoRemove")
        or len(requests) != 1
        or requests[0]["DeviceIDs"] != [config.gpu]
        or requests[0]["Driver"] not in ("", "nvidia")
        or requests[0]["Capabilities"] != [["gpu"]]
        or mounts != expected
        or len(container["Mounts"]) != len(expected)
    ):
        raise ValueError("PageBroker configuration differs; refusing to start or reuse it")
    return container["State"]


def _probe(config: Deployment, deadline: float) -> dict:
    path = config.root / "pagebroker/control/pagebroker.sock"
    if not stat.S_ISSOCK(path.lstat().st_mode):
        raise ValueError("PageBroker control path is not a socket")
    caps = json.loads(_run([str(config.root / "broker-health"), str(path)], deadline))
    if not isinstance(caps, dict) or type(caps.get("custom_storage")) is not bool:
        raise ValueError("Invalid PageBroker capabilities response")
    return caps


def check_health(config: Deployment, *, deadline: float | None = None) -> dict:
    """Check the existing installation and query PageBroker without creating any resources."""
    now = time.monotonic()
    if deadline is not None and (not math.isfinite(deadline) or deadline <= now):
        raise TimeoutError("Deployment deadline expired")
    deadline = min(deadline, now + 20) if deadline is not None else now + 20
    if load_deployment(config.root / "deployment.json") != config:
        raise ValueError("Deployment receipt changed")
    if _host(config.gpu, deadline) != config.docker_root:
        raise ValueError("Docker storage root changed")
    _verify_driver(config.host_image, deadline)
    receipt = config.root / "broker.json"
    _private(receipt, directory=False)
    cid = json.loads(receipt.read_bytes())["id"]
    state = _service(config, cid, deadline)
    if state["Status"] != "running" or state["Running"] is not True or state.get("Paused"):
        raise ValueError("Pinned PageBroker is not running")
    return {
        "status": "healthy",
        "broker_container": cid,
        "broker_capabilities": _probe(config, deadline),
        "snapshot_revision": SNAPSHOT_REVISION,
        "host_image": config.host_image,
        "gpu": config.gpu,
        "process_restore": "NOT_RUN",
    }


def _ensure_service(config: Deployment, deadline: float) -> None:
    receipt = config.root / "broker.json"
    if receipt.exists():
        _private(receipt, directory=False)
        cid = json.loads(receipt.read_bytes())["id"]
    else:
        cid = _docker(
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            f"name=^/{config.broker_name}$",
            "--format",
            "{{.ID}}",
            deadline=deadline,
        )
        if not cid:
            cid = _docker(
                "create",
                "--name",
                config.broker_name,
                "--pull=never",
                "--label",
                f"{_LABEL}={config.deployment_id}",
                *_shared(config),
                BROKER_IMAGE,
                *_BROKER_ARGS,
                deadline=deadline,
            )
    state = _service(config, cid, deadline)
    if not receipt.exists():
        _write(receipt, {"id": cid})
    if state["Status"] == "created":
        _docker("start", cid, deadline=deadline)
    elif state["Status"] != "running":
        raise ValueError("PageBroker previously stopped; inspect retained logs before intervention")


def _build(root: Path, deadline: float) -> str:
    build = root / "build"
    build.mkdir(mode=0o700)
    _write(build / "inputs.json", _inputs())
    tag = f"warp-snapshot-host:build-{secrets.token_hex(16)}"
    with (build / "build.log").open("x") as log:
        _run(
            [
                *DOCKER,
                "buildx",
                "build",
                "--builder",
                "default",
                "--platform=linux/amd64",
                "--load",
                "--progress=plain",
                "--tag",
                tag,
                "--iidfile",
                str(build / "iid"),
                "--build-context",
                f"snapshot=https://github.com/ai-dynamo/snapshot.git#{SNAPSHOT_REVISION}",
                "--label",
                f"{_INPUT_LABEL}={_input_digest()}",
                str(DRIVER),
            ],
            deadline,
            log=log,
        )
    image = _inspect("image", tag, deadline)["Id"]
    if not _IMAGE.fullmatch(image):
        raise ValueError("Expected immutable host image ID from buildx")
    _verify_driver(image, deadline)
    _write(build / "result.json", {"image": image, "tag": tag})
    return image


def _install(root: Path, gpu: str, storage: Path, deadline: float) -> None:
    if any(root.iterdir()):
        raise ValueError(
            "First setup needs an empty private root; retain failed builds for inspection"
        )
    if root.is_relative_to(storage) or storage.is_relative_to(root):
        raise ValueError("Deployment and Docker storage must not overlap")
    for name in _PRIVATE_DIRS:
        (root / name).mkdir(mode=0o700)
    image = _build(root, deadline)
    _docker("pull", "--platform=linux/amd64", BROKER_IMAGE, deadline=deadline)
    # This container is never started. Export only files provided by the pinned image.
    cid = _docker("create", "--pull=never", "--network=none", image, deadline=deadline)
    if not re.fullmatch("[0-9a-f]{64}", cid):
        raise ValueError("No immutable export container ID")
    _write(root / "build/export.json", {"id": cid, "image": image})
    try:
        for name, src in _ASSETS.items():
            _docker("cp", "-L", f"{cid}:{src}", str(root / name), deadline=deadline)
    finally:
        _docker("rm", cid, deadline=time.monotonic() + 10)
    _write(
        root / "deployment.json",
        {
            "version": 1,
            "allow_privileged_host_driver": True,
            "tp": 1,
            "root": str(root),
            "docker_root": str(storage),
            "gpu": gpu,
            "id": secrets.token_hex(16),
            "host_image": image,
            "build_inputs": _inputs(),
            "assets": {name: _hash(root / name) for name in _ASSETS},
        },
    )


def _provision(root: Path, gpu: str) -> dict:
    deadline = time.monotonic() + 1800
    storage = _host(gpu, deadline)
    cfg_path = root / "deployment.json"
    if not cfg_path.exists():
        _install(root, gpu, storage, deadline)
    config = load_deployment(cfg_path)
    if config.gpu != gpu or config.docker_root != storage:
        raise ValueError("Existing deployment belongs to another GPU or Docker root")
    _verify_driver(config.host_image, deadline)
    _ensure_service(config, deadline)
    ready_by = min(deadline, time.monotonic() + 20)
    while True:
        try:
            _probe(config, ready_by)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            if time.monotonic() >= ready_by:
                raise
            time.sleep(0.2)
    return check_health(config, deadline=deadline)


def setup(root: Path, gpu: str, *, allow_privileged_host_driver: bool = False) -> dict:
    """Build pinned inputs or reuse this deployment; only explicit setup can start PageBroker."""
    if allow_privileged_host_driver is not True:
        raise ValueError("Setup requires --allow-privileged-host-driver on an authorized host")
    _private_root(root)
    evidence = root / f"setup-{secrets.token_hex(8)}.json"
    try:
        health = _provision(root, gpu)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        _write(
            evidence,
            {
                "status": "failed",
                "gpu": gpu,
                "error": str(exc),
                "process_restore": "NOT_RUN",
                "stderr": (exc.stderr or "")[-4096:]
                if isinstance(exc, subprocess.CalledProcessError)
                else "",
            },
        )
        raise
    _write(evidence, health)
    return health


def main(argv: list[str] | None = None) -> None:
    """Run explicit operator setup or a read-only health check."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    install = actions.add_parser(
        "setup", allow_abbrev=False, help="Build pinned inputs and create or verify PageBroker"
    )
    install.add_argument("--allow-privileged-host-driver", action="store_true", required=True)
    install.add_argument("--root", type=Path, required=True, help="Existing private 0700 directory")
    install.add_argument("--gpu", required=True, help="One operator-allocated physical GPU UUID")
    health = actions.add_parser(
        "health", help="Inspect deployment and query PageBroker capabilities"
    )
    health.add_argument("--config", type=Path, required=True, help="Absolute deployment.json path")
    args = parser.parse_args(argv)
    if args.action == "setup":
        result = setup(
            args.root, args.gpu, allow_privileged_host_driver=args.allow_privileged_host_driver
        )
    else:
        result = check_health(load_deployment(args.config))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        sys.exit(f"WARP deployment: {exc}")
