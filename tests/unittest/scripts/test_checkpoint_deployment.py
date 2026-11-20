# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only deployment contracts. No daemon, privilege, network, or GPU required."""

import importlib
import json
import stat
import subprocess
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only
GPU = "GPU-00000000-0000-0000-0000-000000000001"
HOST = "sha256:" + "1" * 64
BROKER = "sha256:" + "2" * 64
CID = "3" * 64


@pytest.fixture
def mod(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "examples/serve/checkpoint")
    )
    module = importlib.import_module("deployment")
    for target, name in [
        (module.subprocess, "run"),
        (module.socket, "socket"),
        (module, "_managed_socket"),
    ]:
        monkeypatch.setattr(
            target, name, Mock(side_effect=AssertionError("Unexpected host access"))
        )
    return module


@pytest.fixture
def cfg(mod: ModuleType, tmp_path: Path):
    root = (tmp_path / "deployment").resolve()
    root.mkdir(mode=0o700)
    for name in mod._PRIVATE_DIRS:
        (root / name).mkdir(mode=0o700)
    for name in mod._ASSETS:
        (root / name).write_bytes(name.encode())
    mod._write(
        root / "deployment.json",
        {
            "version": 1,
            "tp": 1,
            "allow_privileged_host_driver": True,
            "root": str(root),
            "docker_root": str((tmp_path / "docker-storage").resolve()),
            "id": "a" * 32,
            "gpu": GPU,
            "host_image": HOST,
            "build_inputs": mod._inputs(),
            "assets": {name: mod._hash(root / name) for name in mod._ASSETS},
        },
    )
    return mod.load_deployment(root / "deployment.json")


def _broker(mod: ModuleType, cfg) -> tuple[dict, dict]:
    image = {
        "Id": BROKER,
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {"User": "", "Env": ["PATH=/bin"], "Entrypoint": ["broker"]},
    }
    ctr = {
        "Id": CID,
        "Name": "/" + cfg.broker_name,
        "Image": BROKER,
        "Config": {
            "Labels": {mod._LABEL: cfg.deployment_id},
            "Entrypoint": ["broker"],
            "Cmd": list(mod._BROKER_ARGS),
            "User": "",
            "Env": ["PATH=/bin", f"NVIDIA_VISIBLE_DEVICES={GPU}"],
        },
        "HostConfig": {
            "Privileged": True,
            "PidMode": "host",
            "IpcMode": "host",
            "CgroupnsMode": "host",
            "NetworkMode": "none",
            "PortBindings": {},
            "RestartPolicy": {"Name": "no"},
            "DeviceRequests": [{"DeviceIDs": [GPU], "Driver": "", "Capabilities": [["gpu"]]}],
        },
        "Mounts": [
            {"Type": "bind", "RW": True, "Source": str(cfg.root / n), "Destination": f"/{n}"}
            for n in ("pagebroker", "checkpoints")
        ],
        "State": {"Status": "running", "Running": True, "Paused": False},
    }
    return image, ctr


def test_setup_requires_authorization_before_access(mod: ModuleType) -> None:
    with pytest.raises(ValueError, match="allow-privileged"):
        mod.setup(Path("/missing"), GPU)
    with pytest.raises(SystemExit):
        mod.main(["setup", "--root", "/missing", "--gpu", GPU])


def test_reject_unsupported_hosts_and_tp(mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="standalone"):
        mod._host(GPU, time.monotonic() + 10)
    for gpu in ("0", "all", f"{GPU},{GPU}", "MIG-" + GPU):
        with pytest.raises(ValueError, match="TP1"):
            mod._host(gpu, time.monotonic() + 10)


def test_subprocess_environment_and_deadline(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://untrusted")
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", "do-not-forward")
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="ready\n"))
    monkeypatch.setattr(mod.subprocess, "run", runner)
    assert mod._docker("info", deadline=time.monotonic() + 1) == "ready"
    args, kwargs = runner.call_args
    assert args[0] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "info"]
    assert set(kwargs["env"]) == {"PATH", "HOME"}
    assert 0 < kwargs["timeout"] <= 1
    with pytest.raises(TimeoutError):
        mod._docker("info", deadline=time.monotonic() - 1)
    assert runner.call_count == 1


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("allow_privileged_host_driver", False),
        ("build_inputs", {}),
        ("root", "/tmp/other"),
        ("tp", 2),
        ("host_image", "host:latest"),
        ("docker_root", "/"),
        ("assets", {}),
    ],
)
def test_reject_receipt_drift(mod: ModuleType, cfg, field: str, bad: object) -> None:
    path = cfg.root / "deployment.json"
    data = json.loads(path.read_bytes())
    data[field] = bad
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        mod.load_deployment(path)


def test_reject_changed_asset_and_unsafe_paths(mod: ModuleType, cfg) -> None:
    path = cfg.root / "deployment.json"
    asset = cfg.root / "broker-health"
    asset.write_text("wrong")
    with pytest.raises(ValueError, match="asset mismatch"):
        mod.load_deployment(path)
    asset.unlink()
    asset.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        mod.load_deployment(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private file"):
        mod.load_deployment(path)


def test_helper_is_scoped_to_managed_socket_and_tp1(mod: ModuleType, cfg) -> None:
    argv = mod.host_helper_args(cfg, "e" * 32)
    assert argv[:3] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "create"]
    assert argv[-1] == HOST and "--pull=never" in argv and "--network=none" in argv
    assert argv[argv.index("--gpus") + 1] == f"device={GPU}"
    assert f"com.nvidia.trtllm.snapshot.attempt={'e' * 32}" in argv
    assert f"type=bind,source={mod.CONTAINERD_SOCKET},target={mod.CONTAINERD_SOCKET}" in argv
    assert not any("source=/run/containerd/" in arg or "sudo" in arg for arg in argv)
    with pytest.raises(ValueError, match="attempt"):
        mod.host_helper_args(cfg, "../sibling")


def test_build_failure_keeps_pinned_inputs_and_logs(
    mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(argv: list[str], deadline: float, *, log) -> str:
        assert argv[:4] == [*mod.DOCKER, "buildx", "build"]
        assert f"snapshot=https://github.com/ai-dynamo/snapshot.git#{mod.SNAPSHOT_REVISION}" in argv
        assert argv[-1] == str(mod.DRIVER) and "--load" in argv and "--push" not in argv
        log.write("test build failure\n")
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(mod, "_run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        mod._build(tmp_path, time.monotonic() + 10)
    inputs = json.loads((tmp_path / "build/inputs.json").read_bytes())
    assert inputs["s2"] == "12582e3977c5019c7c361a4d88ff89e58742ca6d"
    go_files = {"main.go", "main_test.go", "deadline.go", "deadline_test.go"}
    dockerfile = (mod.DRIVER / "Dockerfile").read_text().splitlines()
    host_copy = next(line for line in dockerfile if line.endswith("/src/agent/cmd/trtllm-host/"))
    assert set(host_copy.split()[1:-1]) == go_files
    includes = (mod.DRIVER / ".dockerignore").read_text().splitlines()
    for name in go_files:
        assert f"!{name}" in includes and inputs["files"][name] == mod._hash(mod.DRIVER / name)
    assert inputs["broker"] == mod.BROKER_IMAGE and inputs["agent"] == mod.AGENT_DIGEST
    assert (tmp_path / "build/build.log").read_text() == "test build failure\n"
    assert not (tmp_path / "deployment.json").exists()


@pytest.mark.parametrize(
    "drift",
    [
        "image",
        "label",
        "mount",
        "extra_mount",
        "entrypoint",
        "argv",
        "network",
        "restart",
        "gpu",
        "privileged",
        "env",
        "name",
        "id",
    ],
)
def test_refuse_foreign_service_without_mutation(
    mod: ModuleType, cfg, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    image, ctr = _broker(mod, cfg)
    host, app = ctr["HostConfig"], ctr["Config"]
    changes = {
        "image": (ctr, "Image", HOST),
        "label": (app, "Labels", {}),
        "mount": (ctr["Mounts"][0], "Source", "/wrong"),
        "extra_mount": (ctr, "Mounts", [*ctr["Mounts"], ctr["Mounts"][0]]),
        "entrypoint": (app, "Entrypoint", ["other"]),
        "argv": (app, "Cmd", []),
        "network": (host, "NetworkMode", "host"),
        "restart": (host, "RestartPolicy", {"Name": "always"}),
        "gpu": (host["DeviceRequests"][0], "DeviceIDs", ["all"]),
        "privileged": (host, "Privileged", False),
        "env": (app, "Env", []),
        "name": (ctr, "Name", "/unrelated"),
        "id": (ctr, "Id", "4" * 64),
    }
    target, key, value = changes[drift]
    target[key] = value
    mod._write(cfg.root / "broker.json", {"id": CID})
    monkeypatch.setattr(mod, "_inspect", Mock(side_effect=[image, ctr]))
    with pytest.raises(ValueError, match="refusing"):
        mod._ensure_service(cfg, time.monotonic() + 2)


@pytest.mark.parametrize("existing", [False, True])
def test_create_or_verify_matching_service(
    mod: ModuleType, cfg, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    image, ctr = _broker(mod, cfg)
    if not existing:
        ctr["State"] = {"Status": "created"}
    inspect = Mock(side_effect=[image, ctr])
    docker = Mock(side_effect=[CID] if existing else ["", CID, CID])
    monkeypatch.setattr(mod, "_inspect", inspect)
    monkeypatch.setattr(mod, "_docker", docker)
    mod._ensure_service(cfg, time.monotonic() + 10)
    cmds = [c.args for c in docker.call_args_list]
    assert cmds[0][:2] == ("container", "ls")
    if not existing:
        assert cmds[1][0] == "create" and mod.BROKER_IMAGE in cmds[1]
        assert cmds[2] == ("start", CID)
    else:
        assert len(cmds) == 1
    assert inspect.call_args_list[0].args[:2] == ("image", mod.BROKER_IMAGE)
    assert inspect.call_args_list[1].args[:2] == ("container", CID)
    assert json.loads((cfg.root / "broker.json").read_bytes()) == {"id": CID}


def test_health_probes_capabilities_without_mutation(
    mod: ModuleType, cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod._write(cfg.root / "broker.json", {"id": CID})
    broker, ctr = _broker(mod, cfg)
    driver = {
        "Id": HOST,
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {
            "Entrypoint": ["/usr/local/bin/trtllm-snapshot-host"],
            "Labels": {
                "com.nvidia.trtllm.snapshot.revision": mod.SNAPSHOT_REVISION,
                "com.nvidia.trtllm.snapshot.agent.digest": mod.AGENT_DIGEST,
                mod._INPUT_LABEL: mod._input_digest(),
            },
        },
    }
    monkeypatch.setattr(mod, "_host", Mock(return_value=cfg.docker_root))
    inspect = Mock(side_effect=[driver, broker, ctr])
    monkeypatch.setattr(mod, "_inspect", inspect)
    # Model the socket inode without opening or binding a real socket.
    path = cfg.root / "pagebroker/control/pagebroker.sock"
    original = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda p: SimpleNamespace(st_mode=stat.S_IFSOCK) if p == path else original(p),
    )
    probe = Mock(return_value='{"custom_storage":false}')
    monkeypatch.setattr(mod, "_run", probe)
    before = set(cfg.root.rglob("*"))
    deadline = time.monotonic() + 1
    health = mod.check_health(cfg, deadline=deadline)
    probe.assert_called_once_with([str(cfg.root / "broker-health"), str(path)], deadline)
    assert [c.args[:2] for c in inspect.call_args_list] == [
        ("image", HOST),
        ("image", mod.BROKER_IMAGE),
        ("container", CID),
    ]
    assert health["broker_capabilities"] == {"custom_storage": False}
    assert health["process_restore"] == "NOT_RUN"
    assert set(cfg.root.rglob("*")) == before


def test_probe_fails_closed_for_stale_socket_or_bad_response(
    mod: ModuleType, cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    sock = cfg.root / "pagebroker/control/pagebroker.sock"
    sock.touch()
    with pytest.raises(ValueError, match="not a socket"):
        mod._probe(cfg, time.monotonic() + 1)
    monkeypatch.setattr(Path, "lstat", lambda p: SimpleNamespace(st_mode=stat.S_IFSOCK))
    probe = Mock(side_effect=[subprocess.CalledProcessError(1, "probe"), "{}", "not json"])
    monkeypatch.setattr(mod, "_run", probe)
    for error in (subprocess.CalledProcessError, ValueError, ValueError):
        with pytest.raises(error):
            mod._probe(cfg, time.monotonic() + 1)


def test_setup_builds_once_and_rechecks_on_reuse(
    mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "installation").resolve()
    root.mkdir(mode=0o700)
    storage = (tmp_path / "docker-storage").resolve()
    monkeypatch.setattr(mod, "_host", Mock(return_value=storage))
    inspect = Mock(return_value={"Id": HOST})
    monkeypatch.setattr(mod, "_inspect", inspect)
    verify = Mock()
    monkeypatch.setattr(mod, "_verify_driver", verify)
    cmds = []
    running = False

    def run(argv: list[str], deadline: float, *, log=None) -> str:
        nonlocal running
        assert tuple(argv[:2]) == mod.DOCKER
        cmd = argv[2:]
        cmds.append(cmd)
        if cmd[:2] == ["buildx", "build"]:
            assert "--load" in cmd and "--push" not in cmd
            log.write("mock builder\n")
        elif cmd[0] == "pull":
            assert cmd[-1] == mod.BROKER_IMAGE
        elif cmd[0] == "create" and cmd[-1] == HOST:
            assert "--privileged" not in cmd
            return "e" * 64
        elif cmd[0] == "cp":
            assert cmd[2].startswith("e" * 64 + ":/usr/local/")
            Path(cmd[3]).write_bytes(b"mock image payload")
        elif cmd[0] == "rm":
            assert cmd == ["rm", "e" * 64]
        elif cmd[:2] == ["container", "ls"]:
            return ""
        elif cmd[0] == "create":
            assert "--privileged" in cmd and mod.BROKER_IMAGE in cmd
            return CID
        elif cmd == ["start", CID]:
            running = True
        else:
            pytest.fail(f"Unexpected command: {cmd}")
        return ""

    monkeypatch.setattr(mod, "_run", run)
    monkeypatch.setattr(
        mod,
        "_service",
        lambda config, cid, deadline: {
            "Status": "running" if running else "created",
            "Running": running,
        },
    )
    probe = Mock(return_value={"custom_storage": True})
    monkeypatch.setattr(mod, "_probe", probe)
    first = mod.setup(root, GPU, allow_privileged_host_driver=True)
    new_argv = len(cmds)
    assert mod.setup(root, GPU, allow_privileged_host_driver=True) == first
    assert len(cmds) == new_argv  # Only the mocked read-only checks ran this time.
    assert sum(c[:2] == ["buildx", "build"] for c in cmds) == 1
    assert sum(c[0] == "pull" for c in cmds) == 1
    assert sum(c == ["start", CID] for c in cmds) == 1
    cfg = mod.load_deployment(root / "deployment.json")
    assert cfg.host_image == HOST
    assert first["broker_container"] == CID and first["process_restore"] == "NOT_RUN"
    assert probe.call_count == 4
    assert len(list(root.glob("setup-*.json"))) == 2


def test_failed_setup_retains_diagnostics_without_repair(
    mod: ModuleType, cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "_host", Mock(return_value=cfg.docker_root))
    monkeypatch.setattr(mod, "_verify_driver", Mock())
    monkeypatch.setattr(
        mod,
        "_ensure_service",
        Mock(
            side_effect=subprocess.CalledProcessError(
                1,
                [*mod.DOCKER, "create"],
                stderr="privileged operations denied",
            )
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        mod.setup(cfg.root, GPU, allow_privileged_host_driver=True)
    [report] = cfg.root.glob("setup-*.json")
    value = json.loads(report.read_bytes())
    assert value["status"] == "failed" and value["process_restore"] == "NOT_RUN"
    assert value["stderr"] == "privileged operations denied"
    assert not (cfg.root / "broker.json").exists()


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("Driver", "fuse-overlayfs"),
        ("SecurityOptions", ["name=rootless"]),
        ("SecurityOptions", ["name=userns"]),
        ("Swarm", {"LocalNodeState": "active"}),
        ("Runtimes", {}),
        ("Architecture", "aarch64"),
    ],
)
def test_deny_incompatible_engine(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch, field: str, bad: object
) -> None:
    monkeypatch.setattr(mod, "_local_host", Mock())
    info = {
        "OSType": "linux",
        "Architecture": "x86_64",
        "KernelVersion": mod.platform.release(),
        "Driver": "overlay2",
        "SecurityOptions": [],
        "Runtimes": {"nvidia": {}},
        "Swarm": {"LocalNodeState": "inactive"},
    }
    info[field] = bad
    monkeypatch.setattr(mod, "_docker", Mock(return_value=json.dumps(info)))
    with pytest.raises(ValueError, match="Unsupported Docker"):
        mod._host(GPU, time.monotonic() + 10)
