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
"""CPU-only coverage of the warm source profile. Never invoke a Docker daemon."""

import importlib
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import ANY, Mock, call

import pytest

pytestmark = pytest.mark.cpu_only
TEMPLATE = "1" * 32
CID = "2" * 64
IMAGE = "sha256:" + "3" * 64
PIN = "registry.example:5000/serve@sha256:" + "4" * 64
KEY = "operator-test-secret"
GPU = "GPU-12345678-1234-4321-abcd-0123456789ab"


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "examples/serve/checkpoint")
    )
    return importlib.import_module("source_launcher")


@pytest.fixture
def site(launcher: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    site = Mock()
    site.config = launcher.deployment.Deployment(
        tmp_path, GPU, IMAGE, Path("/var/lib/docker"), TEMPLATE, {}
    )
    for name in ("controls", "warp-state", "checkpoints", "interpose"):
        (tmp_path / name).mkdir(mode=0o700)
    site.info.return_value = {"id": IMAGE, "arch": "amd64", "os": "linux", "volumes": None}
    site.create.return_value = CID
    site.profile.return_value = {"image": IMAGE}
    monkeypatch.setenv("TRTLLM_RUNTIME_CONTROL_API_KEY", KEY)
    monkeypatch.setattr(launcher.deployment, "check_health", site.health)
    monkeypatch.setattr(launcher.Adapter, "profile", site.profile)
    monkeypatch.setattr(launcher, "EngineClient", site.engine)
    monkeypatch.setattr(launcher, "generate_tokens", site.generate)
    configs: set[Path] = set()

    def run(
        argv: list[str], *, input: bytes, env: dict, **kwargs: object
    ) -> subprocess.CompletedProcess:
        command = next(
            i for i, arg in enumerate(argv) if arg in {"image", "create", "exec", "start"}
        )
        options, args = argv[1:command], argv[command:]
        assert argv[0] == "/usr/bin/docker" and "--host=unix:///var/run/docker.sock" in options
        assert options.count("--config") == 1
        config = Path(options[options.index("--config") + 1])
        assert config.is_absolute() and config.is_dir() and not config.is_symlink()
        assert config.stat().st_mode & 0o777 == 0o700 and not any(config.iterdir())
        assert config not in configs
        configs.add(config)
        assert env == {"PATH": os.defpath, "HOME": str(config)}
        assert all(KEY not in arg for arg in argv)
        assert kwargs["check"] and kwargs["timeout"] > 0
        if args[0] == "image":
            assert args[1:3] == ["inspect", "--format"] and args[-1] == PIN and not input
            reply = json.dumps(site.info())
        elif args[0] == "create":
            reply = site.create(args[1:], input)
        elif args[0] == "exec":
            assert args[1:7] == [
                "--env",
                "TRTLLM_RUNTIME_CONTROL_API_KEY=",
                CID,
                "python3",
                "-IS",
                "-c",
            ]
            assert not input
            site.control_binding(CID)
            info = (tmp_path / "controls" / TEMPLATE).stat()
            reply = json.dumps([info.st_dev, info.st_ino, "/snapshot-control", GPU])
        else:
            assert args == ["start", CID] and not input
            site.start(CID)
            reply = CID
        return subprocess.CompletedProcess(argv, 0, reply.encode())

    monkeypatch.setattr(subprocess, "run", run)
    yield site
    assert all(not os.path.lexists(config) for config in configs)


def test_private_warm_source(launcher: ModuleType, site: Mock, tmp_path: Path) -> None:
    assert launcher.launch(site.config, TEMPLATE, PIN, KEY, 60) == CID
    control = tmp_path / "controls" / TEMPLATE
    assert control.stat().st_mode & 0o777 == 0o700 and not any(control.iterdir())
    args, data = site.create.call_args.args
    assert data == f"TRTLLM_RUNTIME_CONTROL_API_KEY={KEY}\n".encode()
    assert args[args.index("--env-file") + 1] == "/dev/stdin"
    assert args[args.index("--name") + 1] == f"warp-source-{TEMPLATE}"
    assert args[args.index("--label") + 1] == f"com.nvidia.trtllm.snapshot.attempt={TEMPLATE}"
    assert args[args.index("--gpus") + 1] == f"device={GPU}"
    assert {
        "--pull=never",
        "--network=none",
        "--ipc=private",
        "--restart=no",
        "--no-healthcheck",
        "--shm-size=1g",
        "--cpus=16",
        "--memory=128g",
        "--memory-swap=128g",
        f"NVIDIA_VISIBLE_DEVICES={GPU}",
        "SNAPSHOT_CONTROL_DIR=/snapshot-control",
        "TLLM_WORKER_USE_SINGLE_PROCESS=0",
        "OMPI_MCA_btl_tcp_if_include=lo",
        "HF_HUB_OFFLINE=1",
    } <= set(args)
    assert not set(args) & {"--publish", "-p", "-P", "--privileged", "-v", "--volume"}
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"] == [
        f"type=bind,src={control},dst=/snapshot-control",
        f"type=bind,src={tmp_path}/interpose,dst=/tmp/snapshot-cuda,readonly",
    ]
    assert args[args.index("--entrypoint") + 1 :] == [
        "/tmp/snapshot-cuda/cuinterpose-launch",
        IMAGE,
        "--library",
        "/tmp/snapshot-cuda/libcuinterpose.so",
        "--",
        "trtllm-serve",
        "/model",
        "--served_model_name",
        "resident-tp1",
        "--config",
        "/etc/trtllm/resident-tp1.yaml",
        "--enable_checkpoint_mode",
        "--num_serve_frontends",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert site.mock_calls == [
        call.health(site.config, deadline=ANY),
        call.info(),
        call.create(ANY, ANY),
        call.start(CID),
        call.engine(8000, KEY, container=CID),
        call.profile(CID, source=True),
        call.engine()._exchange("/ready", ANY),
        call.control_binding(CID),
        call.engine()._status("running", ANY),
    ]


@pytest.mark.parametrize(
    "template,image,key",
    [
        ("../source", PIN, KEY),
        (TEMPLATE, "registry.example/serve:latest", KEY),
        (TEMPLATE, PIN, "key\nOTHER=1"),
        (TEMPLATE, PIN, ""),
    ],
)
def test_invalid_launch_never_contacts_host(
    launcher: ModuleType, site: Mock, template: str, image: str, key: str
) -> None:
    with pytest.raises(ValueError):
        launcher.launch(site.config, template, image, key, 60)
    assert not site.mock_calls


@pytest.mark.parametrize("fault", ["broker", "volume", "architecture"])
def test_preflight_precedes_claim(
    launcher: ModuleType, site: Mock, tmp_path: Path, fault: str
) -> None:
    if fault == "broker":
        site.health.side_effect = ValueError("unhealthy broker")
    elif fault == "volume":
        site.info.return_value["volumes"] = {"/model": {}}
    else:
        site.info.return_value["arch"] = "arm64"
    with pytest.raises(ValueError):
        launcher.launch(site.config, TEMPLATE, PIN, KEY, 60)
    site.create.assert_not_called()
    assert not (tmp_path / "controls" / TEMPLATE).exists()


def test_uncertain_create_never_adopts_or_retries(launcher: ModuleType, site: Mock) -> None:
    site.create.side_effect = TimeoutError("reply lost")
    with pytest.raises(TimeoutError, match="reply lost"):
        launcher.launch(site.config, TEMPLATE, PIN, KEY, 60)
    with pytest.raises(FileExistsError):
        launcher.launch(site.config, TEMPLATE, PIN, KEY, 60)
    site.create.assert_called_once()
    site.start.assert_not_called()


def test_ready_without_signed_s3_status_is_not_success(
    launcher: ModuleType, site: Mock, tmp_path: Path
) -> None:
    output = tmp_path / "probe.json"
    site.engine.return_value._status.side_effect = RuntimeError("Wrong profile")
    with pytest.raises(RuntimeError, match="Wrong profile"):
        launcher.launch(site.config, TEMPLATE, PIN, KEY, 60, probe_out=output)
    site.start.assert_called_once_with(CID)
    site.engine.return_value._exchange.assert_called_once_with("/ready", ANY)
    site.generate.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("collision", [False, True], ids=["publish", "other-writer"])
def test_probe_generation_is_exclusive(
    launcher: ModuleType, site: Mock, tmp_path: Path, collision: bool
) -> None:
    output = tmp_path / "probe.json"

    def generate(
        container: str, port: int, model: str, prompt: list[int], count: int, budget: float
    ) -> list[int]:
        site.engine.return_value._status.assert_called_once_with("running", ANY)
        assert (container, port, model, prompt, count) == (CID, 8000, "resident-tp1", [17], 4)
        assert 0 < budget <= 60
        if collision:
            output.write_text("other probe")
        return [19, 23, 29, 31]

    site.generate.side_effect = generate
    if collision:
        with pytest.raises(FileExistsError):
            launcher.launch(site.config, TEMPLATE, PIN, KEY, 60, probe_out=output)
        assert output.read_text() == "other probe"
    else:
        assert launcher.launch(site.config, TEMPLATE, PIN, KEY, 60, probe_out=output) == CID
        assert json.loads(output.read_bytes()) == {
            "prompt_token_ids": [17],
            "expected_token_ids": [19, 23, 29, 31],
        }
    site.generate.assert_called_once()
