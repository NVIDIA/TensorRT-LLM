# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production protocol; CUDA/CRIU tests require a real Linux host."""

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    """Load a CPU-only module without importing TRTLLM's GPU extension.

    Args:
        name: Test-local module name.
        path: Production module file.

    Returns:
        Executed production module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = _load("snapshot_protocol", _ROOT / "tensorrt_llm/serve/snapshot.py")
sys.modules.setdefault(
    "snapshot_probe", _load("snapshot_probe", _ROOT / "scripts/snapshot_probe.py")
)
host = _load("snapshot_startup", _ROOT / "scripts/snapshot_startup.py")


@pytest.fixture
def control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a real private control directory and template identity."""
    path = tmp_path / "control"
    path.mkdir(mode=0o700)
    monkeypatch.setenv("TRTLLM_SNAPSHOT_DIR", str(path))
    protocol.write_record(path / "template.json", {"template_id": "template"})
    return path


def _executor() -> SimpleNamespace:
    """Supply only the GPU/engine boundary needed by the production protocol."""
    return SimpleNamespace(
        llm_args=SimpleNamespace(
            load_format="auto",
            kv_cache_config=SimpleNamespace(enable_block_reuse=False, host_cache_size=0),
        ),
        dist=SimpleNamespace(world_size=1),
        global_rank=0,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        draft_model_engine=None,
        is_encoder_decoder=False,
        dwdp_manager=None,
        worker_started=False,
        active_requests=[],
        previous_batch=None,
        model_engine=SimpleNamespace(
            cuda_graph_runner=SimpleNamespace(graphs={"batch1": object()})
        ),
    )


def test_startup_waits_for_fresh_restore(control: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook publishes capture evidence before memory/runtime acknowledgements."""
    synchronize = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
    )

    def restore_after_capture(_: float) -> None:
        """Simulate only the external host's completion boundary."""
        assert (control / "capture-0.json").exists()
        assert not (control / "runtime-0.json").exists()
        protocol.write_record(
            control / "restore.json",
            {"template_id": "template", "session_id": "fresh", "validation_token": "secret"},
        )

    monkeypatch.setattr(protocol.time, "sleep", restore_after_capture)
    protocol.startup_checkpoint(_executor())
    assert synchronize.call_count == 2
    assert protocol.read_record(control / "memory-0.json")["session_id"] == "fresh"
    assert protocol.read_record(control / "runtime-0.json")["session_id"] == "fresh"
    assert not (control / "activate.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_started", True),
        ("active_requests", [1]),
        ("kv_cache_transceiver", object()),
        ("kv_connector_manager", object()),
        ("draft_model_engine", object()),
        ("dwdp_manager", object()),
        ("is_encoder_decoder", True),
    ],
)
def test_unsafe_startup_rejected(
    control: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Unsupported live/transfer profiles cannot publish capture permission."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    executor = _executor()
    setattr(executor, field, value)
    with pytest.raises(ValueError):
        protocol.startup_checkpoint(executor)
    assert not (control / "capture-0.json").exists()


@pytest.mark.parametrize(
    "restore",
    [
        {"template_id": "wrong", "session_id": "fresh", "validation_token": "secret"},
        {"template_id": "template"},
    ],
)
def test_stale_restore_rejected(
    control: Path, monkeypatch: pytest.MonkeyPatch, restore: dict
) -> None:
    """Bad identities do not become runtime-ready."""
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
    )
    protocol.write_record(control / "restore.json", restore)
    with pytest.raises(ValueError, match="restore session"):
        protocol.startup_checkpoint(_executor())
    assert not (control / "runtime-0.json").exists()


@pytest.mark.parametrize(
    "state,token,path,expected",
    [
        ("captured", "", "/health", 503),
        ("runtime", "", "/health", 503),
        ("runtime", "wrong", "/v1/completions", 503),
        ("runtime", "secret", "/v1/completions", 200),
        ("runtime", "secret", "/v1/chat/completions", 503),
        ("stale", "", "/v1/completions", 503),
        ("active", "", "/v1/completions", 200),
        ("aborted", "secret", "/v1/completions", 503),
    ],
)
def test_admission_boundaries(
    control: Path, state: str, token: str, path: str, expected: int
) -> None:
    """Memory/runtime restore never grants ordinary serving authority."""
    session = {"template_id": "template", "session_id": "fresh", "validation_token": "secret"}
    if state != "captured":
        protocol.write_record(control / "restore.json", session)
    if state in {"active", "stale", "aborted"}:
        protocol.write_record(
            control / "activate.json",
            {"template_id": "template", "session_id": "old" if state == "stale" else "fresh"},
        )
    if state == "aborted":
        protocol.write_record(control / "abort.json", {"session_id": "fresh"})
    messages = []

    async def send(message: dict) -> None:
        """Record the real middleware's ASGI messages."""
        messages.append(message)

    async def app(scope: dict, receive: object, send: object) -> None:
        """Represent only the wrapped inference application."""
        await send({"type": "http.response.start", "status": 200})

    gate = protocol.SnapshotAdmission(app, control)
    asyncio.run(
        gate(
            {
                "type": "http",
                "path": path,
                "headers": [(b"x-trtllm-snapshot-validation", token.encode())],
            },
            None,
            send,
        )
    )
    assert messages[0]["status"] == expected


def test_private_control_directory(control: Path) -> None:
    """World-readable control tokens are not accepted."""
    assert protocol.control_directory() == control
    control.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        protocol.control_directory()


def test_snapshot_off_does_not_touch_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary startup does not access any checkpoint-only engine state."""
    monkeypatch.delenv("TRTLLM_SNAPSHOT_DIR", raising=False)
    protocol.startup_checkpoint(None)
    protocol.validate_launch("0.0.0.0", {}, False)


@pytest.mark.parametrize(
    "host_name,args,standalone",
    [
        ("0.0.0.0", {"backend": "pytorch"}, True),
        ("127.0.0.1", {"backend": "pytorch", "num_serve_frontends": 2}, True),
        ("127.0.0.1", {"backend": "pytorch", "orchestrator_type": "ray"}, True),
        ("127.0.0.1", {"backend": "pytorch"}, False),
    ],
)
def test_launch_guards(control: Path, host_name: str, args: dict, standalone: bool) -> None:
    """Unsupported launch topology is refused before expensive model loading."""
    with pytest.raises(ValueError):
        protocol.validate_launch(host_name, args, standalone)


def test_host_profile_contract() -> None:
    """The host adapter refuses missing identity and unsupported topology."""
    profile = dict.fromkeys(host._PROFILE_KEYS, "pinned")
    profile["topology"] = {"nodes": 1, "tp": 1, "pp": 1, "cp": 1}
    host.validate_profile(profile)
    profile["topology"]["nodes"] = 2
    with pytest.raises(ValueError):
        host.validate_profile(profile)
    with pytest.raises(ValueError):
        host.validate_profile({})


def test_artifact_hash_detects_changes(tmp_path: Path) -> None:
    """Actual artifact bytes, not filenames, determine the integrity check."""
    path = tmp_path / "image"
    path.write_bytes(b"before")
    before = host.file_digest(path)
    path.write_bytes(b"after")
    assert host.file_digest(path) != before


def test_owned_pid_cleanup_ignores_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reused PID is never killed during failure cleanup."""
    monkeypatch.setattr(host, "process_tree", lambda pid: {pid: "new"})
    kill = Mock()
    monkeypatch.setattr(os, "kill", kill)
    host.terminate_tree({1234: "old"})
    kill.assert_not_called()
