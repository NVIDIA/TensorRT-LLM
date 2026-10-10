# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production protocol; CUDA/CRIU tests require a real Linux host."""

import argparse
import asyncio
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path) -> object:
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
        dist=SimpleNamespace(
            world_size=1,
            mapping=SimpleNamespace(pp_size=1, cp_size=1, gpus_per_node=1),
            allgather=lambda value: [value],
        ),
        enable_attention_dp=False,
        global_rank=0,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        draft_model_engine=None,
        is_encoder_decoder=False,
        dwdp_manager=None,
        worker_started=False,
        active_requests=[],
        previous_batch=None,
        _pending_transfer_responses=[],
        _pending_response_terminations=[],
        request_accumulated=[],
        inflight_req_ids=set(),
        num_scheduled_requests=0,
        num_unscheduled_requests=0,
        kv_cache_manager=SimpleNamespace(_snapshot_startup_state=lambda: {"pools": [123]}),
        model_engine=SimpleNamespace(
            cuda_graph_runner=SimpleNamespace(
                graphs={"batch1": SimpleNamespace(raw_cuda_graph_exec=lambda: 100)}
            ),
            model=SimpleNamespace(named_parameters=lambda: [], named_buffers=lambda: []),
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
        ("expired", "secret", "/v1/completions", 503),
    ],
)
def test_admission_boundaries(
    control: Path, state: str, token: str, path: str, expected: int
) -> None:
    """Memory/runtime restore never grants ordinary serving authority."""
    session = {"template_id": "template", "session_id": "fresh", "validation_token": "secret"}
    protocol.write_record(
        control / "lease.json", {"session_id": "fresh", "expires_at": time.time() + 60}
    )
    if state != "captured":
        protocol.write_record(control / "restore.json", session)
    if state in {"active", "stale", "aborted"}:
        protocol.write_record(
            control / "activate.json",
            {"template_id": "template", "session_id": "old" if state == "stale" else "fresh"},
        )
    if state == "aborted":
        protocol.write_record(control / "abort.json", {"session_id": "fresh"})
    if state == "expired":
        protocol.write_record(
            control / "lease.json", {"session_id": "fresh", "expires_at": time.time() - 1}
        )
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


@pytest.mark.parametrize(
    "command",
    [
        ["trtllm-serve", "model", "{address}"],
        ["trtllm-serve", "model", "--report_addr", "elsewhere", "{address}"],
        ["trtllm-serve", "model", "--report_addr", "{address}", "--report_addr", "elsewhere"],
    ],
)
def test_capture_requires_native_address_contract(tmp_path: Path, command: list[str]) -> None:
    """Reject commands that cannot publish the guarded native HTTP endpoint.

    Args:
        tmp_path: Isolated test directory.
        command: Invalid native serving command.
    """
    profile = dict.fromkeys(host._PROFILE_KEYS, "pinned")
    profile["topology"] = {"nodes": 1, "tp": 1, "pp": 1, "cp": 1}
    path = tmp_path / "profile.json"
    host.probe._write_report(path, profile)
    with pytest.raises(ValueError, match="--report_addr"):
        host.capture(argparse.Namespace(profile=path, command=command))


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


@pytest.mark.parametrize("changed", ["pool", "graph", "scheduler", "peer"])
def test_restore_validity_failure_keeps_rank_closed(
    control: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """Changed memory, metadata or MPI membership never becomes runtime-ready."""
    executor = _executor()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
    )

    def finish_restore(_: float) -> None:
        """Mutate one external restore condition after capture evidence exists."""
        protocol.write_record(
            control / "restore.json",
            {"template_id": "template", "session_id": "fresh", "validation_token": "secret"},
        )
        if changed == "pool":
            executor.kv_cache_manager._snapshot_startup_state = lambda: {"pools": [999]}
        elif changed == "graph":
            executor.model_engine.cuda_graph_runner.graphs["batch1"].raw_cuda_graph_exec = (
                lambda: 101
            )
        elif changed == "scheduler":
            executor.request_accumulated.append(1)
        else:
            executor.dist.allgather = lambda value: [(0, "old", "template")]

    monkeypatch.setattr(protocol.time, "sleep", finish_restore)
    with pytest.raises(ValueError):
        protocol.startup_checkpoint(executor)
    assert not (control / "runtime-0.json").exists()


def _rank(rank: int, count: int = 2) -> dict:
    """Build external rank evidence for the coordinator contract tests.

    Args:
        rank: Logical rank.
        count: Complete cohort size.

    Returns:
        A rank's external checkpoint acknowledgement.
    """
    return {
        "rank": rank,
        "world_size": count,
        "pid": 900000 + rank,
        "hostname": socket.gethostname(),
        "template_id": "template",
        "session_id": "fresh",
    }


def test_partial_cohort_is_not_ready(control: Path) -> None:
    """All configured ranks, not only rank zero, must acknowledge restore."""
    protocol.write_record(control / "runtime-0.json", _rank(0))
    assert host.rank_records(control, "runtime", 2, "template", "fresh") == []
    protocol.write_record(control / "runtime-1.json", _rank(1))
    assert len(host.rank_records(control, "runtime", 2, "template", "fresh")) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("session_id", "old"),
        ("template_id", "old"),
        ("hostname", "another-node"),
        ("world_size", 3),
        ("rank", 0),
        ("pid", 900000),
    ],
)
def test_stale_or_wrong_cohort_rejected(control: Path, key: str, value: object) -> None:
    """No stale, remote, duplicated or mismatched rank can satisfy readiness."""
    protocol.write_record(control / "runtime-0.json", _rank(0))
    bad = _rank(1)
    bad[key] = value
    protocol.write_record(control / "runtime-1.json", bad)
    with pytest.raises(ValueError):
        host.rank_records(control, "runtime", 2, "template", "fresh")


def test_interposer_uses_verified_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real adapter emits the pinned Snapshot coordinator's argv contract."""
    run = Mock()
    monkeypatch.setattr(host, "run_tool", run)
    host.interposer_operation(
        tmp_path, "prepare", [17, 18], tmp_path / "images", tmp_path / "log", 10
    )
    assert run.call_args.args[0] == [
        str(tmp_path / "cuinterpose-coordinator"),
        "--prepare",
        "--socket-dir",
        "/tmp",
        "--checkpoint-dir",
        str(tmp_path / "images"),
        "--process",
        "17",
        "--process",
        "18",
    ]


@pytest.mark.parametrize("fault", [None, "cuda", "partial", "mismatch", "early-admission"])
def test_host_restore_orders_real_adapter_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    """Run the real restore driver, replacing only privileged OS and HTTP boundaries.

    This checks orchestration, not CUDA/CRIU behavior or native inference.
    """
    artifact = tmp_path / "artifact"
    artifact.mkdir(mode=0o700)
    control = artifact / "control"
    control.mkdir(mode=0o700)
    images = artifact / "images"
    images.mkdir()
    (images / "inventory.img").write_bytes(b"external CRIU image boundary")
    profile = dict.fromkeys(host._PROFILE_KEYS, "pinned")
    profile["topology"] = {"nodes": 1, "tp": 2, "pp": 1, "cp": 1}
    requests = [
        {"model": "test", "prompt": "A", "temperature": 0, "stream": False, "max_tokens": 1}
    ]
    outputs = [{"text": "B", "finish_reason": "length", "completion_tokens": 1}]
    ranks = [
        {key: value for key, value in _rank(rank).items() if key != "session_id"}
        for rank in range(2)
    ]
    manifest = {
        "schema_version": 1,
        "profile": profile,
        "profile_digest": host.probe._digest(profile),
        "artifact_path": str(artifact),
        "template_id": "template",
        "root_pid": 900000,
        "pids": [900000, 900001],
        "cuda_pids": [900000, 900001],
        "host": {"same": True},
        "ranks": ranks,
        "interpose": True,
        "images": {"inventory.img": host.file_digest(images / "inventory.img")},
    }
    for name, record in {
        "manifest.json": manifest,
        "profile.json": profile,
        "requests.json": requests,
        "baseline.json": {
            "mode": "cold",
            "generation_probe_status": "PASS",
            "profile_digest": host.probe._digest(profile),
            "requests_digest": host.probe._digest(requests),
            "outputs": outputs,
        },
    }.items():
        host.probe._write_report(artifact / name, record)
    args = argparse.Namespace(
        artifact=artifact,
        profile=artifact / "profile.json",
        baseline=artifact / "baseline.json",
        requests=artifact / "requests.json",
        snapshot_bin=tmp_path / "tools",
        timeout=0.3 if fault == "partial" else 5,
        serve=False,
    )
    events = []
    alive = {}
    monkeypatch.setattr(host, "host_identity", lambda *args: {"same": True})
    monkeypatch.setattr(host, "process_tree", lambda root: dict(alive))

    def terminate(owned: dict) -> None:
        """Observe cleanup of the exact restored cohort."""
        assert owned == alive
        alive.clear()
        events.append("terminate")

    monkeypatch.setattr(host, "terminate_tree", terminate)

    def run(argv: list[str], log: Path, deadline: float) -> str:
        """Model only the privileged process boundary, retaining real argv."""
        if Path(argv[0]).name == "criu":
            events.append("criu-restore")
            Path(argv[argv.index("--pidfile") + 1]).write_text("900000")
            alive.update({900000: "new-root", 900001: "new-rank"})
        elif "--action" in argv:
            action = argv[argv.index("--action") + 1]
            events.append(f"cuda-{action}-{argv[-1]}")
            if fault == "cuda":
                raise RuntimeError("CUDA restore failed")
        else:
            assert "--restore" in argv
            events.append("interposer-restore")
        return ""

    monkeypatch.setattr(host, "run_tool", run)
    write = host.probe._write_report

    def publish(path: Path, record: dict) -> None:
        """Emulate rank acknowledgements only after the host releases the gate."""
        write(path, record)
        if path.name == "restore.json":
            events.append("release-workers")
            assert events[-2] == "interposer-restore"
            for rank in ranks[:1] if fault == "partial" else ranks:
                for phase in ("memory", "runtime"):
                    write(
                        control / f"{phase}-{rank['rank']}.json",
                        {**rank, "session_id": record["session_id"]},
                    )
        elif path.name == "activate.json":
            assert events[-1] == "private-generation"
            events.append("activate")

    monkeypatch.setattr(host.probe, "_write_report", publish)
    monkeypatch.setattr(host.probe, "_read_address", lambda path: ("127.0.0.1", 1234))

    def http(
        address: tuple,
        path: str,
        timeout: float,
        payload: dict | None = None,
        headers: dict | None = None,
    ) -> tuple[int, bytes]:
        """Emulate only the private HTTP candidate boundary."""
        if path == "/health":
            return (
                200
                if headers or fault == "early-admission" or (control / "activate.json").exists()
                else 503
            ), b""
        assert headers and not (control / "activate.json").exists()
        events.append("private-generation")
        return 200, json.dumps(
            {
                "choices": [
                    {"text": "wrong" if fault == "mismatch" else "B", "finish_reason": "length"}
                ],
                "usage": {"completion_tokens": 1},
            }
        ).encode()

    monkeypatch.setattr(host.probe, "_request", http)
    if fault:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            host.restore(args)
        assert not (control / "activate.json").exists()
    else:
        result = host.restore(args)
        assert result["status"] == "PASS"
        assert result["outputs"] == outputs
        assert events[:6] == [
            "criu-restore",
            "cuda-restore-900000",
            "cuda-restore-900001",
            "cuda-unlock-900000",
            "cuda-unlock-900001",
            "interposer-restore",
        ]
    report = json.loads(next(artifact.glob("restore-*/report.json")).read_text())
    assert report["status"] == ("FAIL" if fault else "PASS")
    assert events[-1] == "terminate"
    assert (control / "abort.json").exists()
