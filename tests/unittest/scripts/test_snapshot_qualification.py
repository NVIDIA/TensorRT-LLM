# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise qualification invariants without pretending to execute CUDA restore."""

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _ROOT / "examples/serve/snapshot"


def _load(name: str) -> ModuleType:
    """Load a standalone example without importing the GPU package.

    Args:
        name: Example module basename.

    Returns:
        Loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_EXAMPLE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_EXAMPLE))
    return module


qualification = _load("qualify")
observer = _load("snapshot_observer")
client = _load("private_client")
host_io = _load("host_io")


@pytest.mark.parametrize("tp", [1, 2])
def test_profile_is_bounded(tp: int) -> None:
    """The admitted profile deliberately excludes unqualified KV features."""
    config = qualification.serve_config(tp)
    assert config["tensor_parallel_size"] == tp
    assert config["kv_cache_config"]["use_kv_cache_manager_v2"] is True
    assert config["kv_cache_config"]["enable_block_reuse"] is False
    assert config["kv_cache_config"]["host_cache_size"] == 0
    assert config["sleep_config"]["restore_modes"]["kv_cache"] == "NONE"


@pytest.mark.parametrize("tp", [0, 3, 8, -1])
def test_unsupported_profile_is_rejected(tp: int) -> None:
    """Unsupported ranks cannot be silently mapped to a supported profile."""
    with pytest.raises(ValueError):
        qualification.serve_config(tp)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"rank": 0, "attempt": "fresh"}],
        [{"rank": 0, "attempt": "fresh"}] * 2,
        [{"rank": 0, "attempt": "fresh"}, {"rank": 1, "attempt": "stale"}],
    ],
)
def test_cohort_rejects_missing_duplicate_and_stale(records: list[dict]) -> None:
    """Readiness is a complete fresh cohort, not a healthy rank zero."""
    with pytest.raises(ValueError):
        qualification.validate_cohort(records, "fresh", 2)


def _counts(attempt: str, replay: int) -> list[dict]:
    """Construct observer evidence for checking the actual validator.

    Args:
        attempt: Evidence nonce.
        replay: Graph replay counter.

    Returns:
        Two-rank observed-counter records.
    """
    return [
        {
            "rank": rank,
            "attempt": attempt,
            "counts": {
                "model_load": 1,
                "post_load_walk": 1,
                "graph_capture": 2,
                "graph_replay": replay,
            },
        }
        for rank in range(2)
    ]


def test_every_rank_must_replay_without_startup() -> None:
    """Real reuse evidence must show a positive replay delta for both ranks."""
    before, after = _counts("source", 5), _counts("restore", 7)
    qualification.validate_reuse(before, after, "restore", 2)
    after[1]["counts"]["graph_replay"] = 5
    with pytest.raises(ValueError, match="Rank 1.*no restored graph replay"):
        qualification.validate_reuse(before, after, "restore", 2)


@pytest.mark.parametrize(
    "event",
    [
        "model_load",
        "model_reload",
        "weight_population",
        "post_load_walk",
        "staged_transform_walk",
        "graph_capture",
    ],
)
def test_repeated_startup_is_not_reuse(event: str) -> None:
    """A correct completion cannot conceal reloading or graph recapture."""
    before, after = _counts("source", 5), _counts("restore", 7)
    after[1]["counts"][event] = after[1]["counts"].get(event, 0) + 1
    with pytest.raises(ValueError, match=f"repeated {event}"):
        qualification.validate_reuse(before, after, "restore", 2)


def _state() -> dict:
    """Return a small state record for the real reconstruction validator."""
    return {
        "weights": {"manifest_sha256": "same"},
        "weight_addresses": {"w": 1000},
        "graph_keys": ["batch1"],
        "graph_runner": "CudaGraphRunner",
        "runner_input_storage": {"input_ids_cuda": {"address": 3000}},
        "kv": {"owners": [{"id": -1}], "pools": [{"address": 2000}], "tiers": [{"bytes": 1024}]},
        "weight_observation": {"created_at": "source"},
    }


def test_incidental_weight_observation_is_not_state_identity() -> None:
    """Timestamps and hash duration may differ without invalidating weights."""
    before, after = _state(), _state()
    after["weight_observation"] = {"created_at": "restored", "build_seconds": 7}
    observer._require_same_state(before, after)


@pytest.mark.parametrize(
    "field",
    [
        "weights",
        "weight_addresses",
        "graph_keys",
        "graph_runner",
        "runner_input_storage",
        "owners",
        "pools",
        "tiers",
    ],
)
def test_required_state_changes_fail(field: str) -> None:
    """A changed content, address, owner or graph is an explicit failure."""
    before, after = _state(), _state()
    (after["kv"] if field in ("owners", "pools", "tiers") else after)[field] = "changed"
    with pytest.raises(RuntimeError):
        observer._require_same_state(before, after)


def test_sitecustomize_does_not_initialize_torch_or_mpi() -> None:
    """Observers wait for normal worker imports rather than changing bootstrap order."""
    env = {**os.environ, "PYTHONPATH": str(_EXAMPLE), "TRTLLM_SNAPSHOT_EXPERIMENT": "1"}
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sys; print(any(n in sys.modules for n in ('torch', 'mpi4py', 'tensorrt_llm')))",
        ],
        env=env,
        text=True,
    )
    assert output.strip() == "False"


def test_observer_calls_real_implementation_and_counts_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper preserves returns/errors and counts only completed calls."""

    class Example:
        def operation(self, value: int) -> int:
            """Use a small non-GPU implementation as the observed boundary."""
            if value < 0:
                raise ValueError("negative")
            return value * 2

    monkeypatch.setattr(observer, "_COUNTS", {})
    monkeypatch.setattr(observer, "_RANK", None)
    observer._observe(Example, "operation", "event")
    assert Example().operation(3) == 6
    with pytest.raises(ValueError):
        Example().operation(-1)
    assert observer._COUNTS == {"event": 1}


def test_creator_designation_survives_timer_replacement() -> None:
    """A rebound warmup timer must not reject a final executor as profiling."""
    from types import SimpleNamespace

    class Engine:
        def set_warmup_purpose(self, purpose: str) -> None:
            """Retain the native setter's side effect at the isolated boundary."""
            self.timer = SimpleNamespace(purpose=purpose)

    observer._install_engine_observers(SimpleNamespace(ModelEngine=Engine))
    engine = Engine()
    engine.set_warmup_purpose("memory_profiling")
    assert observer._PURPOSES[engine] == "memory_profiling"
    engine.set_warmup_purpose("final_executor")
    engine.timer = SimpleNamespace(purpose="unspecified")
    assert observer._PURPOSES[engine] == "final_executor"


@pytest.mark.parametrize("purpose", ["memory_profiling", "final_executor", "unknown"])
def test_only_final_worker_enters_capture_gate(
    purpose: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capacity estimation may start workers, but must not enter capture or fail."""
    from types import SimpleNamespace

    events = []

    class Engine:
        """Weak-referenceable stand-in for the designation owner."""

    class Executor:
        def __init__(self) -> None:
            """Bind the designated engine as the native executor does."""
            self.model_engine = Engine()

        def start_worker(self) -> None:
            """Record calls to the original boundary implementation."""
            events.append("start")

    executor = Executor()
    observer._PURPOSES[executor.model_engine] = purpose
    monkeypatch.setattr(observer, "_park", lambda executor: events.append("park"))
    observer._install_executor_observers(SimpleNamespace(PyExecutor=Executor))
    if purpose == "unknown":
        with pytest.raises(RuntimeError, match="identify"):
            executor.start_worker()
        assert events == []
    else:
        executor.start_worker()
        assert events == (["park", "start"] if purpose == "final_executor" else ["start"])


def test_inventory_does_not_mislabel_process_images(tmp_path: Path) -> None:
    """Known GPU payload bytes are separate; CPU-only bytes remain unknown."""
    (tmp_path / "device-0000.bin").write_bytes(b"gpu")
    (tmp_path / "pages-1.img").write_bytes(b"process-plus-conventional-cuda")
    report = qualification.inventory(tmp_path)
    assert report["gpu_device_payload_bytes"] == 3
    assert report["logical_bytes"] == 33
    assert report["cpu_only_payload_bytes"] is None
    assert all(len(item["sha256"]) == 64 for item in report["files"])


def test_inventory_rejects_escaped_symlink(tmp_path: Path) -> None:
    """Artifact traversal cannot hash unrelated host files."""
    root = tmp_path / "artifact"
    root.mkdir()
    (tmp_path / "outside").write_text("outside")
    (root / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes root"):
        qualification.inventory(root)


def test_failed_preflight_is_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable Docker host is a failed real attempt, never a passing mock."""
    args = argparse.Namespace(output=tmp_path / "run", gpu=["GPU-test"], timeout=1)
    experiment = qualification.Experiment(args)

    def unavailable(*args: object, **kwargs: object) -> str:
        """Simulate only the external Docker executable boundary."""
        raise FileNotFoundError("Docker unavailable")

    monkeypatch.setattr(experiment, "command", unavailable)
    assert experiment.run() == 1
    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "FAIL"
    assert "Docker unavailable" in report["error"]
    assert report["trials"] == []
    assert args.output.stat().st_mode & 0o777 == 0o700


def test_command_timeout_retains_partial_output(tmp_path: Path) -> None:
    """A real timed-out child leaves both streams available for failure diagnosis."""
    experiment = qualification.Experiment(argparse.Namespace(output=tmp_path / "run", timeout=1))
    log = experiment.root / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        experiment.command(
            [
                sys.executable,
                "-c",
                "import sys,time; print('partial output', flush=True); "
                "sys.stderr.buffer.write(b'partial error\\xff\\n'); sys.stderr.flush(); time.sleep(60)",
            ],
            log=log,
        )
    assert "partial output" in log.read_text()
    assert "partial error" in log.read_text()
    assert "timed out" in log.read_text()


def test_failed_negative_case_has_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timeouts cannot leave negative cases running or discard partial diagnostics."""
    experiment = qualification.Experiment(
        argparse.Namespace(output=tmp_path / "run", gpu=["test"], timeout=1)
    )
    monkeypatch.setattr(experiment, "command", lambda *args, **kwargs: "host-data")

    def fail_case() -> None:
        """Isolate the external host timeout while retaining the real finalizer."""
        experiment.report["negative_tests"] = [{"case": "stale_target", "status": "RUNNING"}]
        raise subprocess.TimeoutExpired("docker", 1, output=b"partial out", stderr=b"partial err")

    monkeypatch.setattr(experiment, "run_selected_profiles", fail_case)
    assert experiment.run() == 1
    report = json.loads((experiment.root / "report.json").read_text())
    assert report["negative_tests"][0]["status"] == "FAIL"
    assert "timed out" in report["negative_tests"][0]["error"]
    assert report["command_output"] == "partial out"
    assert report["command_stderr"] == "partial err"


def test_cleanup_rejects_changed_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleanup never kills a container that fails immutable ownership checks."""
    experiment = qualification.Experiment(argparse.Namespace(output=tmp_path / "run", timeout=1))
    experiment.containers["owned"] = "attempt"
    calls = []

    def inspect(argv: list[str], **kwargs: object) -> str:
        """Supply a foreign Docker label at the external inspection boundary."""
        calls.append(copy.deepcopy(argv))
        return json.dumps([{"Id": "owned", "Config": {"Labels": {qualification.LABEL: "foreign"}}}])

    monkeypatch.setattr(experiment, "command", inspect)
    with pytest.raises(RuntimeError, match="ownership"):
        experiment.cleanup("owned")
    assert calls == [["docker", "inspect", "owned"]]


@pytest.mark.parametrize("failure", ["timeout", "write"])
def test_cleanup_removes_owned_container_when_log_collection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Diagnostic failures must not leak a verified owned container."""
    experiment = qualification.Experiment(argparse.Namespace(output=tmp_path / "run", timeout=1))
    experiment.containers["owned"] = "attempt"
    calls = []

    def command(argv: list[str], **kwargs: object) -> str:
        """Isolate Docker without replacing the cleanup implementation."""
        calls.append(argv)
        return json.dumps([{"Id": "owned", "Config": {"Labels": {qualification.LABEL: "attempt"}}}])

    def logs(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        """Fail the external log operation or its output destination."""
        if failure == "timeout":
            raise subprocess.TimeoutExpired("docker logs", 30)
        (experiment.root / "logs" / "owned.log").mkdir()
        return subprocess.CompletedProcess([], 0, stdout="logs", stderr="")

    monkeypatch.setattr(experiment, "command", command)
    monkeypatch.setattr(qualification.subprocess, "run", logs)
    with pytest.raises((subprocess.TimeoutExpired, OSError)):
        experiment.cleanup("owned")
    assert calls[-1] == ["docker", "rm", "--force", "owned"]
    assert not experiment.containers


@pytest.mark.parametrize("remaining", [0, 1, 1024])
def test_compact_requires_actual_backing_release(remaining: int) -> None:
    """A no-op release cannot be reported as a compact capture."""
    before = {"tiers": [{"bytes": {"gpu": 1024, "host": 0, "disk": 0}}]}
    after = {"tiers": [{"bytes": {"gpu": remaining, "host": 0, "disk": 0}}]}
    if remaining:
        with pytest.raises(RuntimeError, match="retained"):
            observer.require_compacted(before, after)
    else:
        observer.require_compacted(before, after)


@pytest.mark.parametrize(
    "address",
    ["example.com:8000", "10.0.0.1:8000", "127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:abc"],
)
def test_restore_endpoint_rejects_nonlocal_or_invalid_address(address: str) -> None:
    """Only the retained listener inside the private namespace can be used."""
    with pytest.raises(ValueError):
        client.validate_address(address)
    assert client.validate_address("127.0.0.1:8000\n") == "127.0.0.1:8000"


def test_non_200_success_is_deadline_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 204 must not spin forever outside the exception path."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    now = [0.0]
    monkeypatch.setattr(client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(client.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    opener = SimpleNamespace(open=lambda *args, **kwargs: nullcontext(SimpleNamespace(status=204)))
    with pytest.raises(TimeoutError):
        client.wait_ready(opener, "http://127.0.0.1:8000", 0.3)
    assert now[0] == pytest.approx(0.3)


def test_summary_keeps_failures_and_separates_native_cycles() -> None:
    """Native-only samples never become restore performance evidence."""
    records = [
        {"policy": "compact", "status": "FAIL"},
        {
            "policy": "compact",
            "status": "PASS",
            "cold": {"first_correct_completion_seconds": 20},
            "native_cycle": {"first_correct_completion_seconds": 21},
        },
        {
            "policy": "resident",
            "status": "PASS",
            "cold": {"first_correct_completion_seconds": 20},
            "restore": {"first_correct_completion_seconds": 4},
        },
    ]
    result = qualification.summarize(records)
    assert (result["attempted_trials"], result["passed_trials"]) == (3, 2)
    assert "restore" not in result["policies"]["compact"]
    assert result["policies"]["resident"]["cold_over_restore_ratio"]["p50"] == 5


def test_cleanup_failure_cannot_return_native_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a completed native test is failed if owned-resource cleanup fails."""
    experiment = qualification.Experiment(
        argparse.Namespace(output=tmp_path / "run", gpu=["test"], timeout=1)
    )
    monkeypatch.setattr(experiment, "command", lambda *args, **kwargs: "host-data")

    def selected() -> None:
        """Isolate expensive profile execution, retaining the reporting finalizer."""
        experiment.report["status"] = "NATIVE_ONLY_PASS"
        experiment.containers["leaked"] = "attempt"

    def failed_cleanup(container: str) -> None:
        """Report a failing external container-runtime cleanup."""
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(experiment, "run_selected_profiles", selected)
    monkeypatch.setattr(experiment, "cleanup", failed_cleanup)
    assert experiment.run() == 1
    assert json.loads((experiment.root / "report.json").read_text())["status"] == "FAIL"


def test_device_io_measures_sectors_not_artifact_size(tmp_path: Path) -> None:
    """The real Linux parser and delta calculation preserve device-wide attribution."""
    device = tmp_path.stat().st_dev
    prefix = f"{os.major(device)} {os.minor(device)} test-device"
    stats = tmp_path / "diskstats"
    stats.write_text(f"{prefix} 1 0 8 0 2 0 16 0 0 0 0\n")
    before = host_io.sample_device(tmp_path, stats)
    stats.write_text(f"{prefix} 3 0 24 0 6 0 48 0 0 0 0\n")
    after = host_io.sample_device(tmp_path, stats)
    result = host_io.device_delta(before, after)
    assert result["status"] == "MEASURED"
    assert result["delta"] == {"reads": 2, "read_bytes": 8192, "writes": 4, "write_bytes": 16384}
    assert result["attribution"] == "NOT_PROCESS_OR_ARTIFACT_EXCLUSIVE"
    assert result["interval_seconds"] >= 0


@pytest.mark.parametrize("kind", ["missing", "unmapped", "malformed", "short", "negative"])
def test_device_io_unavailability_is_not_zero_io(tmp_path: Path, kind: str) -> None:
    """Missing or invalid device evidence must not look like an idle disk."""
    device = tmp_path.stat().st_dev
    prefix = f"{os.major(device)} {os.minor(device)} test-device"
    stats = tmp_path / "diskstats"
    if kind == "unmapped":
        stats.write_text("")
    elif kind == "malformed":
        stats.write_text(f"{prefix} bad 0 8 0 2 0 16 0 0 0 0\n")
    elif kind == "short":
        stats.write_text(f"{prefix} 1 0\n")
    elif kind == "negative":
        stats.write_text(f"{prefix} -1 0 8 0 2 0 16 0 0 0 0\n")
    sample = host_io.sample_device(tmp_path, stats)
    assert sample["status"] == "UNAVAILABLE"
    result = host_io.device_delta(sample, sample)
    assert result["status"] == "UNAVAILABLE"
    assert "delta" not in result


@pytest.mark.parametrize("change", ["device_id", "device_name", "counter_reset"])
def test_device_io_rejects_changed_device_or_reset(change: str) -> None:
    """Reinitialized devices cannot create negative or misleading operation costs."""
    before = {
        "status": "AVAILABLE",
        "device_id": [8, 0],
        "device_name": "sda",
        "sample_seconds": 1.0,
        "counters": {"read_bytes": 1024},
    }
    after = copy.deepcopy(before)
    after["sample_seconds"] = 2.0
    if change == "counter_reset":
        after["counters"]["read_bytes"] = 0
    else:
        after[change] = "changed"
    assert host_io.device_delta(before, after)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("fails", [False, True])
def test_host_attempt_always_records_io_and_cleans_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """Successful and failed host operations retain I/O evidence before cleanup."""
    experiment = qualification.Experiment(
        argparse.Namespace(output=tmp_path / "run", timeout=1, containerd_socket="/socket")
    )
    experiment.images = {"host": "host-image", "workload": "workload-image"}
    experiment.gpus = "allocated-gpu"
    experiment.docker_root = "/docker"
    cleaned = []

    def command(argv: list[str], **kwargs: object) -> str:
        """Replace Docker execution only; retain real host orchestration/finalizers."""
        if argv[1] == "create":
            return "helper"
        if fails:
            raise subprocess.CalledProcessError(1, argv, output="host failed")
        return json.dumps({"status": "PASS", "attempt": "attempt"})

    monkeypatch.setattr(experiment, "command", command)
    monkeypatch.setattr(experiment, "cleanup", cleaned.append)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            experiment.host("capture", "target", "attempt", "artifact", experiment.root)
    else:
        assert (
            experiment.host("capture", "target", "attempt", "artifact", experiment.root)["status"]
            == "PASS"
        )
    evidence = json.loads((experiment.root / "capture-device-io.json").read_text())
    assert evidence["status"] in ("MEASURED", "UNAVAILABLE")
    assert cleaned == ["helper"]
