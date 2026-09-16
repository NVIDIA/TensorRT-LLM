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
"""CPU-only orchestration tests; no servers, GPUs, or cache eviction are used."""

import argparse
import copy
import importlib
import io
import json
import signal
import subprocess
import sys
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

pytestmark = pytest.mark.cpu_only
_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "jenkins/scripts/startup_benchmark"
sys.path.insert(0, str(_SCRIPT_DIR))
runner = importlib.import_module("runner")
sys.path.pop(0)


def _matrix() -> dict:
    return {
        "version": 1,
        "cases": [{"name": "small", "model": "${MODEL_ROOT}/weights", "config": {}}],
        "variants": [
            {"name": "native", "config": {"checkpoint_io_policy": "native"}},
            {"name": "striped", "config": {"checkpoint_io_policy": "rank_striped_read_ahead"}},
        ],
    }


def _load(tmp_path: Path, matrix: dict) -> dict:
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
    return runner.load_matrix(path)


def _policy(
    requested: str = "rank_striped_read_ahead",
    selected: str = "rank_striped_read_ahead",
    effective: str = "rank_striped_read_ahead",
    activated: bool = True,
    reason: str = "none",
) -> str:
    return (
        f"Checkpoint I/O policy: requested={requested}, selected={selected}, "
        f"activated={activated}, effective={effective}, fallback_reason={reason}.\n"
    )


def _server_info() -> dict:
    return {
        "startup_metrics": {
            "model_loader": {
                "total_model_loading_seconds": 6.0,
                "checkpoint_preparation_seconds": 1.0,
                "weight_population_seconds": 2.0,
                "checkpoint_finalization_seconds": 0.5,
            }
        }
    }


def test_matrix_defaults_and_placeholders_are_preserved(tmp_path: Path) -> None:
    matrix = _load(tmp_path, _matrix())
    case = matrix["cases"][0]
    assert (case["tp"], case["pp"], case["ep"], case["timeout_seconds"]) == (1, 1, 1, 3600)
    assert case["model"] == "${MODEL_ROOT}/weights"


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("root", "version", 2),
        ("root", "unknown", True),
        ("root", "cases", []),
        ("case", "name", "../unsafe"),
        ("case", "tp", True),
        ("case", "tp", 16),
        ("case", "ep", 2),
        ("case", "timeout_seconds", 0),
        ("case", "optional", "false"),
        ("case", "checkpoint_dirs", "not-a-list"),
        ("case", "config", []),
        ("variant", "config", {}),
    ],
)
def test_invalid_matrix_fails_closed(
    tmp_path: Path, target: str, field: str, value: object
) -> None:
    matrix = _matrix()
    document = (
        matrix if target == "root" else matrix["cases" if target == "case" else "variants"][0]
    )
    document[field] = value
    with pytest.raises(ValueError):
        _load(tmp_path, matrix)


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix["cases"].append(copy.deepcopy(matrix["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        _load(tmp_path, matrix)


def test_case_selection_is_explicit_and_ordered() -> None:
    entries = [{"name": "required"}, {"name": "optional", "optional": True}]
    assert runner.select_names(entries, None) == entries[:1]
    assert runner.select_names(entries, "optional,required") == list(reversed(entries))
    for names in ("missing", "required,required", ""):
        with pytest.raises(ValueError):
            runner.select_names(entries, names)


def test_local_paths_expand_env_but_reject_unresolved_or_non_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STARTUP_TEST_MODEL_ROOT", str(tmp_path))
    monkeypatch.delenv("STARTUP_TEST_UNSET", raising=False)
    assert runner.local_path("${STARTUP_TEST_MODEL_ROOT}") == tmp_path.resolve()
    with pytest.raises(ValueError, match="Unresolved"):
        runner.local_path("${STARTUP_TEST_UNSET}/model")
    path = tmp_path / "weights.bin"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="directory"):
        runner.local_path(str(path))
    with pytest.raises(FileNotFoundError):
        runner.local_path(str(tmp_path / "missing"))


def test_runtime_cache_seed_is_copied_into_independent_trial_directories(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    (seed / "triton").mkdir(parents=True)
    (seed / "triton" / "kernel").write_text("prepared")
    first = runner.runtime_cache_environment(tmp_path / "first", seed)
    second = runner.runtime_cache_environment(tmp_path / "second", seed)
    Path(first["TRITON_CACHE_DIR"], "kernel").write_text("changed")
    assert Path(second["TRITON_CACHE_DIR"], "kernel").read_text() == "prepared"
    assert (seed / "triton" / "kernel").read_text() == "prepared"
    for variable in runner.CACHE_PATHS:
        assert first[variable] != second[variable]
    cold = runner.runtime_cache_environment(tmp_path / "cold", None)
    assert not Path(cold["TRITON_CACHE_DIR"], "kernel").exists()
    with pytest.raises(FileExistsError):
        runner.runtime_cache_environment(tmp_path / "cold", None)


def test_metric_extraction_separates_main_and_draft_and_rejects_invalid_values() -> None:
    info = _server_info()
    info["startup_metrics"]["draft_model_loader"] = {
        "checkpoint_preparation_seconds": 0.1,
        "weight_population_seconds": 0.2,
        "checkpoint_finalization_seconds": 0.3,
    }
    info["startup_metrics"]["model_loader"].update(
        negative_seconds=-1,
        nan_seconds=float("nan"),
        infinite_seconds=float("inf"),
        boolean_seconds=True,
        string_seconds="3",
        unrelated=1,
    )
    metrics = runner.extract_metrics(info)
    assert metrics["checkpoint_pipeline_seconds"] == 3.5
    assert metrics["draft_model_checkpoint_pipeline_seconds"] == pytest.approx(0.6)
    assert not any(
        name in metrics
        for name in (
            "negative_seconds",
            "nan_seconds",
            "infinite_seconds",
            "boolean_seconds",
            "string_seconds",
            "unrelated",
        )
    )
    assert runner.extract_metrics({}) == {}


def test_policy_requires_matching_unambiguous_evidence() -> None:
    requested = "rank_striped_read_ahead"
    assert not runner.parse_policy("no policy evidence", requested)["complete"]
    assert runner.parse_policy(_policy() * 2, requested)["complete"]
    for conflict in (
        _policy(requested="native"),
        _policy(selected="native"),
        _policy(effective="native"),
        _policy(activated=False),
        _policy(reason="fallback"),
    ):
        assert not runner.parse_policy(_policy() + conflict, requested)["complete"]
    inactive = runner.parse_policy(_policy(activated=False), requested)
    assert inactive["complete"] and not inactive["activated"]


def test_readiness_address_file_does_not_imply_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    address = tmp_path / "server.addr"
    address.write_text("127.0.0.1:1234")
    process = Mock(returncode=None)
    process.poll.return_value = None
    opener = Mock()
    opener.open.side_effect = urllib.error.URLError("not ready")
    monkeypatch.setattr(runner.urllib.request, "build_opener", Mock(return_value=opener))
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[0.1, 0.2, 1.1]))
    monkeypatch.setattr(runner.time, "sleep", Mock())
    with pytest.raises(TimeoutError):
        runner.wait_ready(process, address, 0, 1)
    assert opener.open.call_count == 2
    assert opener.open.call_args.args[0] == "http://127.0.0.1:1234/health"


def test_readiness_waits_for_health_and_detects_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    address = tmp_path / "server.addr"
    address.write_text("127.0.0.1:1234")
    process = Mock(returncode=None)
    process.poll.return_value = None
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.side_effect = [urllib.error.URLError("loading"), response]
    monkeypatch.setattr(runner.urllib.request, "build_opener", Mock(return_value=opener))
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[0.1, 0.2, 0.3]))
    monkeypatch.setattr(runner.time, "sleep", Mock())
    assert runner.wait_ready(process, address, 0, 1) == ("127.0.0.1:1234", 0.3)
    process.poll.return_value = 7
    process.returncode = 7
    monkeypatch.setattr(runner.time, "monotonic", Mock(return_value=0.1))
    with pytest.raises(RuntimeError, match="exited before ready: 7"):
        runner.wait_ready(process, address, 0, 1)


def test_stop_server_signals_process_group_and_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    kill = Mock()
    monkeypatch.setattr(runner, "signal_children", kill)
    monkeypatch.setattr(runner, "child_pids", Mock(return_value=[]))
    monkeypatch.setattr(runner.os, "waitpid", Mock(return_value=(0, 0)))
    process = Mock(pid=123)
    process.wait.side_effect = subprocess.TimeoutExpired("server", 30)
    runner.stop_server(process)
    assert [call.args for call in kill.call_args_list] == [
        (signal.SIGTERM,),
        (signal.SIGKILL,),
    ]
    process.wait.assert_called_once_with(timeout=30)
    process.poll.assert_called_once()


def test_unverified_teardown_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "signal_children", Mock())
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[0, 31]))
    with pytest.raises(RuntimeError, match="Unverified process teardown"):
        runner.stop_server(None)


def test_missing_proc_visibility_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.Path, "glob", Mock(return_value=iter(())))
    with pytest.raises(RuntimeError, match="through /proc"):
        runner.child_pids(runner.os.getpid())


@pytest.mark.parametrize("owned", [True, False])
def test_signal_children_checks_parent_and_closes_pidfd(
    monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    parent = runner.os.getpid() + (0 if owned else 1)
    stat = Mock()
    stat.read_text.return_value = f"12345 (worker (with parentheses)) S {parent} 0"
    opened, sent, closed = Mock(return_value=77), Mock(), Mock()
    monkeypatch.setattr(runner, "child_pids", Mock(return_value=[12345]))
    monkeypatch.setattr(runner, "Path", Mock(return_value=stat))
    monkeypatch.setattr(runner.os, "pidfd_open", opened, raising=False)
    monkeypatch.setattr(runner.signal, "pidfd_send_signal", sent, raising=False)
    monkeypatch.setattr(runner.os, "close", closed)
    runner.signal_children(signal.SIGKILL)
    opened.assert_called_once_with(12345)
    closed.assert_called_once_with(77)
    if owned:
        sent.assert_called_once_with(77, signal.SIGKILL)
    else:
        sent.assert_not_called()


@pytest.mark.parametrize("git_available", [False, True])
def test_runtime_identity_uses_installed_commit_and_hashes_runner_without_git(
    monkeypatch: pytest.MonkeyPatch, git_available: bool
) -> None:
    commit = "a" * 40
    metadata = Mock()
    metadata.get_all.return_value = [
        "Homepage, https://github.com/NVIDIA/TensorRT-LLM",
        f"Source Commit, https://github.com/NVIDIA/TensorRT-LLM/commit/{commit}",
    ]
    monkeypatch.setattr(runner.importlib.metadata, "version", Mock(return_value="test-runtime"))
    monkeypatch.setattr(runner.importlib.metadata, "metadata", Mock(return_value=metadata))
    monkeypatch.setattr(
        runner.shutil, "which", Mock(return_value="/bin/git" if git_available else None)
    )
    gpu = subprocess.CompletedProcess([], 0, "GPU-z, B300\nGPU-a, B300\n", "")
    commands = Mock(
        side_effect=[subprocess.CompletedProcess([], 128, "", "no repository"), gpu]
        if git_available
        else [gpu]
    )
    monkeypatch.setattr(runner.subprocess, "run", commands)
    identity = runner.runtime_identity("registry#runtime:pinned")
    assert identity["runtime_git_commit"] == commit
    assert identity["runner_git_commit"] == "unknown"
    assert identity["runtime_version"] == "test-runtime"
    assert identity["gpu_inventory"] == ["GPU-a, B300", "GPU-z, B300"]
    assert len(identity["runner_source_fingerprint"]) == 64
    int(identity["runner_source_fingerprint"], 16)
    assert commands.call_count == (2 if git_available else 1)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper, /proc and pidfd integration")
def test_detached_worker_is_adopted_and_reaped_in_isolated_driver(tmp_path: Path) -> None:
    # Only the subprocess becomes a subreaper; never change pytest's child ownership.
    child = textwrap.dedent("""
        import os, signal, sys, time
        from pathlib import Path
        if os.fork():
            os._exit(0)
        os.setsid()
        if os.fork():
            os._exit(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(sys.argv[1]).write_text(str(os.getpid()))
        time.sleep(20)
        os._exit(0)
    """)
    driver = textwrap.dedent(f"""
        import os, signal, subprocess, sys, time
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        import runner
        runner.enable_subreaper()
        handshake = Path(sys.argv[2])
        process = subprocess.Popen(
            [sys.executable, '-c', {child!r}, str(handshake)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        daemon_fd = None
        try:
            deadline = time.monotonic() + 10
            while not handshake.exists() or not handshake.read_text().strip():
                if time.monotonic() > deadline:
                    raise TimeoutError('Detached worker did not publish its PID')
                time.sleep(0.01)
            daemon_pid = int(handshake.read_text())
            daemon_fd = os.pidfd_open(daemon_pid)
            while daemon_pid not in runner.child_pids(os.getpid()):
                if time.monotonic() > deadline:
                    raise TimeoutError('Detached worker was not adopted')
                time.sleep(0.01)
            runner.stop_server(process)
            assert not runner.child_pids(os.getpid())
            assert not Path(f'/proc/{{daemon_pid}}').exists()
            print('adopted worker reaped')
        finally:
            if daemon_fd is not None:
                try:
                    signal.pidfd_send_signal(daemon_fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(daemon_fd)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    if not os.waitpid(-1, os.WNOHANG)[0]:
                        time.sleep(0.01)
                except ChildProcessError:
                    break
    """)
    # The detached worker also has a finite lifetime if the driver is forcibly killed.
    result = subprocess.run(
        [sys.executable, "-c", driver, str(_SCRIPT_DIR), str(tmp_path / "worker.pid")],
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "adopted worker reaped" in result.stdout


@pytest.fixture
def trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir()
    args = argparse.Namespace(
        output=tmp_path / "output",
        profile="application_cold",
        exclusive_node=True,
        cache_reset_command=None,
        keep_runtime_cache=False,
    )
    case = {
        "name": "small",
        "model": str(model),
        "config": {},
        "tp": 4,
        "pp": 1,
        "ep": 2,
        "timeout_seconds": 1,
    }
    variant = {"name": "striped", "config": {"checkpoint_io_policy": "rank_striped_read_ahead"}}
    manifest = [{"path": "mock.weights", "bytes": 1}]
    state = argparse.Namespace(
        args=args, case=case, variant=variant, manifest=manifest, policy=_policy()
    )
    state.files = [Path("mock.weights")]
    state.seed = None
    state.identity = {
        "runtime_image": "pinned",
        "runtime_cache_seed_fingerprint": None,
        "checkpoint_metadata_fingerprint": runner.metadata_fingerprint([model]),
    }
    state.discovery = Mock(return_value=state.files)
    state.manifest_mock = Mock(return_value=manifest)
    state.evict = Mock(return_value={"verified": True, "backend_cache_verified": False})
    state.stop = Mock()
    state.ready = Mock(return_value=("127.0.0.1:1234", 0.25))
    state.process = Mock(pid=100, returncode=None)
    state.opener = Mock()
    state.opener.open.side_effect = lambda *args, **kwargs: io.StringIO(json.dumps(_server_info()))

    def launch(*args: object, **kwargs: object) -> Mock:
        kwargs["stdout"].write(state.policy)
        kwargs["stdout"].flush()
        return state.process

    state.launch = Mock(side_effect=launch)
    state.runtime_cache = Mock(return_value={"CACHE": "isolated"})
    monkeypatch.setattr(runner, "discover_checkpoint_files", state.discovery)
    monkeypatch.setattr(runner, "checkpoint_manifest", state.manifest_mock)
    monkeypatch.setattr(runner, "evict_and_verify", state.evict)
    monkeypatch.setattr(runner, "runtime_cache_environment", state.runtime_cache)
    monkeypatch.setattr(runner, "stop_server", state.stop)
    monkeypatch.setattr(runner, "child_pids", Mock(return_value=[]))
    monkeypatch.setattr(runner, "wait_ready", state.ready)
    monkeypatch.setattr(runner.subprocess, "Popen", state.launch)
    monkeypatch.setattr(
        runner.subprocess, "run", Mock(side_effect=AssertionError("Unexpected process"))
    )
    monkeypatch.setattr(runner.urllib.request, "build_opener", Mock(return_value=state.opener))
    state.result_path = args.output / "small/repeat_00/striped/result.json"
    return state


def _run_trial(trial: argparse.Namespace) -> dict:
    return runner.run_trial(
        trial.args,
        trial.case,
        trial.variant,
        0,
        trial.identity,
        trial.manifest,
        trial.files,
        trial.seed,
    )


def test_success_records_startup_without_inference_requests(trial: argparse.Namespace) -> None:
    result = _run_trial(trial)
    assert result["status"] == "passed"
    assert result["launch_to_ready_seconds"] == 0.25
    assert result["metrics"]["checkpoint_pipeline_seconds"] == 3.5
    assert json.loads(trial.result_path.read_text()) == result
    trial.stop.assert_called_once_with(trial.process)
    trial.launch.assert_called_once()
    assert trial.launch.call_args.kwargs["start_new_session"] is True
    command = trial.launch.call_args.args[0]
    assert command[0] == "trtllm-serve"
    assert command[command.index("--tensor_parallel_size") + 1] == "4"
    assert command[command.index("--moe_expert_parallel_size") + 1] == "2"
    assert [call.args[0] for call in trial.opener.open.call_args_list] == [
        "http://127.0.0.1:1234/server_info"
    ]


def test_eviction_is_once_before_each_launch_and_after_previous_teardown(
    trial: argparse.Namespace,
) -> None:
    events = Mock()
    for name, operation in (
        ("evict", trial.evict),
        ("launch", trial.launch),
        ("ready", trial.ready),
        ("metrics", trial.opener.open),
        ("stop", trial.stop),
    ):
        events.attach_mock(operation, name)
    for policy in ("native", "rank_striped_read_ahead"):
        trial.variant = {"name": policy, "config": {"checkpoint_io_policy": policy}}
        trial.policy = _policy(
            requested=policy, selected=policy, effective=policy, activated=policy != "native"
        )
        assert _run_trial(trial)["status"] == "passed"
    assert [call[0] for call in events.mock_calls] == [
        "evict",
        "launch",
        "ready",
        "metrics",
        "stop",
    ] * 2


def test_invalid_cache_never_launches_server(trial: argparse.Namespace) -> None:
    evidence = {"verified": False, "errors": ["resident pages"]}
    trial.evict.side_effect = runner.CacheVerificationError("not cold", evidence)
    result = _run_trial(trial)
    assert result["status"] == "invalid" and result["cache"] == evidence
    trial.launch.assert_not_called()
    trial.stop.assert_called_once_with(None)
    assert json.loads(trial.result_path.read_text())["error"] == "not cold"


@pytest.mark.parametrize("failure", [TimeoutError("not ready"), RuntimeError("worker exited")])
def test_failed_startup_is_recorded_and_processes_are_cleaned_up(
    trial: argparse.Namespace, failure: Exception
) -> None:
    trial.ready.side_effect = failure
    result = _run_trial(trial)
    assert result["status"] == "failed" and result["launch_to_ready_seconds"] is None
    trial.stop.assert_called_once_with(trial.process)
    assert trial.result_path.is_file()


@pytest.mark.parametrize("log", ["", _policy(activated=False), _policy(effective="native")])
def test_missing_or_inactive_policy_invalidates_trial(trial: argparse.Namespace, log: str) -> None:
    trial.policy = log
    result = _run_trial(trial)
    assert result["status"] == "invalid"
    trial.stop.assert_called_once_with(trial.process)


def test_http_failure_after_readiness_is_invalid_and_cleaned_up(trial: argparse.Namespace) -> None:
    trial.opener.open.side_effect = urllib.error.URLError("server_info failed")
    result = _run_trial(trial)
    assert result["status"] == "invalid"
    trial.stop.assert_called_once_with(trial.process)


def test_missing_metrics_are_invalid_and_cleaned_up(trial: argparse.Namespace) -> None:
    trial.opener.open.side_effect = lambda *args, **kwargs: io.StringIO("{}")
    result = _run_trial(trial)
    assert result["status"] == "invalid"
    trial.stop.assert_called_once_with(trial.process)


def test_checkpoint_change_blocks_launch(trial: argparse.Namespace) -> None:
    trial.manifest_mock.return_value = [{"path": "different"}]
    result = _run_trial(trial)
    assert result["status"] == "failed"
    trial.launch.assert_not_called()
    trial.evict.assert_not_called()


def test_cleanup_failure_cannot_leave_a_passed_record(trial: argparse.Namespace) -> None:
    trial.stop.side_effect = RuntimeError("worker cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup"):
        _run_trial(trial)
    result = json.loads(trial.result_path.read_text())
    assert result["status"] != "passed"
    assert "cleanup" in result["cleanup_error"].lower()


def test_prelaunch_error_is_recorded_without_starting_process(trial: argparse.Namespace) -> None:
    trial.case["model"] = str(trial.args.output / "missing-model")
    result = _run_trial(trial)
    assert result["status"] == "failed"
    assert trial.result_path.is_file()
    trial.launch.assert_not_called()
    trial.stop.assert_called_once_with(None)


def test_cache_helper_descendants_block_launch(
    trial: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "child_pids", Mock(return_value=[123]))
    result = _run_trial(trial)
    assert result["status"] == "failed"
    assert "descendants" in result["error"]
    trial.launch.assert_not_called()
    trial.stop.assert_called_once_with(None)


def test_draft_path_is_expanded_in_server_config(
    trial: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STARTUP_TEST_DRAFT", trial.case["model"])
    trial.case["config"]["speculative_config"] = {"speculative_model_dir": "${STARTUP_TEST_DRAFT}"}
    assert _run_trial(trial)["status"] == "passed"
    config = yaml.safe_load(trial.result_path.with_name("config.yaml").read_text())
    assert config["speculative_config"]["speculative_model_dir"] == trial.case["model"]


def test_null_speculative_config_is_supported(trial: argparse.Namespace) -> None:
    trial.case["config"]["speculative_config"] = None
    assert _run_trial(trial)["status"] == "passed"


@pytest.mark.parametrize("mutation_stage", ["before", "cloning"])
def test_seed_mutation_blocks_startup(trial: argparse.Namespace, mutation_stage: str) -> None:
    trial.seed = Path(trial.case["model"]).parent / "seed"
    trial.seed.mkdir()
    kernel = trial.seed / "kernel"
    kernel.write_text("prepared")
    trial.identity["runtime_cache_seed_fingerprint"] = runner.seed_fingerprint(trial.seed)
    trial.args.profile = "loader_isolation"
    if mutation_stage == "before":
        kernel.write_text("changed seed contents")
    else:

        def mutate_seed(*args: object) -> dict:
            kernel.write_text("changed seed contents")
            return {"CACHE": "isolated"}

        trial.runtime_cache.side_effect = mutate_seed
    result = _run_trial(trial)
    assert result["status"] == "failed"
    assert "cache seed changed" in result["error"]
    trial.launch.assert_not_called()
    trial.evict.assert_not_called()


@pytest.mark.parametrize("mutation_stage", ["before", "during"])
def test_changed_checkpoint_coverage_invalidates_trial(
    trial: argparse.Namespace, mutation_stage: str
) -> None:
    expanded = [*trial.files, Path("new.weights")]
    trial.discovery.side_effect = (
        [expanded] if mutation_stage == "before" else [trial.files, expanded]
    )
    trial.manifest_mock.side_effect = lambda files: [
        {"path": str(path), "bytes": 1} for path in files
    ]
    result = _run_trial(trial)
    assert result["status"] == ("failed" if mutation_stage == "before" else "invalid")
    assert "Checkpoint changed" in result["error"]
    if mutation_stage == "before":
        trial.launch.assert_not_called()
    else:
        assert trial.discovery.call_count == 2
        trial.stop.assert_called_once_with(trial.process)


@pytest.mark.parametrize("mutation_stage", ["before", "during"])
def test_changed_checkpoint_metadata_invalidates_trial(
    trial: argparse.Namespace, mutation_stage: str
) -> None:
    config = Path(trial.case["model"]) / "config.json"
    config.write_text('{"vocab_size": 1}')
    trial.identity["checkpoint_metadata_fingerprint"] = runner.metadata_fingerprint([config.parent])
    if mutation_stage == "before":
        config.write_text('{"vocab_size": 2}')
    else:

        def mutate_config(*args: object) -> tuple[str, float]:
            config.write_text('{"vocab_size": 2}')
            return "127.0.0.1:1234", 0.25

        trial.ready.side_effect = mutate_config
    result = _run_trial(trial)
    assert result["status"] == ("failed" if mutation_stage == "before" else "invalid")
    assert "Checkpoint" in result["error"]
    if mutation_stage == "before":
        trial.launch.assert_not_called()
    else:
        trial.stop.assert_called_once_with(trial.process)


def test_linux_subreaper_must_be_enabled_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    libc = Mock()
    libc.prctl.return_value = 0
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.ctypes, "CDLL", Mock(return_value=libc))
    runner.enable_subreaper()
    libc.prctl.assert_called_once_with(36, 1, 0, 0, 0)
    libc.prctl.return_value = -1
    with pytest.raises(OSError):
        runner.enable_subreaper()


@pytest.mark.parametrize("keep", [False, True])
def test_runtime_cache_retention_is_opt_in(trial: argparse.Namespace, keep: bool) -> None:
    trial.args.keep_runtime_cache = keep

    def prepare_cache(root: Path, seed: Path | None) -> dict:
        root.mkdir()
        (root / "kernel").write_text("compiled fixture")
        return {"CACHE": str(root)}

    trial.runtime_cache.side_effect = prepare_cache
    assert _run_trial(trial)["status"] == "passed"
    assert trial.result_path.with_name("runtime_cache").exists() == keep
