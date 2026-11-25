# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only QA checks for the experimental disagg timeout state machine."""

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
from test_common.disagg_timeout import PhaseBudget, checked_worker_command, configure_weight_loading

_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.cpu_only
pytest_plugins = ["pytester"]


def _event(root: Path, name: str, completed: float) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(str(completed), encoding="utf-8")
    temporary.replace(path)


def test_waits_for_every_rank_and_uses_last_completion(tmp_path: Path) -> None:
    expected = [
        f"worker-{worker}/model_weight_loading.rank-{rank}.done"
        for worker in range(2)
        for rank in range(2)
    ]
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    budget = PhaseBudget(tmp_path, 100, 20, start=10)
    for name in expected[:-1]:
        _event(tmp_path, name, 30)
    assert budget.remaining(40) == ("weight_loading", 70)
    _event(tmp_path, expected[-1], 39)
    assert budget.remaining(40) == ("post_loading", 19)
    assert budget.remaining(60) == ("post_loading", -1)


def test_late_completion_does_not_reset_expired_budget(tmp_path: Path) -> None:
    name = "worker-0/model_weight_loading.rank-0.done"
    (tmp_path / "expected.json").write_text(json.dumps([name]), encoding="utf-8")
    _event(tmp_path, name, 21)
    assert PhaseBudget(tmp_path, 10, 100, 10).remaining(22) == ("weight_loading", -2)


def test_missing_manifest_stays_bounded(tmp_path: Path) -> None:
    assert PhaseBudget(tmp_path, 10, 100, 10).remaining(21) == ("weight_loading", -1)


@pytest.mark.parametrize("loading,execution", [(0, 1), (1, -1), (float("inf"), 1)])
def test_rejects_invalid_budgets(tmp_path: Path, loading: float, execution: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PhaseBudget(tmp_path, loading, execution, 0)


def test_worker_environments_and_fresh_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "budgets.json").write_text('{"loading": 10, "execution": 20}', encoding="utf-8")
    monkeypatch.setenv("TRTLLM_TEST_DISAGG_TIMEOUT_DIR", str(first))
    context, generation = {}, {}
    readiness = configure_weight_loading([(context, 2), (generation, 1)])
    assert readiness is not None
    assert readiness.safety_timeout == 30
    assert context["TRTLLM_STARTUP_EVENT_DIR"] != generation["TRTLLM_STARTUP_EVENT_DIR"]
    with pytest.raises(RuntimeError, match="one cluster launch"):
        configure_weight_loading([({}, 1)])
    expected = json.loads((first / "expected.json").read_text())
    assert len(expected) == 3
    for name in expected[:-1]:
        _event(first, name, 5)
    assert readiness.remaining(3, 2, 6) is None
    with pytest.raises(RuntimeError, match=r"worker-1/model_weight_loading\.rank-0\.done"):
        readiness.require_complete()
    _event(first, expected[-1], 6)
    readiness.require_complete()
    assert readiness.remaining(3, 2, 7) == 1
    with pytest.raises(TimeoutError, match="within 2s after weight loading"):
        readiness.remaining(3, 2, 8)
    second = tmp_path / "retry"
    second.mkdir()
    (second / "budgets.json").write_text('{"loading": 10, "execution": 20}', encoding="utf-8")
    monkeypatch.setenv("TRTLLM_TEST_DISAGG_TIMEOUT_DIR", str(second))
    configure_weight_loading([({}, 1)])
    assert PhaseBudget(second, 10, 20, 10).remaining(11) == ("weight_loading", 9)


def test_disabled_does_not_mutate_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_TEST_DISAGG_TIMEOUT_DIR", raising=False)
    env = {"EXISTING": "value"}
    assert configure_weight_loading([(env, 2)]) is None
    assert env == {"EXISTING": "value"}
    command = ["trtllm-serve", "serve", "model"]
    assert checked_worker_command(command, {"TRTLLM_STARTUP_EVENT_DIR": "/tmp"}) == command


@pytest.mark.parametrize("supports_events", [False, True], ids=["stale_wheel", "current_wheel"])
def test_checked_worker_uses_its_wheel_and_reports_phase_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supports_events: bool
) -> None:
    package = tmp_path / "wheel" / "tensorrt_llm"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    cwd = tmp_path / "cwd"
    (cwd / "tensorrt_llm").mkdir(parents=True)
    (cwd / "tensorrt_llm" / "__init__.py").write_text(
        'raise AssertionError("source tree shadowed wheel")'
    )
    if supports_events:
        for name in (
            "_startup.py",
            "_startup_events.py",
            "_startup_errors.py",
            "_errors.py",
            "logger.py",
        ):
            shutil.copyfile(_ROOT / "tensorrt_llm" / name, package / name)
        commands = package / "commands"
        commands.mkdir()
        (commands / "serve.py").write_text(
            """\
import sys

from tensorrt_llm._startup import _StartupTimer

def main():
    assert sys.argv == ["trtllm-serve", "serve", "model"]
    with _StartupTimer("executor_creation") as timer:
        with timer.phase("model_engine"):
            with timer.phase("model_weight_loading", completion_rank=0):
                pass
            try:
                with timer.phase("failed_phase", completion_rank=0):
                    raise ValueError("load failed")
            except ValueError:
                pass
""",
            encoding="utf-8",
        )
    else:
        (package / "__init__.py").write_text(
            'raise AssertionError("preflight imported the runtime")'
        )

    (tmp_path / "budgets.json").write_text('{"loading": 10, "execution": 20}', encoding="utf-8")
    monkeypatch.setenv("TRTLLM_TEST_DISAGG_TIMEOUT_DIR", str(tmp_path))
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    readiness = configure_weight_loading([(env, 1)])
    assert readiness is not None
    cmd = checked_worker_command(["trtllm-serve", "serve", "model"], env)
    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=10)
    if not supports_events:
        assert result.returncode != 0
        assert "Install a TensorRT-LLM wheel built from this checkout." in result.stderr
        assert "preflight imported" not in result.stderr
        return

    assert result.returncode == 0, result.stderr
    readiness.require_complete()
    event = tmp_path / "worker-0" / "model_weight_loading.rank-0.done"
    first = event.read_text()
    subprocess.run(cmd, env=env, cwd=cwd, check=True, timeout=10)
    assert event.read_text() == first
    assert float(first) <= time.monotonic()
    assert list(event.parent.iterdir()) == [event]


def _prepare_pytester(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(_ROOT / 'tests')!r})\n"
        f"sys.path.insert(0, {str(_ROOT / 'tests/integration/defs/utils')!r})\n"
        "from test_common.disagg_timeout import register\n"
        "from periodic_junit import register_pytest_timeout_evidence, report_test_timeout, PeriodicJUnitXML\n"
        "def pytest_configure(config):\n"
        "    register(config, report_timeout=report_test_timeout)\n"
        "    register_pytest_timeout_evidence(config)\n"
        "    reporter = PeriodicJUnitXML('reports/results.xml', dump_hang_traceback=True)\n"
        "    reporter.pytest_configure(config)\n"
        "    config.pluginmanager.register(reporter, 'periodic_junit')\n"
    )
    pytester.makepyfile(
        rank_event=f"""
        import os
        import runpy
        from unittest.mock import patch

        publish_phase_completion = runpy.run_path(
            {str(_ROOT / "tensorrt_llm/_startup_events.py")!r}
        )["publish_phase_completion"]

        def publish(env):
            with patch.dict(os.environ, {{"TRTLLM_STARTUP_EVENT_DIR": env["TRTLLM_STARTUP_EVENT_DIR"]}}):
                publish_phase_completion("model_weight_loading", 0)
        """
    )


def test_rerun_gets_new_manifest_and_loading_budget(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pytest_rerunfailures")
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import time
        from pathlib import Path
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import configure_weight_loading

        attempts = []
        @pytest.mark.disagg_weight_loading(5)
        def test_retry():
            env = {}
            configure_weight_loading([(env, 1)])
            directory = Path(env["TRTLLM_STARTUP_EVENT_DIR"])
            assert directory not in attempts
            attempts.append(directory)
            time.sleep(0.6)
            publish(env)
            assert len(attempts) == 2
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "--reruns=1", "-s", timeout=20
    )
    assert result.parseoutcomes() == {"passed": 1, "rerun": 1}


def test_loading_does_not_consume_execution_budget(pytester: pytest.Pytester) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import os
        import time
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading(5)
        def test_two_phases():
            env = {}
            configure_weight_loading([(env, 1)])
            time.sleep(1.2)
            publish(env)

        def test_attempt_environment_was_restored():
            assert "TRTLLM_TEST_DISAGG_TIMEOUT_DIR" not in os.environ
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "-s", timeout=20
    )
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*phase=post_loading*"])


def test_cancelling_before_first_poll_still_reports_the_phase(
    pytester: pytest.Pytester,
) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import _Watchdog, configure_weight_loading

        original_run = _Watchdog._run

        def run_after_cancellation(watchdog):
            if watchdog.stop.wait(5):
                original_run(watchdog)

        _Watchdog._run = run_after_cancellation

        @pytest.mark.disagg_weight_loading(5)
        def test_finishes_before_first_poll():
            env = {}
            configure_weight_loading([(env, 1)])
            publish(env)
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "-s", timeout=20
    )
    result.assert_outcomes(passed=1)
    assert "[disagg-timeout] phase=post_loading" in result.stdout.str()
    assert "TRTLLM_PYTEST_TIMEOUT " not in result.stderr.str()


@pytest.mark.parametrize("phase", ["weight_loading", "post_loading"])
def test_hard_timeout_in_each_phase(pytester: pytest.Pytester, phase: str) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile(f"""
        import time
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading({0.5 if phase == "weight_loading" else 5})
        def test_hangs(request):
            reporter = request.config.pluginmanager.get_plugin("periodic_junit")
            assert reporter._hang_timer is None
            env = {{}}
            configure_weight_loading([(env, 1)])
            if {phase == "post_loading"}:
                publish(env)
            time.sleep(10)
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "-s", timeout=20
    )
    assert result.ret != 0
    assert f"[disagg-timeout] {phase} timeout:" in result.stdout.str()
    marker = "TRTLLM_PYTEST_TIMEOUT "
    evidence = [
        json.loads(line[len(marker) :]) for line in result.stderr.lines if line.startswith(marker)
    ]
    assert evidence == [
        {
            "nodeid": "test_hard_timeout_in_each_phase.py::test_hangs",
            "attempt": 1,
            "timeout": 0.5 if phase == "weight_loading" else 1.0,
        }
    ]
    assert "test_hangs" in (pytester.path / "reports/hang_traceback.txt").read_text()


def test_readiness_timeout_leaves_time_for_cleanup(pytester: pytest.Pytester) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import time
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import configure_weight_loading

        cleaned_up = False

        @pytest.mark.disagg_weight_loading(5)
        def test_ready_hangs():
            global cleaned_up
            env = {}
            readiness = configure_weight_loading([(env, 1)])
            start = time.monotonic()
            publish(env)
            try:
                while True:
                    readiness.remaining(start, 0.05, time.monotonic())
                    time.sleep(0.01)
            finally:
                cleaned_up = True

        def test_next():
            assert cleaned_up
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=2", "--timeout-method=thread", "-s", timeout=20
    )
    result.assert_outcomes(failed=1, passed=1)
    assert "Cluster did not become ready within 0.05s after weight loading" in result.stdout.str()
    assert "TRTLLM_PYTEST_TIMEOUT " not in result.stderr.str()


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="requires POSIX signal timers")
def test_signal_mode_keeps_session_running(pytester: pytest.Pytester) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import os
        import time
        import pytest
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading(5)
        def test_hangs():
            assert configure_weight_loading([({}, 1)]) is None
            time.sleep(10)

        def test_session_continues():
            assert "TRTLLM_TEST_DISAGG_TIMEOUT_DIR" not in os.environ
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=signal", "-s", timeout=20
    )
    result.assert_outcomes(failed=1, passed=1)
    assert "Timeout (>1.0s)" in result.stdout.str()
    assert "TRTLLM_PYTEST_TIMEOUT " in result.stderr.str()
    assert "[disagg-timeout]" not in result.stdout.str()


def test_func_only_keeps_standard_timeout(pytester: pytest.Pytester) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import pytest
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.timeout(func_only=True)
        @pytest.mark.disagg_weight_loading(5)
        def test_no_phase_supervisor():
            assert configure_weight_loading([({}, 1)]) is None
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "-s", timeout=20
    )
    result.assert_outcomes(passed=1)
    assert "[disagg-timeout]" not in result.stdout.str()


def test_invalid_loading_override_is_collection_error(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRTLLM_TEST_WEIGHT_LOADING_TIMEOUT", "NaN")
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.disagg_weight_loading(5)
        def test_example():
            raise AssertionError("must not run")
    """)
    result = pytester.runpytest_subprocess("--timeout=1", timeout=20)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "TRTLLM_TEST_WEIGHT_LOADING_TIMEOUT expects positive seconds" in result.stderr.str()
