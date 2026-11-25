# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only QA checks for the experimental disagg timeout state machine."""

import importlib.util
import json
import signal
import time
from pathlib import Path

import pytest
from test_common.disagg_timeout import PhaseBudget, configure_weight_loading

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
    expected = [f"worker-{worker}/rank-{rank}.done" for worker in range(2) for rank in range(2)]
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    budget = PhaseBudget(tmp_path, 100, 20, start=10)
    for name in expected[:-1]:
        _event(tmp_path, name, 30)
    assert budget.remaining(40) == ("weight_loading", 70)
    _event(tmp_path, expected[-1], 39)
    assert budget.remaining(40) == ("post_loading", 19)
    assert budget.remaining(60) == ("post_loading", -1)


def test_late_completion_does_not_reset_expired_budget(tmp_path: Path) -> None:
    (tmp_path / "expected.json").write_text('["worker-0/rank-0.done"]', encoding="utf-8")
    _event(tmp_path, "worker-0/rank-0.done", 21)
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
    assert context["TRTLLM_TEST_WEIGHT_LOADING_DIR"] != generation["TRTLLM_TEST_WEIGHT_LOADING_DIR"]
    with pytest.raises(RuntimeError, match="one cluster launch"):
        configure_weight_loading([({}, 1)])
    expected = json.loads((first / "expected.json").read_text())
    assert len(expected) == 3
    for name in expected[:-1]:
        _event(first, name, 5)
    assert readiness.remaining(3, 2, 6) is None
    with pytest.raises(RuntimeError, match="worker-1/rank-0.done"):
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


def test_rank_notification_is_atomic_and_keeps_first_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _ROOT / "tensorrt_llm/_torch/pyexecutor/_weight_loading.py"
    spec = importlib.util.spec_from_file_location("_weight_loading", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TRTLLM_TEST_WEIGHT_LOADING_DIR", str(tmp_path))
    module.notify_weight_loading_complete(1)
    event = tmp_path / "rank-1.done"
    first = event.read_text()
    module.notify_weight_loading_complete(1)
    assert event.read_text() == first
    assert float(first) <= time.monotonic()
    assert list(tmp_path.iterdir()) == [event]


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
        rank_event="""
        import time
        from pathlib import Path

        def publish(env):
            directory = Path(env["TRTLLM_TEST_WEIGHT_LOADING_DIR"])
            temporary = directory / "rank-0.tmp"
            temporary.write_text(str(time.monotonic()))
            temporary.replace(directory / "rank-0.done")
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
            directory = Path(env["TRTLLM_TEST_WEIGHT_LOADING_DIR"])
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


@pytest.mark.parametrize("phase", ["weight_loading", "post_loading"])
def test_hard_timeout_in_each_phase(pytester: pytest.Pytester, phase: str) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile(f"""
        import time
        import pytest
        from rank_event import publish
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading({0.5 if phase == "weight_loading" else 5})
        def test_hangs():
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
