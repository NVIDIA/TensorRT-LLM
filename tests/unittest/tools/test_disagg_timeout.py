# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only QA checks for the experimental disagg timeout state machine."""

import importlib.util
import json
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
    path.write_text(str(completed), encoding="utf-8")


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
    assert configure_weight_loading([(context, 2), (generation, 1)]) == 30
    assert context["TRTLLM_TEST_WEIGHT_LOADING_DIR"] != generation["TRTLLM_TEST_WEIGHT_LOADING_DIR"]
    expected = json.loads((first / "expected.json").read_text())
    assert len(expected) == 3
    for name in expected:
        _event(first, name, 5)
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
        "from periodic_junit import register_pytest_timeout_evidence, PeriodicJUnitXML\n"
        "def pytest_configure(config):\n"
        "    register(config)\n"
        "    register_pytest_timeout_evidence(config)\n"
        "    reporter = PeriodicJUnitXML('reports/results.xml', dump_hang_traceback=True)\n"
        "    reporter.pytest_configure(config)\n"
        "    config.pluginmanager.register(reporter, 'periodic_junit')\n"
    )


def test_rerun_gets_new_manifest_and_loading_budget(pytester: pytest.Pytester) -> None:
    pytest.importorskip("pytest_rerunfailures")
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import time
        from pathlib import Path
        import pytest
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
            (directory / "rank-0.done").write_text(str(time.monotonic()))
            assert len(attempts) == 2
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "--reruns=1", "-s", timeout=20
    )
    assert result.parseoutcomes() == {"passed": 1, "rerun": 1}


def test_loading_does_not_consume_execution_budget(pytester: pytest.Pytester) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile("""
        import os, time
        from pathlib import Path
        import pytest
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading(5)
        def test_two_phases():
            env = {}
            configure_weight_loading([(env, 1)])
            time.sleep(1.2)
            (Path(env["TRTLLM_TEST_WEIGHT_LOADING_DIR"]) / "rank-0.done").write_text(str(time.monotonic()))
            time.sleep(0.2)
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=1", "--timeout-method=thread", "-s", timeout=20
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*phase=post_loading*"])


@pytest.mark.parametrize("phase", ["weight_loading", "post_loading"])
def test_hard_timeout_in_each_phase(pytester: pytest.Pytester, phase: str) -> None:
    _prepare_pytester(pytester)
    pytester.makepyfile(f"""
        import time
        from pathlib import Path
        import pytest
        from test_common.disagg_timeout import configure_weight_loading

        @pytest.mark.disagg_weight_loading({0.5 if phase == "weight_loading" else 5})
        def test_hangs():
            env = {{}}
            configure_weight_loading([(env, 1)])
            if {phase == "post_loading"}:
                (Path(env["TRTLLM_TEST_WEIGHT_LOADING_DIR"]) / "rank-0.done").write_text(str(time.monotonic()))
            time.sleep(10)
    """)
    result = pytester.runpytest_subprocess(
        "--timeout=0.5", "--timeout-method=thread", "-s", timeout=20
    )
    assert result.ret != 0
    result.stdout.fnmatch_lines([f"*[disagg-timeout]*{phase} timeout:*"])
    assert "TRTLLM_PYTEST_TIMEOUT " in result.stderr.str()
    assert "test_hangs" in (pytester.path / "reports/hang_traceback.txt").read_text()
