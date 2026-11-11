# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pytest runner: a selection run once and judged from its JUnit report."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import pytest_runner
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.operators.coverage import model_ops_tests_dir
from agent_flow.workflows.trtllm_modeling_v2.run import contract, report

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")

TARGET = "pkg/models/fam/target"
PASSING = "def test_a():\n    assert True\n"
FAILING = "def test_a():\n    assert 1 == 2, 'wrong'\n"
SKIPS_ITSELF = (
    "import pytest\n\npytest.skip('certified on sm_100 only', allow_module_level=True)\n\n\n"
    "def test_never():\n    assert False\n"
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


class _Delivery:
    """What the delivered-tests check reads of a workflow: the checkout, and a shell to run in."""

    _shell_runner_override = True

    def __init__(self, tmp_path: Path, files: dict[str, str]):
        self.repo = tmp_path / "repo"
        for name, text in {".gitignore": "__pycache__/\n", **files}.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        _git(self.repo, "init", "-q")
        _git(self.repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", "-A")
        _git(self.repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "x")
        self.head = _git(self.repo, "rev-parse", "HEAD")
        self.task_data = {
            "trtllm_repo_path": str(self.repo),
            "environment": {"python": sys.executable},
        }
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel=TARGET)
        self.environment = None
        self._status = contract.ContractStatus.load(self.workspace)
        self.ran: list[str] = []

    def _identity(self, commit, tree=None):
        return contract.tree_id(tree or self.repo, commit)

    def _contract_status(self):
        return self._status

    def _shell_runner(self, command, env=None, timeout=0):
        self.ran.append(command)
        # pytest-timeout is not a test dependency here.
        done = subprocess.run(
            command.replace(" --timeout=180", ""), shell=True, capture_output=True, text=True
        )
        return CommandResult(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)

    def check(self):
        return report.delivered_tests(self, self.head)


def test_the_delivered_tests_run_once_at_the_commit_and_hold_on_their_passing(
    tmp_path, monkeypatch
):
    tests = model_ops_tests_dir(TARGET)
    stage = _Delivery(
        tmp_path,
        {f"{tests}/test_forward.py": PASSING, f"{tests}/norm/test_modeling_v2_x.py": PASSING},
    )
    check = stage.check()
    assert check.passed, check.detail
    assert check.detail.startswith("CLEAN (2 passed, 0 skipped, 0 failing)")
    assert "2 test file(s) asked for" in check.detail and check.commit == stage.head
    # One job, from inside the tree, with the tree first on PYTHONPATH.
    [command] = stage.ran
    assert command.startswith(f"cd {stage.repo} && export PYTHONPATH={stage.repo}")
    assert "--junitxml=" in command and "--timeout=180" in command
    # The result is on record like any check: a start cut after them, on the same content,
    # does not run them again; another commit's content does.
    again = stage.check()
    assert again.passed and len(stage.ran) == 1 and again.commit == stage.head
    assert again.detail.endswith("2 test file(s) asked for (on record for this content)")
    assert contract.ContractStatus.load(stage.workspace).last("target:delivered-tests").passed
    (stage.repo / tests / "test_forward.py").write_text(PASSING + "\n# changed\n")
    _git(stage.repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "y")
    stage.head = _git(stage.repo, "rev-parse", "HEAD")
    assert stage.check().passed and len(stage.ran) == 2
    # One-GPU work runs where the site says: behind `environment.launch`, with the spec's
    # variables, and with nothing of the orchestrator's scheduler identity.
    behind = "import os\n\ndef test_site():\n    assert os.environ['SITE_PREFIX'] == '1'\n"
    behind += "    assert os.environ['A'] == 'b' and 'SLURM_JOB_ID' not in os.environ\n"
    (stage.repo / tests / "test_forward.py").write_text(behind)
    _git(stage.repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "z")
    stage.head = _git(stage.repo, "rev-parse", "HEAD")
    assert not stage.check().passed
    site = {"python": sys.executable, "launch": "env SITE_PREFIX=1", "env": {"A": "b"}}
    stage.environment = resolve_environment({"environment": site}, devices=(3,))
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    placed = stage.check()
    assert placed.passed, placed.detail
    assert "; env SITE_PREFIX=1 " in stage.ran[-1] and "CUDA_VISIBLE_DEVICES" not in stage.ran[-1]


@pytest.mark.parametrize(
    "files,verdict",
    [
        ({"test_x.py": FAILING}, "FAILING (0 passed, 0 skipped, 1 failing)"),
        ({"test_x.py": PASSING, "test_y.py": FAILING}, "FAILING (1 passed, 0 skipped, 1 failing)"),
        # A module that skips itself on import has a result, not a hole; alone it proves nothing.
        (
            {"test_x.py": SKIPS_ITSELF},
            "INCOMPLETE (0 passed, 1 skipped, 0 failing): no test passed",
        ),
        (
            {"test_x.py": PASSING, "test_y.py": SKIPS_ITSELF},
            "CLEAN (1 passed, 1 skipped, 0 failing)",
        ),
        ({"test_x.py": "import nothing_there\n"}, "FAILING (0 passed, 0 skipped, 1 failing)"),
    ],
)
def test_a_selection_is_judged_from_its_report(tmp_path, files, verdict):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    report = tmp_path / "out" / "report.xml"
    report.parent.mkdir()
    command = pytest_runner.pytest_command(sys.executable, repo, sorted(files), report)
    subprocess.run(command, shell=True, capture_output=True, text=True)
    assert pytest_runner.read_report(report).describe().startswith(verdict)


def test_a_delivered_file_that_did_not_run_is_not_covered_by_another(tmp_path):
    tests = model_ops_tests_dir(TARGET)
    files = {
        f"{tests}/test_a.py": PASSING,
        f"{tests}/norm/test_b.py": FAILING,
        # The tree's own configuration takes the failing file's only test out of the run.
        "pytest.ini": f"[pytest]\naddopts = --deselect={tests}/norm/test_b.py::test_a\n",
    }
    check = _Delivery(tmp_path, files).check()
    assert not check.passed
    assert check.detail.startswith("INCOMPLETE (1 passed, 0 skipped, 0 failing): no test of ")
    assert f"no test of {tests}/norm/test_b.py ran" in check.detail


def test_a_nonzero_exit_beside_a_clean_report_is_not_a_pass(tmp_path):
    tests = model_ops_tests_dir(TARGET)
    stage = _Delivery(tmp_path, {f"{tests}/test_a.py": PASSING})
    real = stage._shell_runner

    def crashed_after(command, env=None, timeout=0):
        done = real(command, env, timeout)
        return CommandResult(returncode=3, stdout=done.stdout, stderr="worker crashed at exit")

    stage._shell_runner = crashed_after
    check = stage.check()
    assert not check.passed and check.detail.startswith("INCOMPLETE (pytest exited 3): ")
    assert "worker crashed at exit" in check.detail


def test_no_report_and_a_skip_that_must_not_be_are_not_passes(tmp_path):
    unread = pytest_runner.read_report(tmp_path / "none.xml")
    assert unread.verdict() == "INCOMPLETE" and "no test report" in unread.describe()
    (tmp_path / "bad.xml").write_text("<broken", encoding="utf-8")
    assert pytest_runner.read_report(tmp_path / "bad.xml").verdict() == "INCOMPLETE"
    skipped = tmp_path / "skipped.xml"
    skipped.write_text(
        '<testsuite><testcase classname="t" name="hop"><skipped message="2 GPUs"/></testcase>'
        '<testcase classname="t" name="other"/></testsuite>',
        encoding="utf-8",
    )
    outcome = pytest_runner.read_report(skipped)
    assert outcome.verdict() == "CLEAN" and outcome.verdict(skips_fail=True) == "FAILING"
    assert "t::hop skipped" in outcome.describe(skips_fail=True)
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuite></testsuite>", encoding="utf-8")
    assert pytest_runner.read_report(empty).verdict(skips_fail=True) == "INCOMPLETE"


def test_a_test_runs_from_the_directory_pytest_takes_as_its_root(tmp_path):
    deep = tmp_path / "tests/integration/defs/accuracy"
    deep.mkdir(parents=True)
    (tmp_path / "tests/integration/defs/pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    node = "tests/integration/defs/accuracy/test_disagg.py::TestM::test_hop"
    root, argv = pytest_runner.pytest_argv(tmp_path, [node], tmp_path / "r.xml")
    assert root == tmp_path / "tests/integration/defs"
    assert argv[:3] == ["-m", "pytest", "accuracy/test_disagg.py::TestM::test_hop"]
    assert pytest_runner.pytest_root(tmp_path, "tests/unittest/test_x.py") == tmp_path
