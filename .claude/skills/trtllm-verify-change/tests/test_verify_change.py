# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of ``scripts/verify_change.py``, on a throwaway git repository.

  pytest .claude/skills/trtllm-verify-change/tests

Each test builds a small repository with a pytest root and runs the script's
commands against it, so the selection, the judging and the ledger are checked
end to end, with real pytest subprocesses.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_change.py"
_spec = importlib.util.spec_from_file_location("verify_change", SCRIPT)
vc = importlib.util.module_from_spec(_spec)
sys.modules["verify_change"] = vc
_spec.loader.exec_module(vc)

TEST_MOD = """\
import os

import pytest


def test_value():
    assert os.environ.get("EXPECT", "1") == "1"


@pytest.mark.parametrize("member", ["tp1", "tp2"])
def test_member(member):
    assert member


def test_skipped():
    pytest.skip("no device here")


class Base:
    def test_inherited(self):
        assert True


class TestGroup(Base):
    def test_inside(self):
        assert True
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n")
    (root / "pkg" / "other.py").write_text("OTHER = 1\n")
    tests = root / "tests" / "unit"
    tests.mkdir(parents=True)
    (root / "tests" / "pytest.ini").write_text("[pytest]\n")
    (tests / "test_mod.py").write_text(TEST_MOD)
    (tests / "test_slow.py").write_text("import time\n\n\ndef test_slow():\n    time.sleep(120)\n")
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "dev@example.com")
    _git(root, "config", "user.name", "Dev")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    _git(root, "branch", "base")
    return root


def _manifest(tmp_path: Path, tests: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"verify": {"tests": tests}}))
    return path


def _cli(repo: Path, manifest: Path, *args: str) -> list[str]:
    return [*args, "--repo", str(repo), "--manifest", str(manifest), "--base", "base"]


def _ledger(repo: Path) -> list[dict]:
    return vc.read_ledger(vc.state_dir(repo) / vc.LEDGER_NAME)


MOD_TEST = {
    "purpose": "the module's value",
    "pytest": "tests/unit/test_mod.py::test_value",
    "paths": ["pkg/mod.py"],
}
OTHER_TEST = {
    "purpose": "the class's tests",
    "pytest": "tests/unit/test_mod.py::TestGroup",
    "paths": ["pkg/other.py"],
}


# ----------------------------------------------------------------------------- manifest


def test_parse_reports_every_problem():
    errors: list[str] = []
    tests = vc.parse_manifest(
        {
            "verify": {
                "tests": {
                    "core": {
                        "a": {"pytest": "tests/x.py", "colour": 1},
                        "b": {
                            "purpose": "p",
                            "pytest": "tests/x.py::t[{member}]",
                            "stage": "nightly",
                        },
                        "c": {"purpose": "p", "pytest": "/abs/x.py", "members": "all"},
                        "d": {
                            "purpose": "p",
                            "pytest": "tests/x.py",
                            "members": ["tp1"],
                            "devices": 2,
                        },
                        "ok": {"purpose": "p", "pytest": "tests/x.py"},
                    },
                    "other": {"ok": {"purpose": "p", "pytest": "tests/y.py", "timeout": 0}},
                }
            }
        },
        errors,
    )
    text = "\n".join(errors)
    assert "core.a.colour' is not a key" in text
    assert "core.a.purpose' must say" in text
    assert "{member}, which needs 'members'" in text
    assert "core.b.stage' must be one of" in text
    assert "repository-relative test files" in text
    assert "'all' is not supported here" in text
    assert "sets both 'members' and 'devices'" in text
    assert "`ok` is also declared under `core`" in text
    assert "timeout' must be a positive number" in text
    assert [t.name for t in tests] == ["ok"]


def test_a_flow_spec_with_other_keys_is_read():
    errors: list[str] = []
    tests = vc.parse_manifest(
        {"model": "x", "verify": {"tests": {"core": {"t": {"purpose": "p", "pytest": "a.py"}}}}},
        errors,
    )
    assert not errors
    assert tests[0].stage == vc.STAGE_LANDING and tests[0].cases()[0].key == "core:t"


def test_members_expand_into_cases():
    errors: list[str] = []
    (test,) = vc.parse_manifest(
        {
            "verify": {
                "tests": {
                    "moe": {
                        "e2e": {
                            "purpose": "p",
                            "pytest": "t.py::test[{feature}-{member}]",
                            "members": ["tep4", "dep4"],
                        }
                    }
                }
            }
        },
        errors,
    )
    assert [c.key for c in test.cases()] == ["moe:e2e@tep4", "moe:e2e@dep4"]
    assert test.cases()[1].node_ids() == ("t.py::test[moe-dep4]",)


# -------------------------------------------------------------------------------- paths


@pytest.mark.parametrize(
    "path, pattern, expected",
    [
        ("a/b/c.py", "a/**/*.py", True),
        ("a/c.py", "a/**/*.py", True),
        ("a/b/c.py", "a/*.py", False),
        ("a/b/c.py", "a/b", True),
        ("a/bc/d.py", "a/b", False),
        ("c.py", "**/c.py", True),
        ("x/moe_a.py", "x/moe*.py", True),
    ],
)
def test_matches(path, pattern, expected):
    assert vc.matches(path, [pattern]) is expected


# --------------------------------------------------------------------------- definition


def test_defined(repo: Path):
    assert vc.defined(repo, "tests/unit/test_mod.py::test_value")[0] is True
    assert vc.defined(repo, "tests/unit/test_mod.py::test_member[tp1]")[0] is True
    assert vc.defined(repo, "tests/unit/test_mod.py::TestGroup::test_inside")[0] is True
    assert vc.defined(repo, "tests/unit/test_mod.py::TestGroup::test_inherited")[0] is True
    assert vc.defined(repo, "tests/unit/test_mod.py::TestGroup::test_missing")[0] is False
    assert vc.defined(repo, "tests/unit/test_mod.py::test_typo")[0] is False
    assert vc.defined(repo, "tests/unit/test_none.py")[0] is False


def test_pytest_root(repo: Path):
    assert vc.pytest_root(repo, "tests/unit/test_mod.py") == repo / "tests"
    assert vc.pytest_root(repo, "pkg/test_x.py") == repo


# ------------------------------------------------------------------------------- judging


@pytest.mark.parametrize(
    "body, passed, says",
    [
        ("", False, "selected no test"),
        ('<testcase name="a"><skipped message="no gpu"/></testcase>', False, "skipped: no gpu"),
        ('<testcase name="a"/><testcase name="b"><failure message="x"/></testcase>', False, "b"),
        ('<testcase name="a"/><testcase name="b"/>', True, "2 passed"),
    ],
)
def test_judge_report(tmp_path: Path, body: str, passed: bool, says: str):
    report = tmp_path / "r.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    verdict, detail = vc.judge_report(report)
    assert verdict is passed and says in detail


# ------------------------------------------------------------------------------ landing


def test_landing_selects_the_tests_the_change_touches(repo, tmp_path, capsys):
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST, "other": OTHER_TEST}})
    (repo / "pkg" / "mod.py").write_text("VALUE = 2\n")
    assert vc.main(_cli(repo, manifest, "plan")) == vc.EXIT_OK
    out = capsys.readouterr().out
    assert "changed: pkg/mod.py" in out
    assert "the change touches none of its paths" in out
    assert "every node's file, class and function exist" in out

    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_OK
    assert [r["key"] for r in _ledger(repo)] == ["core:mod"]


def test_a_landing_pass_stands_until_its_files_change(repo, tmp_path, capsys):
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST}})
    args = _cli(repo, manifest, "run", "--all")
    assert vc.main(args) == vc.EXIT_OK
    (repo / "pkg" / "other.py").write_text("OTHER = 2\n")
    capsys.readouterr()
    assert vc.main(args) == vc.EXIT_OK
    assert "reused  core:mod" in capsys.readouterr().out

    (repo / "pkg" / "mod.py").write_text("VALUE = 3\n")
    assert vc.main(args) == vc.EXIT_OK
    assert "passed  core:mod" in capsys.readouterr().out
    assert len(_ledger(repo)) == 2


def test_a_skipped_test_fails(repo, tmp_path, capsys):
    skipped = {"purpose": "p", "pytest": "tests/unit/test_mod.py::test_skipped"}
    manifest = _manifest(tmp_path, {"core": {"skipped": skipped}})
    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_FAILED
    assert "skipped: no device here" in capsys.readouterr().out
    assert _ledger(repo)[-1]["passed"] is False


def test_a_member_that_selects_nothing_fails(repo, tmp_path, capsys):
    members = {
        "purpose": "p",
        "pytest": "tests/unit/test_mod.py::test_member[{member}]",
        "members": ["tp1", "tp2", "tp4"],
    }
    manifest = _manifest(tmp_path, {"core": {"members": members}})
    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_FAILED
    out = capsys.readouterr().out
    assert "passed  core:members@tp1" in out and "passed  core:members@tp2" in out
    assert "FAILED  core:members@tp4" in out


def test_each_case_keeps_its_own_logs(repo, tmp_path):
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST, "other": OTHER_TEST}})
    assert vc.main(_cli(repo, manifest, "run", "--all")) == vc.EXIT_OK
    logs = {Path(r["logs"]) for r in _ledger(repo)}
    (run_dir,) = logs
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == [
        f"core.{test}.{kind}"
        for test in ("mod", "other")
        for kind in ("junit.xml", "stderr.log", "stdout.log")
    ]


def test_a_run_without_a_report_says_why(repo, tmp_path, capsys):
    broken = repo / "tests" / "broken"
    broken.mkdir()
    (broken / "conftest.py").write_text("import not_a_module_anywhere\n")
    (broken / "test_b.py").write_text("def test_b():\n    pass\n")
    entry = {"purpose": "p", "pytest": "tests/broken/test_b.py::test_b"}
    manifest = _manifest(tmp_path, {"core": {"broken": entry}})
    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_FAILED
    out = capsys.readouterr().out
    assert "without a test report" in out and "not_a_module_anywhere" in out


def test_env_reaches_the_test(repo, tmp_path):
    entry = {**MOD_TEST, "env": {"EXPECT": "2"}}
    manifest = _manifest(tmp_path, {"core": {"mod": entry}})
    assert vc.main(_cli(repo, manifest, "run", "--all")) == vc.EXIT_FAILED


def test_a_timeout_fails_and_stops_the_test(repo, tmp_path, capsys):
    slow = {"purpose": "p", "pytest": "tests/unit/test_slow.py", "timeout": 2}
    manifest = _manifest(tmp_path, {"core": {"slow": slow}})
    start = time.monotonic()
    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_FAILED
    assert time.monotonic() - start < 60
    assert "timed out after 2s" in capsys.readouterr().out


def test_an_undefined_node_runs_nothing(repo, tmp_path, capsys):
    typo = {"purpose": "p", "pytest": "tests/unit/test_mod.py::test_typo"}
    manifest = _manifest(tmp_path, {"core": {"typo": typo}})
    assert vc.main(_cli(repo, manifest, "run")) == vc.EXIT_USAGE
    assert "defines no `test_typo`" in capsys.readouterr().out
    assert _ledger(repo) == []


def test_only_must_name_a_declared_test(repo, tmp_path, capsys):
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST}})
    assert vc.main(_cli(repo, manifest, "plan", "--only", "nope")) == vc.EXIT_USAGE
    assert "names no declared test" in capsys.readouterr().err


# ----------------------------------------------------------------------------- delivery


def test_delivery_is_satisfied_only_on_the_committed_tree(repo, tmp_path, capsys):
    e2e = {
        "purpose": "the whole model",
        "pytest": "tests/unit/test_mod.py::TestGroup",
        "stage": "delivery",
    }
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST, "e2e": e2e}})
    delivery = ("--stage", "delivery")

    (repo / "pkg" / "mod.py").write_text("VALUE = 2\n")
    assert vc.main(_cli(repo, manifest, "run", *delivery)) == vc.EXIT_OK
    assert "uncommitted changes" in capsys.readouterr().out
    assert vc.main(_cli(repo, manifest, "report", *delivery)) == vc.EXIT_FAILED
    out = capsys.readouterr().out
    assert "NOT satisfied" in out and "which delivery does not accept" in out

    # Committing the measured content satisfies delivery without running again.
    _git(repo, "commit", "-qam", "change")
    assert vc.main(_cli(repo, manifest, "report", *delivery, "--format", "md")) == vc.EXIT_OK
    out = capsys.readouterr().out
    head = _git(repo, "rev-parse", "HEAD").strip()[:10]
    assert f"delivery at `{head}` (clean tree): 2 of 2 case(s) passed; satisfied" in out
    assert "| core | e2e | - | delivery | passed |" in out

    (repo / "pkg" / "other.py").write_text("OTHER = 2\n")
    _git(repo, "commit", "-qam", "another change")
    assert vc.main(_cli(repo, manifest, "report", *delivery)) == vc.EXIT_FAILED
    assert "stale" in capsys.readouterr().out


def test_the_default_manifest_lives_in_the_git_dir(repo, capsys):
    home = vc.state_dir(repo)
    home.mkdir(parents=True)
    (home / vc.MANIFEST_NAME).write_text(
        yaml.safe_dump({"verify": {"tests": {"core": {"mod": MOD_TEST}}}})
    )
    assert vc.main(["plan", "--repo", str(repo), "--base", "base"]) == vc.EXIT_OK
    assert str(home / vc.MANIFEST_NAME) in capsys.readouterr().out
    assert not _git(repo, "status", "--porcelain")


def test_ledger_records_what_was_measured(repo, tmp_path):
    manifest = _manifest(tmp_path, {"core": {"mod": MOD_TEST}})
    assert vc.main(_cli(repo, manifest, "run", "--all")) == vc.EXIT_OK
    (record,) = _ledger(repo)
    assert record["head"] == _git(repo, "rev-parse", "HEAD").strip()
    assert record["clean"] is True and record["passed"] is True
    assert record["nodes"] == ["tests/unit/test_mod.py::test_value"]
    assert Path(record["logs"]).is_dir()
    assert json.loads(json.dumps(record)) == record
