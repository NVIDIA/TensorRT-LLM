"""The start-commit pin: scaffolding from a parent, and checking the drift."""

from __future__ import annotations

import json
import subprocess
import tomllib

import pytest

from agent_flow.ops import project

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@e",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@e",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A three-commit repo; returns (path, [sha1, sha2, sha3])."""
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    shas = []
    for i in range(3):
        (path / f"f{i}").write_text(str(i))
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(path), "commit", "-qm", f"c{i}"], check=True)
        shas.append(
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    return path, shas


@pytest.fixture
def archive_root(tmp_path):
    def make(project_name, head, folder="run-2026-09-05-parent"):
        d = tmp_path / "runs" / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / "MANIFEST.json").write_text(
            json.dumps({"run": project_name, "folder": folder, "repo_head": head})
        )
        return tmp_path / "runs"

    return make


def _toml(root):
    return tomllib.loads((root / "agent-flow-ops.toml").read_text())["project"]


def test_new_refuses_without_a_starting_commit(tmp_path, capsys):
    assert project.main(["--projects-root", str(tmp_path / "p"), "new", "alpha"]) == 2
    assert "no starting commit" in capsys.readouterr().err
    assert not (tmp_path / "p" / "alpha").exists()


def test_no_parent_scaffolds_without_a_pin(tmp_path, capsys):
    root = tmp_path / "p"
    assert project.main(["--projects-root", str(root), "new", "alpha", "--no-parent"]) == 0
    assert "no start commit" in capsys.readouterr().out
    section = _toml(root / "alpha")
    assert "start_commit" not in section and "parent" not in section
    task = (root / "alpha" / "workspace" / "TASK.md").read_text()
    assert "## GIT" not in task


def test_explicit_start_commit_is_written_into_the_toml_and_the_task(tmp_path):
    root = tmp_path / "p"
    assert (
        project.main(
            ["--projects-root", str(root), "new", "alpha", "--start-commit", "abc123def456"]
        )
        == 0
    )
    assert _toml(root / "alpha")["start_commit"] == "abc123def456"
    task = (root / "alpha" / "workspace" / "TASK.md").read_text()
    assert "Branch from `abc123def456`" in task
    assert "frozen evidence" in task
    assert "reopens THAT gate and only that gate" in task


def test_a_parent_supplies_its_archived_final_commit(tmp_path, archive_root):
    root = tmp_path / "p"
    runs = archive_root("parent", "f" * 40)
    assert (
        project.main(
            [
                "--projects-root",
                str(root),
                "--archive-root",
                str(runs),
                "new",
                "child",
                "--parent",
                "parent",
            ]
        )
        == 0
    )
    section = _toml(root / "child")
    assert section["parent"] == "parent" and section["start_commit"] == "f" * 40
    task = (root / "child" / "workspace" / "TASK.md").read_text()
    assert "final commit of `parent`" in task
    assert "run-2026-09-05-parent" in task


def test_an_explicit_start_commit_beats_the_parent_lookup(tmp_path, archive_root):
    root = tmp_path / "p"
    runs = archive_root("parent", "f" * 40)
    args = ["--projects-root", str(root), "--archive-root", str(runs), "new", "child"]
    assert project.main([*args, "--parent", "parent", "--start-commit", "abc1234"]) == 0
    assert _toml(root / "child")["start_commit"] == "abc1234"


def test_an_unarchived_parent_is_refused_with_the_reason(tmp_path, capsys):
    root = tmp_path / "p"
    assert (
        project.main(
            [
                "--projects-root",
                str(root),
                "--archive-root",
                str(tmp_path / "runs"),
                "new",
                "child",
                "--parent",
                "ghost",
            ]
        )
        == 2
    )
    err = capsys.readouterr().err
    assert "no archived run for parent project 'ghost'" in err
    assert "archive freeze ghost" in err


def test_a_parent_archive_without_a_head_is_refused(tmp_path, archive_root, capsys):
    root = tmp_path / "p"
    runs = archive_root("parent", "")
    assert (
        project.main(
            [
                "--projects-root",
                str(root),
                "--archive-root",
                str(runs),
                "new",
                "child",
                "--parent",
                "parent",
            ]
        )
        == 2
    )
    assert "records no repo HEAD" in capsys.readouterr().err


def test_parent_final_commit_reads_the_manifest(tmp_path, archive_root):
    runs = archive_root("parent", "a" * 40)
    assert project.parent_final_commit("parent", runs) == ("a" * 40, "run-2026-09-05-parent")


# -- project check ---------------------------------------------------------


def _project_at(tmp_path, start_commit):
    root = tmp_path / "p"
    project.main(["--projects-root", str(root), "new", "alpha", "--start-commit", start_commit])
    return root / "alpha"


def test_check_reports_head_sitting_on_the_pin(tmp_path, repo, capsys):
    path, shas = repo
    d = _project_at(tmp_path, shas[-1])
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(path)]) == 0
    out = capsys.readouterr().out
    assert "alpha: at" in out and shas[-1] in out


def test_check_reports_descent_and_the_drift(tmp_path, repo, capsys):
    path, shas = repo
    d = _project_at(tmp_path, shas[0])
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(path)]) == 0
    out = capsys.readouterr().out
    assert "alpha: descends" in out
    assert "2 commits ahead, 0 behind" in out


def test_check_flags_a_diverged_checkout(tmp_path, repo, capsys, monkeypatch):
    path, shas = repo
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", "-b", "side", shas[0]], check=True)
    (path / "other").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "side"], check=True)
    d = _project_at(tmp_path, shas[2])
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(path)]) == 1
    out = capsys.readouterr().out
    assert "alpha: diverged" in out
    assert "the parent's verdicts may not apply" in out


def test_check_flags_a_pin_this_checkout_has_never_seen(tmp_path, repo, capsys):
    path, _ = repo
    d = _project_at(tmp_path, "0" * 40)
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(path)]) == 1
    assert "unknown-commit" in capsys.readouterr().out


def test_check_on_a_project_with_no_pin_is_not_a_failure(tmp_path, repo, capsys):
    path, _ = repo
    root = tmp_path / "p"
    project.main(["--projects-root", str(root), "new", "alpha", "--no-parent"])
    capsys.readouterr()
    assert project.main(["check", str(root / "alpha"), "--checkout", str(path)]) == 0
    assert "alpha: no-pin" in capsys.readouterr().out


def test_check_on_something_that_is_not_a_repo(tmp_path, capsys):
    d = _project_at(tmp_path, "a" * 40)
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(tmp_path)]) == 1
    assert "no-repo" in capsys.readouterr().out


def test_check_json_mode(tmp_path, repo, capsys):
    path, shas = repo
    d = _project_at(tmp_path, shas[0])
    capsys.readouterr()
    assert project.main(["check", str(d), "--checkout", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "descends" and out["ahead"] == 2
    assert out["head"] == shas[-1] and out["start_commit"] == shas[0]


def test_check_needs_a_project_and_a_checkout(tmp_path, capsys):
    assert project.main(["check"]) == 2
    assert "no project" in capsys.readouterr().err
    d = _project_at(tmp_path, "a" * 40)
    capsys.readouterr()
    assert project.main(["check", str(d)]) == 2
    assert "no checkout to check" in capsys.readouterr().err


def test_check_uses_the_configured_repo(tmp_path, repo, config_path, cfg, capsys):
    path, shas = repo
    d = _project_at(tmp_path, shas[-1])
    cfg_body = config_path.read_text().replace(f'repo = "{cfg.run_root}/main"', f'repo = "{path}"')
    config_path.write_text(cfg_body)
    (cfg.run_root / "agent-flow-ops.toml").write_text((d / "agent-flow-ops.toml").read_text())
    capsys.readouterr()
    assert project.main(["--config", str(config_path), "check", str(d)]) == 0
    assert "alpha: at" in capsys.readouterr().out


def test_index_shows_the_pin(tmp_path):
    root = tmp_path / "p"
    project.main(["--projects-root", str(root), "new", "alpha", "--start-commit", "abc1234"])
    row = project.index(root)[0]
    assert row["start_commit"] == "abc1234" and row["parent"] is None
