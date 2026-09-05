"""Freezing a synthetic project directory into an archive."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from agent_flow.ops import archive

LEDGER = """
| 09-04 10:00 | AC-01 | | abc1234def | PASS | a.log |
| 09-04 11:00 | AC-02 | | abc1234def | FAIL | b.log |
| 09-04 12:00 | AC-02 | | abc1234def | PASS | c.log |
"""


@pytest.fixture
def project(tmp_path):
    """A project dir with one of everything the archiver knows about."""
    root = tmp_path / "proj"
    (root / "workspace").mkdir(parents=True)
    (root / "logs" / "nested").mkdir(parents=True)
    (root / "notes").mkdir()
    (root / "handoff").mkdir()
    (root / "evidence" / "deep").mkdir(parents=True)
    (root / "workspace" / "PASS-LEDGER.md").write_text(textwrap.dedent(LEDGER))
    (root / "workspace" / "GATE-REASONS.md").write_text(
        "- AC-02 | 09-04 11:05 | reviewer | flaky under load; rerun queued\n"
    )
    (root / "logs" / "nested" / "run.log").write_text("hello\n")
    (root / "notes" / "design.md").write_text("# design\n")
    (root / "handoff" / "HANDOFF.md").write_text("carry on\n")
    (root / "handoff" / "weights.bin").write_bytes(b"\0" * 64)
    (root / "task.yaml").write_text("goal: demo\n")
    (root / "run-2026-09-04.log").write_text("log line\n")
    (root / "evidence" / "small.txt").write_text("small\n")
    (root / "evidence" / "deep" / "big.bin").write_bytes(b"\0" * 4096)
    return root


def freeze(project, dest, **kw):
    kw.setdefault("settings", archive.ArchiveSettings())
    kw["settings"].commit = kw.pop("commit", False)
    return archive.freeze(project, dest, "demo", date="2026-09-05", **kw)


def test_freeze_copies_the_readable_half(project, tmp_path):
    dest = tmp_path / "runs"
    m = freeze(project, dest)
    out = dest / "run-2026-09-05-demo"
    assert out.is_dir()
    assert (out / "workspace" / "PASS-LEDGER.md").is_file()
    assert (out / "logs" / "nested" / "run.log").read_text() == "hello\n"
    assert (out / "notes" / "design.md").is_file()
    assert (out / "task.yaml").is_file()
    assert (out / "run-2026-09-04.log").is_file()
    # handoff/: text documents only, never the bulk beside them
    assert (out / "handoff" / "HANDOFF.md").is_file()
    assert not (out / "handoff" / "weights.bin").exists()
    assert m["counts"]["files_copied"] > 0


def test_scoreboard_comes_from_the_archived_ledger(project, tmp_path):
    m = freeze(project, tmp_path / "runs")
    board = m["scoreboard"]
    assert (board["green"], board["total"]) == (2, 2)
    ac02 = next(g for g in board["gates"] if g["id"] == "AC-02")
    assert ac02["runs"] == 2 and ac02["state"] == "pass"
    assert ac02["last_commit"] == "abc1234def"
    assert ac02["reason"] == "flaky under load; rerun queued"


def test_evidence_over_the_cap_is_listed_not_copied(project, tmp_path):
    st = archive.ArchiveSettings()
    st.evidence_max = 1024
    dest = tmp_path / "runs"
    freeze(project, dest, settings=st)
    out = dest / "run-2026-09-05-demo"
    manifest = json.loads((out / "EVIDENCE-MANIFEST.json").read_text())
    rows = {r["path"]: r for r in manifest["files"]}
    assert rows["small.txt"]["copied"] is True
    assert "sha256" in rows["small.txt"]
    assert rows[os.path.join("deep", "big.bin")]["copied"] is False
    assert rows[os.path.join("deep", "big.bin")]["source"].endswith("big.bin")
    assert (out / "evidence" / "small.txt").is_file()
    assert not (out / "evidence" / "deep" / "big.bin").exists()
    assert manifest["total_files"] == 2 and manifest["copied_files"] == 1


def test_oversize_files_are_gitignored_per_file(project, tmp_path):
    (project / "logs" / "huge.log").write_bytes(b"x" * 3000)
    st = archive.ArchiveSettings()
    st.git_max = 1024
    dest = tmp_path / "runs"
    m = freeze(project, dest, settings=st)
    assert os.path.join("logs", "huge.log") in [o["path"] for o in m["oversize"]]
    ignore = (dest / ".gitignore").read_text()
    assert "run-2026-09-05-demo/logs/huge.log" in ignore
    assert "__pycache__/" in ignore
    # a re-freeze with a bigger limit drops the entry again
    st.git_max = 1 << 30
    freeze(project, dest, settings=st)
    assert "logs/huge.log" not in (dest / ".gitignore").read_text()


def test_gitignore_keeps_other_folders_entries(tmp_path):
    dest = tmp_path / "runs"
    dest.mkdir()
    (dest / ".gitignore").write_text("run-2026-01-01-old/logs/a.log\n")
    archive.write_gitignore(dest, "run-2026-09-05-demo", [{"path": "b.log"}], 1024)
    body = (dest / ".gitignore").read_text()
    assert "run-2026-01-01-old/logs/a.log" in body
    assert "run-2026-09-05-demo/b.log" in body


def test_symlinks_are_never_followed(project, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not copy\n")
    (project / "logs" / "link").symlink_to(outside)
    (project / "notes" / "link.md").symlink_to(outside / "secret.txt")
    (project / "evidence" / "link.txt").symlink_to(outside / "secret.txt")
    dest = tmp_path / "runs"
    m = freeze(project, dest)
    out = dest / "run-2026-09-05-demo"
    assert not (out / "logs" / "link").exists()
    assert not (out / "notes" / "link.md").exists()
    assert not (out / "evidence" / "link.txt").exists()
    assert m["counts"]["symlinks_skipped"] >= 1
    listed = json.loads((out / "EVIDENCE-MANIFEST.json").read_text())
    assert all("link" not in r["path"] for r in listed["files"])


def test_refreezing_is_idempotent(project, tmp_path):
    dest = tmp_path / "runs"
    first = freeze(project, dest)
    second = freeze(project, dest)
    assert second["counts"]["files_copied"] == 0
    assert second["counts"]["files_unchanged"] == first["counts"]["files_copied"]
    rows = [
        ln for ln in (dest / "README.md").read_text().splitlines() if "run-2026-09-05-demo/" in ln
    ]
    assert len(rows) == 1  # amended, not appended


def test_readme_index_rows_accumulate_between_markers(project, tmp_path):
    dest = tmp_path / "runs"
    freeze(project, dest, summary="first pass")
    archive.freeze(
        project,
        dest,
        "other",
        date="2026-09-06",
        settings=_no_commit(),
        summary="second pass",
    )
    body = (dest / "README.md").read_text()
    assert body.count(archive.INDEX_START) == 1 and body.count(archive.INDEX_END) == 1
    table = body.split(archive.INDEX_START)[1].split(archive.INDEX_END)[0]
    assert "run-2026-09-05-demo" in table and "run-2026-09-06-other" in table
    assert "first pass" in table and "second pass" in table
    assert "2/2" in table


def _no_commit():
    st = archive.ArchiveSettings()
    st.commit = False
    return st


def test_dry_run_writes_nothing(project, tmp_path):
    dest = tmp_path / "runs"
    m = freeze(project, dest, dry_run=True)
    assert m["dry_run"] is True and m["counts"]["files_copied"] > 0
    assert not (dest / "run-2026-09-05-demo").exists()


def test_manifest_records_repo_head_and_branch(project, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }
    subprocess.run(["git", "init", "-q", "-b", "trunk", str(repo)], check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "c"], check=True, env=env)
    m = freeze(project, tmp_path / "runs", repo=repo)
    assert len(m["repo_head"]) == 40
    assert m["repo_branch"] == "trunk"
    assert m["repo"] == str(repo)


def test_read_manifest_and_find_archives(project, tmp_path):
    dest = tmp_path / "runs"
    freeze(project, dest)
    found = archive.find_archives(dest)
    assert [d.name for d in found] == ["run-2026-09-05-demo"]
    assert archive.read_manifest(found[0])["run"] == "demo"
    assert archive.read_manifest(tmp_path / "nope") == {}


def test_cli_freeze_and_list(project, tmp_path, capsys):
    dest = tmp_path / "runs"
    argv = ["--dest", str(dest), "freeze", "demo", "--source", str(project), "--date", "2026-09-05"]
    assert archive.main(argv) == 0
    assert "gates green: 2/2" in capsys.readouterr().out
    assert archive.main(["--dest", str(dest), "list"]) == 0
    assert "run-2026-09-05-demo" in capsys.readouterr().out


def test_cli_needs_a_source_and_dest(tmp_path, capsys):
    assert archive.main(["freeze", "demo"]) == 2
    assert "need a source project dir" in capsys.readouterr().err
    assert archive.main(["list"]) == 2
    assert "no archive root" in capsys.readouterr().err


def test_cli_reports_a_missing_source(tmp_path, capsys):
    assert (
        archive.main(
            ["--dest", str(tmp_path / "d"), "freeze", "x", "--source", str(tmp_path / "nope")]
        )
        == 2
    )
    assert "no such project directory" in capsys.readouterr().err


def test_settings_come_from_the_config(cfg):
    st = archive.ArchiveSettings(cfg)
    assert st.trees == archive.DEFAULT_TREES
    assert st.evidence_max == archive.DEFAULT_EVIDENCE_MAX


def test_commit_lands_in_the_archive_repo(project, tmp_path):
    dest = tmp_path / "runs"
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
    }
    os.environ.update(env)
    try:
        st = archive.ArchiveSettings()
        st.commit = True
        archive.freeze(project, dest, "demo", date="2026-09-05", settings=st)
    finally:
        for k in env:
            os.environ.pop(k, None)
    log = subprocess.run(
        ["git", "-C", str(dest), "log", "--oneline"], capture_output=True, text=True, check=False
    ).stdout
    assert "Archive run-2026-09-05-demo: 2/2 gates green" in log
    assert Path(dest / ".git").is_dir()
