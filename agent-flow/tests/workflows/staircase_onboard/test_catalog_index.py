# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``catalog/index.yaml`` is generated from per-entry fragments (F013/F014).

Concurrent smiths each appended their row to the one shared index and
committed the whole file, so the first to commit swept in its siblings'
uncommitted rows: HEAD named entries whose files were not in HEAD, and a
later smith's commit message named rows it did not write.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import catalog_index as ci
from agent_flow.workflows.staircase_onboard.committer import install_committer
from agent_flow.workflows.staircase_onboard.delivery import CATALOG_DIR

from .test_delivery import _build_repo

REPO_ROOT = Path(__file__).resolve().parents[4]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path):
    root = _build_repo(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    _commit_all(root, "seed")
    return root


@pytest.fixture
def commit(tmp_path, repo):
    script = install_committer(tmp_path / "ws", repo)
    return ci.commit_with_script(script, repo)


def _catalog(repo: Path) -> Path:
    return repo / CATALOG_DIR


def _entry(repo: Path, category: str, name: str, summary: str) -> list[str]:
    """What a smith writes: the wrapper, the contract, and its own fragment."""
    catalog = _catalog(repo)
    (catalog / category).mkdir(parents=True, exist_ok=True)
    (catalog / category / f"{name}.py").write_text("def f(): ...\n", encoding="utf-8")
    (catalog / category / f"{name}.md").write_text("# contract\n", encoding="utf-8")
    fragment = catalog / ci.fragment_rel(f"{category}/{name}.py")
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(
        yaml.safe_dump(
            {
                "path": f"{category}/{name}.py",
                "impl": f"torch.ops.trtllm.{name}",
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )
    return [
        f"{CATALOG_DIR}/{category}/{name}.py",
        f"{CATALOG_DIR}/{category}/{name}.md",
        f"{CATALOG_DIR}/{ci.fragment_rel(f'{category}/{name}.py')}",
    ]


def _index_paths(text: str) -> list[str]:
    return [row["path"] for row in yaml.safe_load(text)["entries"]]


def _head_index(repo: Path) -> str:
    return _git(repo, "show", f"HEAD:{CATALOG_DIR}/index.yaml")


def _last_commit_files(repo: Path) -> list[str]:
    return sorted(_git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").split())


# --------------------------------------------------------------- migration


def test_the_real_catalog_migrates_without_losing_a_row_or_a_comment(tmp_path):
    """Run against this checkout's catalog: the conventions have to be the ones in use."""
    source = REPO_ROOT / CATALOG_DIR
    if not (source / "index.yaml").is_file():
        pytest.skip("no catalog in this checkout")
    copy = tmp_path / "catalog"
    shutil.copytree(source, copy)
    original = (copy / "index.yaml").read_text(encoding="utf-8")
    if (copy / ci.FRAGMENTS_DIRNAME).exists():
        pytest.skip("this checkout's catalog is already migrated")

    ci.migrate(copy)
    regenerated = (copy / "index.yaml").read_text(encoding="utf-8")

    assert yaml.safe_load(regenerated)["entries"] == yaml.safe_load(original)["entries"]
    # Byte-identical apart from the notice that says the file is generated.
    assert regenerated.replace(ci.GENERATED_NOTICE, "") == original
    assert ci.problems(ci.snapshot_worktree(copy)) == []


def test_migration_is_idempotent(tmp_path):
    catalog = _catalog(_build_repo(tmp_path))
    ci.migrate(catalog)
    before = {p: p.read_text() for p in catalog.rglob("*.yaml")}
    ci.migrate(catalog)
    assert {p: p.read_text() for p in catalog.rglob("*.yaml")} == before


def test_a_new_entry_sorts_after_the_migrated_ones_in_its_category(tmp_path):
    root = _build_repo(tmp_path)
    ci.migrate(_catalog(root))
    _entry(root, "norm", "aaa_first_by_name", "a later entry")
    text = ci.render(ci.snapshot_worktree(_catalog(root)))
    paths = _index_paths(text)
    assert paths.index("norm/aaa_first_by_name.py") == paths.index("norm/flashinfer_rmsnorm.py") + 1


# ------------------------------------------------------------------- check


def test_a_hand_edit_to_the_index_is_a_problem(tmp_path):
    catalog = _catalog(_build_repo(tmp_path))
    ci.migrate(catalog)
    index = catalog / "index.yaml"
    index.write_text(index.read_text() + "\n  - path: norm/ghost.py\n", encoding="utf-8")
    (problem,) = ci.problems(ci.snapshot_worktree(catalog))
    assert "not what its fragments generate" in problem


def test_misplaced_duplicate_and_orphaned_fragments_are_problems(tmp_path):
    catalog = _catalog(_build_repo(tmp_path))
    ci.migrate(catalog)
    (catalog / "index.d" / "norm" / "elsewhere.yaml").write_text(
        "path: norm/flashinfer_rmsnorm.py\n", encoding="utf-8"
    )
    (catalog / "index.d" / "gemm" / "ghost.yaml").write_text(
        "path: gemm/ghost.py\n", encoding="utf-8"
    )
    found = "\n".join(ci.problems(ci.snapshot_worktree(catalog)))
    assert "whose fragment belongs at index.d/norm/flashinfer_rmsnorm.yaml" in found
    assert "also claimed by" in found
    assert "entry gemm/ghost.py does not exist" in found


def test_an_unmigrated_catalog_says_so():
    snapshot = ci.Snapshot(header=None, fragments={}, files=frozenset(), index_text="entries: []\n")
    assert "not migrated" in ci.problems(snapshot)[0]


def test_the_check_cli_exit_codes(tmp_path, capsys):
    root = _build_repo(tmp_path)
    assert ci.main(["check", "--repo", str(root)]) == 1
    assert ci.main(["migrate", "--repo", str(root)]) == 0
    assert ci.main(["check", "--repo", str(root)]) == 0
    assert "0 problem(s)" in capsys.readouterr().out


# -------------------------------------------------------------------- sync


def test_without_a_fragment_the_catalog_is_not_touched(repo, commit):
    """A run that adds no catalog entry must not commit anything to the catalog."""
    head = _git(repo, "rev-parse", "HEAD")
    assert ci.sync(repo, commit=commit).action == "skipped"
    assert _git(repo, "rev-parse", "HEAD") == head
    assert not (_catalog(repo) / "index.d").exists()


def _first_smith(repo, commit):
    paths = _entry(repo, "norm", "first_fragment", "the first smith of the run")
    ok, _ = commit("feat(catalog): onboard first_fragment", paths)
    assert ok


def test_the_first_fragment_migrates_the_catalog_in_one_commit_of_its_own(repo, commit):
    _first_smith(repo, commit)
    result = ci.sync(repo, commit=commit)
    assert result.action == "committed"
    files = _last_commit_files(repo)
    assert f"{CATALOG_DIR}/index.yaml" in files
    assert f"{CATALOG_DIR}/index.d/_header.yaml" in files
    assert all(f.startswith(f"{CATALOG_DIR}/index") for f in files)
    paths = _index_paths(_head_index(repo))
    assert "norm/first_fragment.py" in paths and "gemm/cublas_mm.py" in paths
    assert ci.problems(ci.snapshot_at(repo)) == []
    assert ci.sync(repo, commit=commit).action == "current"


def test_concurrent_smiths_never_cross_attribute(repo, commit):
    """F013 as measured, replayed: smith A commits while smith B is still writing."""
    _first_smith(repo, commit)
    ci.sync(repo, commit=commit)
    a_paths = _entry(repo, "norm", "rmsnorm_a", "smith A's entry")
    _entry(repo, "gemm", "gemm_b", "smith B's entry, not committed yet")
    ok, _ = commit("feat(catalog): onboard rmsnorm_a", a_paths)
    assert ok
    assert _last_commit_files(repo) == sorted(a_paths)

    result = ci.sync(repo, commit=commit)
    assert result.action == "committed"
    assert _last_commit_files(repo) == [f"{CATALOG_DIR}/index.yaml"]
    head_index = _head_index(repo)
    assert "norm/rmsnorm_a.py" in _index_paths(head_index)
    # B's fragment is on disk but not in HEAD, so HEAD's index does not name it.
    assert "gemm/gemm_b.py" not in _index_paths(head_index)


def test_a_fragment_whose_entry_is_not_committed_is_left_out(repo, commit):
    _first_smith(repo, commit)
    ci.sync(repo, commit=commit)
    paths = _entry(repo, "norm", "half_landed", "fragment committed, wrapper not")
    ok, _ = commit("feat(catalog): fragment only", [paths[2]])
    assert ok
    ci.sync(repo, commit=commit)
    assert "norm/half_landed.py" not in _index_paths(_head_index(repo))
    assert any("does not exist" in p for p in ci.problems(ci.snapshot_at(repo)))


def test_a_row_committed_the_old_way_is_adopted_not_dropped(repo, commit):
    _first_smith(repo, commit)
    ci.sync(repo, commit=commit)
    catalog = _catalog(repo)
    (catalog / "norm" / "legacy.py").write_text("def f(): ...\n", encoding="utf-8")
    index = catalog / "index.yaml"
    index.write_text(
        index.read_text()
        + '\n  - path: norm/legacy.py\n    impl: torch.ops.trtllm.legacy\n    summary: "old way"\n',
        encoding="utf-8",
    )
    _commit_all(repo, "smith that still edits index.yaml")

    assert ci.sync(repo, commit=commit).action == "committed"
    assert f"{CATALOG_DIR}/index.d/norm/legacy.yaml" in _last_commit_files(repo)
    assert "norm/legacy.py" in _index_paths(_head_index(repo))


def test_an_index_with_uncommitted_edits_is_left_alone(repo, commit):
    _first_smith(repo, commit)
    index = _catalog(repo) / "index.yaml"
    index.write_text(index.read_text() + "# someone is editing\n", encoding="utf-8")
    assert ci.sync(repo, commit=commit).action == "deferred"
    assert index.read_text().endswith("# someone is editing\n")


def test_a_failed_commit_restores_the_tree(repo, commit):
    _first_smith(repo, commit)
    before = (_catalog(repo) / "index.yaml").read_text()
    result = ci.sync(repo, commit=lambda message, paths: (False, "lock timeout"))
    assert result.action == "failed"
    assert (_catalog(repo) / "index.yaml").read_text() == before
    assert not (_catalog(repo) / "index.d" / ci.HEADER_FILENAME).exists()
    assert _git(repo, "status", "--porcelain", "--", CATALOG_DIR) == ""


def test_outside_git_nothing_is_written(tmp_path):
    root = _build_repo(tmp_path)
    result = ci.sync(root, commit=lambda message, paths: (True, ""))
    assert result.action == "skipped"
    assert not (_catalog(root) / "index.d").exists()


# ------------------------------------------------------------ in the workflow


def test_a_run_generates_the_index_and_delivery_checks_it(tmp_path, task_file, monkeypatch):
    from agent_flow.workflows.staircase_onboard.state import load_state

    from .test_workflow import Script, _install, _plan_then_vocabulary, _workflow

    root = Path(yaml.safe_load(task_file.read_text())["trtllm_repo_path"])
    # An earlier smith's entry, committed with its fragment the new way.
    paths = _entry(root, "norm", "earlier_smith", "committed before this run")
    _git(root, "add", *paths)
    _git(root, "commit", "-qm", "feat(catalog): onboard earlier_smith")
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, [])
    with workflow:
        workflow.run(task_file)

    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").verdict == "delivered"
    assert "index.d/_header.yaml" in _git(root, "ls-files", CATALOG_DIR)
    assert "norm/earlier_smith.py" in _index_paths(_head_index(root))
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "catalog index committed" in journal
    assert "[ok] catalog:index-generated" in journal


# ------------------------------------------- idempotent, and safe under concurrency


def _migration_commits(repo: Path) -> list[str]:
    return [
        line
        for line in _git(repo, "log", "--format=%s").splitlines()
        if line.startswith("chore(catalog): split index.yaml")
    ]


def test_a_second_sync_on_the_same_head_changes_nothing(tmp_path, repo, commit):
    _first_smith(repo, commit)
    lock = tmp_path / "ws" / ci.LOCK_FILENAME
    assert ci.sync(repo, commit=commit, lock_path=lock).action == "committed"
    head = _git(repo, "rev-parse", "HEAD")
    assert ci.sync(repo, commit=commit, lock_path=lock).action == "current"
    assert _git(repo, "rev-parse", "HEAD") == head


def test_racing_syncs_migrate_exactly_once(tmp_path, repo, commit):
    """Two orchestrators (or a resume racing a dying run) on one repo."""
    import threading

    _first_smith(repo, commit)
    lock = tmp_path / "ws" / ci.LOCK_FILENAME
    results: list[str] = []
    barrier = threading.Barrier(6)

    def run():
        barrier.wait()
        results.append(ci.sync(repo, commit=commit, lock_path=lock).action)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["committed"] + ["current"] * 5
    assert len(_migration_commits(repo)) == 1
    assert _git(repo, "status", "--porcelain", "--", CATALOG_DIR) == ""


def test_smiths_landing_concurrently_with_syncs_never_cross_attribute(tmp_path, repo, commit):
    """F013 under load: six smiths commit while syncs regenerate the index."""
    import threading

    lock = tmp_path / "ws" / ci.LOCK_FILENAME
    errors: list[str] = []
    landed: dict[str, list[str]] = {}

    def smith(index: int) -> None:
        paths = _entry(repo, "norm", f"concurrent_{index}", f"smith {index}")
        ok, output = commit(f"feat(catalog): onboard concurrent_{index}", paths)
        if not ok:
            errors.append(output)
        landed[f"concurrent_{index}"] = sorted(paths)

    def syncer() -> None:
        for _ in range(3):
            if ci.sync(repo, commit=commit, lock_path=lock).action == "failed":
                errors.append("sync failed")

    threads = [threading.Thread(target=smith, args=(i,)) for i in range(6)]
    threads += [threading.Thread(target=syncer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    ci.sync(repo, commit=commit, lock_path=lock)
    assert ci.sync(repo, commit=commit, lock_path=lock).action == "current"

    paths = _index_paths(_head_index(repo))
    assert all(f"norm/{name}.py" in paths for name in landed)
    assert len(_migration_commits(repo)) == 1
    for name, files in landed.items():
        sha = _git(repo, "log", "--format=%H", "--grep", f"onboard {name}$").split()[0]
        changed = sorted(
            _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
        )
        assert changed == files, name
    assert ci.problems(ci.snapshot_at(repo)) == []
    assert _git(repo, "status", "--porcelain", "--", CATALOG_DIR) == ""


def test_a_commit_that_landed_but_reported_failure_is_not_undone(repo, commit):
    _first_smith(repo, commit)

    def lands_then_complains(message, paths):
        commit(message, paths)
        return False, "post-commit check complained"

    result = ci.sync(repo, commit=lands_then_complains)
    assert result.action == "committed"
    assert (_catalog(repo) / "index.d" / ci.HEADER_FILENAME).is_file()
    assert _git(repo, "status", "--porcelain", "--", CATALOG_DIR) == ""


def test_a_helper_that_stages_then_fails_leaves_nothing_staged(repo, commit):
    """A helper that resets its own commit leaves the paths staged; the undo unstages them."""
    _first_smith(repo, commit)

    def stages_then_fails(message, paths):
        _git(repo, "add", "--", *paths)
        return False, "refused after staging"

    assert ci.sync(repo, commit=stages_then_fails).action == "failed"
    assert _git(repo, "status", "--porcelain", "--", CATALOG_DIR) == ""
    assert ci.sync(repo, commit=commit).action == "committed"
