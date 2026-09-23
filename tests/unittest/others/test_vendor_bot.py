# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for metadata, attribution, local state, and author feedback."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = "vendor_bot_test"
_SPEC = importlib.util.spec_from_file_location(
    _PACKAGE,
    _ROOT / "scripts/vendor/__init__.py",
    submodule_search_locations=[str(_ROOT / "scripts/vendor")],
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_PACKAGE] = _MODULE
_SPEC.loader.exec_module(_MODULE)
b = importlib.import_module(f"{_PACKAGE}.bot")
m = importlib.import_module(f"{_PACKAGE}.metadata")
p = importlib.import_module(f"{_PACKAGE}.promote")

pytestmark = pytest.mark.cpu_only


def _body(content: str) -> str:
    return f"{m.BEGIN}\n```yaml\nschema_version: 1\nvendors:\n  toy:\n{content}\n```\n{m.END}"


def test_metadata_only_needs_upstream_urls() -> None:
    entry = m.parse(
        _body("    upstream_prs:\n      - https://github.com/org/library/pull/42"),
        "toy",
        "org/library",
    )
    assert entry == {"upstream_prs": [42], "unpaired_reason": None, "resolutions": {}}


@pytest.mark.parametrize(
    "content",
    [
        "    upstream_prs: []",
        "    upstream_prs: https://github.com/org/library/pull/42",
        "    upstream_prs: [https://github.com/other/library/pull/42]",
        "    upstream_prs: [https://github.com/org/library/pull/42, https://github.com/org/library/pull/42]",
        "    unpaired_reason: ''",
        "    unpaired_reason: test\n    upstream_prs: []",
        "    canonical_repo: attacker/repo",
        "    upstream_prs: []\n    upstream_prs: []",
        "    upstream_prs: [",
        "    upstream_prs: [https://github.com/org/library/pull/42]\n    resolutions: []",
        "    upstream_prs: &prs [https://github.com/org/library/pull/42]",
        "    upstream_prs: &prs [*prs]",
    ],
)
def test_invalid_metadata_rejected(content: str) -> None:
    with pytest.raises(ValueError):
        m.parse(_body(content), "toy", "org/library")


@pytest.mark.parametrize(
    "body", ["", "missing", m.template("toy", "org/library") * 2, m.END + m.BEGIN]
)
def test_missing_or_duplicate_marked_blocks(body: str) -> None:
    with pytest.raises(ValueError):
        m.parse(body, "toy", "org/library")


def test_author_resolution_requires_explanation() -> None:
    content = (
        "    upstream_prs: [https://github.com/org/library/pull/42]\n    resolutions:\n      "
        + "a" * 40
        + ":\n        upstream_pr: https://github.com/org/library/pull/42\n        reason: adapted ABI"
    )
    entry = m.parse(_body(content), "toy", "org/library")
    assert entry["resolutions"]["a" * 40] == {"upstream_pr": 42, "reason": "adapted ABI"}
    with pytest.raises(ValueError):
        m.parse(_body(content.replace("adapted ABI", "''")), "toy", "org/library")


def test_unpaired_and_template() -> None:
    entry = m.parse(
        _body("    unpaired_reason: downstream-only compatibility"), "toy", "org/library"
    )
    assert entry["unpaired_reason"] == "downstream-only compatibility"
    assert m.parse(m.template("toy", "org/library"), "toy", "org/library")["upstream_prs"] == [123]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "core.hooksPath", str(tmp_path / "no-hooks"))
    return repo


def _commit(repo: Path, text: str, filename: str = "code.py") -> str:
    (repo / filename).write_text(text)
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", "test")
    return _git(repo, "rev-parse", "HEAD")


def _pr(base: str, head: str) -> dict:
    return {"base": {"sha": base, "ref": "main"}, "head": {"sha": head}, "state": "open"}


def test_match_multiple_upstream_prs_without_manual_map(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    first = _commit(source, "VALUE = 1\n")
    second = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(
        source, base, second, {41: _pr(base, first), 42: _pr(first, second)}
    )
    assert assignments == {first: 41, second: 42}
    assert all(item["method"] == "same-commit" for item in evidence.values())


def test_match_rebased_patch(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    context = _commit(source, "context\n", "other.txt")
    upstream = _commit(source, "VALUE = 1\n")
    assignments, evidence = m.match(source, base, downstream, {42: _pr(context, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"


@pytest.mark.parametrize("different", ["    VALUE = 1\n", "VALUE = 1 \n", "VALUE = 1\r\n"])
def test_matching_preserves_whitespace(source: Path, different: str) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, different)
    with pytest.raises(ValueError, match="Author action required"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


def test_identical_edits_in_different_functions_do_not_match(source: Path) -> None:
    original = "def first():\n    return False\n\n\ndef second():\n    return False\n"
    base = _commit(source, original)
    downstream = _commit(source, original.replace("return False", "return True", 1))
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(
        source,
        original.replace("def second():\n    return False", "def second():\n    return True"),
    )
    with pytest.raises(ValueError, match="Author action required"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


def test_matching_squashed_series_and_ambiguous_duplicates(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    first = _commit(source, "VALUE = 1\n")
    second = _commit(source, "VALUE = 2\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    squash = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(source, base, second, {42: _pr(base, squash)})
    assert assignments == {first: 42, second: 42}
    assert evidence[first]["method"] == "source-series-aggregate"
    with pytest.raises(ValueError, match="#42, #43"):
        m.match(source, base, second, {42: _pr(base, squash), 43: _pr(base, squash)})


def test_author_can_resolve_adapted_commit(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(
        source,
        base,
        downstream,
        {42: _pr(base, upstream)},
        {downstream: {"upstream_pr": 42, "reason": "downstream ABI adaptation"}},
    )
    assert assignments == {downstream: 42}
    assert evidence[downstream]["refresh_policy"] == "retain-until-verified"
    with pytest.raises(ValueError, match="outside"):
        m.match(source, base, downstream, {42: _pr(base, upstream)}, {base: {"upstream_pr": 42}})


class FakeGitHub(p.GitHub):
    def __init__(self) -> None:
        super().__init__("org/consumer", "toy", "org/library")
        self.pr = {
            "number": 7,
            "user": {"login": "author"},
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "base": {"ref": "main"},
            "body": "",
            "updated_at": "2099-01-01T00:00:00Z",
        }
        self.comments: list[dict] = []
        self.reviews: list[dict] = []
        self.writes: list[tuple[str, str, dict]] = []

    def api(
        self, endpoint: str, *, method: str = "GET", payload: dict | None = None
    ) -> dict | list:
        if method != "GET":
            self.writes.append((endpoint, method, payload))
        if endpoint == "user":
            return {"login": "operator"}
        if endpoint.endswith("/pulls/7"):
            return copy.deepcopy(self.pr)
        if "/pulls?" in endpoint:
            return [copy.deepcopy(self.pr)]
        if "/files?" in endpoint:
            return [{"filename": p._LOCK}]
        if "/reviews/" in endpoint and endpoint.endswith("/dismissals"):
            number = int(endpoint.split("/")[-2])
            self.reviews[number - 1]["state"] = "DISMISSED"
            return {}
        if "/reviews" in endpoint:
            if method == "GET":
                return copy.deepcopy(self.reviews)
            self.reviews.append(
                {
                    "id": len(self.reviews) + 1,
                    "user": {"login": "operator"},
                    "body": payload["body"],
                    "state": "CHANGES_REQUESTED",
                }
            )
            return self.reviews[-1]
        if "/comments/" in endpoint:
            number = int(endpoint.rsplit("/", 1)[1])
            self.comments[number - 1]["body"] = payload["body"]
            return {}
        if "/comments" in endpoint:
            if method == "GET":
                return copy.deepcopy(self.comments)
            self.comments.append(
                {
                    "id": len(self.comments) + 1,
                    "user": {"login": "operator"},
                    "body": payload["body"],
                }
            )
            return self.comments[-1]
        raise AssertionError((endpoint, method, payload))


@pytest.fixture
def monitor(tmp_path: Path):
    args = argparse.Namespace(
        canonical_repo="operator/library",
        canonical_branch="dev",
        fork="operator/consumer",
        repo=tmp_path,
        workdir=tmp_path,
        publish=True,
        since=None,
    )
    store = b.Store(tmp_path / "state.sqlite")
    gh = FakeGitHub()
    instance = b.Monitor(args, gh, store)
    yield instance
    store.close()


def test_feedback_requests_changes_once_and_clears_only_metadata_review(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, "Missing marked block")
    monitor._feedback(gh.pr, "Missing marked block")
    assert len(gh.comments) == len(gh.reviews) == 1
    assert "PR description" in gh.comments[0]["body"]
    assert "resolutions:" in gh.comments[0]["body"]
    assert gh.writes[1][2]["event"] == "REQUEST_CHANGES"
    monitor._feedback(gh.pr, None)
    assert gh.reviews[0]["state"] == "DISMISSED"
    assert all(payload.get("event") != "APPROVE" for _, _, payload in gh.writes)


def test_bot_does_not_override_operator_manual_review(monitor: b.Monitor) -> None:
    monitor.gh.reviews.append(
        {
            "id": 1,
            "user": {"login": "operator"},
            "body": "Manual code review",
            "state": "CHANGES_REQUESTED",
        }
    )
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    monitor._feedback(monitor.gh.pr, None)
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    assert len(monitor.gh.reviews) == 1


def test_dry_run_and_self_review(monitor: b.Monitor) -> None:
    monitor.args.publish = False
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    assert not monitor.gh.writes
    monitor.args.publish = True
    monitor.gh.pr["user"]["login"] = "operator"
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    assert monitor.gh.comments
    assert not monitor.gh.reviews


def test_feedback_rechecks_mutable_description(monitor: b.Monitor) -> None:
    stale = copy.deepcopy(monitor.gh.pr)
    monitor.gh.pr["body"] = "updated while checking"
    monitor._feedback(stale, "Missing metadata")
    assert not monitor.gh.writes


def test_discovery_and_persistent_operator_scope(monitor: b.Monitor) -> None:
    assert monitor._discover() == [7]
    assert monitor._discover() == [7]
    assert not monitor.gh.writes
    changed = copy.copy(monitor.args)
    changed.canonical_repo = "different/library"
    with pytest.raises(ValueError, match="different operator"):
        b.Monitor(changed, monitor.gh, monitor.store)


def _vendor(commit: str, url: str, branch: str):
    return m.manage._validate_vendor(
        "toy",
        {
            "url": url,
            "branch": branch,
            "commit": commit,
            "source": "src",
            "destination": "vendored/toy",
            "include": ["**/*.py"],
            "digest": "sha256-tree-v1:" + "0" * 64,
        },
    )


def test_monitor_reuses_matching_and_publishes_once_after_merge(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gh = monitor.gh
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    gh.pr["body"] = _body("    upstream_prs: [https://github.com/org/library/pull/42]")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    matches, publications = [], []

    def resolve(*args):
        matches.append(True)
        return {reviewed.commit: 42}, {reviewed.commit: {"method": "same-edits"}}

    def make_plan(args, github):
        assert args.auto_merge and not args.wait
        assert args.map_upstream == [f"{reviewed.commit}=42"]
        return p.Promotion(
            7,
            "c" * 40,
            previous,
            reviewed,
            "operator/library",
            "operator/consumer",
            {},
            consumer_repo="org/consumer",
        )

    def publish(args, github, plan):
        publications.append(plan)
        assert plan.record["attribution"][reviewed.commit]["method"] == "same-edits"
        return {"number": 8}

    original = gh.get
    monkeypatch.setattr(
        gh,
        "get",
        lambda endpoint: {
            "number": 8,
            "merged": True,
            "merge_commit_sha": "d" * 40,
            "html_url": "https://github.com/org/consumer/pull/8",
        }
        if endpoint.endswith("/pulls/8")
        else original(endpoint),
    )
    monkeypatch.setattr(m, "resolve", resolve)
    monkeypatch.setattr(p, "_make_plan", make_plan)
    monkeypatch.setattr(p, "_publish", publish)
    monkeypatch.setattr(p, "_verify_remote_commit", lambda *args: "verified")
    monitor.inspect(7)
    monitor.inspect(7)
    assert len(matches) == 1
    assert not publications
    gh.pr.update(merged=True, state="closed")
    monitor.inspect(7)
    monitor.inspect(7)
    assert len(publications) == 1
    assert monitor.store.get("complete:7").endswith("/8")


def test_monitor_reports_attribution_to_author_and_retries_after_correction(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monitor.gh.pr["body"] = _body("    upstream_prs: [https://github.com/org/library/pull/42]")

    def unresolved(*args):
        raise ValueError("Author action required: no exact match for " + reviewed.commit)

    monkeypatch.setattr(m, "resolve", unresolved)
    monitor.inspect(7)
    assert "@author" in monitor.gh.comments[0]["body"]
    assert reviewed.commit in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    monkeypatch.setattr(m, "resolve", lambda *args: ({reviewed.commit: 42}, {}))
    monitor.inspect(7)
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


def test_forbidden_dismissal_informs_author_without_approving(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor._feedback(monitor.gh.pr, "Missing block")
    original = monitor.gh.api

    def forbidden(endpoint, **kwargs):
        if endpoint.endswith("/dismissals"):
            raise RuntimeError("HTTP 403")
        return original(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", forbidden)
    monitor._feedback(monitor.gh.pr, None)
    assert "lacks dismissal permission" in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = len(monitor.gh.writes)
    monitor._feedback(monitor.gh.pr, None)
    assert len(monitor.gh.writes) == writes
    monkeypatch.setattr(monitor.gh, "api", original)
    monitor._feedback(monitor.gh.pr, None)
    assert "lacks dismissal permission" not in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


def test_old_promotion_messages_remain_readable() -> None:
    record = {
        "version": 1,
        "vendor": "flashinfer-prims-ts",
        "trtllm_pr": "https://github.com/NVIDIA/TensorRT-LLM/pull/17",
        "trtllm_merge": "a" * 40,
    }
    message = (
        "----- BEGIN PRIMTS PROMOTION V1 -----\n"
        + json.dumps(record)
        + "\n----- END PRIMTS PROMOTION V1 -----"
    )
    normalized = p._record_from_message(message)
    assert normalized["consumer_pr"].endswith("/17")
    assert normalized["upstream_repository"] == "flashinfer-ai/flashinfer"


def test_workdir_locks_and_refuses_unrelated_data(tmp_path: Path) -> None:
    workdir = tmp_path / "state"
    b._initialize_workdir(workdir)
    with b._locked(workdir):
        assert b._running(workdir)
        with pytest.raises(ValueError, match="already owns"):
            with b._locked(workdir):
                pass
    assert not b._running(workdir)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "data").write_text("preserve")
    with pytest.raises(ValueError, match="empty workdir"):
        b._initialize_workdir(unrelated)
    assert (unrelated / "data").read_text() == "preserve"


def test_workdir_refuses_checkout(source: Path) -> None:
    with pytest.raises(ValueError, match="outside a Git worktree"):
        b._initialize_workdir(source / "bot-state")


def test_daemon_launch_status_stop_without_systemd(tmp_path: Path) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        "print(json.dumps({'login': 'operator'} if sys.argv[-1] == 'user' else []))\n"
    )
    gh.chmod(0o700)
    workdir = tmp_path / "state"
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "GH_CONFIG_DIR": str(tmp_path / "private-auth"),
    }
    command = [sys.executable, str(_ROOT / "scripts/vendor/bot.py")]
    arguments = [
        "run",
        "--workdir",
        str(workdir),
        "--vendor",
        "toy",
        "--upstream-repo",
        "org/library",
        "--canonical-repo",
        "operator/library",
        "--canonical-branch",
        "dev",
        "--fork",
        "operator/consumer",
        "--daemon",
    ]
    try:
        result = subprocess.run(
            command + arguments, env=env, capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, result.stderr
        assert "Started daemon PID" in result.stdout
        status = subprocess.run(
            command + ["status", "--workdir", str(workdir)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(status.stdout)["running"]
        assert "private-auth" not in (workdir / "bot.log").read_text()
    finally:
        subprocess.run(
            command + ["stop", "--workdir", str(workdir)], env=env, capture_output=True, timeout=5
        )
        deadline = time.monotonic() + 10
        while b._running(workdir) and time.monotonic() < deadline:
            time.sleep(0.1)
    assert not b._running(workdir)
