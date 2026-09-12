# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Promotion lifecycle tests: real local Git, fake GitHub, no network or GPUs."""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

# tests/unittest/scripts shadows the repository's scripts namespace. Load the
# standalone tool as an isolated package, without editing sys.path.
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "prims_ts_maintenance_test",
    _SCRIPTS / "maintain_prims_ts.py",
    submodule_search_locations=[str(_SCRIPTS)],
)
m = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m
_SPEC.loader.exec_module(m)

pytestmark = pytest.mark.cpu_only


def _git(repo: Path, *args: str, text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=text, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Promotion Test")
    _git(path, "config", "user.email", "promotion@example.invalid")
    _git(path, "config", "core.hooksPath", str(path / "empty-hooks"))
    return path


def _write(repo: Path, name: str, content: str) -> None:
    file = repo / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-s", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class World(m.GitHub):
    """GitHub facade whose immutable data comes from temporary Git repositories."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = _init(root / "flashinfer")
        _write(self.source, "prims_ts/kernel.py", "VALUE = 0\n")
        self.upstream = _commit(self.source, "upstream baseline")
        _write(self.source, "prims_ts/kernel.py", "VALUE = 1\n")
        self.previous = _commit(self.source, "previous downstream fix")
        _write(self.source, "prims_ts/kernel.py", "VALUE = 2\n")
        self.first = _commit(self.source, "first fix")
        _write(self.source, "prims_ts/kernel.py", "VALUE = 3\n")
        self.reviewed = _commit(self.source, "second fix")
        self.consumer = _init(root / "trtllm")
        assert (
            m.vendor.main(
                [
                    "--lock",
                    str(self.consumer / m._LOCK),
                    "create",
                    m._VENDOR,
                    "--url",
                    "https://github.com/maintainer/flashinfer.git",
                    "--branch",
                    "trtllm-prims-ts-dev",
                    "--commit",
                    self.previous,
                    "--source",
                    "prims_ts",
                    "--destination",
                    "vendored/prims_ts",
                    "--include",
                    "**/*.py",
                    "--repo",
                    str(self.source),
                ]
            )
            == 0
        )
        self.base = _commit(self.consumer, "previous vendor pin")
        _write(self.consumer, "vendored/prims_ts/kernel.py", "VALUE = 3\n")
        assert (
            m.vendor.main(
                [
                    "--lock",
                    str(self.consumer / m._LOCK),
                    "pin",
                    m._VENDOR,
                    "--url",
                    "https://github.com/dev/flashinfer.git",
                    "--branch",
                    "tmp-fix",
                    "--commit",
                    self.reviewed,
                    "--repo",
                    str(self.source),
                ]
            )
            == 0
        )
        self.merged = _commit(self.consumer, "merged source update")
        self.main = self.merged
        self.canonical = root / "canonical.git"
        self.fork = root / "trtllm-fork.git"
        _git(root, "clone", "-q", "--bare", str(self.source), str(self.canonical))
        _git(root, "clone", "-q", "--bare", str(self.consumer), str(self.fork))
        _git(self.canonical, "update-ref", "refs/heads/trtllm-prims-ts-dev", self.previous)
        _git(
            self.consumer, "remote", "add", "fork", "https://github.com/maintainer/TensorRT-LLM.git"
        )
        self.prs: dict[int, dict] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.overrides: dict[str, dict | list] = {}
        self.state = "BLOCKED"
        self.tamper_auto = False
        self.fail_create = False
        self.comments: list[dict] = []
        self.fail_comment_before = False
        self.fail_comment_after = False
        self.git_calls: list[tuple[str, ...]] = []

    def args(self, **overrides: object) -> argparse.Namespace:
        values = dict(
            trtllm_pr="17",
            canonical_repo="maintainer/flashinfer",
            canonical_branch="trtllm-prims-ts-dev",
            upstream_pr=["4829"],
            unpaired_reason=None,
            map_upstream=[],
            repo=self.consumer,
            fork=None,
            flashinfer_repo=self.source,
            worktree=self.root / "followup",
            publish=False,
            auto_merge=False,
            wait=False,
            timeout=30,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def commit_data(self, repo: Path, sha: str) -> dict:
        return {
            "sha": sha,
            "parents": [{"sha": s} for s in _git(repo, "show", "-s", "--format=%P", sha).split()],
            "commit": {"message": _git(repo, "show", "-s", "--format=%B", sha)},
            "files": [
                {"filename": f}
                for f in _git(
                    repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha
                ).splitlines()
            ],
        }

    def compare(self, repo: str, base: str, head: str, *, all_commits: bool = False) -> dict:
        source = self.consumer if repo == m._TRTLLM else self.source
        commits = _git(source, "rev-list", "--reverse", f"{base}..{head}").splitlines()
        return {
            "merge_base_commit": {"sha": _git(source, "merge-base", base, head)},
            "commits": [self.commit_data(source, sha) for sha in commits],
            "total_commits": len(commits),
        }

    def routed_git(self, repo: Path, *args: str, text: str | None = None) -> str:
        self.git_calls.append(args)
        replacements = {
            "https://github.com/NVIDIA/TensorRT-LLM.git": str(self.consumer),
            "https://github.com/maintainer/TensorRT-LLM.git": str(self.fork),
            "https://github.com/maintainer/flashinfer.git": str(self.canonical),
            "https://github.com/dev/flashinfer.git": str(self.source),
        }
        if args[:3] == ("remote", "get-url", "fork"):
            return "https://github.com/maintainer/TensorRT-LLM.git"
        return _git(repo, *[replacements.get(arg, arg) for arg in args], text=text)

    def api(
        self, endpoint: str, *, method: str = "GET", payload: dict | None = None
    ) -> dict | list:
        self.calls.append((method, endpoint, payload))
        if method == "GET" and endpoint in self.overrides:
            return copy.deepcopy(self.overrides[endpoint])
        if endpoint == "user":
            return {"login": "maintainer"}
        if endpoint == "graphql":
            query = payload["query"]
            pr = self.prs[99]
            if query.startswith("query"):
                return {
                    "data": {
                        "node": {"headRefOid": pr["head"]["sha"], "mergeStateStatus": self.state}
                    }
                }
            if "disablePullRequestAutoMerge" in query:
                pr["auto_merge"] = None
                return {"data": {}}
            data = payload["variables"]["input"]
            assert data["expectedHeadOid"] == pr["head"]["sha"]
            if "enablePullRequestAutoMerge" in query:
                pr["auto_merge"] = {
                    "merge_method": "squash",
                    "commit_title": data["commitHeadline"],
                    "commit_message": "lost metadata" if self.tamper_auto else data["commitBody"],
                }
            else:
                message = f"{data['commitHeadline']}\n\n{data['commitBody']}\n"
                tree = _git(self.consumer, "rev-parse", f"{pr['head']['sha']}^{{tree}}")
                sha = _git(self.consumer, "commit-tree", tree, "-p", self.main, text=message)
                self.main = sha
                pr.update(merged=True, merged_at="today", state="closed", merge_commit_sha=sha)
            return {"data": {}}
        prefix = "repos/"
        assert endpoint.startswith(prefix), endpoint
        owner, name, *rest = endpoint[len(prefix) :].split("/")
        repository = f"{owner}/{name}"
        route = "/".join(rest)
        if route.startswith("issues/99/comments"):
            if method == "GET":
                page = int(route.rsplit("page=", 1)[1])
                return copy.deepcopy(self.comments[(page - 1) * 100 : page * 100])
            if self.fail_comment_before:
                raise RuntimeError("simulated comment failure before posting")
            comment = {
                "body": payload["body"],
                "user": {"login": "maintainer"},
                "html_url": f"https://github.com/NVIDIA/TensorRT-LLM/pull/99#issuecomment-{len(self.comments) + 1}",
            }
            self.comments.append(comment)
            if self.fail_comment_after:
                raise RuntimeError("simulated comment failure after posting")
            return copy.deepcopy(comment)
        if not route:
            root = m._FLASHINFER if name == "flashinfer" else m._TRTLLM
            return {
                "source": {"full_name": root},
                "allow_auto_merge": True,
                "allow_squash_merge": True,
            }
        if route == "rules/branches/main":
            return [{"type": "required_status_checks"}]
        if route.startswith("git/ref/heads/"):
            branch = route[len("git/ref/heads/") :]
            if repository == m._TRTLLM:
                return {"object": {"sha": self.main}}
            if repository == m._FLASHINFER:
                return {"object": {"sha": self.upstream}}
            remote = self.canonical if name == "flashinfer" else self.fork
            result = subprocess.run(
                ["git", "-C", str(remote), "rev-parse", "--verify", f"refs/heads/{branch}"],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError("HTTP 404")
            return {"object": {"sha": result.stdout.strip()}}
        if route.startswith("contents/"):
            path, sha = route[len("contents/") :].split("?ref=")
            content = _git(self.consumer, "show", f"{sha}:{path}") + "\n"
            return {"encoding": "base64", "content": base64.b64encode(content.encode()).decode()}
        if route.startswith("commits/"):
            return self.commit_data(self.consumer, route.split("/")[1])
        if route.startswith("pulls?"):
            return copy.deepcopy(list(self.prs.values()))
        if route == "pulls" and method == "POST":
            if self.fail_create:
                raise RuntimeError("simulated PR creation failure")
            branch = payload["head"].split(":")[1]
            head = _git(self.fork, "rev-parse", f"refs/heads/{branch}")
            self.prs[99] = dict(
                number=99,
                node_id="PR_test",
                state="open",
                merged=False,
                head={"sha": head},
                html_url="https://github.com/NVIDIA/TensorRT-LLM/pull/99",
                body=payload["body"],
                auto_merge=None,
            )
            return copy.deepcopy(self.prs[99])
        if route.startswith("pulls/"):
            number = int(route.split("/")[1])
            if repository == m._FLASHINFER:
                return {
                    "number": number,
                    "base": {"ref": "main", "sha": self.upstream},
                    "head": {"sha": self.reviewed},
                    "state": "open",
                    "merged": False,
                    "draft": True,
                }
            if number == 17:
                return {
                    "number": 17,
                    "merged": True,
                    "base": {"ref": "main"},
                    "merge_commit_sha": self.merged,
                    "head": {"sha": "a" * 40},
                }
            return copy.deepcopy(self.prs[number])
        raise AssertionError((method, endpoint, payload))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    instance = World(tmp_path)
    monkeypatch.setattr(m, "_git", instance.routed_git)
    monkeypatch.setattr(m, "GitHub", lambda: instance)
    return instance


def test_dry_run_has_no_writes_and_pins_merged_not_live_head(
    world: World, capsys: pytest.CaptureFixture
) -> None:
    assert (
        m.main(
            [
                "promote",
                "--trtllm-pr",
                "17",
                "--canonical-repo",
                "maintainer/flashinfer",
                "--upstream-pr",
                "4829",
                "--repo",
                str(world.consumer),
            ]
        )
        == 0
    )
    assert "DRY RUN" in capsys.readouterr().out
    assert all(method == "GET" for method, _, _ in world.calls)
    assert not any(args[0] in ("push", "commit", "worktree") for args in world.git_calls)
    plan = m._make_plan(world.args(), world)
    assert plan.reviewed.commit == world.reviewed
    assert plan.record["changes"][0]["commits"] == [world.first, world.reviewed]
    assert plan.record["upstream_base"] == world.upstream


def test_publish_signed_lock_only_and_repeat(world: World) -> None:
    args = world.args(publish=True)
    plan = m._make_plan(args, world)
    pr = m._publish(args, world, plan)
    assert _git(world.canonical, "rev-parse", "trtllm-prims-ts-dev") == world.reviewed
    head = pr["head"]["sha"]
    assert _git(world.consumer, "diff", "--name-only", f"{head}^", head) == m._LOCK
    assert m._lock_at(world, m._TRTLLM, head).to_mapping() == plan.promoted
    message = _git(world.consumer, "show", "-s", "--format=%B", head)
    assert m._record_from_message(message) == plan.record
    assert "Signed-off-by: Promotion Test <promotion@example.invalid>" in message
    assert not any("graphql" == endpoint for _, endpoint, _ in world.calls)
    m._publish(args, world, m._make_plan(args, world))
    assert (
        sum(method == "POST" and endpoint.endswith("/pulls") for method, endpoint, _ in world.calls)
        == 1
    )


@pytest.mark.parametrize("state", ["BLOCKED", "CLEAN"])
def test_auto_merge_message_and_immediate_merge(world: World, state: str) -> None:
    world.state = state
    args = world.args(publish=True, auto_merge=True)
    plan = m._make_plan(args, world)
    m._publish(args, world, plan)
    if state == "CLEAN":
        assert world.prs[99]["merged"]
        m._verify_remote_commit(world, plan, m._TRTLLM, world.main)
        m._publish(args, world, m._make_plan(args, world))
    else:
        saved = world.prs[99]["auto_merge"]
        assert saved["merge_method"] == "squash"
        assert m._record_from_message(saved["commit_message"]) == plan.record
        assert "Signed-off-by:" in saved["commit_message"]


def test_resume_after_upstream_head_moves_retains_snapshot(world: World) -> None:
    args = world.args(publish=True)
    original = m._make_plan(args, world)
    m._publish(args, world, original)
    changed = world.get(f"repos/{m._FLASHINFER}/pulls/4829")
    changed["head"]["sha"] = "b" * 40
    world.overrides[f"repos/{m._FLASHINFER}/pulls/4829"] = changed
    resumed = m._make_plan(args, world)
    assert resumed.record != original.record
    m._publish(args, world, resumed)
    assert resumed.record == original.record


def test_resume_after_push_before_pr_creation(world: World) -> None:
    args = world.args(publish=True)
    world.fail_create = True
    with pytest.raises(RuntimeError, match="creation failure"):
        m._publish(args, world, m._make_plan(args, world))
    assert _git(world.canonical, "rev-parse", "trtllm-prims-ts-dev") == world.reviewed
    world.fail_create = False
    m._publish(args, world, m._make_plan(args, world))
    assert len(world.prs) == 1


def test_auto_merge_tampering_disables_request(world: World) -> None:
    world.tamper_auto = True
    args = world.args(publish=True, auto_merge=True)
    with pytest.raises(ValueError, match="verification failed"):
        m._publish(args, world, m._make_plan(args, world))
    assert world.prs[99]["auto_merge"] is None


@pytest.mark.parametrize("field,value", [("merged", False), ("base", {"ref": "release"})])
def test_unmerged_or_wrong_base_rejected(world: World, field: str, value: object) -> None:
    endpoint = f"repos/{m._TRTLLM}/pulls/17"
    changed = world.get(endpoint)
    changed[field] = value
    world.overrides[endpoint] = changed
    with pytest.raises(ValueError, match="already merged"):
        m._make_plan(world.args(), world)


@pytest.mark.parametrize("target", ["main", "canonical"])
def test_concurrent_updates_block_before_publication(world: World, target: str) -> None:
    plan = m._make_plan(world.args(), world)
    if target == "main":
        world.main = world.base
    else:
        _git(world.canonical, "update-ref", "refs/heads/trtllm-prims-ts-dev", world.first)
    with pytest.raises(ValueError, match="superseded|moved unexpectedly"):
        m._publish(world.args(publish=True), world, plan)
    assert not any(method != "GET" for method, _, _ in world.calls)
    assert not any(args[0] == "push" for args in world.git_calls)


def test_divergent_source_rejected(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    original = world.compare

    def divergent(*args: object, **kwargs: object) -> dict:
        result = original(*args, **kwargs)
        result["merge_base_commit"]["sha"] = world.upstream
        return result

    monkeypatch.setattr(world, "compare", divergent)
    with pytest.raises(ValueError, match="not a fast-forward"):
        m._make_plan(world.args(), world)


def test_upstream_closed_unmerged_rejected(world: World) -> None:
    endpoint = f"repos/{m._FLASHINFER}/pulls/4829"
    changed = world.get(endpoint)
    changed.update(state="closed", merged=False)
    world.overrides[endpoint] = changed
    with pytest.raises(ValueError, match="Upstream PR"):
        m._make_plan(world.args(), world)


@pytest.mark.parametrize("setting", ["auto", "squash", "queue"])
def test_unsupported_auto_merge_settings_fail_before_push(world: World, setting: str) -> None:
    if setting == "queue":
        world.overrides[f"repos/{m._TRTLLM}/rules/branches/main"] = [{"type": "merge_queue"}]
    else:
        data = world.get(f"repos/{m._TRTLLM}")
        data[f"allow_{'auto_merge' if setting == 'auto' else 'squash_merge'}"] = False
        world.overrides[f"repos/{m._TRTLLM}"] = data
    args = world.args(publish=True, auto_merge=True)
    with pytest.raises(ValueError, match="allow|Merge queues"):
        m._publish(args, world, m._make_plan(args, world))
    assert not any(args[0] == "push" for args in world.git_calls)


def test_multiple_upstream_pr_mapping(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m.sys.stdin, "isatty", lambda: False)
    args = world.args(upstream_pr=["4829", "4830"])
    with pytest.raises(ValueError, match="--map-upstream"):
        m._make_plan(args, world)
    args.map_upstream = [f"{world.first}=4829", f"{world.reviewed}=4830"]
    plan = m._make_plan(args, world)
    assert [group["commits"] for group in plan.record["changes"]] == [
        [world.first],
        [world.reviewed],
    ]


@pytest.mark.parametrize("mapping", ["1234567=4829", "bad=4829", "abcdefg=4829"])
def test_bad_mapping_rejected(world: World, mapping: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        m._make_plan(world.args(map_upstream=[mapping]), world)


def test_modified_vendor_files_block_before_push(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = world.routed_git

    def tamper(repo: Path, *args: str, text: str | None = None) -> str:
        result = original(repo, *args, text=text)
        if args[:2] == ("worktree", "add"):
            _write(world.root / "followup", "vendored/prims_ts/kernel.py", "UNREVIEWED = True\n")
        return result

    monkeypatch.setattr(m, "_git", tamper)
    args = world.args(publish=True)
    with pytest.raises(ValueError, match="outside the vendor lock"):
        m._publish(args, world, m._make_plan(args, world))
    assert not any(args[0] == "push" for args in world.git_calls)


def test_commit_hook_adds_unrelated_file_rejected(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = world.routed_git

    def tamper(repo: Path, *args: str, text: str | None = None) -> str:
        if args[0] == "commit":
            _write(repo, "unrelated.txt", "unexpected hook edit\n")
            _git(repo, "add", "unrelated.txt")
        return original(repo, *args, text=text)

    monkeypatch.setattr(m, "_git", tamper)
    args = world.args(publish=True)
    with pytest.raises(ValueError, match="outside the vendor lock"):
        m._publish(args, world, m._make_plan(args, world))
    assert not any(args[0] == "push" for args in world.git_calls)


@pytest.mark.parametrize(
    "value",
    [
        "http://github.com/user/repo",
        "https://token@github.com/a/b",
        "file:///tmp/source",
        "https://example.com/a/b",
    ],
)
def test_non_github_or_credential_urls_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="credential-free"):
        m._repo_name(value)


@pytest.mark.parametrize("suffix", ["", "/", ".git"])
def test_github_repo_url_parsing(suffix: str) -> None:
    assert m._repo_name(f"https://github.com/owner/repo{suffix}") == "owner/repo"


@pytest.mark.parametrize("flags", [["--auto-merge"], ["--wait"], ["--timeout", "0"]])
def test_cli_invalid_flags(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        m.main(
            [
                "promote",
                "--trtllm-pr",
                "17",
                "--canonical-repo",
                "maintainer/flashinfer",
                "--upstream-pr",
                "4829",
                *flags,
            ]
        )
    assert raised.value.code == 2


def test_command_preserves_identity_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_CONFIG_DIR", "/tmp/test-gh-config")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Real Maintainer")
    monkeypatch.setenv("GIT_DIR", "/unrelated/git")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        env = kwargs["env"]
        assert env["GH_CONFIG_DIR"] == "/tmp/test-gh-config"
        assert env["GIT_AUTHOR_NAME"] == "Real Maintainer"
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert "GIT_DIR" not in env
        assert "shell" not in kwargs
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(m.subprocess, "run", run)
    assert m._run(["git", "status"]) == "ok"


def test_metadata_missing_and_duplicate_rejected() -> None:
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        m._record_from_message("ordinary commit")
    record = m._record_text({"version": 1})
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        m._record_from_message(record + record)


def test_compare_paginates_all_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = object.__new__(World)

    def get(endpoint: str) -> dict:
        page = int(endpoint.rsplit("=", 1)[1])
        return {"total_commits": 101, "commits": list(range(100)) if page == 1 else [100]}

    monkeypatch.setattr(gh, "get", get)
    result = m.GitHub.compare(gh, "a/b", "a" * 40, "b" * 40, all_commits=True)
    assert result["commits"] == list(range(101))


def test_existing_auto_merge_is_not_reenabled(world: World) -> None:
    args = world.args(publish=True, auto_merge=True)
    m._publish(args, world, m._make_plan(args, world))
    m._publish(args, world, m._make_plan(args, world))
    enables = [
        payload
        for _, endpoint, payload in world.calls
        if endpoint == "graphql" and "enablePullRequestAutoMerge" in payload["query"]
    ]
    assert len(enables) == 1


def test_remote_lock_cannot_change_other_vendor_entries(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = world.args(publish=True)
    plan = m._make_plan(args, world)
    pr = m._publish(args, world, plan)
    original = m._lock_document_at

    def tampered(gh: m.GitHub, repo: str, sha: str) -> dict:
        data = original(gh, repo, sha)
        if sha == pr["head"]["sha"]:
            data["vendors"]["unexpected"] = {}
        return data

    monkeypatch.setattr(m, "_lock_document_at", tampered)
    with pytest.raises(ValueError, match="branch/URL-only"):
        m._verify_remote_commit(world, plan, m._TRTLLM, pr["head"]["sha"])


def test_unmerged_parent_cannot_hide_unrelated_commits(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = world.args(publish=True)
    plan = m._make_plan(args, world)
    pr = m._publish(args, world, plan)
    original = world.compare

    def unrelated(repo: str, base: str, head: str, **kwargs: object) -> dict:
        result = original(repo, base, head, **kwargs)
        if repo == m._TRTLLM:
            result["merge_base_commit"]["sha"] = world.base
        return result

    monkeypatch.setattr(world, "compare", unrelated)
    with pytest.raises(ValueError, match="not on TRT-LLM main"):
        m._verify_remote_commit(world, plan, m._TRTLLM, pr["head"]["sha"])


def test_wait_checks_final_commit(world: World) -> None:
    world.state = "CLEAN"
    args = world.args(publish=True, auto_merge=True, wait=True)
    m._publish(args, world, m._make_plan(args, world))
    assert world.prs[99]["merged"]


def test_wait_timeout_preserves_enabled_auto_merge(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0, 2])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks))
    args = world.args(publish=True, auto_merge=True, wait=True, timeout=1)
    with pytest.raises(TimeoutError, match="still pending"):
        m._publish(args, world, m._make_plan(args, world))
    assert world.prs[99]["auto_merge"]


def test_source_fetch_uses_reviewed_sha_when_no_local_checkout(world: World) -> None:
    args = world.args(publish=True, flashinfer_repo=None)
    m._publish(args, world, m._make_plan(args, world))
    assert any(args[0] == "fetch" and args[-1] == world.reviewed for args in world.git_calls)


def test_incomplete_local_worktree_is_preserved(world: World) -> None:
    args = world.args(publish=True)
    args.worktree.mkdir()
    marker = args.worktree / "important.txt"
    marker.write_text("preserve\n")
    with pytest.raises((RuntimeError, AssertionError)):
        m._publish(args, world, m._make_plan(args, world))
    assert marker.read_text() == "preserve\n"


def test_unpaired_promotion_publishes_and_resumes(world: World) -> None:
    args = world.args(
        publish=True, upstream_pr=[], unpaired_reason="No upstream PR for the DSL API migration."
    )
    plan = m._make_plan(args, world)
    change = plan.record["changes"][0]
    assert change["upstream_pr"] is None
    assert change["upstream_head"] is None
    assert change["refresh_policy"] == "retain"
    assert change["unpaired_reason"] == args.unpaired_reason
    assert change["commits"] == [world.first, world.reviewed]
    m._publish(args, world, plan)
    m._publish(args, world, m._make_plan(args, world))
    assert len(world.prs) == 1
    assert "No paired FlashInfer PR" in world.prs[99]["body"]
    assert not any(f"repos/{m._FLASHINFER}/pulls/" in call[1] for call in world.calls)
    changed = copy.deepcopy(plan.record)
    changed["changes"][0]["refresh_policy"] = "drop"
    with pytest.raises(ValueError, match="reason/policy"):
        m._adopt_record(plan, m._record_text(changed))


@pytest.mark.parametrize(
    "options",
    [
        {"upstream_pr": [], "unpaired_reason": None},
        {"upstream_pr": [], "unpaired_reason": "  "},
        {"upstream_pr": ["4829"], "unpaired_reason": "No pairing"},
        {"upstream_pr": [], "unpaired_reason": "No pairing", "map_upstream": ["abcdef0=4829"]},
    ],
)
def test_unpaired_options_require_explicit_unambiguous_reason(world: World, options: dict) -> None:
    with pytest.raises(ValueError, match="unpaired-reason"):
        m._make_plan(world.args(**options), world)
    assert not world.calls


@pytest.mark.parametrize("same_fork", [False, True])
def test_canonical_branch_name_on_developer_fork(world: World, same_fork: bool) -> None:
    lock = world.consumer / m._LOCK
    text = lock.read_text().replace("branch: tmp-fix", "branch: trtllm-prims-ts-dev")
    if same_fork:
        text = text.replace("github.com/dev/", "github.com/maintainer/")
    lock.write_text(text)
    _git(world.consumer, "add", m._LOCK)
    _git(world.consumer, "commit", "--amend", "--no-edit", "-s")
    world.main = world.merged = _git(world.consumer, "rev-parse", "HEAD")
    args = world.args(publish=True)
    if same_fork:
        with pytest.raises(ValueError, match="temporary branch"):
            m._make_plan(args, world)
        return
    plan = m._make_plan(args, world)
    m._publish(args, world, plan)
    actual = m.vendor._load_lock(args.worktree / m._LOCK).vendors[m._VENDOR].to_mapping()
    assert actual == {**plan.reviewed.to_mapping(), "url": plan.previous.url}


def test_unpaired_cli_dry_run(world: World, capsys: pytest.CaptureFixture) -> None:
    assert (
        m.main(
            [
                "promote",
                "--trtllm-pr",
                "17",
                "--canonical-repo",
                "maintainer/flashinfer",
                "--repo",
                str(world.consumer),
                "--unpaired-reason",
                "No paired upstream PR.",
            ]
        )
        == 0
    )
    assert '"refresh_policy": "retain"' in capsys.readouterr().out
    assert not any(args[0] == "push" for args in world.git_calls)


def test_previous_developer_fork_cannot_become_canonical(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = m._lock_at

    def lock_at(gh: m.GitHub, repo: str, sha: str) -> m.vendor.Vendor:
        entry = original(gh, repo, sha)
        if sha == world.base:
            return dataclasses.replace(entry, url="https://github.com/another-dev/flashinfer.git")
        return entry

    monkeypatch.setattr(m, "_lock_at", lock_at)
    with pytest.raises(ValueError, match="explicit canonical"):
        m._make_plan(world.args(), world)
    assert not any(call[0] == "push" for call in world.git_calls)


@pytest.mark.parametrize("branch", ["tmp-fix", "trtllm-prims-ts-dev-20260910"])
def test_canonical_branch_must_be_explicit_and_match(world: World, branch: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        m._make_plan(world.args(canonical_branch=branch), world)


def test_completed_promotion_ignores_later_live_state(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    world.state = "CLEAN"
    args = world.args(publish=True, auto_merge=True)
    plan = m._make_plan(args, world)
    m._publish(args, world, plan)
    merged = world.main
    assert f"Squash commit: {merged}" in capsys.readouterr().out
    _git(world.consumer, "switch", "--detach", world.main)
    _write(world.consumer, "later.txt", "later main commit\n")
    world.main = _commit(world.consumer, "later main update")
    world.overrides[f"repos/{m._TRTLLM}"] = {"allow_auto_merge": False}
    world.overrides["repos/maintainer/flashinfer/git/ref/heads/trtllm-prims-ts-dev"] = {
        "object": {"sha": "b" * 40}
    }
    world.overrides["repos/dev/flashinfer"] = {}
    world.overrides[f"repos/{m._FLASHINFER}/pulls/4829"] = {"state": "closed", "merged": False}
    original_compare = world.compare

    def compare(repo: str, base: str, head: str, **kwargs: object) -> dict:
        assert repo == m._TRTLLM, "Completed retry must not query old source history"
        return original_compare(repo, base, head, **kwargs)

    monkeypatch.setattr(world, "compare", compare)
    world.calls.clear()
    world.git_calls.clear()
    args.flashinfer_repo = world.root / "missing-source"
    resumed = m._make_plan(args, world)
    assert resumed.completed_pr["merge_commit_sha"] == merged
    assert resumed.record == plan.record
    m._publish(args, world, resumed)
    assert all(method == "GET" for method, _, _ in world.calls)
    assert not any(call[0] in ("fetch", "push", "commit", "worktree") for call in world.git_calls)
    assert len(world.comments) == 1


@pytest.mark.parametrize("phase", ["worktree_created", "before_commit", "after_commit"])
@pytest.mark.parametrize("main_advances", [False, True])
def test_interrupted_preparation_resumes_without_extra_commits(
    world: World, monkeypatch: pytest.MonkeyPatch, phase: str, main_advances: bool
) -> None:
    def interrupted(repo: Path, *args: str, text: str | None = None) -> str:
        if phase == "before_commit" and args[0] == "commit":
            raise RuntimeError("simulated interruption")
        result = world.routed_git(repo, *args, text=text)
        if (phase == "worktree_created" and args[:2] == ("worktree", "add")) or (
            phase == "after_commit" and args[0] == "commit"
        ):
            raise RuntimeError("simulated interruption")
        return result

    args = world.args(publish=True)
    monkeypatch.setattr(m, "_git", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        m._publish(args, world, m._make_plan(args, world))
    assert not any(call[0] == "push" for call in world.git_calls)
    monkeypatch.setattr(m, "_git", world.routed_git)
    if main_advances:
        _write(world.consumer, "later.txt", "later main commit\n")
        world.main = _commit(world.consumer, "later main update")
    pr = m._publish(args, world, m._make_plan(args, world))
    head = pr["head"]["sha"]
    assert _git(world.consumer, "rev-list", "--count", f"{world.merged}..{head}") == "1"
    assert sum(call[:2] == ("worktree", "add") for call in world.git_calls) == 1
    assert len(world.comments) == 1


@pytest.mark.parametrize("edit", ["tracked", "untracked", "index_only", "index_only_other"])
def test_incomplete_recovery_preserves_unexpected_edits(
    world: World, monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    def interrupted(repo: Path, *args: str, text: str | None = None) -> str:
        if args[0] == "commit":
            raise RuntimeError("simulated interruption")
        return world.routed_git(repo, *args, text=text)

    args = world.args(publish=True)
    monkeypatch.setattr(m, "_git", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        m._publish(args, world, m._make_plan(args, world))
    monkeypatch.setattr(m, "_git", world.routed_git)
    if edit.startswith("index_only"):
        file = m._LOCK if edit == "index_only" else "vendored/prims_ts/kernel.py"
        target = args.worktree / file
        original = target.read_text()
        target.write_text(
            original.replace(world.reviewed, world.previous)
            if edit == "index_only"
            else "preserve this staged edit\n"
        )
        _git(args.worktree, "add", file)
        target.write_text(original)
        preserved = _git(args.worktree, "show", f":{file}")
    else:
        file = "vendored/prims_ts/kernel.py" if edit == "tracked" else "important.txt"
        _write(args.worktree, file, "preserve this edit\n")
    with pytest.raises(ValueError):
        m._publish(args, world, m._make_plan(args, world))
    if edit.startswith("index_only"):
        assert _git(args.worktree, "show", f":{file}") == preserved
    else:
        assert (args.worktree / file).read_text() == "preserve this edit\n"
    assert not any(call[0] == "push" for call in world.git_calls)
    assert _git(args.worktree, "rev-parse", "HEAD") == world.merged


def test_ci_skip_is_exact_once_and_precedes_auto_merge(world: World) -> None:
    args = world.args(publish=True, auto_merge=True)
    m._publish(args, world, m._make_plan(args, world))
    m._publish(args, world, m._make_plan(args, world))
    assert [comment["body"] for comment in world.comments] == [m._CI_SKIP_COMMAND]
    create_index = next(
        i
        for i, (verb, route, _) in enumerate(world.calls)
        if verb == "POST" and route.endswith("/pulls")
    )
    comment_index = next(
        i
        for i, (verb, route, _) in enumerate(world.calls)
        if verb == "POST" and route.endswith("/comments")
    )
    auto_index = next(
        i
        for i, (_, route, payload) in enumerate(world.calls)
        if route == "graphql" and "enablePullRequestAutoMerge" in payload["query"]
    )
    assert create_index < comment_index < auto_index


@pytest.mark.parametrize("after", [False, True])
def test_ci_comment_failure_is_resumable_without_duplicates(world: World, after: bool) -> None:
    world.fail_comment_before = not after
    world.fail_comment_after = after
    args = world.args(publish=True, auto_merge=True)
    with pytest.raises(RuntimeError, match="comment failure"):
        m._publish(args, world, m._make_plan(args, world))
    assert len(world.prs) == 1
    assert not world.prs[99]["auto_merge"]
    world.fail_comment_before = world.fail_comment_after = False
    m._publish(args, world, m._make_plan(args, world))
    assert len(world.comments) == 1
    assert world.prs[99]["auto_merge"]


@pytest.mark.parametrize("same_actor", [False, True])
def test_ci_comment_search_paginates_and_checks_author(world: World, same_actor: bool) -> None:
    world.comments = [{"body": "unrelated", "user": {"login": "maintainer"}}] * 100
    world.comments.append(
        {
            "body": m._CI_SKIP_COMMAND,
            "user": {"login": "maintainer" if same_actor else "someone-else"},
        }
    )
    args = world.args(publish=True)
    m._publish(args, world, m._make_plan(args, world))
    assert len(world.comments) == (101 if same_actor else 102)
    assert any("comments?per_page=100&page=2" in route for _, route, _ in world.calls)


@pytest.mark.parametrize("change", ["closed", "head"])
def test_ci_request_rejects_changed_pr(world: World, change: str) -> None:
    world.fail_comment_before = True
    args = world.args(publish=True)
    plan = m._make_plan(args, world)
    with pytest.raises(RuntimeError, match="comment failure"):
        m._publish(args, world, plan)
    verified = copy.deepcopy(world.prs[99])
    if change == "closed":
        world.prs[99]["state"] = "closed"
    else:
        world.prs[99]["head"]["sha"] = "b" * 40
    world.calls.clear()
    with pytest.raises(ValueError, match="closed or changed"):
        m._ensure_ci_skip(world, plan, verified)
    assert all(method == "GET" for method, _, _ in world.calls)
    assert not world.comments


def test_generated_body_passes_repository_checklist(world: World) -> None:
    checker = runpy.run_path(str(_SCRIPTS.parent / ".github/scripts/pr_checklist_check.py"))
    plan = m._make_plan(world.args(), world)
    body = m._pr_body(plan, f"{plan.title}\n\n{m._record_text(plan.record)}")
    assert checker["find_unresolved_tasks"](body) == []
    assert checker["check_pr_checklist_section"](body) == (True, "")
    assert "Required owner reviews and checks completed" not in body
    assert m._CI_SKIP_COMMAND in body
    assert "Automatic GitHub checks" in body


def test_mutating_git_commands_stream_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert kwargs["capture_output"] is False
        assert kwargs["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"
        return subprocess.CompletedProcess(command, 0, None, None)

    monkeypatch.setattr(m.subprocess, "run", run)
    assert m._git(Path("/test/repository"), "worktree", "add", "new") == ""
    assert m._git(Path("/test/repository"), "commit", "-s", "-F", "-", text="test") == ""
