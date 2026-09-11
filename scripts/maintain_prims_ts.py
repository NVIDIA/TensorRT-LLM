#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Promote reviewed PrimTS source revisions; dry-run unless --publish is supplied.

Only Git, gh, and PyYAML are required. GitHub reads and temporary source fetches
are allowed in dry-run; remote writes and consumer worktrees require --publish.
Provenance lives in signed-off commits, not in a separate tracking file.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import yaml

if __package__:
    from . import vendor_sources as vendor
else:
    import vendor_sources as vendor

_TRTLLM = "NVIDIA/TensorRT-LLM"
_FLASHINFER = "flashinfer-ai/flashinfer"
_VENDOR = "flashinfer-prims-ts"
_LOCK = "3rdparty/vendor_sources.lock.yaml"
_BEGIN = "----- BEGIN PRIMTS PROMOTION V1 -----"
_END = "----- END PRIMTS PROMOTION V1 -----"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_CANONICAL = re.compile(r"trtllm-prims-ts-dev(?:-[0-9]{8})?\Z")
_CI_SKIP_COMMAND = '/bot skip --comment "skip CI since no code change"'


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    text: str | None = None,
    timeout: int = 300,
    stream: bool = False,
) -> str:
    # Do not use vendor_sources._run_git here: its synthetic identity is for
    # patch generation. Real promotion commits must use the maintainer's identity.
    env = os.environ.copy()
    for key in vendor._GIT_LOCAL_ENVIRONMENT_VARIABLES:
        env.pop(key, None)
    env.setdefault("GH_CONFIG_DIR", str(Path.home() / ".config/gh"))
    env.update(GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1", GIT_NO_REPLACE_OBJECTS="1")
    if command[0] == "git":
        # Lock-only worktrees need source text, not unrelated LFS assets.
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=text,
        text=True,
        capture_output=not stream,
        timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "See streamed output above.").strip()
        raise RuntimeError(f"{command[0]} failed: {detail}")
    return (result.stdout or "").strip()


def _git(repo: Path, *args: str, text: str | None = None) -> str:
    return _run(
        ["git", "-C", str(repo), *args],
        text=text,
        timeout=1800 if args[:2] == ("worktree", "add") else 300,
        stream=args[:2] == ("worktree", "add") or args[0] in ("fetch", "commit", "push"),
    )


class GitHub:
    """Small gh-backed client; respects GH_CONFIG_DIR without handling tokens."""

    def api(
        self, endpoint: str, *, method: str = "GET", payload: dict | None = None
    ) -> dict | list:
        command = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
        if payload is not None:
            command += ["--input", "-"]
        result = json.loads(_run(command, text=None if payload is None else json.dumps(payload)))
        if not isinstance(result, (dict, list)):
            raise ValueError(f"Unexpected GitHub response for {endpoint}")
        return result

    def get(self, endpoint: str) -> dict:
        result = self.api(endpoint)
        if not isinstance(result, dict):
            raise ValueError(f"Expected an object from {endpoint}")
        return result

    def optional(self, endpoint: str) -> dict | None:
        try:
            return self.get(endpoint)
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise
            return None

    def compare(self, repo: str, base: str, head: str, *, all_commits: bool = False) -> dict:
        first = self.get(f"repos/{repo}/compare/{base}...{head}?per_page=100&page=1")
        if all_commits:
            count = int(first["total_commits"])
            for page in range(2, (count + 99) // 100 + 1):
                more = self.get(f"repos/{repo}/compare/{base}...{head}?per_page=100&page={page}")
                first["commits"].extend(more["commits"])
            if len(first["commits"]) != count:
                raise ValueError("Incomplete GitHub comparison; refusing partial provenance.")
        return first


def _sha(value: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Expected a full immutable commit SHA, got {value!r}.")
    return value


def _repo_name(url: str) -> str:
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?", url
    )
    if not match:
        raise ValueError(f"Expected a credential-free GitHub repository URL, got {url!r}.")
    return match[1]


def _pr_number(value: str, repo: str) -> int:
    for prefix in (f"https://github.com/{repo}/pull/", f"{repo}#"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Expected a PR number or PR URL in {repo}: {value!r}.")
    return int(value)


def _lock_document_at(gh: GitHub, repo: str, commit: str) -> dict:
    response = gh.get(f"repos/{repo}/contents/{_LOCK}?ref={_sha(commit)}")
    if response.get("encoding") != "base64":
        raise ValueError("GitHub did not return the complete vendor lock.")
    data = yaml.load(base64.b64decode(response["content"]), Loader=vendor._UniqueKeyLoader)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported vendor lock schema.")
    return data


def _lock_at(gh: GitHub, repo: str, commit: str) -> vendor.Vendor:
    data = _lock_document_at(gh, repo, commit)
    return vendor._validate_vendor(_VENDOR, data["vendors"][_VENDOR])


def _main_sha(gh: GitHub) -> str:
    return _sha(gh.get(f"repos/{_TRTLLM}/git/ref/heads/main")["object"]["sha"])


def _check_fork(gh: GitHub, repo: str, upstream: str) -> None:
    info = gh.get(f"repos/{repo}")
    if (
        repo.lower() == upstream.lower()
        or info.get("source", {}).get("full_name", "").lower() != upstream.lower()
    ):
        raise ValueError(f"{repo} must be a personal fork of {upstream}, not upstream itself.")


def _record_from_message(message: str) -> dict:
    if message.count(_BEGIN) != 1 or message.count(_END) != 1:
        raise ValueError("Missing or ambiguous PrimTS promotion metadata in commit message.")
    record = json.loads(message.split(_BEGIN, 1)[1].split(_END, 1)[0])
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("Unsupported PrimTS promotion metadata version.")
    return record


def _record_text(record: dict) -> str:
    return f"{_BEGIN}\n{json.dumps(record, indent=2, sort_keys=True)}\n{_END}"


def _map_commits(commits: list[dict], prs: dict[int, dict], mappings: list[str]) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for mapping in mappings:
        prefix, separator, reference = mapping.partition("=")
        matches = [item["sha"] for item in commits if item["sha"].startswith(prefix)]
        if not separator or not re.fullmatch(r"[0-9a-f]{7,40}", prefix) or len(matches) != 1:
            raise ValueError(f"Ambiguous or invalid --map-upstream {mapping!r}.")
        number = _pr_number(reference, _FLASHINFER)
        if number not in prs or matches[0] in assignments:
            raise ValueError(f"Duplicate mapping or PR missing from --upstream-pr: {mapping!r}.")
        assignments[matches[0]] = number
    for item in commits:
        sha = item["sha"]
        if sha in assignments:
            continue
        if len(prs) == 1:
            assignments[sha] = next(iter(prs))
        elif sys.stdin.isatty():
            title = item["commit"]["message"].splitlines()[0]
            answer = input(f"Upstream PR for {sha[:12]} ({title}), choose {sorted(prs)}: ")
            number = _pr_number(answer, _FLASHINFER)
            if number not in prs:
                raise ValueError("Answer must identify a supplied upstream PR.")
            assignments[sha] = number
        else:
            raise ValueError(f"Provide --map-upstream {sha}=PR for every unmapped commit.")
    if set(assignments.values()) != set(prs):
        raise ValueError("Every supplied upstream PR must map to at least one new source commit.")
    return assignments


@dataclass
class Promotion:
    """An immutable source revision and the expected consumer/canonical states."""

    number: int
    main_sha: str
    previous: vendor.Vendor
    reviewed: vendor.Vendor
    canonical_repo: str
    fork: str
    record: dict
    completed_pr: dict | None = None

    @property
    def branch(self) -> str:
        return f"chore/prims-ts-promote-{self.number}-{self.reviewed.commit[:12]}"

    @property
    def title(self) -> str:
        return f"[None][chore] promote PrimTS source from TRT-LLM #{self.number}"

    @property
    def promoted(self) -> dict:
        result = self.reviewed.to_mapping()
        result.update(url=self.previous.url, branch=self.previous.branch)
        return result


def _check_current(gh: GitHub, plan: Promotion) -> None:
    current = _lock_at(gh, _TRTLLM, _main_sha(gh)).to_mapping()
    if current not in (plan.reviewed.to_mapping(), plan.promoted):
        raise ValueError("TRT-LLM main's vendor lock has been superseded; do not promote this PR.")
    ref = gh.get(f"repos/{plan.canonical_repo}/git/ref/heads/{plan.previous.branch}")
    if ref["object"]["sha"] not in (plan.previous.commit, plan.reviewed.commit):
        raise ValueError(
            "Canonical branch moved unexpectedly; coordinate with the other maintainer."
        )


def _make_plan(args: argparse.Namespace, gh: GitHub) -> Promotion:
    if args.unpaired_reason is not None:
        if not args.unpaired_reason.strip() or args.upstream_pr or args.map_upstream:
            raise ValueError(
                "--unpaired-reason requires a nonempty reason and cannot be combined "
                "with --upstream-pr or --map-upstream."
            )
    elif not args.upstream_pr:
        raise ValueError("Provide --upstream-pr or an explicit --unpaired-reason.")
    number = _pr_number(args.trtllm_pr, _TRTLLM)
    pr = gh.get(f"repos/{_TRTLLM}/pulls/{number}")
    if not pr.get("merged") or pr["base"]["ref"] != "main":
        raise ValueError("--trtllm-pr must identify a PR already merged into TRT-LLM main.")
    merge_sha = _sha(pr["merge_commit_sha"])
    merge = gh.get(f"repos/{_TRTLLM}/commits/{merge_sha}")
    previous = _lock_at(gh, _TRTLLM, merge["parents"][0]["sha"])
    reviewed = _lock_at(gh, _TRTLLM, merge_sha)
    canonical_repo = _repo_name(previous.url)
    source_repo = _repo_name(reviewed.url)
    expected_repo = _repo_name(f"https://github.com/{args.canonical_repo}")
    if not _CANONICAL.fullmatch(args.canonical_branch):
        raise ValueError("--canonical-branch must name a supported canonical branch.")
    if canonical_repo.lower() != expected_repo.lower() or previous.branch != args.canonical_branch:
        raise ValueError(
            "Previous lock does not match the explicit canonical repository/branch; "
            "finish the preceding promotion first."
        )
    if (
        previous.commit == reviewed.commit
        or not reviewed.branch
        or (canonical_repo.lower() == source_repo.lower() and _CANONICAL.fullmatch(reviewed.branch))
    ):
        raise ValueError("The PR must introduce a new source SHA on a temporary branch.")
    fork = args.fork or _repo_name(_git(args.repo, "remote", "get-url", "fork"))
    record = {
        "version": 1,
        "kind": "promotion",
        "vendor": _VENDOR,
        "trtllm_pr": f"https://github.com/{_TRTLLM}/pull/{number}",
        "trtllm_merge": merge_sha,
        "canonical_url": previous.url,
        "canonical_branch": previous.branch,
        "previous": previous.commit,
        "promoted": reviewed.commit,
        "source_url": reviewed.url,
        "source_branch": reviewed.branch,
    }
    plan = Promotion(number, _main_sha(gh), previous, reviewed, canonical_repo, fork, record)
    # A completed operation is historical: later promotions, deleted temporary
    # forks, or changed upstream PR state must not make it run again or fail.
    existing = _find_followup(gh, plan)
    if existing:
        existing = gh.get(f"repos/{_TRTLLM}/pulls/{existing['number']}")
        if existing.get("merged"):
            _verify_remote_commit(gh, plan, _TRTLLM, existing["merge_commit_sha"])
            plan.completed_pr = existing
            return plan
    _check_fork(gh, canonical_repo, _FLASHINFER)
    _check_fork(gh, source_repo, _FLASHINFER)
    _check_fork(gh, fork, _TRTLLM)
    comparison = gh.compare(canonical_repo, previous.commit, reviewed.commit, all_commits=True)
    if comparison["merge_base_commit"]["sha"] != previous.commit:
        raise ValueError(
            "Reviewed source is not a fast-forward of the previous canonical revision."
        )
    commits = comparison["commits"]
    if not commits or any(len(item["parents"]) != 1 for item in commits):
        raise ValueError(
            "Promote requires a nonempty, linear source commit range; rebase merges first."
        )
    prs = {}
    for reference in args.upstream_pr:
        upstream_number = _pr_number(reference, _FLASHINFER)
        if upstream_number in prs:
            raise ValueError("Duplicate --upstream-pr.")
        upstream = gh.get(f"repos/{_FLASHINFER}/pulls/{upstream_number}")
        if upstream["base"]["ref"] != "main" or (
            upstream["state"] == "closed" and not upstream.get("merged")
        ):
            raise ValueError("Upstream PR must target main and be open (draft allowed) or merged.")
        prs[upstream_number] = upstream
    assignments = _map_commits(commits, prs, args.map_upstream) if prs else {}
    upstream_main = _sha(gh.get(f"repos/{_FLASHINFER}/git/ref/heads/main")["object"]["sha"])
    upstream_base = gh.compare(canonical_repo, previous.commit, upstream_main)["merge_base_commit"][
        "sha"
    ]
    groups = []
    if args.unpaired_reason is not None:
        groups.append(
            {
                "commits": [item["sha"] for item in commits],
                "upstream_pr": None,
                "upstream_base": None,
                "upstream_head": None,
                "upstream_merge": None,
                "unpaired_reason": args.unpaired_reason.strip(),
                "refresh_policy": "retain",
            }
        )
    for upstream_number, upstream in prs.items():
        groups.append(
            {
                "commits": [
                    item["sha"] for item in commits if assignments[item["sha"]] == upstream_number
                ],
                "upstream_pr": f"https://github.com/{_FLASHINFER}/pull/{upstream_number}",
                "upstream_base": _sha(upstream["base"]["sha"]),
                "upstream_head": _sha(upstream["head"]["sha"]),
                "upstream_merge": upstream.get("merge_commit_sha")
                if upstream.get("merged")
                else None,
            }
        )
    record.update(
        {
            "upstream_base": _sha(upstream_base),
            "upstream_observed_main": upstream_main,
            "changes": groups,
        }
    )
    _check_current(gh, plan)
    return plan


def _find_followup(gh: GitHub, plan: Promotion) -> dict | None:
    query = urlencode(
        {"state": "all", "base": "main", "head": f"{plan.fork.split('/')[0]}:{plan.branch}"}
    )
    prs = gh.api(f"repos/{_TRTLLM}/pulls?{query}")
    if not isinstance(prs, list) or len(prs) > 1:
        raise ValueError("Ambiguous existing promotion PRs.")
    if prs and prs[0]["state"] == "closed" and not prs[0].get("merged_at"):
        raise ValueError("The promotion PR was closed without merging; inspect it before retrying.")
    return prs[0] if prs else None


def _adopt_record(plan: Promotion, message: str) -> None:
    """Keep the original upstream snapshot when resuming an already-published run."""
    existing = _record_from_message(message)
    stable_keys = set(plan.record) - {"upstream_base", "upstream_observed_main", "changes"}
    if any(existing.get(key) != plan.record[key] for key in stable_keys):
        raise ValueError("Existing promotion metadata does not describe this source update.")
    _validate_snapshot(existing)
    if "changes" not in plan.record:
        # Only the read-only, already-merged path restores an entire snapshot.
        plan.record = existing
        return
    expected = {
        group["upstream_pr"]: (
            group["commits"],
            group.get("unpaired_reason"),
            group.get("refresh_policy"),
        )
        for group in plan.record["changes"]
    }
    actual = {
        group["upstream_pr"]: (
            group["commits"],
            group.get("unpaired_reason"),
            group.get("refresh_policy"),
        )
        for group in existing["changes"]
    }
    if actual != expected:
        raise ValueError("Existing promotion has different upstream PR/commit mappings.")
    plan.record = existing


def _validate_snapshot(record: dict) -> None:
    _sha(record["upstream_base"])
    _sha(record["upstream_observed_main"])
    commits = []
    if not isinstance(record["changes"], list) or not record["changes"]:
        raise ValueError("Promotion metadata must contain source changes.")
    for group in record["changes"]:
        if not isinstance(group["commits"], list) or not group["commits"]:
            raise ValueError("Promotion change group has no commits.")
        commits.extend(_sha(sha) for sha in group["commits"])
        if group["upstream_pr"] is None:
            if (
                group.get("refresh_policy") != "retain"
                or not group.get("unpaired_reason", "").strip()
            ):
                raise ValueError(
                    "Unpaired promotion changes must retain an explicit reason/policy."
                )
        else:
            _pr_number(group["upstream_pr"], _FLASHINFER)
            _sha(group["upstream_base"])
            _sha(group["upstream_head"])
            if group["upstream_merge"] is not None:
                _sha(group["upstream_merge"])
    if len(set(commits)) != len(commits) or record["promoted"] not in commits:
        raise ValueError("Promotion metadata has duplicate or missing source commits.")


def _verify_remote_commit(gh: GitHub, plan: Promotion, repo: str, sha: str) -> str:
    commit = gh.get(f"repos/{repo}/commits/{_sha(sha)}")
    if len(commit["parents"]) != 1 or [item["filename"] for item in commit["files"]] != [_LOCK]:
        raise ValueError("Promotion commit must change only the vendor lock and have one parent.")
    parent = commit["parents"][0]["sha"]
    comparison = gh.compare(_TRTLLM, parent, _main_sha(gh))
    if comparison["merge_base_commit"]["sha"] != parent:
        raise ValueError(
            "Promotion parent is not on TRT-LLM main; unrelated commits may be present."
        )
    before = _lock_document_at(gh, repo, parent)
    after = _lock_document_at(gh, repo, sha)
    _check_document_change(before, after, plan)
    message = commit["commit"]["message"]
    _adopt_record(plan, message)
    if not re.search(r"^Signed-off-by: .+ <.+>$", message, re.MULTILINE):
        raise ValueError("Promotion commit has no DCO sign-off.")
    return message


def _check_document_change(before: dict, after: dict, plan: Promotion) -> None:
    if before["vendors"][_VENDOR] != plan.reviewed.to_mapping():
        raise ValueError("Promotion parent does not contain the reviewed temporary pin.")
    expected = {**before, "vendors": {**before["vendors"], _VENDOR: plan.promoted}}
    if expected != after:
        raise ValueError("Promotion commit is not the expected branch/URL-only lock change.")


@contextlib.contextmanager
def _source_repo(args: argparse.Namespace, plan: Promotion) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="prims-ts-promote-source-") as temporary:
        repo = args.flashinfer_repo or Path(temporary)
        if args.flashinfer_repo is None:
            print(f"Fetching reviewed FlashInfer source {plan.reviewed.commit}...", flush=True)
            _git(repo, "init", "--quiet")
            _git(repo, "fetch", "--quiet", "--no-tags", plan.reviewed.url, plan.reviewed.commit)
        _git(repo, "cat-file", "-e", f"{plan.reviewed.commit}^{{commit}}")
        _git(repo, "merge-base", "--is-ancestor", plan.previous.commit, plan.reviewed.commit)
        yield repo


def _local_lock_document(worktree: Path, revision: str | None = None) -> dict:
    content = (
        (worktree / _LOCK).read_text()
        if revision is None
        else _git(worktree, "show", f"{revision}:{_LOCK}")
    )
    return yaml.load(content, Loader=vendor._UniqueKeyLoader)


def _pending_paths(worktree: Path) -> set[str]:
    # A file restored in the working tree can still have an unrelated staged edit.
    return set(_git(worktree, "diff", "--name-only", "HEAD").splitlines()) | set(
        _git(worktree, "diff", "--cached", "--name-only", "HEAD").splitlines()
    )


def _check_lock_only(worktree: Path, before: dict, plan: Promotion) -> None:
    _check_document_change(before, _local_lock_document(worktree), plan)
    if _pending_paths(worktree) != {_LOCK}:
        raise ValueError("Promotion must change only the vendor lock.")
    if _git(worktree, "ls-files", "--others", "--exclude-standard"):
        raise ValueError("Unexpected untracked files in the promotion worktree.")


def _prepare_commit(args: argparse.Namespace, plan: Promotion, source: Path) -> tuple[Path, str]:
    print(f"Fetching TRT-LLM base {plan.main_sha}...", flush=True)
    _git(
        args.repo,
        "fetch",
        "--quiet",
        "--no-tags",
        f"https://github.com/{_TRTLLM}.git",
        plan.main_sha,
    )
    worktree = (
        args.worktree
        or args.repo.parent / f"prims-ts-promote-{plan.number}-{plan.reviewed.commit[:12]}"
    )
    worktree = worktree.resolve()
    if worktree.exists():
        if _git(worktree, "branch", "--show-current") != plan.branch or _git(
            worktree, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ) != _git(args.repo, "rev-parse", "--path-format=absolute", "--git-common-dir"):
            raise ValueError(
                f"Existing worktree is not this repository's promotion branch: {worktree}"
            )
        head = _git(worktree, "rev-parse", "HEAD")
        if _git(worktree, "merge-base", head, plan.main_sha) != head:
            if _git(worktree, "status", "--porcelain"):
                raise ValueError(
                    f"Completed promotion worktree contains unexpected edits: {worktree}"
                )
            _adopt_record(plan, _git(worktree, "log", "-1", "--format=%B"))
            parent = _git(worktree, "rev-parse", "HEAD^")
            _git(worktree, "merge-base", "--is-ancestor", parent, plan.main_sha)
            before = _local_lock_document(worktree, "HEAD^")
            _check_committed_lock(worktree, before, plan)
            lock = vendor._load_lock(worktree / _LOCK)
            vendor._verify_source(lock, lock.vendors[_VENDOR], source)
            print(f"Reusing completed promotion commit {head} in {worktree}.", flush=True)
            return worktree, head
        print(f"Resuming incomplete promotion in {worktree}...", flush=True)
    else:
        print(f"Creating promotion worktree {worktree} (LFS downloads disabled)...", flush=True)
        _git(args.repo, "worktree", "add", "-b", plan.branch, str(worktree), plan.main_sha)
    before = _local_lock_document(worktree, "HEAD")
    if before["vendors"][_VENDOR] != plan.reviewed.to_mapping():
        raise ValueError("Worktree base no longer contains the reviewed temporary pin.")
    # Recover only the two states this tool writes. Preserve arbitrary user or
    # hook edits, including staged-only changes, instead of overwriting them.
    for revision in (None, ""):
        pending = _local_lock_document(worktree, revision)
        if pending != before:
            _check_document_change(before, pending, plan)
    if _pending_paths(worktree) - {_LOCK}:
        raise ValueError(
            "Incomplete promotion has changes outside the vendor lock; inspect before retrying."
        )
    if _git(worktree, "ls-files", "--others", "--exclude-standard"):
        raise ValueError("Incomplete promotion has untracked files; inspect before retrying.")
    lock = vendor._load_lock(worktree / _LOCK)
    print("Verifying source materialization and the lock-only change...", flush=True)
    vendor._verify_source(lock, lock.vendors[_VENDOR], source)
    result = vendor.main(
        [
            "--lock",
            str(worktree / _LOCK),
            "pin",
            _VENDOR,
            "--url",
            plan.previous.url,
            "--branch",
            plan.previous.branch,
            "--commit",
            plan.reviewed.commit,
            "--repo",
            str(source),
        ]
    )
    if result:
        raise ValueError("Vendor pin failed; worktree preserved for inspection.")
    _check_lock_only(worktree, before, plan)
    _git(worktree, "add", "--", _LOCK)
    print("Creating signed-off promotion commit...", flush=True)
    _git(worktree, "commit", "-s", "-F", "-", text=f"{plan.title}\n\n{_record_text(plan.record)}\n")
    if _git(worktree, "status", "--porcelain"):
        raise ValueError(f"Commit hooks left changes in {worktree}; inspect and retry.")
    _check_committed_lock(worktree, before, plan)
    return worktree, _git(worktree, "rev-parse", "HEAD")


def _check_committed_lock(worktree: Path, before: dict, plan: Promotion) -> None:
    _check_document_change(before, _local_lock_document(worktree), plan)
    if len(_git(worktree, "rev-list", "--parents", "-n", "1", "HEAD").split()) != 2:
        raise ValueError("Promotion commit must have exactly one parent.")
    if _git(worktree, "diff", "--name-only", "HEAD^", "HEAD") != _LOCK:
        raise ValueError("Commit hooks added changes outside the vendor lock.")
    message = _git(worktree, "log", "-1", "--format=%B")
    if _record_from_message(message) != plan.record:
        raise ValueError("Commit hooks changed the promotion provenance.")
    if not re.search(r"^Signed-off-by: .+ <.+>$", message, re.MULTILINE):
        raise ValueError("Promotion commit has no DCO sign-off.")


def _check_auto_merge(gh: GitHub) -> None:
    settings = gh.get(f"repos/{_TRTLLM}")
    if not settings.get("allow_auto_merge") or not settings.get("allow_squash_merge"):
        raise ValueError("Repository must allow auto-merge and squash merging.")
    rules = gh.api(f"repos/{_TRTLLM}/rules/branches/main")
    if not isinstance(rules, list) or any(rule["type"] == "merge_queue" for rule in rules):
        raise ValueError(
            "Merge queues are unsupported: custom squash-message retention is required."
        )


def _disable_auto_merge(gh: GitHub, pr: dict) -> None:
    gh.api(
        "graphql",
        method="POST",
        payload={
            "query": (
                "mutation($id:ID!){disablePullRequestAutoMerge("
                "input:{pullRequestId:$id}){clientMutationId}}"
            ),
            "variables": {"id": pr["node_id"]},
        },
    )


def _report_merged(pr: dict) -> None:
    print(
        f"Promotion merged: {pr['html_url']}\n"
        f"Squash commit: {pr['merge_commit_sha']}\n"
        "Final lock-only change, provenance, and DCO sign-off verified.",
        flush=True,
    )


def _arm_auto_merge(gh: GitHub, plan: Promotion, pr: dict, message: str) -> None:
    _check_auto_merge(gh)
    _check_current(gh, plan)
    number, head = pr["number"], pr["head"]["sha"]
    state = gh.api(
        "graphql",
        method="POST",
        payload={
            "query": "query($id:ID!){node(id:$id){... on PullRequest{headRefOid mergeStateStatus}}}",
            "variables": {"id": pr["node_id"]},
        },
    )
    node = state["data"]["node"]
    if node["headRefOid"] != head:
        raise ValueError("Follow-up PR head changed; revalidate before enabling auto-merge.")
    title, _, body = message.partition("\n")
    saved = pr.get("auto_merge")
    if saved:
        if (
            saved.get("merge_method") == "squash"
            and saved.get("commit_title") == title
            and saved.get("commit_message", "").strip() == body.strip()
        ):
            print(f"Existing squash auto-merge message verified: {pr['html_url']}")
            return
        _disable_auto_merge(gh, pr)
        raise ValueError("Existing auto-merge message differs; disabled. Inspect before retrying.")
    # Query only necessary fields: gh pr merge's broader PR query can require
    # read:org even when the token already has permission to merge this PR.
    operation = (
        "mergePullRequest" if node["mergeStateStatus"] == "CLEAN" else "enablePullRequestAutoMerge"
    )
    input_type = (
        "MergePullRequestInput"
        if operation == "mergePullRequest"
        else "EnablePullRequestAutoMergeInput"
    )
    gh.api(
        "graphql",
        method="POST",
        payload={
            "query": f"mutation($input:{input_type}!){{{operation}(input:$input){{clientMutationId}}}}",
            "variables": {
                "input": {
                    "pullRequestId": pr["node_id"],
                    "expectedHeadOid": head,
                    "mergeMethod": "SQUASH",
                    "commitHeadline": title,
                    "commitBody": body.strip(),
                }
            },
        },
    )
    updated = gh.get(f"repos/{_TRTLLM}/pulls/{number}")
    if updated.get("merged"):
        _verify_remote_commit(gh, plan, _TRTLLM, updated["merge_commit_sha"])
        _report_merged(updated)
    else:
        saved = updated.get("auto_merge") or {}
        if (
            updated["head"]["sha"] != head
            or saved.get("merge_method") != "squash"
            or saved.get("commit_title") != title
            or saved.get("commit_message", "").strip() != body.strip()
        ):
            _disable_auto_merge(gh, pr)
            raise ValueError("Auto-merge message/head verification failed; auto-merge disabled.")
        print(f"Squash auto-merge enabled and message verified: {updated['html_url']}")


def _pr_body(plan: Promotion, message: str) -> str:
    unpaired = ""
    if any(group["upstream_pr"] is None for group in plan.record["changes"]):
        unpaired = (
            "No paired FlashInfer PR exists for this source update. Its provenance "
            "explicitly records it as unpaired, with `refresh_policy: retain`. A future "
            "refresh must carry it forward until an upstream pairing or equivalence "
            "is explicitly established.\n\n"
        )
    return (
        "@coderabbitai summary\n\n## Description\n\n"
        f"Promote the exact source revision reviewed in {plan.record['trtllm_pr']}.\n"
        "Only the PrimTS lock URL/branch changes. The source SHA, selected vendor files, "
        "compatibility patch, and digests are unchanged.\n\n"
        f"{unpaired}"
        "## Test Coverage\n\nSource materialization and lock-only diff verified by the promotion tool. "
        "No new kernel behavior. The tool posts "
        f"`{_CI_SKIP_COMMAND}` to request the bot's no-code-change CI path. "
        "Automatic GitHub checks and required reviews still apply.\n\n"
        "## PR Checklist\n\n"
        "- [x] No runtime/API/dependency changes.\n"
        "- [x] Immutable source pin and vendor content preserved.\n"
        "- [x] Please check this after reviewing the above items as appropriate for this PR.\n\n"
        "## Required squash commit message\n\n"
        "Preserve the following message, including the provenance block and DCO sign-off.\n\n"
        f"```text\n{message.strip()}\n```\n"
    )


def _ensure_ci_skip(gh: GitHub, plan: Promotion, pr: dict) -> dict:
    """Request the no-code-change bot path once across sequential retries."""
    current = gh.get(f"repos/{_TRTLLM}/pulls/{pr['number']}")
    if current.get("merged"):
        _verify_remote_commit(gh, plan, _TRTLLM, current["merge_commit_sha"])
        _report_merged(current)
        return current
    if current["state"] != "open" or current["head"]["sha"] != pr["head"]["sha"]:
        raise ValueError("Follow-up PR closed or changed before the CI request; inspect it.")
    _check_current(gh, plan)
    _verify_remote_commit(gh, plan, _TRTLLM, current["head"]["sha"])
    actor = gh.get("user")["login"]
    endpoint = f"repos/{_TRTLLM}/issues/{pr['number']}/comments"
    page = 1
    while True:
        comments = gh.api(f"{endpoint}?per_page=100&page={page}")
        if not isinstance(comments, list):
            raise ValueError("Unexpected issue-comment response; refusing a duplicate CI request.")
        if any(
            comment.get("body", "").strip() == _CI_SKIP_COMMAND
            and comment.get("user", {}).get("login") == actor
            for comment in comments
        ):
            print(f"Existing CI skip request found: {current['html_url']}", flush=True)
            return current
        if len(comments) < 100:
            break
        page += 1
    comment = gh.api(endpoint, method="POST", payload={"body": _CI_SKIP_COMMAND})
    print(f"Posted CI skip request: {comment['html_url']}", flush=True)
    return current


def _publish(args: argparse.Namespace, gh: GitHub, plan: Promotion) -> dict:
    if plan.completed_pr is not None:
        _report_merged(plan.completed_pr)
        return plan.completed_pr
    existing = _find_followup(gh, plan)
    remote = gh.optional(f"repos/{plan.fork}/git/ref/heads/{plan.branch}")
    if existing:
        existing = gh.get(f"repos/{_TRTLLM}/pulls/{existing['number']}")
        sha = existing["merge_commit_sha"] if existing.get("merged") else existing["head"]["sha"]
        message = _verify_remote_commit(gh, plan, _TRTLLM, sha)
        if existing.get("merged"):
            _report_merged(existing)
            return existing
        if remote is None or remote["object"]["sha"] != sha:
            raise ValueError("Follow-up branch is missing or changed; inspect before resuming.")
    elif remote:
        sha = remote["object"]["sha"]
        message = _verify_remote_commit(gh, plan, plan.fork, sha)
    else:
        message = ""
    if args.auto_merge:
        _check_auto_merge(gh)
    _check_current(gh, plan)
    with _source_repo(args, plan) as source:
        if remote is None:
            worktree, sha = _prepare_commit(args, plan, source)
            message = _git(worktree, "log", "-1", "--format=%B")
        _check_current(gh, plan)
        # No force push: Git also enforces the fast-forward constraint remotely.
        print(f"Ensuring canonical branch points to {plan.reviewed.commit}...", flush=True)
        _git(
            source,
            "push",
            plan.previous.url,
            f"{plan.reviewed.commit}:refs/heads/{plan.previous.branch}",
        )
        print(f"Canonical branch now contains reviewed revision {plan.reviewed.commit}.")
        if remote is None:
            print(f"Publishing follow-up branch {plan.fork}:{plan.branch}...", flush=True)
            _git(
                worktree,
                "push",
                f"https://github.com/{plan.fork}.git",
                f"{sha}:refs/heads/{plan.branch}",
            )
    _check_current(gh, plan)
    if existing is None:
        print("Creating the lock-only follow-up PR...", flush=True)
        existing = gh.api(
            f"repos/{_TRTLLM}/pulls",
            method="POST",
            payload={
                "title": plan.title,
                "body": _pr_body(plan, message),
                "base": "main",
                "head": f"{plan.fork.split('/')[0]}:{plan.branch}",
                "head_repo": plan.fork.split("/")[1],
            },
        )
    print(f"Lock-only promotion PR: {existing['html_url']}")
    existing = _ensure_ci_skip(gh, plan, existing)
    if existing.get("merged"):
        return existing
    if args.auto_merge:
        _arm_auto_merge(gh, plan, existing, message)
    if args.wait:
        deadline = time.monotonic() + args.timeout
        while True:
            current = gh.get(f"repos/{_TRTLLM}/pulls/{existing['number']}")
            if current.get("merged"):
                _verify_remote_commit(gh, plan, _TRTLLM, current["merge_commit_sha"])
                _report_merged(current)
                break
            if current["state"] == "closed" or current["head"]["sha"] != existing["head"]["sha"]:
                if args.auto_merge and current.get("auto_merge"):
                    _disable_auto_merge(gh, current)
                raise ValueError(
                    "Follow-up PR closed or changed while waiting; inspect before proceeding."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Promotion PR is still pending; rerun promote to verify after merge."
                )
            time.sleep(min(15, max(0, deadline - time.monotonic())))
    return existing


def main(argv: list[str] | None = None) -> int:
    """Run the maintainer CLI; dry-run unless publication is explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    promote = commands.add_parser("promote", help="Promote a merged TRT-LLM source update.")
    promote.add_argument("--trtllm-pr", required=True)
    promote.add_argument(
        "--canonical-repo",
        required=True,
        help="Explicit canonical FlashInfer fork OWNER/REPO (not the developer's temporary fork).",
    )
    promote.add_argument(
        "--canonical-branch",
        default="trtllm-prims-ts-dev",
        help="Expected canonical branch; specify a dated branch when applicable.",
    )
    provenance = promote.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--upstream-pr", action="append", default=[])
    provenance.add_argument(
        "--unpaired-reason",
        help="Explicit exception for changes with no upstream PR; retain them during refresh.",
    )
    promote.add_argument("--map-upstream", action="append", default=[], metavar="COMMIT=PR")
    promote.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    promote.add_argument(
        "--fork", help="TRT-LLM personal fork OWNER/REPO; defaults to remote 'fork'."
    )
    promote.add_argument(
        "--flashinfer-repo",
        type=Path,
        help="Local source repository containing both immutable SHAs.",
    )
    promote.add_argument("--worktree", type=Path, help="New/preserved follow-up worktree path.")
    promote.add_argument(
        "--publish",
        action="store_true",
        help="Publish the lock-only PR and post the no-code-change /bot skip request.",
    )
    promote.add_argument("--auto-merge", action="store_true")
    promote.add_argument(
        "--wait", action="store_true", help="Wait for merge and verify final commit metadata."
    )
    promote.add_argument(
        "--timeout", type=int, default=3600, help="Maximum --wait duration in seconds."
    )
    args = parser.parse_args(argv)
    if (args.auto_merge or args.wait) and not args.publish:
        parser.error("--auto-merge and --wait require --publish")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.repo = args.repo.resolve()
    try:
        gh = GitHub()
        plan = _make_plan(args, gh)
        mode = (
            "ALREADY COMPLETED" if plan.completed_pr else "PUBLISH" if args.publish else "DRY RUN"
        )
        print(f"{mode}: {plan.canonical_repo}:{plan.previous.branch}", flush=True)
        print(f"Source: {plan.previous.commit} -> {plan.reviewed.commit}")
        print(f"Follow-up: {plan.fork}:{plan.branch}")
        print(_record_text(plan.record))
        if args.publish:
            _publish(args, gh, plan)
        elif plan.completed_pr is not None:
            _report_merged(plan.completed_pr)
        else:
            print(f"On publication: post {_CI_SKIP_COMMAND}")
            print(f"Squash auto-merge: {'enabled' if args.auto_merge else 'off'}")
            print("No remote writes or consumer worktree changes. Use --publish to proceed.")
        return 0
    except (
        RuntimeError,
        ValueError,
        OSError,
        KeyError,
        subprocess.TimeoutExpired,
        vendor.VendorError,
    ) as error:
        print(f"Promotion stopped: {error}", file=sys.stderr)
        print(
            "Any completed pushes/PRs are preserved. Inspect the error before retrying.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
