# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open one draft pull request through the GitHub REST API. Opt-in only.

**This module never pushes and never opens anything by default.** Without
``yes=True`` it validates the request and returns what it *would* send; the
network is not touched unless ``check_remote`` (read-only ``GET`` requests)
or ``yes`` (one ``POST``) is asked for explicitly. Pushing the branch to the
fork is the operator's own step, done by hand beforehand.

It exists because the login node this was built on has no ``gh`` binary; the
token is read from ``gh``'s own ``hosts.yml`` (``$GH_CONFIG_DIR``, default
``~/.config/gh``) or from ``$GH_TOKEN`` / ``$GITHUB_TOKEN``, and is never
printed or logged.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

API = "https://api.github.com"

#: ``[JIRA/NVBUG/None][type] description`` — the repository's PR title rule.
TITLE_PATTERN = re.compile(
    r"^\[(?:[A-Z][A-Z0-9]+-\d+|https://nvbugs/\d+|nvbug/\d+|None)(?:,[^\]]+)*\]"
    r"\[(?:feat|fix|doc|docs|test|chore|perf|refactor|infra|ci|build|style|revert)\] \S"
)


class PrOpenError(ValueError):
    """The request is not valid, or no token is available."""


@dataclass(frozen=True)
class PrRequest:
    upstream: str
    owner: str
    branch: str
    title: str
    body: str
    base: str = "main"
    draft: bool = True

    def payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "head": f"{self.owner}:{self.branch}",
            "base": self.base,
            "body": self.body,
            "draft": self.draft,
        }


def validate(request: PrRequest) -> list[str]:
    problems = []
    if not TITLE_PATTERN.match(request.title):
        problems.append(
            f"title {request.title!r} does not match `[JIRA/NVBUG/None][type] description`"
        )
    if not request.body.strip():
        problems.append("the body is empty")
    if "/" not in request.upstream:
        problems.append(f"upstream {request.upstream!r} is not OWNER/REPO")
    if not request.owner or not request.branch:
        problems.append("the fork owner and the branch are both required")
    if re.search(r"(?im)^co-authored-by:", request.body):
        problems.append("the body carries a Co-authored-by line")
    return problems


def read_token(environ: Mapping[str, str] | None = None) -> str:
    """The GitHub token, from the environment or ``gh``'s hosts file. Never printed."""
    env = os.environ if environ is None else environ
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if env.get(name):
            return env[name]
    config = Path(
        env.get("GH_CONFIG_DIR") or Path(env.get("HOME", "~")).expanduser() / ".config/gh"
    )
    hosts = config / "hosts.yml"
    if not hosts.is_file():
        raise PrOpenError(f"no token: $GH_TOKEN/$GITHUB_TOKEN unset and {hosts} does not exist")
    data = yaml.safe_load(hosts.read_text(encoding="utf-8")) or {}
    host = data.get("github.com") or {}
    user = host.get("user")
    token = ((host.get("users") or {}).get(user) or {}).get("oauth_token") or host.get(
        "oauth_token"
    )
    if not token:
        raise PrOpenError(f"no oauth_token for github.com in {hosts}")
    return str(token)


Opener = Callable[[urllib.request.Request], Any]


def _call(opener: Opener, method: str, path: str, token: str, data: dict | None = None) -> Any:
    request = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with opener(request) as response:
        return json.loads(response.read().decode("utf-8") or "null")


def open_pr(
    request: PrRequest,
    *,
    yes: bool = False,
    check_remote: bool = False,
    token: str | None = None,
    opener: Opener = urllib.request.urlopen,
) -> dict[str, Any]:
    """Validate, optionally check the remote, and only with ``yes`` create the PR.

    Returns a report: ``{"dry_run": bool, "payload": ..., "checks": [...],
    "url": ...}``. Raises :class:`PrOpenError` on an invalid request.
    """
    problems = validate(request)
    if problems:
        raise PrOpenError("; ".join(problems))
    report: dict[str, Any] = {"dry_run": not yes, "payload": request.payload(), "checks": []}
    if not (yes or check_remote):
        return report
    token = token or read_token()
    fork_repo = request.upstream.split("/", 1)[1]
    branch = urllib.parse.quote(request.branch, safe="")
    found = _call(opener, "GET", f"/repos/{request.owner}/{fork_repo}/branches/{branch}", token)
    report["checks"].append(
        f"fork branch {request.owner}:{request.branch}: "
        + (
            f"at {found['commit']['sha'][:10]}"
            if isinstance(found, dict) and "commit" in found
            else "MISSING"
        )
    )
    head = urllib.parse.quote(f"{request.owner}:{request.branch}", safe=":")
    existing = _call(
        opener, "GET", f"/repos/{request.upstream}/pulls?state=open&head={head}", token
    )
    urls = [p.get("html_url") for p in existing or [] if isinstance(p, dict)]
    report["checks"].append(f"open PRs with this head: {urls or 'none'}")
    if not yes:
        return report
    if urls:
        raise PrOpenError(f"an open PR already uses this head: {urls}")
    created = _call(opener, "POST", f"/repos/{request.upstream}/pulls", token, request.payload())
    report["url"] = created.get("html_url") if isinstance(created, dict) else None
    report["response"] = (
        {k: created.get(k) for k in ("number", "state", "draft")}
        if isinstance(created, dict)
        else created
    )
    return report


__all__ = ["PrOpenError", "PrRequest", "TITLE_PATTERN", "open_pr", "read_token", "validate"]
