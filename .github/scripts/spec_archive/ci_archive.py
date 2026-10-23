#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Archive inert spec blobs from GitHub/GitLab APIs using trusted CI code.

No source branch is checked out and no downloaded file is executed. Keep this
script and git-spec on the protected default branch. See ci/README.md.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import sys
import tempfile
from collections.abc import Iterator
from http.client import HTTPMessage
from typing import BinaryIO, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOTS = ("specs/", "openspec/changes/")
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILES = 10000
SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


class ArtifactClient(Protocol):
    """Archive storage interface used by the provider adapters."""

    def upload(self, data: bytes, descriptor: dict) -> dict:
        """Upload verified bytes and publish their finalized descriptor."""
        ...


def core_api() -> dict:
    path = Path(os.environ.get("SPEC_ARCHIVE_CORE", Path(__file__).resolve().parents[1] / "git-spec"))
    return runpy.run_path(str(path))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def checked_sha(value: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Expected a complete lowercase Git commit/object SHA")
    return value


def checked_repo(value: str) -> str:
    if not isinstance(value, str) or len(value.split("/")) < 2 or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in (".", "..") for part in value.split("/")
    ):
        raise ValueError("Expected namespace/repository (nested GitLab namespaces are supported)")
    return value


def safe_path(path: str) -> str:
    if not isinstance(path, str) or "\\" in path or "\x00" in path:
        raise ValueError("Unsafe spec blob path")
    parts = path.split("/")
    if any(p in ("", ".", "..") or p.lower() in (".git", ".github") for p in parts):
        raise ValueError("Unsafe spec blob path")
    if PurePosixPath(path).is_absolute() or not any(path.startswith(root) for root in ROOTS):
        raise ValueError("Blob is outside allowed spec roots")
    return path


def design_links(description: str | None, files: dict[str, bytes]) -> list[str]:
    texts = [description or ""] + [data.decode("utf-8", errors="replace") for name, data in files.items()
                                      if name.lower().endswith(".md")]
    links = set()
    for text in texts:
        for match in re.finditer(r"(?im)^\s*design-doc:\s*(https://[^\s<>]+)", text):
            value = match.group(1).rstrip(").,")
            parsed = urlsplit(value)
            if parsed.hostname and not parsed.username and not parsed.password:
                links.add(value)
    return sorted(links)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: BinaryIO, code: int, msg: str,
                         headers: HTTPMessage, newurl: str) -> None:
        # Do not send a CI credential to a Location supplied by remote content.
        return None


class API:
    def __init__(self, base_url: str, token: str | None, provider: str) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Provider API URL must be a credential-free HTTPS URL")
        if not token:
            raise ValueError(f"Configure a personal/project read API token for {provider}")
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": "spec-archive/1", "Accept": "application/json"}
        self.headers["Authorization" if provider == "github" else "PRIVATE-TOKEN"] = (
            f"Bearer {token}" if provider == "github" else token
        )
        self.opener = build_opener(NoRedirects())

    def get(self, path: str, *, params: dict | None = None,
            raw: bool = False) -> dict | list | bytes:
        url = self.base_url + "/" + path.lstrip("/")
        if params:
            url += "?" + urlencode(params, doseq=True)
        request = Request(url, headers=self.headers)
        with self.opener.open(request, timeout=60) as response:
            data = response.read(MAX_FILE_BYTES * 2 + 1)
            if len(data) > MAX_FILE_BYTES * 2:
                raise ValueError("Provider response exceeds archive limits")
            return data if raw else json.loads(data)

    def pages(self, path: str, params: dict | None = None) -> Iterator[dict]:
        for page in range(1, 1001):
            result = self.get(path, params={**(params or {}), "per_page": 100, "page": page})
            if not isinstance(result, list):
                raise ValueError("Expected a provider API list response")
            yield from result
            if len(result) < 100:
                return
        raise ValueError("Provider list exceeds pagination limit")


class Provider:
    def __init__(self, api: API, repo: str, host: str) -> None:
        self.api, self.repo, self.host = api, checked_repo(repo), host
        parsed = urlsplit("https://" + host)
        if not parsed.hostname or parsed.netloc != host or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Invalid forge host")

    def identity(self, number: int) -> dict:
        if not isinstance(number, int) or number < 1:
            raise ValueError("PR/MR number must be a positive integer")
        return {"provider": self.name, "host": self.host, "repo": self.repo, "pr": number}

    def manifest(self, number: int, record: dict, head: str, base: str,
                 files: dict[str, bytes], now: str | None = None) -> dict:
        paths = [root for root in ROOTS if any(path.startswith(root) for path in files)]
        tool = "mixed" if len(paths) == 2 else "openspec" if paths == [ROOTS[1]] else "speckit"
        status, merge = self.lifecycle(record)
        return {"schema_version": 1, **self.identity(number),
                "repo_url": f"https://{self.host}/{self.repo}",
                "head_sha": checked_sha(head), "base_sha": checked_sha(base),
                "merge_sha": merge, "status": status, "title": record["title"],
                "author": record["user"]["login"] if self.name == "github" else record["author"]["username"],
                "created": record["created_at"], "archived_at": now or utc_now(),
                "tool": tool, "paths": paths,
                "design_docs": design_links(record.get("body") if self.name == "github" else record.get("description"), files),
                "sha256": None}

    @staticmethod
    def check_files(files: dict[str, bytes]) -> None:
        if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_TOTAL_BYTES:
            raise ValueError("Spec snapshot exceeds archive limits")
        for path, data in files.items():
            safe_path(path)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Spec blob exceeds file-size limit")


class GitHub(Provider):
    name = "github"

    def path(self, suffix: str, repo: str | None = None) -> str:
        repository = checked_repo(repo or self.repo)
        return "repos/" + "/".join(quote(p, safe="") for p in repository.split("/")) + "/" + suffix

    def record(self, number: int) -> dict:
        self.identity(number)
        return self.api.get(self.path(f"pulls/{number}"))

    def list_open(self) -> list[int]:
        return [r["number"] for r in self.api.pages(self.path("pulls"), {"state": "open"})]

    @staticmethod
    def lifecycle(record: dict) -> tuple[str, str | None]:
        status = "merged" if record.get("merged") or record.get("merged_at") else "open" if record["state"] == "open" else "abandoned"
        # GitHub's open PR merge_commit_sha can identify a synthetic test merge.
        return status, checked_sha(record["merge_commit_sha"]) if status == "merged" and record.get("merge_commit_sha") else None

    def tree(self, repo: str, treeish: str, recursive: bool = False) -> list[dict]:
        result = self.api.get(self.path("git/trees/" + checked_sha(treeish), repo), params={"recursive": "1"} if recursive else None)
        if result.get("truncated"):
            raise ValueError("GitHub spec tree was truncated; refusing incomplete archival")
        return result["tree"]

    def entries(self, repo: str, commit: str) -> dict[str, tuple[str, str]]:
        root_tree = self.tree(repo, commit)
        entries = {}
        for root in ROOTS:
            tree = root_tree
            folder_sha = None
            for part in root.rstrip("/").split("/"):
                entry = next((e for e in tree if e["path"] == part), None)
                if entry is None:
                    folder_sha = None
                    break
                if entry["type"] != "tree":
                    raise ValueError("Spec root must be a directory")
                folder_sha = checked_sha(entry["sha"])
                tree = self.tree(repo, folder_sha)
            if folder_sha is None:
                continue
            for entry in self.tree(repo, folder_sha, recursive=True):
                if entry["type"] == "tree":
                    continue
                if entry["type"] != "blob" or entry.get("mode") not in ("100644", "100755"):
                    raise ValueError("Refusing symlink/submodule in spec snapshot")
                path = safe_path(root + entry["path"])
                entries[path] = (checked_sha(entry["sha"]), entry["mode"])
                if len(entries) > MAX_FILES:
                    raise ValueError("Spec tree exceeds file-count limit")
        return entries

    def snapshot(self, number: int, head: str | None = None,
                 now: str | None = None) -> tuple[dict, dict[str, bytes]]:
        record = self.record(number)
        if self.lifecycle(record)[0] != "open":
            raise ValueError("Pull request is no longer open; refusing removal-bound archival")
        if head is not None and head != record["head"]["sha"]:
            raise ValueError("GitHub source head moved; retry archival with the current PR head")
        head = checked_sha(head or record["head"]["sha"])
        source = checked_repo(record["head"]["repo"]["full_name"])
        comparison = self.api.get(self.path("compare/" + checked_sha(record["base"]["sha"]) + "..." + head))
        merge_base = checked_sha(comparison["merge_base_commit"]["sha"])
        head_entries = self.entries(source, head)
        base_entries = self.entries(self.repo, merge_base)
        files = {}
        # Cleanup heads can still contain baseline specs. Never publish those
        # as a new design snapshot after the PR's actual delta was archived.
        if head_entries != base_entries:
            for path, (blob_sha, _) in head_entries.items():
                result = self.api.get(self.path("git/blobs/" + blob_sha, source))
                if result.get("encoding") != "base64":
                    raise ValueError("Expected base64 GitHub blob")
                files[path] = base64.b64decode("".join(result["content"].split()), validate=True)
                self.check_files(files)
        return self.manifest(number, record, head, merge_base, files, now), files


class GitLab(Provider):
    name = "gitlab"

    def path(self, suffix: str, project: str | int | None = None) -> str:
        return "projects/" + quote(str(project or self.repo), safe="") + "/" + suffix

    def record(self, number: int) -> dict:
        self.identity(number)
        return self.api.get(self.path(f"merge_requests/{number}"))

    def list_open(self) -> list[int]:
        return [r["iid"] for r in self.api.pages(self.path("merge_requests"), {"state": "opened"})]

    @staticmethod
    def lifecycle(record: dict) -> tuple[str, str | None]:
        status = {"opened": "open", "locked": "open", "merged": "merged", "closed": "abandoned"}[record["state"]]
        merge = record.get("merge_commit_sha") or record.get("squash_commit_sha")
        return status, checked_sha(merge) if status == "merged" and merge else None

    def entries(self, project: str | int, commit: str) -> dict[str, tuple[str, str]]:
        blobs = {}
        for root in ROOTS:
            try:
                entries = list(self.api.pages(self.path("repository/tree", project),
                                              {"path": root.rstrip("/"), "ref": checked_sha(commit), "recursive": "true"}))
            except HTTPError as error:
                # Missing roots are legitimate; this handler cannot swallow a
                # failed blob download because downloads happen below it.
                if error.code != 404:
                    raise
                continue
            if len(entries) > MAX_FILES:
                raise ValueError("GitLab spec tree exceeds file-count limit")
            for entry in entries:
                if entry["type"] == "tree":
                    continue
                if entry["type"] != "blob" or entry.get("mode") not in ("100644", "100755"):
                    raise ValueError("Refusing symlink/submodule in spec snapshot")
                path = safe_path(entry["path"])
                if not path.startswith(root):
                    raise ValueError("GitLab returned a blob outside the requested spec root")
                blobs[path] = (checked_sha(entry["id"]), entry["mode"])
                if len(blobs) > MAX_FILES:
                    raise ValueError("Spec tree exceeds file-count limit")
        return blobs

    def snapshot(self, number: int, head: str | None = None,
                 now: str | None = None) -> tuple[dict, dict[str, bytes]]:
        record = self.record(number)
        if self.lifecycle(record)[0] != "open":
            raise ValueError("Merge request is no longer open; refusing removal-bound archival")
        diff = record.get("diff_refs") or {}
        current_head = checked_sha(diff.get("head_sha") or record.get("sha"))
        if record.get("sha") and record["sha"] != current_head:
            raise ValueError("GitLab MR diff refs are not ready for the current source head; retry later")
        if head is not None and head != current_head:
            raise ValueError("GitLab source head moved; retry archival with the current MR head")
        source = record.get("source_project_id")
        if not isinstance(source, int) or source < 1:
            raise ValueError("GitLab source project is unavailable")
        merge_base = checked_sha(diff.get("base_sha"))
        head_entries = self.entries(source, current_head)
        base_entries = self.entries(record.get("target_project_id") or self.repo, merge_base)
        files = {}
        if head_entries != base_entries:
            for path, (blob_sha, _) in head_entries.items():
                files[path] = self.api.get(self.path("repository/blobs/" + blob_sha + "/raw", source), raw=True)
                self.check_files(files)
        return self.manifest(number, record, current_head, merge_base, files, now), files


def provider_from_env(name: str, repo: str | None = None) -> GitHub | GitLab:
    if name == "github":
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        host = urlsplit(server).netloc
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        token = os.environ.get("SPEC_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        return GitHub(API(api_url, token, name), repo or os.environ.get("GITHUB_REPOSITORY"), host)
    server = os.environ.get("SPEC_GITLAB_URL") or os.environ.get("CI_SERVER_URL")
    if not server:
        raise ValueError("Configure SPEC_GITLAB_URL or CI_SERVER_URL")
    host = urlsplit(server).netloc
    api_url = os.environ.get("CI_API_V4_URL", server.rstrip("/") + "/api/v4")
    token = os.environ.get("SPEC_GITLAB_TOKEN") or os.environ.get("GITLAB_TOKEN")
    return GitLab(API(api_url, token, name), repo or os.environ.get("CI_PROJECT_PATH"), host)


def archive(provider: GitHub | GitLab, number: int, client: ArtifactClient,
            api: dict, head: str | None = None) -> dict | None:
    manifest, files = provider.snapshot(number, head=head)
    if not files:
        return None
    api["validate_manifest"](manifest)
    with tempfile.TemporaryDirectory(prefix="spec-archive-ci-") as directory:
        staging = Path(directory) / "source"
        staging.mkdir()
        for name, data in files.items():
            destination = staging / safe_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        output = Path(directory) / "specs.tar.gz"
        descriptor = api["build_archive"](staging, manifest, output)
        client.upload(output.read_bytes(), descriptor)
    return descriptor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("archive", "event"))
    parser.add_argument("--provider", choices=("github", "gitlab"), required=True)
    parser.add_argument("--repo")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--head")
    args = parser.parse_args(argv)
    number = args.pr
    if args.command == "event":
        if args.provider != "github":
            parser.error("GitLab events must trigger a protected default-branch pipeline with --pr IID")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        number = number or event.get("number")
        # Identity comes from the trusted runner repository, never a PR URL.
        if event.get("repository", {}).get("full_name") != os.environ.get("GITHUB_REPOSITORY"):
            raise ValueError("Event repository does not match the trusted workflow repository")
    if number is None:
        parser.error("--pr is required outside a GitHub PR event")
    provider = provider_from_env(args.provider, args.repo)
    api = core_api()
    client = api["Client"].from_env()
    record = provider.record(number)
    status, merge = provider.lifecycle(record)
    if status != "open":
        if args.command == "archive":
            raise ValueError("PR/MR is no longer open; refusing removal-bound archival")
        client.update_status(provider.identity(number), status, merge)
        print(json.dumps({"identity": provider.identity(number), "status": status}))
        return 0
    if args.command == "event":
        # Reopening a PR clears terminal lifecycle state on every prior version.
        client.update_status(provider.identity(number), "open", None)
    descriptor = archive(provider, number, client, api, args.head)
    print(json.dumps(descriptor or {"identity": provider.identity(number), "outcome": "no-specs"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"spec archive: {error}", file=sys.stderr)
        raise SystemExit(1)
