# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict contributor metadata and conservative upstream commit attribution.

Matching establishes provenance, not permission to drop changes during refresh.
Source code and commit messages are data and are never executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

if __package__:
    from . import manage, promote
else:
    import manage
    import promote

BEGIN = "<!-- vendor-promotion:start -->"
END = "<!-- vendor-promotion:end -->"


def template(vendor_name: str, upstream_repo: str) -> str:
    """Return a copyable PR-description block; repository settings stay operator-owned."""
    return (
        f"{BEGIN}\n```yaml\nschema_version: 1\nvendors:\n  {vendor_name}:\n"
        f"    upstream_prs:\n      - https://github.com/{upstream_repo}/pull/123\n"
        f"```\n{END}"
    )


def parse(body: str, vendor_name: str, upstream_repo: str) -> dict:
    """Read exactly one marked block and validate the selected vendor's inputs."""
    if len(body) > 65536:
        raise ValueError("PR description exceeds the supported metadata size.")
    if body.count(BEGIN) != 1 or body.count(END) != 1:
        raise ValueError("PR description must contain exactly one vendor-promotion marked block.")
    start, end = body.index(BEGIN) + len(BEGIN), body.index(END)
    block = body[start:end].strip()
    if start >= end or not block.startswith("```yaml\n") or not block.endswith("\n```"):
        raise ValueError("The marked block must contain one fenced yaml mapping.")
    try:
        content = block[len("```yaml\n") : -len("\n```")]
        if any(
            isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for token in yaml.scan(content)
        ):
            raise ValueError("Promotion YAML aliases and anchors are not allowed.")
        data = yaml.load(content, Loader=manage._UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid promotion YAML: {error}") from error
    if not isinstance(data, dict) or set(data) != {"schema_version", "vendors"}:
        raise ValueError("Promotion YAML requires only schema_version and vendors.")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported promotion schema_version; expected 1.")
    entries = data["vendors"]
    if not isinstance(entries, dict) or vendor_name not in entries:
        raise ValueError(f"Promotion YAML must include vendors.{vendor_name}.")
    if any(
        not isinstance(name, str) or not manage._NAME_PATTERN.fullmatch(name) for name in entries
    ):
        raise ValueError("Invalid vendor key in promotion YAML.")
    entry = entries[vendor_name]
    if not isinstance(entry, dict) or set(entry) - {
        "upstream_prs",
        "unpaired_reason",
        "resolutions",
    }:
        raise ValueError(
            "Vendor metadata allows upstream_prs, unpaired_reason, and resolutions only."
        )
    if "unpaired_reason" in entry:
        if (
            set(entry) != {"unpaired_reason"}
            or not isinstance(entry["unpaired_reason"], str)
            or not entry["unpaired_reason"].strip()
        ):
            raise ValueError("unpaired_reason must be nonempty and cannot accompany upstream PRs.")
        return {
            "upstream_prs": [],
            "resolutions": {},
            "unpaired_reason": entry["unpaired_reason"].strip(),
        }
    references = entry.get("upstream_prs")
    if not isinstance(references, list) or not references:
        raise ValueError("upstream_prs must be a nonempty list of upstream PR URLs.")
    numbers = []
    for reference in references:
        if not isinstance(reference, str) or not reference.startswith(
            f"https://github.com/{upstream_repo}/pull/"
        ):
            raise ValueError(f"Each upstream PR must be a URL in {upstream_repo}.")
        numbers.append(promote._pr_number(reference, upstream_repo))
    if len(set(numbers)) != len(numbers):
        raise ValueError("upstream_prs contains duplicate PRs.")
    resolutions = entry.get("resolutions", {})
    if not isinstance(resolutions, dict):
        raise ValueError("resolutions must be a mapping keyed by full source commit SHA.")
    normalized = {}
    for sha, resolution in resolutions.items():
        promote._sha(sha)
        if not isinstance(resolution, dict) or set(resolution) != {"upstream_pr", "reason"}:
            raise ValueError("Each resolution requires upstream_pr and reason.")
        reference, reason = resolution["upstream_pr"], resolution["reason"]
        if not isinstance(reference, str):
            raise ValueError("Resolution upstream_pr must be a PR URL.")
        number = promote._pr_number(reference, upstream_repo)
        if number not in numbers or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Resolution must name a listed PR and explain the unmatched patch.")
        normalized[sha] = {"upstream_pr": number, "reason": reason.strip()}
    return {"upstream_prs": sorted(numbers), "resolutions": normalized, "unpaired_reason": None}


def fingerprint(value: dict) -> str:
    """Hash normalized metadata without recording credentials or unrelated PR prose."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _patch(repo: Path, base: str, head: str) -> str:
    """Normalize hunk coordinates, retaining whitespace, paths, modes, and binary data."""
    diff = promote._run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--binary",
            "--full-index",
            "--no-color",
            "--unified=3",
            base,
            head,
            "--",
        ],
        strip=False,
        binary_output=True,
    )
    # Blob IDs and hunk positions change on rebases. Keep surrounding lines and
    # function context so identical edits at different code sites do not match.
    return "\n".join(
        re.sub(r"^@@ -[0-9]+(?:,[0-9]+)? \+[0-9]+(?:,[0-9]+)? @@", "@@", line)
        for line in diff.split("\n")
        if not line.startswith("index ")
    )


def _commits(repo: Path, base: str, head: str) -> list[str]:
    return promote._git(repo, "rev-list", "--reverse", f"{base}..{head}").splitlines()


def match(
    repo: Path,
    previous: str,
    reviewed: str,
    upstream_prs: dict[int, dict],
    resolutions: dict[str, dict] | None = None,
) -> tuple[dict[str, int], dict[str, dict]]:
    """Match a linear source range against fetched immutable upstream PR snapshots.

    Exact edits are compared across all files, not only selected vendored files.
    Ambiguities remain author-actionable; titles and cherry-pick trailers are not proof.
    """
    commits = _commits(repo, previous, reviewed)
    if not commits:
        raise ValueError("Source update has no new commits to attribute.")
    for sha in commits:
        if len(promote._git(repo, "show", "-s", "--format=%P", sha).split()) != 1:
            raise ValueError("Source history must be linear; rebase merge commits first.")
    resolutions = resolutions or {}
    if set(resolutions) - set(commits):
        raise ValueError("Resolutions include commits outside the reviewed source delta.")
    candidates: dict[str, set[int]] = {sha: set() for sha in commits}
    patches = {sha: _patch(repo, f"{sha}^", sha) for sha in commits}
    methods: dict[tuple[str, int], str] = {}
    for number, pr in upstream_prs.items():
        base, head = promote._sha(pr["base"]["sha"]), promote._sha(pr["head"]["sha"])
        merge_base = promote._git(repo, "merge-base", base, head)
        upstream_commits = _commits(repo, merge_base, head)
        upstream_patches = set()
        for sha in upstream_commits:
            parents = promote._git(repo, "show", "-s", "--format=%P", sha).split()
            if len(parents) == 1:
                upstream_patches.add(_patch(repo, parents[0], sha))
        aggregate = _patch(repo, merge_base, head)
        for sha in commits:
            if sha in upstream_commits:
                candidates[sha].add(number)
                methods[sha, number] = "same-commit"
            elif patches[sha] and patches[sha] in upstream_patches:
                candidates[sha].add(number)
                methods[sha, number] = "same-edits"
            elif patches[sha] and patches[sha] == aggregate:
                candidates[sha].add(number)
                methods[sha, number] = "upstream-aggregate"
        # A whole reviewed series may be squashed into one upstream PR. Do not
        # invent individual correspondences: identify the shared group explicitly.
        if aggregate and _patch(repo, previous, reviewed) == aggregate:
            for sha in commits:
                candidates[sha].add(number)
                methods.setdefault((sha, number), "source-series-aggregate")
    assignments, evidence, unresolved = {}, {}, []
    for sha in commits:
        if sha in resolutions:
            resolution = resolutions[sha]
            number = resolution["upstream_pr"]
            if number not in upstream_prs:
                raise ValueError("Resolution names an unlisted upstream PR.")
            assignments[sha] = number
            evidence[sha] = {
                "method": "author-assertion",
                "reason": resolution["reason"],
                "refresh_policy": "retain-until-verified",
            }
        elif len(candidates[sha]) == 1:
            number = next(iter(candidates[sha]))
            assignments[sha] = number
            evidence[sha] = {
                "method": methods[sha, number],
                "upstream_head": upstream_prs[number]["head"]["sha"],
                "upstream_base": upstream_prs[number]["base"]["sha"],
            }
        else:
            options = (
                ", ".join(f"#{number}" for number in sorted(candidates[sha])) or "no exact match"
            )
            unresolved.append(f"- `{sha}`: {options}")
    if unresolved:
        raise ValueError(
            "Author action required: upstream attribution is unresolved. Add missing upstream PRs, "
            "update/rebase the source pin, or add a targeted resolution with an upstream_pr and "
            "reason for each commit below. No complete commit_map is required.\n"
            + "\n".join(unresolved)
        )
    if set(assignments.values()) != set(upstream_prs):
        raise ValueError(
            "Some listed upstream PRs cover no new source commits; remove unrelated PRs."
        )
    return assignments, evidence


def resolve(
    gh: promote.GitHub,
    previous: manage.Vendor,
    reviewed: manage.Vendor,
    entry: dict,
    cache: Path,
) -> tuple[dict[str, int], dict[str, dict]]:
    """Fetch only Git data into a private bare cache and resolve declared PRs."""
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "HEAD").exists():
        promote._git(cache, "init", "--bare", "--quiet")
    if promote._git(cache, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("Matching cache must be a dedicated bare repository.")
    promote._check_source_repo(gh, promote._repo_name(reviewed.url))
    for sha in (previous.commit, reviewed.commit):
        promote._git(cache, "fetch", "--quiet", "--no-tags", reviewed.url, sha)
    try:
        common = promote._git(cache, "merge-base", previous.commit, reviewed.commit)
    except RuntimeError as error:
        raise ValueError(
            "Source revisions have no merge base; rebase the temporary source branch onto the canonical pin."
        ) from error
    if common != previous.commit:
        raise ValueError(
            "Source update is not a fast-forward; rebase the temporary source branch onto the canonical pin."
        )
    if entry["unpaired_reason"]:
        commits = _commits(cache, previous.commit, reviewed.commit)
        if not commits or any(
            len(promote._git(cache, "show", "-s", "--format=%P", sha).split()) != 1
            for sha in commits
        ):
            raise ValueError("Unpaired source updates still require a nonempty linear history.")
        return {}, {}
    prs = {}
    for number in entry["upstream_prs"]:
        pr = gh.get(f"repos/{gh.upstream_repo}/pulls/{number}")
        if pr["base"]["ref"] != gh.upstream_branch or (
            pr["state"] == "closed" and not pr.get("merged")
        ):
            raise ValueError(
                f"Upstream PR #{number} must target {gh.upstream_branch} and be open or merged."
            )
        for sha in (pr["base"]["sha"], pr["head"]["sha"]):
            promote._git(
                cache,
                "fetch",
                "--quiet",
                "--no-tags",
                f"https://github.com/{gh.upstream_repo}.git",
                promote._sha(sha),
            )
        prs[number] = pr
    return match(cache, previous.commit, reviewed.commit, prs, entry["resolutions"])
