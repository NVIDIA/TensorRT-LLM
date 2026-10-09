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
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

import yaml

if __package__:
    from . import manage, promote
else:
    import manage
    import promote

BEGIN = "<!-- vendor-promotion:start -->"
END = "<!-- vendor-promotion:end -->"
SIMILARITY_THRESHOLD = 0.9
_SIMILARITY_METRIC = "sequence-matcher-lines-v1"


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
    body = body.replace("\r\n", "\n").replace("\r", "\n")
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


def _patch(
    repo: Path, base: str, head: str, *, source: str = ".", include: tuple[str, ...] = ("**/*",)
) -> str:
    """Compare lock-selected edits, retaining bytes, paths, modes, and hunk boundaries."""
    changed = promote._run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            head,
            "--",
        ],
        strip=False,
        binary_output=True,
    )
    # Use the vendoring glob semantics rather than Git's different pathspec
    # rules. Enumerate both sides without rename detection to include deletions
    # and moves across the selection boundary; pass the results as literal paths.
    selected = []
    for path in changed.split("\0"):
        if not path or not PurePosixPath(path).is_relative_to(source):
            continue
        relative = PurePosixPath(path).relative_to(source).as_posix()
        if relative != "." and manage._matches(relative, include):
            selected.append(f":(top,literal){path}")
    if not selected:
        return ""
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
            "--unified=0",
            "--inter-hunk-context=0",
            base,
            head,
            "--",
            *selected,
        ],
        strip=False,
        binary_output=True,
    )
    # Hunk labels are Git-generated context, not changed code. Keep only the
    # boundary marker; edit bytes, including changed function definitions, remain.
    return "\n".join(
        re.sub(r"^@@ -[0-9]+(?:,[0-9]+)? \+[0-9]+(?:,[0-9]+)? @@.*$", "@@", line)
        for line in diff.split("\n")
        if not line.startswith("index ")
    )


def _commits(repo: Path, base: str, head: str) -> list[str]:
    return promote._git(repo, "rev-list", "--reverse", f"{base}..{head}").splitlines()


def _similarity(source_patch: str, upstream_patch: str) -> float:
    """Score normalized patch lines without discarding whitespace or frequent lines."""
    if not source_patch or not upstream_patch:
        return 0.0
    if source_patch == upstream_patch:
        return 1.0
    return SequenceMatcher(
        None,
        source_patch.removesuffix("\n").split("\n"),
        upstream_patch.removesuffix("\n").split("\n"),
        autojunk=False,
    ).ratio()


def _upstream_reference(number: int, repository: str | None) -> str:
    label = f"upstream PR #{number}"
    return f"[{label}](https://github.com/{repository}/pull/{number})" if repository else label


def similarity_feedback(evidence: dict[str, dict], upstream_repo: str | None = None) -> str:
    """Describe accepted non-exact matches for feedback and durable provenance."""
    lines = [
        f"- `{sha}`: {_upstream_reference(item['upstream_pr'], upstream_repo)}, similarity "
        f"**{item['similarity']:.2%}** — accepted "
        f"(threshold {item['similarity_threshold']:.2%}; {item['method']})."
        for sha, item in evidence.items()
        if "similarity" in item
    ]
    if not lines:
        return ""
    return (
        "No exact match was found for the following commits; similarity matching accepted them:\n\n"
        + "\n".join(lines)
        + "\n\nScores compare normalized patch lines in lock-selected files, not semantic "
        "equivalence. These changes retain `retain-until-verified` refresh policy."
    )


def match(
    repo: Path,
    previous: str,
    reviewed: str,
    upstream_prs: dict[int, dict],
    resolutions: dict[str, dict] | None = None,
    *,
    source: str = ".",
    include: tuple[str, ...] = ("**/*",),
    upstream_repo: str | None = None,
) -> tuple[dict[str, int | None], dict[str, dict]]:
    """Match a linear source range against fetched immutable upstream PR snapshots.

    Only lock-selected files are compared. None assignments record commits with
    no selected changes, not upstream equivalence for the rest of those commits.
    Ambiguities remain author-actionable; titles and cherry-pick trailers are not proof.
    """
    commits = _commits(repo, previous, reviewed)
    if not commits:
        raise ValueError("Source update has no new commits to attribute.")
    source_parents = {}
    for sha in commits:
        parents = promote._git(repo, "show", "-s", "--format=%P", sha).split()
        if len(parents) != 1:
            raise ValueError("Source history must be linear; rebase merge commits first.")
        source_parents[sha] = parents[0]
    resolutions = resolutions or {}
    if set(resolutions) - set(commits):
        raise ValueError("Resolutions include commits outside the reviewed source delta.")
    candidates: dict[str, set[int]] = {sha: set() for sha in commits}
    patches = {sha: _patch(repo, f"{sha}^", sha, source=source, include=include) for sha in commits}
    if any(not patches[sha] for sha in resolutions):
        raise ValueError("Remove resolutions for commits with no lock-selected file changes.")
    series = _patch(repo, previous, reviewed, source=source, include=include)
    methods: dict[tuple[str, int], str] = {}
    comparisons: dict[int, list[tuple[str, str, str, str]]] = {}
    for number, pr in upstream_prs.items():
        base, head = promote._sha(pr["base"]["sha"]), promote._sha(pr["head"]["sha"])
        merge_base = promote._git(repo, "merge-base", base, head)
        upstream_commits = _commits(repo, merge_base, head)
        upstream_patches = set()
        comparisons[number] = []
        for sha in upstream_commits:
            parents = promote._git(repo, "show", "-s", "--format=%P", sha).split()
            if len(parents) == 1:
                patch = _patch(repo, parents[0], sha, source=source, include=include)
                upstream_patches.add(patch)
                if patch:
                    comparisons[number].append(("similar-edits", parents[0], sha, patch))
        aggregate = _patch(repo, merge_base, head, source=source, include=include)
        if aggregate:
            comparisons[number].append(("similar-upstream-aggregate", merge_base, head, aggregate))
        for sha in commits:
            if not patches[sha]:
                continue
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
        if aggregate and series == aggregate:
            for sha in commits:
                if not patches[sha]:
                    continue
                candidates[sha].add(number)
                methods.setdefault((sha, number), "source-series-aggregate")
    # Exact matches take priority across all listed PRs. Approximate attribution
    # is considered only when none exists, and is not permission to drop changes.
    similarities: dict[str, dict[int, dict]] = {}
    for sha in commits:
        if not patches[sha] or sha in resolutions or candidates[sha]:
            continue
        similarities[sha] = {}
        for number, choices in comparisons.items():
            best = {"similarity": 0.0}
            for method, base, head, patch in choices:
                score = _similarity(patches[sha], patch)
                if score > best["similarity"]:
                    best = {
                        "method": method,
                        "similarity": score,
                        "comparison": {
                            "source_base": source_parents[sha],
                            "source_head": sha,
                            "upstream_base": base,
                            "upstream_head": head,
                        },
                    }
            similarities[sha][number] = best
            if best["similarity"] >= SIMILARITY_THRESHOLD:
                candidates[sha].add(number)
    assignments, evidence, unresolved = {}, {}, []
    for sha in commits:
        if not patches[sha]:
            assignments[sha] = None
            evidence[sha] = {"method": "outside-vendor-scope", "refresh_policy": "retain"}
        elif sha in resolutions:
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
                "upstream_head": upstream_prs[number]["head"]["sha"],
                "upstream_base": upstream_prs[number]["base"]["sha"],
            }
            if sha in similarities:
                evidence[sha].update(
                    similarities[sha][number],
                    upstream_pr=number,
                    similarity_threshold=SIMILARITY_THRESHOLD,
                    similarity_metric=_SIMILARITY_METRIC,
                    refresh_policy="retain-until-verified",
                )
            else:
                evidence[sha]["method"] = methods[sha, number]
        else:
            options = (
                ", ".join(f"#{number}" for number in sorted(candidates[sha])) or "no exact match"
            )
            detail = f"- `{sha}`: {options}"
            if sha in similarities:
                closest = sorted(
                    similarities[sha].items(), key=lambda item: (-item[1]["similarity"], item[0])
                )[:3]
                scores = (
                    ", ".join(
                        f"{_upstream_reference(number, upstream_repo)}: {item['similarity']:.2%}"
                        for number, item in closest
                    )
                    or "no upstream candidates"
                )
                outcome = "multiple PRs meet" if candidates[sha] else "no PR meets"
                detail += (
                    f"; best similarities: {scores}; {outcome} the "
                    f"{SIMILARITY_THRESHOLD:.2%} similarity threshold"
                )
            unresolved.append(detail)
    accepted = similarity_feedback(evidence, upstream_repo)
    if unresolved:
        raise ValueError(
            "Author action required: upstream attribution is unresolved. Add missing upstream PRs, "
            "update/rebase the source pin, or add a targeted resolution with an upstream_pr and "
            "reason for each commit below. No complete commit_map is required.\n"
            + "\n".join(unresolved)
            + ("\n\n" + accepted if accepted else "")
        )
    if set(assignments.values()) - {None} != set(upstream_prs):
        raise ValueError(
            "Some listed upstream PRs cover no new lock-selected changes; remove unrelated PRs."
            + ("\n\n" + accepted if accepted else "")
        )
    return assignments, evidence


def resolve(
    gh: promote.GitHub,
    previous: manage.Vendor,
    reviewed: manage.Vendor,
    entry: dict,
    cache: Path,
) -> tuple[dict[str, int | None], dict[str, dict]]:
    """Fetch only Git data into a private bare cache and resolve declared PRs."""
    if previous.source != reviewed.source or set(previous.include) != set(reviewed.include):
        raise ValueError(
            "Automatic attribution requires unchanged lock source/include selection; "
            "separate the selection change from the source update."
        )
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
    return match(
        cache,
        previous.commit,
        reviewed.commit,
        prs,
        entry["resolutions"],
        source=reviewed.source,
        include=reviewed.include,
        upstream_repo=gh.upstream_repo,
    )
