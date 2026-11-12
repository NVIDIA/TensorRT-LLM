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
    from .diagnostics import Diagnostic, field, yaml_problem
else:
    import manage
    import promote
    from diagnostics import Diagnostic, field, yaml_problem

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


def _vendor_example(vendor_name: str, content: str) -> str:
    return f"```yaml\nvendors:\n  {vendor_name}:\n{content}\n```"


def _resolution_example(vendor_name: str, upstream_repo: str) -> str:
    return _vendor_example(
        vendor_name,
        f"    upstream_prs:\n      - https://github.com/{upstream_repo}/pull/123\n"
        "    resolutions:\n      FULL_40_CHARACTER_SOURCE_SHA:\n"
        f"        upstream_pr: https://github.com/{upstream_repo}/pull/123\n"
        "        reason: Explain the downstream adaptation or why this PR owns the change.",
    )


def _pr_reference(
    reference: object, upstream_repo: str, path: str, *, require_url: bool = True
) -> int:
    if isinstance(reference, str) and (
        not require_url or reference.startswith(f"https://github.com/{upstream_repo}/pull/")
    ):
        try:
            return promote._pr_number(reference, upstream_repo)
        except ValueError:
            pass
    raise Diagnostic(
        f"Invalid upstream PR URL at {field(path)}.",
        f"In the PR description, use a full URL such as "
        f"`https://github.com/{upstream_repo}/pull/123` with the actual PR number "
        "and no query string or fragment.",
    )


def parse(body: str, vendor_name: str, upstream_repo: str) -> dict:
    """Read exactly one marked block and validate the selected vendor's inputs."""
    if len(body) > 65536:
        raise Diagnostic(
            "PR description exceeds the supported metadata size (65,536 characters).",
            "Shorten the PR description to at most 65,536 characters, keeping the complete "
            "vendor-promotion block. Move long logs to an attachment or a comment.",
        )
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    counts = {BEGIN: body.count(BEGIN), END: body.count(END)}
    missing = [marker for marker, count in counts.items() if count == 0]
    duplicates = [marker for marker, count in counts.items() if count > 1]
    full_example = f"````markdown\n{template(vendor_name, upstream_repo)}\n````"
    if missing or duplicates:
        problems, actions = [], []
        if missing:
            problems.append(
                "Missing vendor-promotion markers in the PR description: "
                + ", ".join(f"`{marker}`" for marker in missing)
                + "."
            )
            if len(missing) == 2:
                actions.append(
                    "Edit this PR's description—not a comment—and insert the complete block "
                    "below. Replace the example upstream PR URL with your actual PR URL. "
                    "If there is genuinely no paired upstream PR, replace `upstream_prs` "
                    "with a nonempty `unpaired_reason` explaining why."
                )
            else:
                position = "before" if missing[0] == BEGIN else "after"
                actions.append(
                    f"Edit this PR's description—not a comment—and insert `{missing[0]}` "
                    f"on its own line {position} the existing fenced YAML block. "
                    "Keep the existing metadata and the other marker."
                )
        if duplicates:
            problems.append(
                "Duplicate vendor-promotion markers: "
                + ", ".join(f"`{marker}` appears {counts[marker]} times" for marker in duplicates)
                + "."
            )
            actions.append(
                "Consolidate all vendor entries into one `vendors` mapping in one marked "
                "block in the PR description. Remove extra markers/blocks, preserving the "
                "other vendors' metadata; keep one start marker and one end marker."
            )
        raise Diagnostic(
            " ".join(problems),
            " ".join(actions),
            example=full_example if len(missing) == 2 else "",
        )
    start, end = body.index(BEGIN) + len(BEGIN), body.index(END)
    if start >= end:
        raise Diagnostic(
            "The vendor-promotion end marker appears before the start marker.",
            f"Move `{BEGIN}` before the fenced YAML block and `{END}` after it "
            "in the PR description. Preserve the metadata between them.",
        )
    block = body[start:end].strip()
    if not block.startswith("```yaml\n") or not block.endswith("\n```"):
        raise Diagnostic(
            "The marked block must contain one fenced yaml mapping.",
            "In the PR description, put the metadata between an opening line of three "
            "backticks followed by `yaml` and a closing line of three backticks. "
            "Keep both markers outside that fence, as shown below; retain your real PR URLs.",
            example=full_example,
        )
    try:
        content = block[len("```yaml\n") : -len("\n```")]
        if any(
            isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for token in yaml.scan(content)
        ):
            raise Diagnostic(
                "Promotion YAML aliases and anchors are not allowed.",
                "In the PR description's metadata block, replace YAML anchors (`&name`) "
                "and aliases (`*name`) with explicit values.",
            )
        data = yaml.load(content, Loader=manage._UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise Diagnostic(
            f"Invalid promotion YAML: {yaml_problem(error)}.",
            "Correct the metadata block in the PR description at the location above "
            "(line 1 is the first line inside the YAML fence). Use space indentation, "
            "balanced brackets/quotes, and unique mapping keys.",
        ) from error
    if not isinstance(data, dict):
        raise Diagnostic(
            "Promotion YAML must be a mapping, not a list or scalar.",
            "Use top-level `schema_version: 1` and `vendors:` fields in the PR description, "
            "following this structure. Replace the example URL.",
            example=full_example,
        )
    missing_keys = {"schema_version", "vendors"} - set(data)
    extra_keys = set(data) - {"schema_version", "vendors"}
    if missing_keys or extra_keys:
        raise Diagnostic(
            "Invalid top-level promotion YAML fields. "
            + (
                "Missing: " + ", ".join(field(key) for key in sorted(missing_keys)) + ". "
                if missing_keys
                else ""
            )
            + (
                "Unsupported: " + ", ".join(field(key) for key in sorted(extra_keys, key=str)) + "."
                if extra_keys
                else ""
            ),
            "In the PR description, keep only top-level `schema_version: 1` and `vendors:`; "
            "put vendor-specific fields under `vendors.<vendor-name>`.",
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise Diagnostic(
            "Unsupported promotion schema_version; expected integer 1.",
            "Set `schema_version: 1` (without quotes) in the PR description's metadata block.",
        )
    entries = data["vendors"]
    if not isinstance(entries, dict) or vendor_name not in entries:
        raise Diagnostic(
            f"Promotion YAML must include vendors.{vendor_name} as a mapping entry.",
            f"Add `{vendor_name}:` under `vendors:` in the existing PR-description block. "
            "Preserve any other vendor entries and replace the example URL.",
            example=_vendor_example(
                vendor_name,
                f"    upstream_prs:\n      - https://github.com/{upstream_repo}/pull/123",
            ),
        )
    invalid_names = [
        name
        for name in entries
        if not isinstance(name, str) or not manage._NAME_PATTERN.fullmatch(name)
    ]
    if invalid_names:
        raise Diagnostic(
            "Invalid vendor keys under `vendors`: "
            + ", ".join(field(name) for name in invalid_names)
            + ".",
            "Use the exact vendor names from `3rdparty/vendor_sources.lock.yaml` as "
            "string keys under `vendors` in the PR description.",
        )
    entry = entries[vendor_name]
    path = f"vendors.{vendor_name}"
    if not isinstance(entry, dict) or set(entry) - {
        "upstream_prs",
        "unpaired_reason",
        "resolutions",
    }:
        extra = (
            set(entry) - {"upstream_prs", "unpaired_reason", "resolutions"}
            if isinstance(entry, dict)
            else set()
        )
        raise Diagnostic(
            f"Invalid vendor metadata at {field(path)}. "
            + (
                "Unsupported fields: "
                + ", ".join(field(key) for key in sorted(extra, key=str))
                + "."
                if extra
                else "Expected a mapping."
            ),
            "Use only `upstream_prs`, `unpaired_reason`, and `resolutions` under this vendor "
            "in the PR description. Repository/operator settings do not belong here.",
        )
    if "unpaired_reason" in entry:
        if (
            set(entry) != {"unpaired_reason"}
            or not isinstance(entry["unpaired_reason"], str)
            or not entry["unpaired_reason"].strip()
        ):
            raise Diagnostic(
                f"Invalid {field(path + '.unpaired_reason')}: it must be nonempty and "
                "cannot accompany `upstream_prs` or `resolutions`.",
                "If there genuinely is no paired upstream PR, keep only `unpaired_reason` "
                "with an explanation under this vendor. Otherwise remove `unpaired_reason` "
                "and list the actual upstream PRs. Do not use it to bypass failed matching.",
            )
        return {
            "upstream_prs": [],
            "resolutions": {},
            "unpaired_reason": entry["unpaired_reason"].strip(),
        }
    references = entry.get("upstream_prs")
    if not isinstance(references, list) or not references:
        raise Diagnostic(
            f"{field(path + '.upstream_prs')} must be a nonempty list of upstream PR URLs.",
            "In the PR description, list every related upstream PR using YAML list items "
            "(`- URL`); replace the example URL. If there genuinely is no paired PR, use "
            "only a nonempty `unpaired_reason` instead.",
            example=_vendor_example(
                vendor_name,
                f"    upstream_prs:\n      - https://github.com/{upstream_repo}/pull/123",
            ),
        )
    numbers = []
    for index, reference in enumerate(references):
        number = _pr_reference(reference, upstream_repo, f"{path}.upstream_prs[{index}]")
        if number in numbers:
            raise Diagnostic(
                f"{field(path + '.upstream_prs')} contains duplicate upstream PR #{number}.",
                f"Remove the repeated `https://github.com/{upstream_repo}/pull/{number}` "
                "entry from the PR description; keep each upstream PR once.",
            )
        numbers.append(number)
    resolutions = entry.get("resolutions", {})
    if not isinstance(resolutions, dict):
        raise Diagnostic(
            f"{field(path + '.resolutions')} must be a mapping keyed by full source commit SHA.",
            "Replace it with the structure below inside the existing vendor entry. Use a "
            "full 40-character source SHA, a listed upstream PR, and a specific explanation.",
            example=_resolution_example(vendor_name, upstream_repo),
        )
    normalized = {}
    for sha, resolution in resolutions.items():
        resolution_path = f"{path}.resolutions.{sha}"
        try:
            promote._sha(sha)
        except ValueError as error:
            raise Diagnostic(
                f"Invalid source commit key at {field(resolution_path)}.",
                "Use the full lowercase 40-character SHA from the bot's unresolved-commit "
                "list, not a short SHA, branch name, or upstream commit SHA.",
            ) from error
        if not isinstance(resolution, dict) or set(resolution) != {"upstream_pr", "reason"}:
            details = "Expected a mapping."
            if isinstance(resolution, dict):
                absent = {"upstream_pr", "reason"} - set(resolution)
                unknown = set(resolution) - {"upstream_pr", "reason"}
                details = (
                    "Missing: " + ", ".join(field(key) for key in sorted(absent)) + ". "
                    if absent
                    else ""
                ) + (
                    "Unsupported: "
                    + ", ".join(field(key) for key in sorted(unknown, key=str))
                    + "."
                    if unknown
                    else ""
                )
            raise Diagnostic(
                f"{field(resolution_path)} requires exactly `upstream_pr` and `reason`. {details}",
                "Add the missing fields or remove extra fields in this resolution, "
                "following the structure below. Preserve other vendor metadata.",
                example=_resolution_example(vendor_name, upstream_repo),
            )
        reference, reason = resolution["upstream_pr"], resolution["reason"]
        number = _pr_reference(
            reference, upstream_repo, f"{resolution_path}.upstream_pr", require_url=False
        )
        if number not in numbers:
            raise Diagnostic(
                f"{field(resolution_path)} names upstream PR #{number}, which is not listed.",
                f"Add `https://github.com/{upstream_repo}/pull/{number}` to "
                f"`{path}.upstream_prs` if it owns this change, or correct the resolution "
                "to reference a listed PR.",
            )
        if not isinstance(reason, str) or not reason.strip():
            raise Diagnostic(
                f"{field(resolution_path + '.reason')} must be a nonempty string.",
                "Explain the downstream adaptation or why this upstream PR owns the "
                "unmatched change. Author assertions are retained until separately verified.",
            )
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
    vendor_name: str = "VENDOR_NAME",
) -> tuple[dict[str, int | None], dict[str, dict]]:
    """Match a linear source range against fetched immutable upstream PR snapshots.

    Only lock-selected files are compared. None assignments record commits with
    no selected changes, not upstream equivalence for the rest of those commits.
    Ambiguities remain author-actionable; titles and cherry-pick trailers are not proof.
    """
    commits = _commits(repo, previous, reviewed)
    if not commits:
        raise Diagnostic(
            "Source update has no new commits to attribute.",
            "Check the source SHA in `3rdparty/vendor_sources.lock.yaml`. Pin the intended "
            "new source commit, or remove the unnecessary source update from this PR.",
        )
    source_parents = {}
    for sha in commits:
        parents = promote._git(repo, "show", "-s", "--format=%P", sha).split()
        if len(parents) != 1:
            raise Diagnostic(
                f"Source history must be linear; commit `{sha}` has {len(parents)} parents. "
                "Temporary-fork updates require a linear range for commit-by-commit upstream "
                "attribution; containing the dev tip alone is not sufficient.",
                "Fetch the latest canonical dev branch and rebase your temporary branch's "
                "changes onto its tip without merge commits, or cherry-pick the intended "
                "changes onto a fresh branch from that tip. Do not rewrite canonical history "
                f"or assume the consumer lock pin `{previous}` is still the latest dev tip. "
                "Test and push the source branch, then use `scripts/vendor/manage.py` to "
                "update and verify the lock and vendored snapshot at the new source SHA. "
                "Push the consumer PR. Editing the upstream PR list or `resolutions` alone "
                "cannot remove a merge commit.",
            )
        source_parents[sha] = parents[0]
    resolutions = resolutions or {}
    if set(resolutions) - set(commits):
        raise Diagnostic(
            "Resolutions include commits outside the reviewed source delta: "
            + ", ".join(field(sha) for sha in sorted(set(resolutions) - set(commits)))
            + ".",
            f"Remove those stale entries from `vendors.{vendor_name}.resolutions` in the PR "
            "description. After rebasing, use the new source SHAs reported by the bot.",
        )
    candidates: dict[str, set[int]] = {sha: set() for sha in commits}
    patches = {sha: _patch(repo, f"{sha}^", sha, source=source, include=include) for sha in commits}
    if any(not patches[sha] for sha in resolutions):
        raise Diagnostic(
            "Remove resolutions for commits with no lock-selected file changes: "
            + ", ".join(f"`{sha}`" for sha in resolutions if not patches[sha])
            + ".",
            f"Delete only these entries from `vendors.{vendor_name}.resolutions`. "
            "Excluded-only changes do not need upstream attribution.",
        )
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
                raise Diagnostic(
                    f"Resolution for `{sha}` names unlisted upstream PR #{number}.",
                    f"Add that PR to `vendors.{vendor_name}.upstream_prs` if it owns this "
                    "change, or correct the resolution to reference a listed PR.",
                )
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
                ", ".join(
                    _upstream_reference(number, upstream_repo) for number in sorted(candidates[sha])
                )
                or "no exact match"
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
        raise Diagnostic(
            "Upstream attribution is unresolved for these source commits:\n\n"
            + "\n".join(unresolved)
            + ("\n\n" + accepted if accepted else ""),
            f"In the PR description, add missing PRs to `vendors.{vendor_name}.upstream_prs`, "
            "remove incorrectly listed PRs, or correct/rebase the source pin and update the "
            "vendored snapshot. For an intentional adaptation or ambiguous ownership, merge "
            "a targeted `resolutions` entry into the existing vendor metadata as shown below. "
            "Replace the SHA and URL and give a specific reason; preserve other entries. "
            "No complete commit_map is required. A resolution records an author assertion, "
            "not verified equivalence; it is retained during refresh until separately verified.",
            example=_resolution_example(vendor_name, upstream_repo or "OWNER/REPO"),
        )
    if set(assignments.values()) - {None} != set(upstream_prs):
        unused = set(upstream_prs) - set(assignments.values())
        raise Diagnostic(
            "Some listed upstream PRs cover no new lock-selected changes: "
            + ", ".join(_upstream_reference(number, upstream_repo) for number in sorted(unused))
            + "."
            + ("\n\n" + accepted if accepted else ""),
            f"Remove these unrelated PRs from `vendors.{vendor_name}.upstream_prs` in the "
            "PR description. PRs covering only excluded tests/docs do not need to be listed.",
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
    promote._check_selection(previous, reviewed)
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "HEAD").exists():
        promote._git(cache, "init", "--bare", "--quiet")
    if promote._git(cache, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("Matching cache must be a dedicated bare repository.")
    source_repo = promote._repo_name(reviewed.url)
    try:
        promote._check_source_repo(gh, source_repo)
    except Diagnostic as error:
        raise Diagnostic(
            f"The source repository {field(source_repo)} is not a fork of {field(gh.upstream_repo)}.",
            "Push the temporary source branch to a fork of the configured upstream and "
            "update the lock URL, branch, SHA, and vendored snapshot with "
            "`scripts/vendor/manage.py`.",
        ) from error
    for sha in (previous.commit, reviewed.commit):
        promote._git(cache, "fetch", "--quiet", "--no-tags", reviewed.url, sha)
    # An empty successful query means unrelated histories; Git failures remain
    # operator errors rather than being relabeled as author mistakes.
    common = promote._git(
        cache, "merge-base", previous.commit, reviewed.commit, allowed_returncodes=(0, 1)
    )
    rebase_action = (
        f"Fetch canonical repository {field(previous.url)}, branch {field(previous.branch)}, "
        f"and rebase the temporary source branch {field(reviewed.branch)} onto its latest "
        f"tip without merge commits, not just the consumer lock's pin `{previous.commit}`. "
        "Then use `scripts/vendor/manage.py` to update "
        "`3rdparty/vendor_sources.lock.yaml` and the vendored snapshot to the rebased SHA, "
        "and push the source branch and consumer PR. Editing metadata alone will not fix this."
    )
    if not common:
        raise Diagnostic("Source revisions have no merge base.", rebase_action)
    if common != previous.commit:
        raise Diagnostic("Source update is not a fast-forward.", rebase_action)
    if entry["unpaired_reason"]:
        commits = _commits(cache, previous.commit, reviewed.commit)
        if not commits or any(
            len(promote._git(cache, "show", "-s", "--format=%P", sha).split()) != 1
            for sha in commits
        ):
            raise Diagnostic(
                "Unpaired source updates still require a nonempty linear history.", rebase_action
            )
        return {}, {}
    prs = {}
    for number in entry["upstream_prs"]:
        pr = gh.get(f"repos/{gh.upstream_repo}/pulls/{number}")
        if pr["base"]["ref"] != gh.upstream_branch or (
            pr["state"] == "closed" and not pr.get("merged")
        ):
            raise Diagnostic(
                f"{_upstream_reference(number, gh.upstream_repo)} must target "
                f"`{gh.upstream_branch}` and be open or merged; currently targets "
                f"{field(pr['base']['ref'])} and is "
                + (
                    "closed without merging."
                    if pr["state"] == "closed" and not pr.get("merged")
                    else f"{field(pr['state'])}."
                ),
                "Correct the upstream PR target or reopen the upstream PR if appropriate; "
                "otherwise replace its URL in the consumer PR description with the correct "
                "open/merged upstream PR. Do not remove attribution for changes still included.",
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
        vendor_name=gh.vendor_name,
    )
