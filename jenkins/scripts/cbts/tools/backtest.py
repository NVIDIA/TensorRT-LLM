#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Replay current CBTS on a pinned historical cohort and a merged coverage DB.

Uses historical source/test inventories and freezes the working-tree policy,
including uncommitted edits. Results are retrospective candidate decisions,
not observed CI savings. Does not post telemetry or trigger CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import dryrun
from report_cbts_decision import _case_counts

sys.path.insert(0, str(dryrun.CBTS_DIR / "coverage_selection"))
import artifact  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _freeze_policy(out: Path, repo: Path, policy_ref: str | None) -> str:
    if policy_ref:
        prefix = dryrun.CBTS_DIR.relative_to(dryrun.DEFAULT_REPO_ROOT)
        paths = dryrun._git(repo, "ls-tree", "-r", "--name-only", policy_ref, "--", str(prefix))
        files = [repo / path for path in paths.stdout.splitlines() if path.endswith(".py")]
    else:
        files = sorted(dryrun.CBTS_DIR.rglob("*.py"))
    hashes = {}
    for source in files:
        relative = source.relative_to(
            repo / "jenkins/scripts/cbts" if policy_ref else dryrun.CBTS_DIR
        )
        destination = out / "policy" / "cbts" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if policy_ref:
            content = dryrun._git(repo, "show", f"{policy_ref}:{source.relative_to(repo)}").stdout
            destination.write_text(content)
        else:
            shutil.copy2(source, destination)
        hashes[relative.as_posix()] = _sha256(destination)
    _write_json(out / "policy_files.json", hashes)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _latest_pair() -> dict:
    newest = artifact.latest_build_number()
    if newest is None:
        raise ValueError("latest PostMerge build number could not be resolved")
    for build in range(newest, max(0, newest - artifact._MAX_PROBE), -1):
        urls = artifact.tarball_urls(build)
        if all(artifact._exists(url) for url in urls):
            commit = artifact.build_commit(build)
            if commit:
                return {
                    "build": build,
                    "commit": commit,
                    "urls": urls,
                    "latest_observed_build": newest,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
    raise ValueError("latest complete architecture pair could not be resolved")


def _coverage(args: argparse.Namespace, repo: Path, out: Path) -> tuple[Path, dict]:
    if args.coverage_db:
        database = Path(args.coverage_db).resolve()
        if not database.is_file():
            raise ValueError(f"coverage DB missing: {database}")
        if not args.coverage_meta:
            raise ValueError("--coverage-db requires --coverage-meta with build/commit/urls")
        metadata = json.loads(Path(args.coverage_meta).read_text())
        if not metadata.get("build") or not metadata.get("commit"):
            raise ValueError("coverage metadata must identify build and commit")
        expected = metadata.get("sha256")
        if expected and expected != _sha256(database):
            raise ValueError("coverage DB hash differs from metadata")
    else:
        metadata = _latest_pair()
        destination = out / "coverage"
        destination.mkdir()
        sources = []
        for index, url in enumerate(metadata["urls"]):
            tarball = artifact.download(url, destination)
            unpacked = destination / str(index)
            unpacked.mkdir()
            if tarball is None or not artifact.extract(tarball, unpacked):
                raise ValueError(f"coverage download/extraction failed: {url}")
            sources.append(unpacked / artifact.DB_NAME)
        database = destination / artifact.DB_NAME
        artifact.merge_databases(sources, database).close()
    metadata = dict(metadata, sha256=_sha256(database), path=str(database))
    _write_json(out / "coverage.json", metadata)
    return database, metadata


def _replay(
    repo: Path,
    sha: str,
    out: Path,
    database: Path,
    metadata: dict,
    post_merge: bool,
    check_compatibility: bool,
) -> dict:
    subject = dryrun._git(repo, "log", "-1", "--pretty=%s", sha).stdout.strip()
    label, pr_url = dryrun._resolve_pr(subject, sha)
    directory = out / "commits" / sha
    directory.mkdir(parents=True)
    files = dryrun._git(repo, "diff", "--name-only", f"{sha}^", sha).stdout.splitlines()
    diffs = {path: dryrun._file_diff(repo, sha, path) for path in files}
    payload = {"changed_files": files, "diffs": diffs, "post_merge": post_merge}
    _write_json(directory / "input.json", payload)
    status = "post_merge" if post_merge else "pre_merge"
    row = {
        "sha": sha,
        "label": label,
        "pr_url": pr_url,
        "subject": subject,
        "core_python": any(p.startswith("tensorrt_llm/") and p.endswith(".py") for p in files),
    }
    with tempfile.TemporaryDirectory(prefix="cbts_backtest_") as temporary:
        snapshot = Path(temporary) / "wt"
        dryrun._git(repo, "worktree", "add", "--detach", str(snapshot), sha)
        try:
            result = dryrun._run_cbts(
                payload,
                snapshot / dryrun.TEST_DB_REL,
                snapshot / dryrun.GROOVY_REL,
                snapshot,
                str(database),
            )
            _write_json(directory / "decision.json", result)
            (directory / "summary.txt").write_text(
                dryrun._fmt_summary(
                    pr_url, sha, subject, files, result, post_merge, dryrun._is_tests_only(files)
                )
            )
            row["scope"] = result.get("scope")
            row["decline_category"] = result.get("coverage_decline_category") or "none"
            row["decline_reason"] = result.get("coverage_decline_reason") or ""
            row["error"] = result.get("_error")
            row["compatibility"] = "not_checked"
            if result.get("scope") == "coverage" and check_compatibility:
                base = dryrun._git(repo, "rev-parse", f"{sha}^").stdout.strip()
                row["compatibility"] = artifact._patch_apply_status(
                    base,
                    sha,
                    metadata["commit"],
                    repo_root=snapshot,
                    upstream_url=str(repo),
                    relevant_paths=result["coverage_residual_files"],
                )
            row["gated_scope"] = (
                None if row["compatibility"] in {"conflict", "unknown"} else row["scope"]
            )
            baseline = dict(
                result, affected_stages=[], sanity_required=True, perfsanity_required=True
            )
            _, total = _case_counts(baseline, status, str(snapshot), post_merge)
            if row["scope"] and not row["error"]:
                kept, counted_total = _case_counts(result, status, str(snapshot), post_merge)
                if total != counted_total or not 0 < total or not 0 <= kept <= total:
                    raise ValueError(
                        f"invalid case counts: {kept}/{counted_total}, baseline={total}"
                    )
            else:
                kept = total
            row.update(kept_cases=kept, total_cases=total, skipped_cases=total - kept)
            filtered = snapshot / "cbts_test_db"
            if filtered.is_dir():
                shutil.copytree(filtered, directory / "test-db")
            selection = {
                "stages": result.get("affected_stages", []),
                "counts": result.get("affected_stage_test_counts", {}),
                "test_db": {path.name: _sha256(path) for path in sorted(filtered.glob("*.yml"))},
            }
            row["selection_sha256"] = hashlib.sha256(
                json.dumps(selection, sort_keys=True).encode()
            ).hexdigest()
        finally:
            dryrun._git(repo, "worktree", "remove", "--force", str(snapshot), check=False)
    _write_json(directory / "metrics.json", row)
    return row


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    hits = [row for row in rows if row.get("scope") and not row.get("error")]
    errors = sum(bool(row.get("error")) for row in rows)
    baseline = sum(row.get("total_cases", 0) for row in rows)
    skipped = sum(row.get("skipped_cases", 0) for row in hits)
    gated_hits = sum(bool(row.get("gated_scope")) and not row.get("error") for row in rows)
    gated_skipped = sum(
        row.get("skipped_cases", 0)
        for row in rows
        if row.get("gated_scope") and not row.get("error")
    )
    return {
        "commits": count,
        "hits": len(hits),
        "hit_rate": len(hits) / count if count else 0,
        "tier1_hits": sum(row["scope"] != "coverage" for row in hits),
        "tier2_hits": sum(row["scope"] == "coverage" for row in hits),
        "effective_hits": sum(row.get("skipped_cases", 0) > 0 for row in hits),
        "conservative_no_skip": sum(row.get("skipped_cases", 0) == 0 for row in hits),
        "fallbacks": count - len(hits) - errors,
        "errors": errors,
        "compatibility_gated_hits": gated_hits,
        "compatibility_gated_hit_rate": gated_hits / count if count else 0,
        "baseline_case_entries": baseline,
        "skipped_case_entries": skipped,
        "weighted_case_skip_rate": skipped / baseline if baseline else None,
        "compatibility_gated_case_skip_rate": gated_skipped / baseline if baseline else None,
        "scopes": dict(
            Counter("ERROR" if row.get("error") else row.get("scope") or "fallback" for row in rows)
        ),
        "fallback_categories": dict(
            Counter(
                row.get("decline_category", "other")
                for row in rows
                if not row.get("scope") and not row.get("error")
            )
        ),
    }


def _compare(rows: list[dict], manifest: dict, previous: Path) -> dict:
    old_manifest = json.loads((previous / "manifest.json").read_text())
    for key in ("commits", "coverage_sha256", "post_merge", "check_compatibility"):
        if manifest[key] != old_manifest[key]:
            raise ValueError(f"comparison requires identical {key}; reuse the previous cohort/DB")
    old = {row["sha"]: row for row in json.loads((previous / "results.json").read_text())}
    changes = []
    for row in rows:
        prior = old[row["sha"]]
        if (
            prior.get("scope") != row.get("scope")
            or prior.get("skipped_cases") != row.get("skipped_cases")
            or prior.get("selection_sha256") != row.get("selection_sha256")
            or prior.get("gated_scope") != row.get("gated_scope")
        ):
            changes.append(
                {
                    "sha": row["sha"],
                    "label": row.get("label"),
                    "before_scope": prior.get("scope"),
                    "after_scope": row.get("scope"),
                    "before_skipped": prior.get("skipped_cases"),
                    "after_skipped": row.get("skipped_cases"),
                    "before_selection_sha256": prior.get("selection_sha256"),
                    "after_selection_sha256": row.get("selection_sha256"),
                }
            )
    return {
        "previous": str(previous),
        "before": _aggregate(list(old.values())),
        "after": _aggregate(rows),
        "changed_decisions": changes,
    }


def _report(out: Path, manifest: dict, rows: list[dict], summary: dict) -> None:
    stats = summary["overall"]
    cohort = (
        f"pinned {manifest.get('cohort_name', 'cohort')}, {len(rows)} commits"
        if manifest.get("cohort_source")
        else f"latest {len(rows)} first-parent commits at `{manifest['ref_sha']}`"
    )
    lines = [
        "# CBTS retrospective backtest",
        "",
        f"Coverage: build **{manifest['coverage_build']}**, `{manifest['coverage_commit']}`.",
        f"Policy: `{manifest['policy_sha256']}` (frozen working-tree Python files).",
        f"Cohort: {cohort}.",
        "Historical source, test-db, and stage inventory; current candidate policy.",
        "Latest coverage can postdate samples. These are retrospective candidate results,",
        "not historical CI measurements or proof of selection safety.",
        "",
        f"Hit rate: **{stats['hits']}/{stats['commits']} ({stats['hit_rate']:.1%})**; "
        f"Tier 1: {stats['tier1_hits']}; Tier 2: {stats['tier2_hits']}.",
        f"Fallbacks: {stats['fallbacks']}; errors: {stats['errors']} (included in denominator).",
        f"Effective hits: {stats['effective_hits']}; no-skip hits: {stats['conservative_no_skip']}.",
        "Case counts are YAML entries per stage family, not expanded pytest cases;",
        "pre-merge assumes the multi-GPU label gate is closed, post-merge includes multi-GPU.",
        "Coverage rollout/pilot eligibility is not simulated.",
        "",
    ]
    if stats["weighted_case_skip_rate"] is not None:
        lines.append(
            f"Weighted candidate case-entry skip: **{stats['weighted_case_skip_rate']:.1%}**."
        )
    if manifest["check_compatibility"]:
        lines.append(
            f"After residual patch compatibility: **{stats['compatibility_gated_hits']}/"
            f"{stats['commits']} ({stats['compatibility_gated_hit_rate']:.1%})**."
        )
    else:
        lines.append("Residual patch compatibility was not checked.")
    lines += ["", "## Fallback categories", "", "| Category | Count |", "|---|---:|"]
    lines += [
        f"| {name} | {count} |"
        for name, count in sorted(stats["fallback_categories"].items(), key=lambda item: -item[1])
    ]
    comparison_path = out / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text())
        before, after = comparison["before"], comparison["after"]
        lines += [
            "",
            "## Optimization comparison",
            "",
            f"Same cohort, DB and mode: **{before['hits']}/{before['commits']} "
            f"({before['hit_rate']:.1%}) → {after['hits']}/{after['commits']} "
            f"({after['hit_rate']:.1%})**.",
            f"Effective hits: {before['effective_hits']} → {after['effective_hits']}.",
            "",
            "| Commit | Before | After | Skipped entries before → after |",
            "|---|---|---|---:|",
        ]
        for change in comparison["changed_decisions"]:
            lines.append(
                f"| {change.get('label') or change['sha'][:10]} | "
                f"{change['before_scope'] or 'fallback'} | "
                f"{change['after_scope'] or 'fallback'} | "
                f"{change['before_skipped']} → {change['after_skipped']} |"
            )
    lines += [
        "",
        "## Commits",
        "",
        "| SHA | Scope | Compatibility | Skipped entries | Subject |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        sha = row["sha"]
        subject = row.get("subject", "").replace("|", "\\|")
        lines.append(
            f"| [{sha[:10]}](commits/{sha}/metrics.json) | "
            f"{row.get('scope') or ('ERROR' if row.get('error') else 'fallback')} | "
            f"{row.get('compatibility', 'not_checked')} | "
            f"{row.get('skipped_cases', 0)} | {subject} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(dryrun.DEFAULT_REPO_ROOT))
    parser.add_argument("--ref", default="upstream/main")
    parser.add_argument("--window", type=dryrun._positive_int, default=200)
    parser.add_argument("--cohort", help="previous manifest.json; reuse its exact commit list")
    parser.add_argument(
        "--policy-ref", help="evaluate committed CBTS at a ref instead of working tree"
    )
    parser.add_argument("--coverage-db", help="merged architecture DB; otherwise resolve latest")
    parser.add_argument("--coverage-meta", help="JSON provenance for an explicitly supplied DB")
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--jobs", type=dryrun._positive_int, default=4)
    parser.add_argument("--post-merge", action="store_true")
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="also check residual patch compatibility for coverage hits locally",
    )
    parser.add_argument("--compare", help="previous result directory; cohort/DB/mode must match")
    args = parser.parse_args(argv)
    repo, out = Path(args.repo_root).resolve(), Path(args.out).resolve()
    if out.exists():
        parser.error("--out must be a new directory to preserve prior runs")
    out.mkdir(parents=True)
    database, metadata = _coverage(args, repo, out)
    if args.cohort:
        previous = json.loads(Path(args.cohort).read_text())
        commits, ref_sha = previous["commits"], previous["ref_sha"]
    else:
        ref_sha = dryrun._git(repo, "rev-parse", args.ref).stdout.strip()
        commits = dryrun._git(
            repo, "rev-list", "--first-parent", f"--max-count={args.window}", ref_sha
        ).stdout.splitlines()
        if len(commits) != args.window:
            raise ValueError(f"expected {args.window} commits, found {len(commits)}")
    fingerprint = _freeze_policy(out, repo, args.policy_ref)
    dryrun.CBTS_MAIN = out / "policy/cbts/main.py"
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ref": args.ref,
        "ref_sha": ref_sha,
        "commits": commits,
        "policy_head": dryrun._git(repo, "rev-parse", "HEAD").stdout.strip(),
        "policy_sha256": fingerprint,
        "policy_ref": args.policy_ref or "working-tree",
        "coverage_build": metadata["build"],
        "coverage_commit": metadata["commit"],
        "coverage_sha256": metadata["sha256"],
        "post_merge": args.post_merge,
        "check_compatibility": args.check_compatibility,
    }
    if args.cohort:
        manifest["cohort_source"] = str(Path(args.cohort).resolve())
        manifest["cohort_name"] = previous.get("cohort_name", "cohort")
    _write_json(out / "manifest.json", manifest)
    (out / "working-tree.patch").write_text(dryrun._git(repo, "diff", "HEAD").stdout)
    rows_by_sha = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _replay,
                repo,
                sha,
                out,
                database,
                metadata,
                args.post_merge,
                args.check_compatibility,
            ): sha
            for sha in commits
        }
        for future in as_completed(futures):
            sha = futures[future]
            try:
                row = future.result()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                detail = (
                    error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
                )
                row = {"sha": sha, "error": detail, "scope": None}
                (out / "commits" / sha).mkdir(parents=True, exist_ok=True)
                _write_json(out / "commits" / sha / "error.json", row)
            rows_by_sha[sha] = row
            print(
                f"[{len(rows_by_sha)}/{len(commits)}] {sha[:10]} "
                f"{row.get('scope') or ('ERROR' if row.get('error') else 'fallback')}",
                flush=True,
            )
            _write_json(out / "results.json", [rows_by_sha[s] for s in commits if s in rows_by_sha])
    rows = [rows_by_sha[sha] for sha in commits]
    summary = {
        "overall": _aggregate(rows),
        "core_python": _aggregate([row for row in rows if row.get("core_python")]),
    }
    _write_json(out / "summary.json", summary)
    if args.compare:
        _write_json(out / "comparison.json", _compare(rows, manifest, Path(args.compare)))
    _report(out, manifest, rows, summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["overall"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
