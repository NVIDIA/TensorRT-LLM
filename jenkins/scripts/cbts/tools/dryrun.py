#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
r"""CBTS dry-run debug tool.

Replays CBTS over historical commits and writes per-PR artifacts (summary
+ filtered test-db YAMLs) plus a top-level INDEX.md, so the rule set can
be inspected without firing real CI.

Each commit is replayed with its own snapshot of
`tests/integration/test_lists/test-db/` and `jenkins/L0_Test.groovy`
(extracted via `git show`), so the result reflects what CBTS would have
decided at the time the PR landed.

Examples:
--------
Reproduce `cbts_dryrun/` (tests-only PRs in the last 500 commits)::

    python3 jenkins/scripts/cbts/tools/dryrun.py \\
        --window 500 --filter tests-only --out cbts_dryrun

Reproduce `cbts_dryrun_recent10/` (last 10 commits, any kind)::

    python3 jenkins/scripts/cbts/tools/dryrun.py \\
        --window 10 --filter all --out cbts_dryrun_recent10

Single commit by SHA (debug a specific PR)::

    python3 jenkins/scripts/cbts/tools/dryrun.py \\
        --sha 219559c --out /tmp/cbts_one
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

THIS_FILE = Path(__file__).resolve()
CBTS_DIR = THIS_FILE.parent.parent  # jenkins/scripts/cbts
DEFAULT_REPO_ROOT = CBTS_DIR.parents[2]  # repo root
CBTS_MAIN = CBTS_DIR / "main.py"

TEST_DB_REL = "tests/integration/test_lists/test-db"
GROOVY_REL = "jenkins/L0_Test.groovy"
PR_RE = re.compile(r"\(#(\d+)\)\s*$")


# --- git helpers ------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=check,
    )


def _list_changed_files(repo: Path, sha: str) -> list[str]:
    out = _git(repo, "show", "--name-only", "--pretty=format:", sha).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _file_diff(repo: Path, sha: str, path: str) -> str:
    return _git(repo, "diff", f"{sha}^", sha, "--", path, check=False).stdout


def _resolve_pr(subject: str, sha: str) -> tuple[str, str]:
    m = PR_RE.search(subject)
    if m:
        pr = m.group(1)
        return f"pr-{pr}", f"https://github.com/NVIDIA/TensorRT-LLM/pull/{pr}"
    return f"sha-{sha[:8]}", f"(no PR number; sha={sha})"


# --- run main.py ------------------------------------------------------------


def _run_cbts(payload: dict, test_db: Path, groovy: Path, repo: Path) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        res = subprocess.run(
            [
                sys.executable,
                str(CBTS_MAIN),
                path,
                "--repo-root",
                str(repo),
                "--test-db",
                str(test_db),
                "--groovy-file",
                str(groovy),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(path)
    if res.returncode != 0:
        return {"_error": res.stderr.strip(), "_returncode": res.returncode}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {"_error": "non-json stdout", "_stdout": res.stdout[:1000]}


# --- formatting -------------------------------------------------------------


def _fmt_summary(
    pr_url: str,
    sha: str,
    subject: str,
    files: list[str],
    result: dict,
    post_merge: bool,
    tests_only: bool,
) -> str:
    trigger = "/bot run --post-merge" if post_merge else "/bot run"
    lines = [
        f"PR:                {pr_url}",
        f"Subject:           {subject}",
        f"SHA:               {sha}",
        f"Trigger:           {trigger} (post_merge={post_merge})",
        f"Tests-only:        {tests_only}",
        f"Files changed ({len(files)}):",
    ]
    lines.extend(f"  - {f}" for f in files)
    lines.append("")
    if "_error" in result:
        lines.append(f"ERROR: {result['_error']}")
        return "\n".join(lines) + "\n"
    scope = result.get("scope")
    lines.append(f"scope:                {scope!r}")
    lines.append(f"sanity_required:      {result.get('sanity_required')}")
    lines.append(f"perfsanity_required:  {result.get('perfsanity_required')}")
    override = result.get("test_db_dir_override")
    lines.append(f"test_db_dir_override: {override!r}")
    stages = result.get("affected_stages", [])
    lines.append(f"affected_stages ({len(stages)}):")
    lines.extend(f"  - {s}" for s in stages)
    lines.append("reasons:")
    lines.extend(f"  - {r}" for r in result.get("reasons", []))
    return "\n".join(lines) + "\n"


def _index_row(label: str, pr_url: str, files: list[str], subject: str, result: dict) -> str:
    if "_error" in result:
        return (
            f"| [{label}](./{label}/) | `ERROR` | - | - | - | - | {len(files)} | "
            f"[link]({pr_url}) | {subject[:70]} |"
        )
    scope = result.get("scope")
    scope_disp = f"`{scope}`" if scope else "`None`"
    n_stages = len(result.get("affected_stages", []))
    n_yamls = len(result.get("affected_stage_test_counts", {})) or n_stages
    sanity = "Y" if result.get("sanity_required") else "n"
    perf = "Y" if result.get("perfsanity_required") else "n"
    short = subject[:70] + ("…" if len(subject) > 70 else "")
    return (
        f"| [{label}](./{label}/) | {scope_disp} | {n_stages} | {n_yamls} | "
        f"{sanity} | {perf} | {len(files)} | [link]({pr_url}) | {short} |"
    )


# --- commit discovery -------------------------------------------------------


def _walk_commits(repo: Path, ref: str, window: int) -> list[str]:
    out = _git(repo, "log", ref, f"-{window}", "--pretty=format:%H").stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _is_tests_only(files: list[str]) -> bool:
    return bool(files) and all(f.startswith("tests/") for f in files)


# --- main loop --------------------------------------------------------------


def _replay_one(
    repo: Path,
    sha: str,
    out_dir: Path,
    post_merge: bool,
) -> tuple[str, str, list[str], str, dict, bool]:
    subject = _git(repo, "log", "-1", "--pretty=%s", sha).stdout.strip()
    label, pr_url = _resolve_pr(subject, sha)
    files = _list_changed_files(repo, sha)
    tests_only = _is_tests_only(files)

    pr_dir = out_dir / label
    pr_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale artifacts from a previous run before writing fresh ones.
    for old in pr_dir.iterdir():
        if old.name == "summary.txt":
            continue
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()

    if not files:
        result: dict = {"_error": "commit touched no files"}
    else:
        diffs = {f: _file_diff(repo, sha, f) for f in files}
        payload = {"changed_files": files, "diffs": diffs, "post_merge": post_merge}
        with tempfile.TemporaryDirectory() as td:
            wt = Path(td) / "wt"
            # Materialize the SHA's tree as a detached worktree so every
            # path CBTS reads (post-PR YAML, .py source, accuracy refs)
            # is the state at PR-decision time, not whatever HEAD is now.
            _git(repo, "worktree", "add", "--detach", str(wt), sha)
            try:
                test_db = wt / TEST_DB_REL
                groovy = wt / GROOVY_REL
                shared_out = wt / "cbts_test_db"
                result = _run_cbts(payload, test_db, groovy, wt)
                # Copy filtered test-db YAMLs out before the worktree
                # is torn down in the finally block.
                if shared_out.exists():
                    for yml in shared_out.glob("*.yml"):
                        shutil.copy2(yml, pr_dir / yml.name)
            finally:
                _git(repo, "worktree", "remove", "--force", str(wt), check=False)

    (pr_dir / "summary.txt").write_text(
        _fmt_summary(pr_url, sha, subject, files, result, post_merge, tests_only)
    )
    return label, pr_url, files, subject, result, tests_only


def _write_index(
    out_dir: Path,
    rows: list[tuple[str, str, list[str], str, dict, bool]],
    ref: str,
    window: int,
    filter_mode: str,
    post_merge: bool,
) -> None:
    counts: dict[Optional[str], int] = {}
    for _, _, _, _, result, _ in rows:
        scope = result.get("scope") if "_error" not in result else "ERROR"
        counts[scope] = counts.get(scope, 0) + 1

    trigger = "/bot run --post-merge" if post_merge else "/bot run"
    lines = [
        "# CBTS dry-run results",
        "",
        f"Source ref: `{ref}` (last {window} commits), simulated as "
        f"`{trigger}` (post_merge={post_merge}).",
        f"Filter: `{filter_mode}` — replayed commits: **{len(rows)}**",
        "",
        "| Label | Scope | Stages | YAMLs | Sanity | Perf | Files | PR | Subject |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def _sort_key(row):
        label = row[0]
        if label.startswith("pr-") and label[3:].isdigit():
            return (0, int(label[3:]))
        return (1, label)

    for label, pr_url, files, subject, result, _ in sorted(rows, key=_sort_key):
        lines.append(_index_row(label, pr_url, files, subject, result))
    lines += ["", "## Scope distribution", ""]
    for scope, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {scope or 'None (fallback)'}: {n}")
    (out_dir / "INDEX.md").write_text("\n".join(lines) + "\n")


def _positive_int(s: str) -> int:
    try:
        n = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected integer, got {s!r}") from e
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer (>= 1), got {n}")
    return n


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ref", default="upstream/main", help="git ref to walk (default: upstream/main)"
    )
    ap.add_argument(
        "--window",
        type=_positive_int,
        default=500,
        help="commits back from --ref, must be >= 1 (default: 500)",
    )
    ap.add_argument(
        "--filter",
        choices=["tests-only", "all"],
        default="tests-only",
        help="commit filter (default: tests-only)",
    )
    ap.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="cap commits after filter, must be >= 1 (default: no cap)",
    )
    ap.add_argument(
        "--sha",
        action="append",
        default=[],
        help="replay specific SHA (repeatable; ignores --ref/--window/--filter)",
    )
    ap.add_argument("--post-merge", action="store_true", help="set post_merge=True")
    ap.add_argument("--out", default="cbts_dryrun", help="output directory (default: cbts_dryrun)")
    ap.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help=f"TRT-LLM repo root (default: {DEFAULT_REPO_ROOT})",
    )
    ap.add_argument(
        "--keep-stale", action="store_true", help="don't wipe existing pr-*/sha-* dirs in --out"
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    if not (repo / GROOVY_REL).exists():
        print(f"error: --repo-root does not look like a TRT-LLM checkout: {repo}", file=sys.stderr)
        return 2
    if not CBTS_MAIN.is_file():
        print(f"error: cbts main.py not found at {CBTS_MAIN}", file=sys.stderr)
        return 2

    # Drop stale worktree metadata from interrupted prior runs so
    # `git worktree add` doesn't trip on dangling entries.
    _git(repo, "worktree", "prune", check=False)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_stale:
        for old in out_dir.iterdir():
            if old.is_dir() and old.name.startswith(("pr-", "sha-")):
                shutil.rmtree(old)

    if args.sha:
        shas = args.sha
        print(f"Replaying {len(shas)} explicit SHA(s)", file=sys.stderr)
    else:
        all_shas = _walk_commits(repo, args.ref, args.window)
        if args.filter == "tests-only":
            shas = [s for s in all_shas if _is_tests_only(_list_changed_files(repo, s))]
        else:
            shas = all_shas
        if args.limit is not None:
            shas = shas[: args.limit]
        suffix = f", limit={args.limit}" if args.limit else ""
        print(
            f"Window: last {args.window} commits on {args.ref} → "
            f"{len(shas)} commits after filter={args.filter}{suffix}",
            file=sys.stderr,
        )

    rows: list[tuple[str, str, list[str], str, dict, bool]] = []
    for sha in shas:
        try:
            row = _replay_one(repo, sha, out_dir, args.post_merge)
        except subprocess.CalledProcessError as e:
            print(f"  {sha[:8]}: git error: {e.stderr.strip()}", file=sys.stderr)
            continue
        rows.append(row)
        label, _, _, _, result, _ = row
        scope = result.get("scope", "ERROR") if "_error" not in result else "ERROR"
        print(f"  {label}: {scope}", file=sys.stderr)

    _write_index(out_dir, rows, args.ref, args.window, args.filter, args.post_merge)

    counts: dict[Optional[str], int] = {}
    for _, _, _, _, result, _ in rows:
        scope = result.get("scope") if "_error" not in result else "ERROR"
        counts[scope] = counts.get(scope, 0) + 1
    print(f"\nDone. Output: {out_dir}", file=sys.stderr)
    print(f"Scopes: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
