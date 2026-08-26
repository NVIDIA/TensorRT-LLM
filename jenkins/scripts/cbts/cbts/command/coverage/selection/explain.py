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
r"""Explain a coverage-selection decision for one commit.

For each instrumented stage, prints why each known case is kept (it entered a
changed function), forced-kept (its capture is untrusted), or removed (it is in
the DB, entered no changed function, and its capture is trusted). The
justification is the forward touch lookup — the audit view that makes
`cbts_removed_cases.txt` self-verifying.

Mirrors `CoverageSelector.decide()`'s safety gates: a changed file with no DB
rows refuses the whole change, and untrusted tests are never removable.

Example::

    PYTHONPATH=jenkins/scripts/cbts python3 -m cbts.command coverage selection explain \\
        --db /tmp/cbts_inspect/cbts_touchmap.sqlite --sha 890e1089 --show-kept
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from cbts.coverage.selection.qualname_map import qualnames_for_lines
from cbts.coverage.selection.selector import CoverageSelector
from cbts.coverage.selection.touch_db import TouchDB, canon, stage_family
from cbts.rules._helpers import iter_diff_post_line_numbers

# jenkins/scripts/cbts/cbts/command/coverage/selection/explain.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[7]


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=check
    ).stdout


def _src_at(repo: Path, sha: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=str(repo), capture_output=True, text=True, check=False
    )
    return r.stdout if r.returncode == 0 else None


@click.command("explain", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--db", required=True)
@click.option("--sha", required=True)
@click.option("--repo-root", default=str(_REPO_ROOT))
@click.option("--stage", default=None, help="limit to one stage")
@click.option("--show-kept", is_flag=True, help="also list kept (impacted) cases")
def main(db, sha, repo_root, stage, show_kept):
    """Explain a coverage-selection decision for one commit."""
    args = _Args(db, sha, repo_root, stage, show_kept)
    _run(args)


class _Args:
    def __init__(self, db, sha, repo_root, stage, show_kept):
        self.db = db
        self.sha = sha
        self.repo_root = repo_root
        self.stage = stage
        self.show_kept = show_kept


def _run(args: _Args) -> int:
    repo = Path(args.repo_root).resolve()
    db = TouchDB.open(args.db)

    files = [
        ln
        for ln in _git(repo, "show", "--name-only", "--pretty=format:", args.sha).splitlines()
        if ln.strip()
    ]
    core = [f for f in files if f.endswith(".py") and canon(f).startswith("tensorrt_llm/")]
    non_core = [f for f in files if f not in core]

    # Delegate the decision itself so the gates cannot drift from the selector.
    diffs = {f: _git(repo, "diff", f"{args.sha}^", args.sha, "--", f, check=False) for f in core}
    selector = CoverageSelector(db, repo, read_source=lambda f: _src_at(repo, args.sha, f))
    res = selector.decide(core, diffs)
    if not res.ok:
        print(f"commit {args.sha[:12]} — coverage selection REFUSES this change:")
        print(f"  {res.reason}")
        print("\nNo case is removable; every stage runs in full.")
        return 0

    # Forward lookup for the per-case justification the selector does not return.
    impact_funcs: set[tuple[str, str]] = set()
    impact_files: set[str] = set()
    changed_files: set[str] = set()
    for f in core:
        cf = canon(f)
        changed_files.add(cf)
        lines = iter_diff_post_line_numbers(diffs.get(f, ""))
        src = _src_at(repo, args.sha, f)
        qns, ok = qualnames_for_lines(src, lines) if (lines and src) else (set(), False)
        if not ok:
            impact_files.add(cf)
            continue
        for q in qns:
            if db.tests_touching_func(cf, q):
                impact_funcs.add((cf, q))
            else:
                impact_files.add(cf)
    no_data = res.no_data_funcs

    print(f"commit {args.sha[:12]} — {len(core)} core file(s), impact set:")
    for cf, q in sorted(impact_funcs):
        print(f"  {cf} :: {q}")
    for cf in sorted(impact_files):
        print(f"  {cf} :: <file-level>")
    if no_data:
        print(f"  ({len(no_data)} changed function(s) with no DB rows -> file-level fallback)")
        for s in sorted(no_data):
            print(f"    no-data: {s}")
    if non_core:
        print(
            f"\n  NOTE: {len(non_core)} non-core file(s) in this commit are not evaluated here. "
            "Tier-1 rules claim them; any left as residual makes the coverage tier refuse."
        )
        for f in sorted(non_core):
            print(f"    {f}")

    untrusted_fam = selector.untrusted_families()

    def entered_changed(nodeid: str, stage: str) -> tuple[int, int, list[str]]:
        """(total rows, funcs entered in changed files, changed qualnames entered)."""
        touched = db.files_touched_by(f"{stage}/{nodeid}")
        in_changed = sum(1 for f, _ in touched if f in changed_files)
        hits = [f"{f.rsplit('/', 1)[-1]}::{q}" for f, q in touched if (f, q) in impact_funcs]
        hits += [f"{f.rsplit('/', 1)[-1]}::<file>" for f, q in touched if f in impact_files]
        return len(touched), in_changed, sorted(set(hits))

    for stage in sorted(db.known_by_stage()):
        if args.stage and stage != args.stage:
            continue
        known_s = db.known_by_stage()[stage]
        fam = stage_family(stage)
        imp_s = res.impacted.get(fam, set()) & known_s
        forced_s = {n for n in known_s - imp_s if f"{fam}/{n}" in untrusted_fam}
        skip_s = known_s - imp_s - forced_s
        print(
            f"\n=== {stage}  known={len(known_s)}  kept={len(imp_s)}  "
            f"forced={len(forced_s)}  removed={len(skip_s)} ==="
        )
        if args.show_kept and imp_s:
            print("  KEPT (impacted):")
            for n in sorted(imp_s):
                _, _, hits = entered_changed(n, stage)
                print(f"    {n}\n        entered: {', '.join(hits) or '(file-level)'}")
        if forced_s:
            print("  FORCED-KEPT (untrusted capture; not impacted but never removable):")
            for n in sorted(forced_s):
                print(f"    {n}")
        print("  REMOVED (safe to skip):")
        for n in sorted(skip_s):
            total, in_changed, _ = entered_changed(n, stage)
            if in_changed == 0:
                why = f"in DB (rows={total}); never entered any changed file"
            else:
                why = f"in DB (rows={total}); entered {in_changed} func(s) in changed file(s), none the changed one"
            print(f"    {n}\n        {why}")
    return 0


if __name__ == "__main__":
    main()
