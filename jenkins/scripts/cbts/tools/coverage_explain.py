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

    python3 jenkins/scripts/cbts/tools/coverage_explain.py \\
        --db /tmp/cbts_inspect/cbts_touchmap.sqlite --sha 890e1089 --show-kept
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
CBTS = THIS.parent.parent
sys.path.insert(0, str(CBTS))
sys.path.insert(0, str(CBTS / "coverage_selection"))

from qualname_map import (  # noqa: E402
    analyze_python_changes,
    closure_attributed_qualnames,
    import_executed_qualnames,
    qualnames_for_lines,
)
from rules._helpers import iter_diff_deleted_post_lines, iter_diff_post_line_numbers  # noqa: E402
from selector import CoverageSelector  # noqa: E402
from touch_db import TouchDB, canon, stage_family  # noqa: E402


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=check
    ).stdout


def _src_at(repo: Path, sha: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=str(repo), capture_output=True, text=True, check=False
    )
    return r.stdout if r.returncode == 0 else None


def _caller_frontier(db: TouchDB, cf: str, qualname: str, callers, visiting=None):
    """Return row-bearing local callers only when every branch resolves."""
    visiting = set() if visiting is None else visiting
    if qualname in visiting or not callers.get(qualname):
        return None
    frontier: set[str] = set()
    for caller in callers[qualname]:
        if db.tests_touching_func(cf, caller):
            frontier.add(caller)
            continue
        nested = _caller_frontier(db, cf, caller, callers, visiting | {qualname})
        if nested is None:
            return None
        frontier.update(nested)
    return frontier or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--repo-root", default=str(CBTS.parents[2]))
    ap.add_argument("--stage", default=None, help="limit to one stage")
    ap.add_argument("--show-kept", action="store_true", help="also list kept (impacted) cases")
    args = ap.parse_args(argv)

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
    caller_bounded = set(res.caller_bounded_funcs)
    for f in core:
        cf = canon(f)
        changed_files.add(cf)
        lines = iter_diff_post_line_numbers(diffs.get(f, ""))
        src = _src_at(repo, args.sha, f)
        qns, ok = qualnames_for_lines(src, lines) if (lines and src) else (set(), False)
        if not ok:
            impact_files.add(cf)
            continue
        dependencies = analyze_python_changes(
            src, lines, iter_diff_deleted_post_lines(diffs.get(f, ""))
        )
        import_executed = import_executed_qualnames(src)
        closures = closure_attributed_qualnames(src, lines)

        def record_impact(q: str) -> None:
            if db.tests_touching_func(cf, q):
                impact_funcs.add((cf, q))
                return
            if f"{cf}::{q}" in caller_bounded:
                frontier = _caller_frontier(db, cf, q, dependencies.callers)
                if frontier:
                    impact_funcs.update((cf, caller) for caller in frontier)
                    return
            impact_files.add(cf)

        for q in qns:
            if q in import_executed:
                for consumer in dependencies.import_consumers:
                    record_impact(consumer)
            elif q in closures:
                impact_files.add(cf)
            else:
                record_impact(q)
    no_data = res.no_data_funcs

    print(f"commit {args.sha[:12]} — {len(core)} core file(s), impact set:")
    for cf, q in sorted(impact_funcs):
        print(f"  {cf} :: {q}")
    for cf in sorted(impact_files):
        print(f"  {cf} :: <file-level>")
    if no_data:
        file_bounded = set(no_data) - caller_bounded
        print(
            f"  ({len(no_data)} changed function(s) with no DB rows: "
            f"{len(caller_bounded)} caller-bounded, {len(file_bounded)} file-level fallback)"
        )
        for s in sorted(no_data):
            bound = "caller-bound" if s in caller_bounded else "file-level"
            print(f"    no-data ({bound}): {s}")
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
    sys.exit(main())
