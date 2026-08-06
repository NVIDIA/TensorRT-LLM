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

from qualname_map import qualnames_for_lines  # noqa: E402
from rules._helpers import iter_diff_post_line_numbers  # noqa: E402
from touch_db import (  # noqa: E402
    _LAUNCH_MARKERS,
    _MIN_FUNCS,
    _SERVING_PATH_MARKERS,
    _UNTRUSTED_STAGE_MARKERS,
    _WORKER_SENTINEL,
    TouchDB,
    canon,
    split_stage,
    stage_family,
)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=check
    ).stdout


def _src_at(repo: Path, sha: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=str(repo), capture_output=True, text=True, check=False
    )
    return r.stdout if r.returncode == 0 else None


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

    # `decide()` refuses the whole change here, so no removal below would be reachable.
    zero_touch = [f for f in core if not db.file_has_touch_rows(canon(f))]
    if zero_touch:
        print(f"commit {args.sha[:12]} — coverage selection REFUSES this change:")
        for f in zero_touch:
            print(f"  zero-touch residual file (new/uninstrumented): {f}")
        print("\nNo case is removable; every stage runs in full.")
        return 0

    # Impact set: (file, qualname) per function, or the whole file when it falls back.
    impact_funcs: set[tuple[str, str]] = set()
    impact_files: set[str] = set()  # file-level fallback
    changed_files: set[str] = set()
    impacted: set[str] = set()
    no_data: list[str] = []
    for f in core:
        cf = canon(f)
        changed_files.add(cf)
        diff = _git(repo, "diff", f"{args.sha}^", args.sha, "--", f, check=False)
        lines = iter_diff_post_line_numbers(diff)
        src = _src_at(repo, args.sha, f)
        if not lines or src is None:
            impact_files.add(cf)
            impacted |= db.tests_touching_file(cf)
            continue
        qns, ok = qualnames_for_lines(src, lines)
        if not ok:
            impact_files.add(cf)
            impacted |= db.tests_touching_file(cf)
            continue
        for q in qns:
            impact_funcs.add((cf, q))
            tests = db.tests_touching_func(cf, q)
            impacted |= tests
            # Mirrors the selector's default no_data_policy="file".
            if not tests and q != "<module>":
                no_data.append(f"{cf}::{q}")
                impact_files.add(cf)
                impacted |= db.tests_touching_file(cf)

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

    # Untrusted capture is forced to run by the selector; it is never removable.
    untrusted = db.untrusted_tests(
        _WORKER_SENTINEL,
        _LAUNCH_MARKERS,
        _SERVING_PATH_MARKERS,
        _MIN_FUNCS,
        _UNTRUSTED_STAGE_MARKERS,
    )
    # Keyed by family: untrusted on any shard means untrusted for the family.
    untrusted_fam = {f"{stage_family(s)}/{n}" for s, n in map(split_stage, untrusted) if s}

    impacted_by_stage: dict[str, set[str]] = {}
    for t in impacted:
        stage, nodeid = split_stage(t)
        impacted_by_stage.setdefault(stage, set()).add(nodeid)

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
        imp_s = impacted_by_stage.get(stage, set()) & known_s
        fam = stage_family(stage)
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
