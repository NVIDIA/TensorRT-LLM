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
r"""Audit a CBTS touch DB (`cbts_touchmap.sqlite`) — format, scale, and coverage completeness.

Reports the format (stage prefix, schema_version, collection commit), scale,
per-stage known counts, the per-test footprint distribution, and the tests
whose capture looks incomplete (the same heuristic the selector uses).

Example::

    python3 jenkins/scripts/cbts/tools/coverage_audit.py \\
        --db cbts_touchmap.sqlite --list-untrusted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
CBTS = THIS.parent.parent
sys.path.insert(0, str(CBTS))
sys.path.insert(0, str(CBTS / "coverage_selection"))

from blocks import YAMLIndex, block_matches_stage, parse_stages_from_groovy  # noqa: E402
from touch_db import (  # noqa: E402
    _LAUNCH_MARKERS,
    _MIN_FUNCS,
    _SERVING_PATH_MARKERS,
    _WORKER_SENTINEL,
    TouchDB,
    db_key,
    split_stage,
)

_DEFAULT_TEST_DB = CBTS.parents[2] / "tests/integration/test_lists/test-db"
_DEFAULT_GROOVY = CBTS.parents[2] / "jenkins/L0_Test.groovy"


def _fmt_pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", required=True, help="path to cbts_touchmap.sqlite")
    ap.add_argument("--list-untrusted", action="store_true", help="print every untrusted test")
    ap.add_argument(
        "--min-funcs", type=int, default=_MIN_FUNCS, help=f"near-empty floor (default {_MIN_FUNCS})"
    )
    ap.add_argument(
        "--test-db",
        default=str(_DEFAULT_TEST_DB),
        help="test-db dir to diff against the DB (HEAD coverage gap); '' to skip",
    )
    ap.add_argument(
        "--groovy", default=str(_DEFAULT_GROOVY), help="Groovy file to parse stage defs from"
    )
    ap.add_argument(
        "--list-not-in-db", action="store_true", help="print every gap case (default: first 15)"
    )
    args = ap.parse_args(argv)

    db = TouchDB.open(args.db)
    known = db.known_tests()
    stages = db.known_by_stage()
    footprint = db.per_test_footprint()

    print(f"=== CBTS coverage DB audit: {args.db} ===\n")

    # -- Format --
    print("## Format")
    if stages:
        print(f"  test field: stage-prefixed  ({len(stages)} instrumented stage(s) derivable)")
    elif known:
        print(
            "  test field: BARE nodeid  !! WARNING: no stage prefix -> per-stage narrowing impossible"
        )
    sv = db.schema_version()
    commit = db.collection_commit()
    print(f"  schema_version: {sv or 'MISSING (selector cannot hard-fail on format drift)'}")
    print(
        f"  collection commit: {commit or 'MISSING (no staleness gating; zero-touch lever stays off)'}"
    )

    # -- Scale --
    print("\n## Scale")
    print(
        f"  known tests: {len(known)}  |  meta: tests={db.meta('tests')} files={db.meta('files')} "
        f"functions={db.meta('functions')}"
    )
    fr, qr = db.meta("file_rate_pct"), db.meta("func_rate_pct")
    if fr or qr:
        print(f"  coverage rate: files {fr}%  functions {qr}%")

    # -- Per-stage --
    print("\n## Instrumented stages")
    for stage in sorted(stages):
        print(f"  {stage}: {len(stages[stage])} known")

    # -- Completeness --
    untrusted = db.untrusted_tests(
        _WORKER_SENTINEL, _LAUNCH_MARKERS, _SERVING_PATH_MARKERS, args.min_funcs
    )

    def reason(test: str) -> str:
        if any(m in test for m in _SERVING_PATH_MARKERS):
            return "disagg-path (servers uninstrumented)"
        if footprint[test] < args.min_funcs:
            return f"near-empty (<{args.min_funcs} funcs)"
        return "worker-lost (drove inference, no py_executor)"

    print("\n## Coverage completeness")
    if footprint:
        print(
            f"  per-test footprint (functions entered): min={min(footprint.values())} "
            f"max={max(footprint.values())}  (few funcs => likely lost subprocess capture)"
        )
    else:
        print("  per-test footprint: none (no test != '' rows — no usable per-test coverage)")
    trusted_fp = [footprint[t] for t in known if t not in untrusted]
    untrusted_fp = [footprint[t] for t in untrusted]
    if trusted_fp and untrusted_fp:
        print(
            f"  footprint gap: untrusted max={max(untrusted_fp)}  |  trusted min={min(trusted_fp)}"
        )
    print(
        f"  UNTRUSTED (incomplete capture): {len(untrusted)}/{len(known)} ({_fmt_pct(len(untrusted), len(known))})"
    )
    by_reason: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for t in untrusted:
        by_reason[reason(t)] = by_reason.get(reason(t), 0) + 1
        by_stage[split_stage(t)[0]] = by_stage.get(split_stage(t)[0], 0) + 1
    for r, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    - {r}: {n}")
    print(f"  by stage: {dict(sorted(by_stage.items()))}")
    print(
        f"\n  TRUSTED skippable universe: {len(known) - len(untrusted)}/{len(known)} "
        f"(only these may ever be skipped)"
    )

    if args.list_untrusted:
        print("\n## Untrusted tests")
        for t in sorted(untrusted):
            print(f"  [{footprint[t]:>5} funcs] {t}\n      -> {reason(t)}")

    # -- HEAD coverage gap: cases on an instrumented stage with no DB row --
    if args.test_db and Path(args.test_db).is_dir() and Path(args.groovy).is_file():
        yaml_index = YAMLIndex.load(Path(args.test_db))
        all_stages = parse_stages_from_groovy(Path(args.groovy), include_post_merge=True)
        bare_known = {split_stage(t)[1] for t in known}
        per_stage: dict[str, set[str]] = {}
        for name in sorted(set(all_stages) & set(stages)):
            stage = all_stages[name]
            missing = {
                entry
                for block in yaml_index.blocks
                if block.yaml_stem == stage.yaml_stem and block_matches_stage(block, stage)
                for entry in block.tests
                if (k := db_key(entry)) is not None and k not in bare_known
            }
            if missing:
                per_stage[name] = missing
        gap = sorted(set().union(*per_stage.values())) if per_stage else []
        print("\n## HEAD coverage gap on instrumented stages")
        print(
            f"  {len(gap)} unique case(s) render on an instrumented single-GPU stage but have "
            f"NO DB row -> always must-run (new/renamed or never captured)"
        )
        for name in sorted(per_stage, key=lambda n: -len(per_stage[n])):
            print(f"    {name}: {len(per_stage[name])} not-in-DB")
        preview = gap if args.list_not_in_db else gap[:15]
        if preview:
            print("  cases:")
        for t in preview:
            print(f"    - {t}")
        if not args.list_not_in_db and len(gap) > 15:
            print(f"    ... (+{len(gap) - 15}; use --list-not-in-db)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
