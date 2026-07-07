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
"""Post a CBTS decision to OpenSearch (CI-health monitoring).

--status <pre_merge|post_merge|fallback|deferred|disabled> [--reason <text>]
[--decision <main.py output>] [--pr-number <n>] [--repo-root <dir>].
Context + creds come from env. Exits 0 on failure (never blocks CI).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # jenkins/scripts/

logger = logging.getLogger("report_cbts_decision")


def _case_counts(
    affected: list[str], kept_per_stage: dict, status: str, repo_root: str
) -> tuple[int, int]:
    """(cbts_cases, total_cases): cases CBTS runs vs all cases in the mode.

    cbts = sum over hit stages of the Layer-3-kept count (or full if a stage
    wasn't narrowed). total = full case count over the whole trigger-mode
    universe: all non-Post-Merge stages for pre_merge, all stages for
    post_merge. Only meaningful when CBTS ran; returns (0, 0) otherwise or on
    any failure so the record still posts.
    """
    if status not in ("pre_merge", "post_merge"):
        return 0, 0
    try:
        root = Path(repo_root)
        sys.path.insert(0, str(root / "jenkins/scripts/cbts"))
        from blocks import YAMLIndex, block_matches_stage, parse_stages_from_groovy

        stages = parse_stages_from_groovy(root / "jenkins/L0_Test.groovy", include_post_merge=True)
        index = YAMLIndex.load(root / "tests/integration/test_lists/test-db")
        by_stem: dict[str, list] = {}
        for b in index.blocks:
            by_stem.setdefault(b.yaml_stem, []).append(b)

        def full(stage) -> int:
            return sum(
                len(b.tests)
                for b in by_stem.get(stage.yaml_stem, [])
                if block_matches_stage(b, stage)
            )

        post = status == "post_merge"
        cbts = sum(
            kept_per_stage.get(name, full(stages[name])) for name in affected if name in stages
        )
        total = sum(full(st) for name, st in stages.items() if post or "Post-Merge" not in name)
        return cbts, total
    except Exception as exc:  # noqa: BLE001 - case rate is best-effort
        logger.info("CBTS case-count failed (non-fatal): %s", exc)
        return 0, 0


def build_document(
    decision: dict,
    status: str,
    reason: str,
    pr_number: str,
    cbts_cases: int,
    total_cases: int,
) -> dict:
    """Build the typed OpenSearch doc (field prefixes: s_=str, l_=int, d_=float, flat_=dict)."""
    scope = decision.get("scope")
    affected = sorted(decision.get("affected_stages") or [])
    # deferred has no decision; fall back to --reason.
    if not reason:
        reason = " | ".join(decision.get("reasons") or [])

    case_skip_rate = (1 - cbts_cases / total_cases) if total_cases else 0.0

    return {
        "@timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "s_commit": os.getenv("gitlabCommit", ""),
        "s_pr_number": pr_number,
        "s_build_url": os.getenv("BUILD_URL", ""),
        "s_cbts_status": status,
        "s_scope": str(scope) if scope is not None else "",
        "s_reason": reason,
        "l_hit_stages": len(affected),
        "l_total_cases": total_cases,
        "l_cbts_cases": cbts_cases,
        "d_case_skip_rate": round(case_skip_rate, 4),
        "flat_detail": {
            "hit_stages": affected,
            "scopes": list(decision.get("scopes") or []),
            "split_counts": decision.get("affected_stage_split_counts") or {},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post a CBTS decision to OpenSearch.")
    parser.add_argument(
        "--status", required=True, help="pre_merge / post_merge / fallback / deferred / disabled."
    )
    parser.add_argument("--reason", default="", help="Reason text (deferred only).")
    parser.add_argument("--decision", default=None, help="Path to cbts/main.py decision JSON.")
    parser.add_argument("--pr-number", default="", help="PR / MR number for s_pr_number.")
    parser.add_argument("--repo-root", default=".", help="Repo root (for the case-rate counts).")
    args = parser.parse_args(argv)

    # Lazy import: any failure surfaces here and is caught by the __main__
    # guard, so telemetry never blocks CI. open_search_db imports cleanly even
    # without requests; postToOpenSearchDB falls back to urllib on pods (e.g.
    # the Setup Environment pod) that don't ship requests.
    from open_search_db import CBTS_PROJECT_NAME, OpenSearchDB

    decision = json.loads(Path(args.decision).read_text()) if args.decision else {}
    cbts_cases, total_cases = _case_counts(
        decision.get("affected_stages") or [],
        decision.get("affected_stage_test_counts") or {},
        args.status,
        args.repo_root,
    )
    doc = build_document(
        decision, args.status, args.reason, args.pr_number, cbts_cases, total_cases
    )
    OpenSearchDB.add_id_of_json(doc)
    ok = OpenSearchDB.postToOpenSearchDB(doc, CBTS_PROJECT_NAME)
    logger.info(
        "CBTS report %s: status=%s hit_stages=%d case_skip=%s (cbts %d / total %d)",
        "posted" if ok else "post returned False",
        doc["s_cbts_status"],
        doc["l_hit_stages"],
        doc["d_case_skip_rate"],
        cbts_cases,
        total_cases,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - telemetry must never break CI
        logger.info("CBTS telemetry failed (non-fatal): %s", exc)
        sys.exit(0)
