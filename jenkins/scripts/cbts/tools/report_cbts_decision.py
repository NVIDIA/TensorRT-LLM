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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # jenkins/scripts/

from cbts.rules.base import format_reason  # noqa: E402

logger = logging.getLogger("report_cbts_decision")


_MULTI_GPU_RE = re.compile(r"\d+_GPUs")
_STAGE_SHARD_SUFFIX_RE = re.compile(r"-\d+$")
_ON_DEMAND_MARKER = "-OnDemand-"
_POST_MERGE_MARKER = "Post-Merge"
_PERF_SANITY_MARKER = "PerfSanity"
_PACKAGE_SANITY_MARKER = "PackageSanityCheck"


def _is_multi_gpu_stage(name: str) -> bool:
    return bool(_MULTI_GPU_RE.search(name))


def _multi_gpu_scheduled(
    status: str, multi_gpu_required: bool, multi_gpu_label_gate_open: bool
) -> bool:
    """Whether multi-GPU belongs to the non-CBTS baseline at decision time."""
    return status == "post_merge" or (multi_gpu_required and multi_gpu_label_gate_open)


def _stage_group(name: str) -> str:
    """Strip the trailing pytest-split shard id from a stage name."""
    return _STAGE_SHARD_SUFFIX_RE.sub("", name)


def _is_scheduled_stage(name: str, status: str, multi_gpu_scheduled: bool) -> bool:
    """Whether the baseline trigger mode makes a stage eligible to run."""
    if _ON_DEMAND_MARKER in name:
        return False
    if status == "pre_merge" and _POST_MERGE_MARKER in name:
        return False
    return status == "post_merge" or multi_gpu_scheduled or not _is_multi_gpu_stage(name)


def _scheduled_affected_stages(decision: dict, status: str, multi_gpu_scheduled: bool) -> list[str]:
    """Return decision stages that survive the baseline scheduling gates."""
    return sorted(
        name
        for name in decision.get("affected_stages") or []
        if _is_scheduled_stage(name, status, multi_gpu_scheduled)
    )


def _case_counts(
    decision: dict,
    status: str,
    repo_root: str,
    multi_gpu_scheduled: bool = False,
) -> tuple[int, int]:
    r"""(cbts_cases, total_cases): cases CBTS runs vs all cases in the mode.

    cbts = sum over the stages expected to pass Layer 2 at decision time,
    using the Layer-3 kept count for narrowed stages and the full count for
    force-kept stages.
    total = the full case count over the same trigger-mode universe. OnDemand
    stages are never in that universe. Multi-GPU stages are included only
    when normal CI requires them and the approval-label gate is open.

    A trailing stage number is a pytest-split shard, not another copy of the
    test list. Counts are therefore aggregated once per stage family; summing
    the same YAML entries once per shard would over-weight highly sharded
    platforms in both the numerator and denominator.

    PerfSanity and PackageSanityCheck stages can be force-kept even when they
    are not in affected_stages. Coverage selection can similarly request that
    baseline multi-GPU stages be re-added; all such stages use their full
    counts in the numerator.

    Only meaningful when CBTS ran; returns (0, 0) otherwise or on any failure
    so the record still posts.
    """
    if status not in ("pre_merge", "post_merge"):
        return 0, 0
    multi_gpu_scheduled = status == "post_merge" or multi_gpu_scheduled
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

        scheduled_groups: dict[str, dict[str, object]] = {}
        for name, stage in stages.items():
            if _is_scheduled_stage(name, status, multi_gpu_scheduled):
                scheduled_groups.setdefault(_stage_group(name), {})[name] = stage

        affected_names = set(decision.get("affected_stages") or [])
        affected_groups = {
            group for group, members in scheduled_groups.items() if affected_names & members.keys()
        }
        force_kept_groups: set[str] = set()
        if decision.get("sanity_required", True):
            force_kept_groups.update(
                group for group in scheduled_groups if _PACKAGE_SANITY_MARKER in group
            )
        if decision.get("perfsanity_required", True):
            force_kept_groups.update(
                group
                for group in scheduled_groups
                if _PERF_SANITY_MARKER in group and _POST_MERGE_MARKER not in group
            )
        if decision.get("enable_multi_gpu", False) and multi_gpu_scheduled:
            force_kept_groups.update(
                group for group in scheduled_groups if _is_multi_gpu_stage(group)
            )

        # A stage explicitly affected by CBTS may be narrowed. The force-keep
        # paths add only otherwise-unaffected stages at their full size.
        force_kept_groups -= affected_groups
        kept_per_stage = decision.get("affected_stage_test_counts") or {}

        def full_group(group: str) -> int:
            return full(next(iter(scheduled_groups[group].values())))

        def kept_group(group: str) -> int:
            counts = [
                kept_per_stage[name] for name in scheduled_groups[group] if name in kept_per_stage
            ]
            return max(counts) if counts else full_group(group)

        cbts = sum(kept_group(group) for group in affected_groups) + sum(
            full_group(group) for group in force_kept_groups
        )
        total = sum(full_group(group) for group in scheduled_groups)
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
    multi_gpu_required: bool = False,
    multi_gpu_label_gate_open: bool = False,
) -> dict:
    """Build the typed OpenSearch doc (field prefixes: s_=str, l_=int, d_=float, flat_=dict)."""
    scope = decision.get("scope")
    multi_gpu_scheduled = _multi_gpu_scheduled(
        status, multi_gpu_required, multi_gpu_label_gate_open
    )
    affected = _scheduled_affected_stages(decision, status, multi_gpu_scheduled)
    # deferred has no decision; fall back to --reason.
    if not reason:
        reason = " | ".join(format_reason(r) for r in decision.get("reasons") or [])

    case_skip_rate_valid = (
        status in ("pre_merge", "post_merge") and total_cases > 0 and 0 <= cbts_cases <= total_cases
    )
    case_skip_rate = (1 - cbts_cases / total_cases) if case_skip_rate_valid else 0.0

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
        # Post-merge build of the consulted touch DB; 0 when no DB was used.
        "l_coverage_db_build": int(decision.get("coverage_db_build") or 0),
        "s_coverage_db_commit": decision.get("coverage_db_commit") or "",
        # Residual files whose patch the forge API omitted (binary / rename / oversized).
        "l_coverage_no_diff_files": int(decision.get("coverage_no_diff_files") or 0),
        # Commits main gained since that DB was collected; -1 when unmeasurable.
        "l_coverage_db_lag": int(
            decision["coverage_db_lag"] if decision.get("coverage_db_lag") is not None else -1
        ),
        # Commits between that DB and the PR's base — what the gate decides on; -1 when unmeasurable.
        "s_coverage_db_base_commit": decision.get("coverage_db_base_commit") or "",
        "l_coverage_db_drift": int(
            decision["coverage_db_drift"] if decision.get("coverage_db_drift") is not None else -1
        ),
        "s_coverage_db_drift_status": decision.get("coverage_db_drift_status") or "",
        # Freshness-gate verdict on that drift: ok / stale / unknown; empty when no DB was consulted.
        "s_coverage_freshness": decision.get("coverage_freshness") or "",
        # This field is consumed directly by the CBTS OpenSearch dashboard.
        "d_case_skip_rate": round(case_skip_rate, 4),
        "b_case_skip_rate_valid": case_skip_rate_valid,
        "b_non_cbts_multi_gpu_required": multi_gpu_required,
        "b_multi_gpu_label_gate_open": multi_gpu_label_gate_open,
        "flat_detail": {
            "hit_stages": affected,
            "scopes": list(decision.get("scopes") or []),
            "split_counts": {
                name: count
                for name, count in (decision.get("affected_stage_split_counts") or {}).items()
                if name in affected
            },
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
    parser.add_argument(
        "--multi-gpu-required",
        action="store_true",
        default=False,
        help="Pass when normal non-CBTS policy requires multi-GPU pre-merge stages.",
    )
    parser.add_argument(
        "--multi-gpu-label-gate-open",
        action="store_true",
        default=False,
        help="Pass when the multi-GPU approval-label gate is open at CBTS decision time.",
    )
    args = parser.parse_args(argv)

    # Lazy import: any failure surfaces here and is caught by the __main__
    # guard, so telemetry never blocks CI. open_search_db imports cleanly even
    # without requests; postToOpenSearchDB falls back to urllib on pods (e.g.
    # the Setup Environment pod) that don't ship requests.
    from open_search_db import CBTS_PROJECT_NAME, OpenSearchDB

    decision = json.loads(Path(args.decision).read_text()) if args.decision else {}
    multi_gpu_scheduled = _multi_gpu_scheduled(
        args.status,
        args.multi_gpu_required,
        args.multi_gpu_label_gate_open,
    )
    cbts_cases, total_cases = _case_counts(
        decision,
        args.status,
        args.repo_root,
        multi_gpu_scheduled=multi_gpu_scheduled,
    )
    doc = build_document(
        decision,
        args.status,
        args.reason,
        args.pr_number,
        cbts_cases,
        total_cases,
        multi_gpu_required=args.multi_gpu_required,
        multi_gpu_label_gate_open=args.multi_gpu_label_gate_open,
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
