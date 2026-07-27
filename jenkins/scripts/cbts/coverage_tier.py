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
"""CBTS Tier 2: coverage-based test-db narrowing on the Tier-1 fallback residual."""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from blocks import (
    TARGET_SHARD_SECONDS,
    Stage,
    YAMLIndex,
    _avg_duration,
    _entry_applies_to_waive,
    _entry_target,
    _estimate_entries_seconds,
    _target_in_filter_subtree,
    block_matches_stage,
)
from rules.base import PRInputs, RuleResult

sys.path.insert(0, str(Path(__file__).resolve().parent / "coverage_selection"))
from selector import CoverageSelector  # noqa: E402
from touch_db import TouchDB, db_key  # noqa: E402


def open_db(path: str) -> TouchDB:
    """Open a touch DB read-only."""
    return TouchDB.open(path)


@dataclass
class CoverageTierResult:
    affected_stages: set[str]
    removed: dict[tuple[str, int], set[str]] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    reason: str = ""
    must_run_reasons: dict[str, int] = field(default_factory=dict)
    detail: dict[str, object] = field(default_factory=dict)


# Mirrors L0_Test.groovy multiGpuJobs: pre-merge stages with N_GPUs token.
_MULTI_GPU_RE = re.compile(r"\d+_GPUs")


def _is_multi_gpu(stage_name: str) -> bool:
    return bool(_MULTI_GPU_RE.search(stage_name)) and "Post-Merge" not in stage_name


def _rule_kept_entries(block, prefix_to_waives: dict[str, set[str]]) -> set[str]:
    """Entries a rule's block_filters would keep."""
    kept: set[str] = set()
    for t in block.tests:
        target = _entry_target(t)
        matched: set[str] = set()
        for prefix, waives in prefix_to_waives.items():
            if _target_in_filter_subtree(target, prefix):
                matched |= waives
        if matched and any(_entry_applies_to_waive(t, w) for w in matched):
            kept.add(t)
    return kept


_R_SAFE = "safe"
_R_IMPACTED = "impacted"
_R_UNTRUSTED = "untrusted"
_R_NO_DATA = "no_data"
_R_RULE_KEPT = "rule_kept"
_R_COARSE = "coarse"


def _entry_reason(
    entry: str,
    served: list[str],
    keep_rule: set[str],
    cov,
    known: dict[str, set[str]],
    untrusted: set[str],
) -> str:
    """Return SAFE or the must-run cause for a candidate YAML entry."""
    if entry in keep_rule:
        return _R_RULE_KEPT
    dbk = db_key(entry)
    if dbk is None:
        return _R_COARSE
    for name in served:
        if dbk not in known.get(name, frozenset()):
            return _R_NO_DATA
        if dbk in cov.impacted.get(name, frozenset()):
            return _R_IMPACTED
        if f"{name}/{dbk}" in untrusted:
            return _R_UNTRUSTED
    return _R_SAFE


def _build_narrowing(
    cov,
    stages: dict[str, Stage],
    yaml_index: YAMLIndex,
    rule_block_filters: dict[tuple[str, int], dict[str, set[str]]],
    known: dict[str, set[str]],
    untrusted: set[str],
) -> tuple[dict[tuple[str, int], set[str]], set[str], Counter]:
    """Classify every candidate entry; remove only SAFE ones.

    Returns (removed per block, fully-emptied instrumented stages, must-run tally).
    """
    instrumented = set(cov.skippable)
    rule_kept = {
        key: _rule_kept_entries(b, rule_block_filters[key])
        for b in yaml_index.blocks
        if (key := (b.yaml_stem, b.block_index)) in rule_block_filters
    }

    removed: dict[tuple[str, int], set[str]] = {}
    must_run: Counter = Counter()
    for block in yaml_index.blocks:
        served = [
            s.name
            for s in stages.values()
            if s.yaml_stem == block.yaml_stem and block_matches_stage(block, s)
        ]
        # shared-block rule: prune only when every served stage is instrumented
        if not served or any(name not in instrumented for name in served):
            continue
        key = (block.yaml_stem, block.block_index)
        keep_rule = rule_kept.get(key, set())
        rm: set[str] = set()
        for entry in block.tests:
            reason = _entry_reason(entry, served, keep_rule, cov, known, untrusted)
            if reason == _R_SAFE:
                rm.add(entry)
            else:
                must_run[reason] += 1
        if rm:
            removed[key] = rm

    dropped: set[str] = set()
    for name in instrumented:
        stage = stages.get(name)
        if stage is None:
            continue
        total = kept = 0
        for block in yaml_index.blocks:
            if block.yaml_stem != stage.yaml_stem or not block_matches_stage(block, stage):
                continue
            rm = removed.get((block.yaml_stem, block.block_index), set())
            for entry in block.tests:
                total += 1
                if entry not in rm:
                    kept += 1
        if total > 0 and kept == 0:
            dropped.add(name)
    return removed, dropped, must_run


def apply_coverage_tier(
    pr: PRInputs,
    pairs: list[tuple[object, RuleResult]],
    handled: set[str],
    stages: dict[str, Stage],
    yaml_index: YAMLIndex,
    repo_root: Path,
    db: TouchDB,
) -> tuple[CoverageTierResult | None, str]:
    """Return (narrowing, note); narrowing is None when the tier keeps the Tier-1 result."""
    if any(r.scope is None for _, r in pairs):
        return None, "coverage tier skipped: a rule forced fallback (scope=null)"
    residual = sorted(set(pr.changed_files) - handled)
    if not residual:
        return None, "coverage tier skipped: no residual (all files handled by rules)"

    selector = CoverageSelector(db, repo_root)
    cov = selector.decide(residual, pr.diffs)
    if not cov.ok:
        return None, f"coverage tier declined: {cov.reason}"

    rule_block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
    for _, r in pairs:
        for key, prefix_to_waives in r.block_filters.items():
            dst = rule_block_filters.setdefault(key, {})
            for prefix, waives in prefix_to_waives.items():
                dst.setdefault(prefix, set()).update(waives)

    nd = ""
    if cov.no_data_funcs:
        shown = ", ".join(cov.no_data_funcs[:3])
        more = f" (+{len(cov.no_data_funcs) - 3})" if len(cov.no_data_funcs) > 3 else ""
        nd = f"; new/uncovered function(s), bounded via importers: {shown}{more}"

    known = db.known_by_stage()
    untrusted = selector.untrusted_tests()
    removed, dropped, must_run_reasons = _build_narrowing(
        cov, stages, yaml_index, rule_block_filters, known, untrusted
    )
    # exclude multi-GPU and post-merge stages from the drop set (not coverage's to decide)
    dropped = {s for s in dropped if not _is_multi_gpu(s)}
    if not pr.post_merge:
        dropped = {s for s in dropped if "Post-Merge" not in s}

    n_impacted = sum(len(v) for v in cov.impacted.values())
    n_removed = sum(len(v) for v in removed.values())
    narrowed = bool(removed or dropped)
    if narrowed:
        reason = (
            f"coverage: {len(residual)} core file(s), {n_impacted} impacted test(s), "
            f"{cov.n_untrusted} untrusted forced-run; "
            f"removed {n_removed} case(s), dropped {len(dropped)} single-GPU stage(s){nd}"
        )
    else:
        reason = (
            f"coverage: {len(residual)} core file(s), {n_impacted} impacted test(s), "
            f"{cov.n_untrusted} untrusted; nothing removable (all impacted / untrusted / not-in-DB){nd}"
        )
    result = CoverageTierResult(
        # single-GPU only; multi-GPU re-added in Groovy under MULTI_GPU_FILE_CHANGED gate
        affected_stages={s for s in stages if not _is_multi_gpu(s)} - dropped,
        removed=removed,
        dropped=dropped,
        reason=reason,
        must_run_reasons=dict(must_run_reasons),
        detail={
            "source": "coverage",
            "files": len(residual),
            "impacted": n_impacted,
            "untrusted": cov.n_untrusted,
            "removed_cases": n_removed,
            "dropped_stages": len(dropped),
            "outcome": "narrowed" if narrowed else "nothing_removable",
            **({"no_data_funcs": list(cov.no_data_funcs)} if cov.no_data_funcs else {}),
        },
    )
    return result, result.reason


def write_coverage_test_db(
    src_dir: Path, out_dir: Path, removed: dict[tuple[str, int], set[str]]
) -> None:
    """Write narrowed YAMLs with removed entries dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in sorted({stem for stem, _ in removed}):
        src = src_dir / f"{stem}.yml"
        if not src.exists():
            continue
        data = yaml.safe_load(src.read_text()) or {}
        blocks = data.get(stem)
        if not isinstance(blocks, list):
            continue
        for i, block_data in enumerate(blocks):
            if not isinstance(block_data, dict):
                continue
            rm = removed.get((stem, i))
            if not rm:
                continue
            block_data["tests"] = [t for t in (block_data.get("tests") or []) if t not in rm]
        (out_dir / src.name).write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        )


def compute_coverage_stage_counts(
    affected_stages: set[str],
    stages: dict[str, Stage],
    yaml_index: YAMLIndex,
    removed: dict[tuple[str, int], set[str]],
    durations: dict[str, float],
    target_seconds: int = TARGET_SHARD_SECONDS,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return per-stage (kept-entry count, resized split count) for narrowed stages only."""
    avg = _avg_duration(durations)
    test_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for name in affected_stages:
        stage = stages.get(name)
        if stage is None:
            continue
        entries: list[str] = []
        had_removal = False
        for block in yaml_index.blocks:
            if block.yaml_stem != stage.yaml_stem or not block_matches_stage(block, stage):
                continue
            rm = removed.get((block.yaml_stem, block.block_index), set())
            if rm:
                had_removal = True
            entries.extend(t for t in block.tests if t not in rm)
        if not had_removal:
            continue
        test_counts[name] = len(entries)
        seconds = _estimate_entries_seconds(entries, durations, avg)
        split_counts[name] = max(1, min(math.ceil(seconds / target_seconds), stage.total_splits))
    return test_counts, split_counts
