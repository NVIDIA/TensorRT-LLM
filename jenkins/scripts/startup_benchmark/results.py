# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect startup trials without pooling different cases or runtime identities."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import median

_IDENTITY_FIELDS = (
    "runtime_image",
    "runner_source_fingerprint",
    "runtime_version",
    "hostname",
    "checkpoint_fingerprint",
    "case_config_fingerprint",
    "gpu_inventory",
)
_CSV_FIELDS = (
    "source",
    "schema_version",
    "case",
    "variant",
    "repetition",
    "profile",
    "status",
    "launch_to_ready_seconds",
    "metric_scope",
    "metrics",
    "cache",
    "policy",
    "identity",
    "error",
    "exclusion",
)


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _exclusion(record: dict) -> str:
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        return "unsupported schema_version"
    if record.get("status") != "passed":
        return f"trial status: {record.get('status', 'missing')}"
    if not all(
        isinstance(record.get(key), str) and record[key] for key in ("case", "variant", "profile")
    ):
        return "missing case, variant, or profile"
    if type(record.get("repetition")) is not int or record["repetition"] < 0:
        return "invalid repetition"
    if not isinstance(record.get("cache"), dict) or record["cache"].get("verified") is not True:
        return "checkpoint cache state unverified"
    identity = record.get("identity")
    if (
        not isinstance(identity, dict)
        or any(
            not isinstance(identity.get(key), str) or identity[key] in {"", "unknown"}
            for key in _IDENTITY_FIELDS[:-1]
        )
        or not isinstance(identity.get("gpu_inventory"), list)
        or not identity["gpu_inventory"]
    ):
        return "incomplete comparison identity"
    policy = record.get("policy", {})
    native = record["variant"] == "native"
    if not isinstance(policy, dict):
        return "invalid policy object"
    requested, effective = policy.get("requested"), policy.get("effective")
    expected = "rank_striped_read_ahead" if requested == "auto" else requested
    if not (
        isinstance(requested, str)
        and isinstance(effective, str)
        and (
            (requested == effective == "native")
            if native
            else (
                effective == expected
                and effective not in {"native", "auto", "unknown", "mixed", "none", ""}
            )
        )
        and policy.get("activated") is (not native)
        and policy.get("complete") is True
        and isinstance(policy.get("scope"), str)
        and policy.get("scope") not in (None, "", "unknown", "mixed")
    ):
        return "effective policy unverified or not the intended treatment"
    if not _numeric(record.get("launch_to_ready_seconds")) or record["launch_to_ready_seconds"] < 0:
        return "invalid launch_to_ready_seconds"
    if not isinstance(record.get("metrics"), dict):
        return "invalid metrics object"
    if not isinstance(record.get("metric_scope"), str) or record["metric_scope"] in {"", "unknown"}:
        return "unknown metric scope"
    return ""


def _values(record: dict) -> dict:
    return {
        key: float(value)
        for key, value in {
            **record["metrics"],
            "launch_to_ready_seconds": record["launch_to_ready_seconds"],
        }.items()
        if _numeric(value) and value >= 0
    }


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _missing_trials(root: Path, found: set[str]) -> list[dict]:
    missing = []
    for path in sorted([*root.rglob("run_manifest.json"), *root.rglob("submission.json")]):
        submission = path.name == "submission.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            jobs = {job["case"]: job for job in manifest["jobs"]} if submission else {}
            cases = list(jobs) if submission else [entry["name"] for entry in manifest["cases"]]
            variants = (
                manifest["variants"]
                if submission
                else [entry["name"] for entry in manifest["variants"]]
            )
            if type(manifest["repeats"]) is not int or manifest["repeats"] <= 0:
                raise ValueError("Invalid manifest repeats")
            if any(
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in {".", ".."}
                for name in cases + variants
            ):
                raise ValueError("Invalid manifest case or variant name")
            error_path = path.parent / "run_error.json"
            run_error = (
                json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else None
            )
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            missing.append(
                {
                    "source": str(path.relative_to(root)),
                    "status": "invalid",
                    "error": str(error),
                    "exclusion": "invalid run/submission manifest or run error",
                }
            )
            continue
        for case, variant, repetition in product(cases, variants, range(manifest["repeats"])):
            trial_root, status, error = path.parent, "missing", run_error
            if submission:
                trial_root = path.parent / "results" / case
                if (trial_root / "run_manifest.json").exists():
                    continue
                job = jobs[case]
                if manifest.get("status") == "dry_run":
                    status = "not_executed"
                elif job.get("status") in {"planned", "submission_failed"}:
                    status = "not_submitted"
                error = {
                    "job_id": job.get("job_id"),
                    "job_status": job.get("status"),
                    "reason": job.get("error")
                    or job.get("stderr")
                    or (
                        "Dry-run plan only"
                        if status == "not_executed"
                        else "Job was not submitted"
                        if status == "not_submitted"
                        else "No run manifest; job may be queued, running, or failed before runner startup"
                    ),
                }
            source = str(
                (
                    trial_root / case / f"repeat_{repetition:02d}" / variant / "result.json"
                ).relative_to(root)
            )
            if source in found:
                continue
            missing.append(
                {
                    "schema_version": 1,
                    "source": source,
                    "case": case,
                    "variant": variant,
                    "repetition": repetition,
                    "profile": manifest.get("profile"),
                    "status": status,
                    "identity": manifest.get(
                        "identity", {"runtime_image": manifest.get("image_reference")}
                    ),
                    "error": error or "Planned trial has no result.json",
                    "exclusion": "missing planned trial" if status == "missing" else status,
                }
            )
            found.add(source)
    return missing


def collect_results(root: Path) -> dict:
    """Write raw CSV, matched-pair JSON and descriptive Markdown under root.

    Only passed, cache-verified trials with exact matching identity, policy and
    metric scopes are compared. Missing manifest trials remain in the report.
    """
    records = []
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            record = {"status": "invalid", "error": str(error)}
        if not isinstance(record, dict):
            record = {"status": "invalid", "error": "result must be a JSON object"}
        record = {**record, "source": str(path.relative_to(root))}
        record["exclusion"] = _exclusion(record)
        records.append(record)
        if not record["exclusion"]:
            identity = json.dumps(record["identity"], sort_keys=True)
            key = (
                record["case"],
                record["repetition"],
                record["profile"],
                identity,
                record["policy"]["scope"],
                record["metric_scope"],
            )
            groups[key][record["variant"]].append(record)
    records.extend(_missing_trials(root, {record["source"] for record in records}))

    pairs, matched = [], set()
    for variants in groups.values():
        for entries in variants.values():
            if len(entries) > 1:
                for record in entries:
                    record["exclusion"] = "duplicate trial identity"
        baseline = variants.get("native", [])
        if len(baseline) != 1:
            continue
        native = baseline[0]
        for variant, candidates in variants.items():
            if variant == "native" or len(candidates) != 1:
                continue
            candidate = candidates[0]
            before, after = _values(native), _values(candidate)
            metrics = {
                name: {
                    "native": before[name],
                    "candidate": after[name],
                    "reduction_percent": 100 * (1 - after[name] / before[name])
                    if before[name] > 0
                    else None,
                }
                for name in sorted(before.keys() & after.keys())
            }
            pairs.append(
                {
                    "case": candidate["case"],
                    "variant": variant,
                    "profile": candidate["profile"],
                    "repetition": candidate["repetition"],
                    "identity": candidate["identity"],
                    "policy_scope": candidate["policy"]["scope"],
                    "metric_scope": candidate["metric_scope"],
                    "native_source": native["source"],
                    "candidate_source": candidate["source"],
                    "metrics": metrics,
                }
            )
            matched.update((native["source"], candidate["source"]))
    for record in records:
        if not record["exclusion"] and record["source"] not in matched:
            record["exclusion"] = (
                "no unique eligible counterpart with matching identity and policy/metric scopes"
            )

    strata = defaultdict(list)
    for pair in pairs:
        key = (
            pair["case"],
            pair["profile"],
            pair["variant"],
            json.dumps(pair["identity"], sort_keys=True),
            pair["policy_scope"],
            pair["metric_scope"],
        )
        strata[key].append(pair)
    comparisons = []
    for (case, profile, variant, identity, scope, metric_scope), entries in sorted(strata.items()):
        for name in sorted({name for entry in entries for name in entry["metrics"]}):
            values = [entry["metrics"][name] for entry in entries if name in entry["metrics"]]
            reductions = [
                value["reduction_percent"]
                for value in values
                if value["reduction_percent"] is not None
            ]
            comparisons.append(
                {
                    "case": case,
                    "profile": profile,
                    "variant": variant,
                    "identity": json.loads(identity),
                    "policy_scope": scope,
                    "metric_scope": metric_scope,
                    "metric": name,
                    "pair_count": len(values),
                    "native_median": median(value["native"] for value in values),
                    "candidate_median": median(value["candidate"] for value in values),
                    "reduction_pair_count": len(reductions),
                    "median_paired_reduction_percent": median(reductions) if reductions else None,
                }
            )
    summary = {
        "schema_version": 1,
        "trial_count": len(records),
        "status_counts": dict(Counter(str(record.get("status", "invalid")) for record in records)),
        "matched_pair_count": len(pairs),
        "pairs": pairs,
        "comparisons": comparisons,
        "exclusions": [
            {
                "source": record["source"],
                "reason": record["exclusion"],
                "error": record.get("error"),
            }
            for record in records
            if record["exclusion"]
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(record.get(key))
                    if isinstance(record.get(key), (dict, list))
                    else record.get(key, "")
                    for key in _CSV_FIELDS
                }
            )
    lines = [
        "# Startup benchmark results",
        "",
        f"Trials: {len(records)}; matched pairs: {len(pairs)}; excluded trials: {len(summary['exclusions'])}.",
        "",
        "Descriptive matched-pair results only; no pooled fleet speedup or significance claim. "
        "Positive reduction means faster. Medians use matched trials only; zero native values have no percentage. "
        "Cache verification covers the client checkpoint page cache, not storage-backend caches. "
        "Launch-to-ready is not launch-to-first-token; loader timings retain their reported rank scope.",
        "",
        "| Case / profile | Variant | Identity | Metric | Pairs | Native median | Candidate median "
        "| Median paired reduction |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        identity_tag = hashlib.sha256(
            json.dumps(row["identity"], sort_keys=True).encode()
        ).hexdigest()[:12]
        reduction = row["median_paired_reduction_percent"]
        cells = [
            f"{row['case']} / {row['profile']}",
            row["variant"],
            identity_tag,
            f"{row['metric']} ({row['metric_scope']})",
            row["pair_count"],
            f"{row['native_median']:.4f}",
            f"{row['candidate_median']:.4f}",
            "n/a" if reduction is None else f"{reduction:.2f}%",
        ]
        lines.append("| " + " | ".join(map(_markdown, cells)) + " |")
    lines.extend(["", "## Exclusions and failures", ""])
    lines.extend(
        f"- {_markdown(entry['source'])}: {_markdown(entry['reason'])}; {_markdown(entry['error'] or '')}"
        for entry in summary["exclusions"]
    )
    if not summary["exclusions"]:
        lines.append("None.")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
