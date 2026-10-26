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

"""Reclassify named test failures and timeouts covered by new target-branch waives."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from open_search_db import WAIVE_PROJECT_NAME, OpenSearchDB
from test_rerun import parse_name


@dataclass(frozen=True)
class WaiveRule:
    nodeid: str
    marker: str
    reason: str


def _parse_rules(content: str) -> list[WaiveRule]:
    rules = []
    for line in content.splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"(.+?)\s+(SKIP|XFAIL|UNSTABLE|TIMEOUT)\s+\((.+)\)", line)
        if not match:
            raise ValueError(f"Unsupported waive rule: {line}")
        rules.append(WaiveRule(*match.groups()))
    return rules


def _matching_rules(
    rules: list[WaiveRule], stage_name: str, sm_version: int | None
) -> dict[str, WaiveRule]:
    result = {}
    for rule in rules:
        nodeid = rule.nodeid
        if nodeid.startswith("full:"):
            scope, separator, nodeid = nodeid[5:].partition("/")
            matches_stage = scope == stage_name or stage_name.startswith(scope + "-")
            matches_sm = sm_version is not None and scope == f"sm{sm_version}"
            if not separator or not (matches_stage or matches_sm):
                continue
        result[nodeid] = rule
    return result


def _test_name(case: ET.Element, stage_name: str) -> str:
    filename = case.get("file", "")
    if not filename:
        return ""
    name = parse_name(case.get("classname", ""), case.get("name", ""), filename)
    return name.removeprefix(stage_name + "/")


def _failed_cases(root: ET.Element) -> list[ET.Element]:
    return [
        case
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]


def _update_counts(root: ET.Element) -> None:
    for node in reversed(list(root.iter())):
        if node.tag not in {"testsuites", "testsuite"}:
            continue
        cases = list(node.iter("testcase"))
        node.set("tests", str(len(cases)))
        for attribute, child in (
            ("failures", "failure"),
            ("errors", "error"),
            ("skipped", "skipped"),
        ):
            node.set(attribute, str(sum(case.find(child) is not None for case in cases)))


def _latest_states(records: list[dict]) -> list[dict]:
    """Treat missing and empty scopes as the same rule, retaining its latest state."""
    latest_by_scope = {}
    for record in records:
        fields = record["_source"]
        key = (fields["s_turtle_name"], fields.get("s_full_scope") or "")
        previous = latest_by_scope.get(key)
        if previous is not None:
            for candidate in (previous, record):
                if len(candidate.get("sort", [])) != 2 or not all(
                    isinstance(value, (int, float)) for value in candidate["sort"]
                ):
                    raise ValueError("Cannot order duplicate OpenDB waive scopes")
            if (
                previous["sort"] == record["sort"]
                and fields["s_status"] != previous["_source"]["s_status"]
            ):
                raise ValueError("Conflicting OpenDB waive states have the same timestamp")
            if previous["sort"] > record["sort"]:
                continue
        latest_by_scope[key] = record
    return list(latest_by_scope.values())


def query_latest_waives(test_names: list[str], branch: str) -> tuple[str, dict]:
    """Read the latest state per test and scope before filtering active waives.

    Filtering for Waived first would resurrect revoked rules. All pages use the
    same ingestion cutoff so updates arriving during the query are excluded.
    """
    cutoff = datetime.now(timezone.utc).isoformat()
    filters = [
        {"term": {"s_record_type": "waive_state"}},
        {"term": {"s_branch": branch}},
        {"terms": {"s_turtle_name": sorted(set(test_names))}},
        {"range": {"@timestamp": {"lte": cutoff}}},
    ]
    latest = {
        "size": 1,
        "sort": [{"ts_created": "desc"}, {"@timestamp": "desc"}],
        "_source": [
            "s_branch",
            "s_turtle_name",
            "s_full_scope",
            "s_status",
            "s_match_type",
            "s_reason",
            "s_bug_url",
            "s_commit_sha",
            "ts_created",
            "@timestamp",
        ],
    }
    composite = {
        "size": 500,
        "sources": [
            {"test": {"terms": {"field": "s_turtle_name"}}},
            {"scope": {"terms": {"field": "s_full_scope", "missing_bucket": True}}},
        ],
    }
    records = []
    while True:
        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {"states": {"composite": composite, "aggs": {"latest": {"top_hits": latest}}}},
        }
        response = OpenSearchDB.queryFromOpenSearchDB(body, WAIVE_PROJECT_NAME)
        if response is None:
            raise RuntimeError("OpenDB waive query failed")
        data = response.json()
        if data.get("timed_out") is not False or data.get("_shards", {}).get("failed", 1):
            raise RuntimeError("OpenDB returned incomplete waive states")
        states = data["aggregations"]["states"]
        for bucket in states["buckets"]:
            record = bucket["latest"]["hits"]["hits"][0]
            fields = record["_source"]
            if fields["s_branch"] != branch or fields["s_turtle_name"] not in test_names:
                raise ValueError("OpenDB returned a different branch or test")
            records.append(record)
        next_after = states.get("after_key")
        if not states["buckets"] or next_after is None:
            break
        if composite.get("after") == next_after:
            raise RuntimeError("OpenDB waive pagination did not advance")
        composite = {**composite, "after": next_after}

    lines = []
    for record in _latest_states(records):
        fields = record["_source"]
        if fields["s_status"] != "Waived" or fields.get("s_match_type") != "exact":
            continue
        name = fields["s_turtle_name"]
        if fields.get("s_full_scope"):
            name = f"full:{fields['s_full_scope']}/{name}"
        reason = fields.get("s_reason") or fields.get("s_bug_url") or "OpenDB waive"
        lines.append(f"{name} SKIP ({reason})")
    return "\n".join(lines), {"branch": branch, "cutoff": cutoff, "records": records}


def _detect_sm_version() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = result.stdout.splitlines()
    if result.returncode or not values or not re.fullmatch(r"\d+\.\d+", values[0].strip()):
        return None
    return int(values[0].strip().replace(".", ""))


def _apply_waive(
    case: ET.Element, rule: WaiveRule, result_file: Path, stage_name: str, source: dict[str, str]
) -> dict:
    """Replace a failure with a skip and record the applied waive rule."""
    for failure in case.findall("failure") + case.findall("error"):
        case.remove(failure)
    message = f"Skipped by new TOT waive: {rule.reason}"
    ET.SubElement(case, "skipped", {"type": "tot-waive", "message": message})
    properties = case.find("properties")
    if properties is None:
        properties = ET.SubElement(case, "properties")
    for key, value in {"waive_rule": rule.nodeid, **source}.items():
        ET.SubElement(properties, "property", {"name": key, "value": value})
    return {
        "test": _test_name(case, stage_name),
        "file": result_file.name,
        "reason": rule.reason,
        "rule": rule.nodeid,
    }


def reclassify_failures(
    result_dir: Path,
    latest_waives: str,
    initial_waives: str,
    stage_name: str,
    source: dict[str, str],
    sm_version: int | None = None,
) -> dict:
    """Replace newly waived failures with audited skips; retain all other errors.

    All input reports are parsed before any are changed. Original XML files are
    retained with a .original suffix so Jenkins results*.xml globs cannot ingest
    them. Historical rerun reports remain unchanged. Waives already covering the
    stage in its original TOT snapshot cannot grant a new skip, even if their
    reason, scope spelling, or marker changes.
    """
    audit_dir = result_dir / "waive_refresh"
    if (audit_dir / "report.json").exists():
        raise ValueError("Waive refresh has already been applied to this stage attempt")
    latest_rules = _matching_rules(_parse_rules(latest_waives), stage_name, sm_version)
    initial_rules = _matching_rules(_parse_rules(initial_waives), stage_name, sm_version)
    added_rules = {
        name: rule
        for name, rule in latest_rules.items()
        if rule.marker == "SKIP" and name not in initial_rules
    }
    report_files = sorted(result_dir.glob("results*.xml"))
    trees = {path: ET.parse(path) for path in report_files}
    report = {
        "stage_name": stage_name,
        "sm_version": sm_version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "added_waives": [
            {"test": name, "rule": rule.nodeid, "reason": rule.reason}
            for name, rule in sorted(added_rules.items())
        ],
        "waived": [],
        "remaining_failures": [],
    }
    changed_paths = set()
    reported_timeouts = set()
    for path, tree in trees.items():
        for case in _failed_cases(tree.getroot()):
            name = _test_name(case, stage_name)
            if path.name == "results-timeout.xml":
                reported_timeouts.add(name)
            if path.name == "results-stage.xml":
                reason = "synthetic failure"
            elif name in initial_rules:
                reason = "already present in initial TOT"
            elif name not in added_rules:
                reason = "not covered by a new TOT waive"
            else:
                entry = _apply_waive(case, added_rules[name], path, stage_name, source)
                report["waived"].append(entry)
                changed_paths.add(path)
                continue
            report["remaining_failures"].append(
                {"test": name or case.get("name", ""), "file": path.name, "reason": reason}
            )

    unfinished_path = result_dir / "unfinished_test.txt"
    if unfinished_path.exists():
        unfinished_tests = {
            line.strip().removeprefix(stage_name + "/")
            for line in unfinished_path.read_text().splitlines()
            if line.strip()
        }
        report["remaining_failures"].extend(
            {
                "test": name,
                "file": unfinished_path.name,
                "reason": "unfinished test missing from timeout report",
            }
            for name in sorted(unfinished_tests - reported_timeouts)
        )

    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "initial_tot_waives.txt").write_text(initial_waives)
    (audit_dir / "latest_tot_waives.txt").write_text(latest_waives)
    (audit_dir / "added_tot_waives.txt").write_text(
        "".join(
            f"{rule.nodeid} {rule.marker} ({rule.reason})\n"
            for _, rule in sorted(added_rules.items())
        )
    )
    for path in changed_paths:
        shutil.copyfile(path, audit_dir / (path.name + ".original"))
        _update_counts(trees[path].getroot())
        temporary = path.with_suffix(".xml.tmp")
        trees[path].write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
    report["all_failures_waived"] = bool(report["waived"]) and not report["remaining_failures"]
    (audit_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--initial-waives-url", required=True)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--pr-commit", required=True)
    parser.add_argument("--sm-version", type=int)
    args = parser.parse_args()
    names = set()
    for path in args.result_dir.glob("results*.xml"):
        if path.name == "results-stage.xml":
            continue
        for case in _failed_cases(ET.parse(path).getroot()):
            name = _test_name(case, args.stage_name)
            if name:
                names.add(name)
    if not names:
        return 1
    if not args.target_branch or not args.pr_commit:
        raise ValueError("Missing target branch or PR commit")
    expected_suffix = f"/waives_TOT_initial_{urllib.parse.quote(args.pr_commit, safe='')}.txt"
    if not urllib.parse.urlsplit(args.initial_waives_url).path.endswith(expected_suffix):
        raise ValueError("Initial TOT waive URL belongs to a different PR commit")
    with urllib.request.urlopen(args.initial_waives_url, timeout=15) as response:
        initial_waives = response.read().decode("utf-8")
    _parse_rules(initial_waives)
    latest_waives, snapshot = query_latest_waives(sorted(names), args.target_branch)
    report = reclassify_failures(
        args.result_dir,
        latest_waives,
        initial_waives,
        args.stage_name,
        {
            "branch": snapshot["branch"],
            "cutoff": snapshot["cutoff"],
            "project": WAIVE_PROJECT_NAME,
            "pr_commit": args.pr_commit,
            "initial_waives_url": args.initial_waives_url,
            "query_scope": "failed_tests",
        },
        args.sm_version if args.sm_version is not None else _detect_sm_version(),
    )
    (args.result_dir / "waive_refresh" / "opendb_snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )
    for entry in report["waived"]:
        print(f"[TOT-WAIVE] {entry['test']}: skipped; {entry['reason']}")
    return 0 if report["all_failures_waived"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
