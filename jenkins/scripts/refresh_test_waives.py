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

"""Reclassify named test failures and timeouts covered by new target-branch waives.

Exit status: 0 when every failure is covered by a new waive, 1 when at least one
failure is retained, and 2 when the refresh itself could not run (missing
inputs, OpenDB unavailable, malformed reports, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import open_search_db
from open_search_db import WAIVE_PROJECT_NAME, OpenSearchDB
from test_rerun import parse_name

EXIT_ALL_WAIVED = 0
EXIT_FAILURES_RETAINED = 1
EXIT_UNAVAILABLE = 2

_TIMEOUT_REPORT = "results-timeout.xml"
_SNAPSHOT_TIME = re.compile(r"^# snapshot-time: (\S+)$", re.MULTILINE)
_UNITTEST_PREFIX = "test_unittests.py::test_unittests_v2["
# Trailing test-list directives, e.g. "test TIMEOUT (90) ISOLATION".
_LIST_DIRECTIVES = re.compile(r"(?:\s+(?:TIMEOUT\s*\(\s*\d+\s*\)|ISOLATION))+\s*$")


@dataclass(frozen=True)
class WaiveRule:
    nodeid: str
    reason: str
    # When the waive tracker recorded the rule (ts_created), in epoch milliseconds.
    recorded_ms: int = 0


def _snapshot_ms(initial_waives: str) -> int:
    """Return the "# snapshot-time: <ISO 8601>" header of the initial TOT snapshot in epoch ms."""
    match = _SNAPSHOT_TIME.search(initial_waives)
    if not match:
        raise ValueError("Initial TOT snapshot has no snapshot time")
    moment = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Initial TOT snapshot time has no time zone")
    return int(moment.timestamp() * 1000)


def _parse_rules(content: str) -> list[WaiveRule]:
    """Parse baseline markers, allowing omitted reasons except for numeric TIMEOUT values."""
    rules = []
    for line in content.splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"(.+?)\s+(SKIP|XFAIL|UNSTABLE|TIMEOUT)(?:\s*\((.+)\))?", line)
        if not match:
            raise ValueError(f"Unsupported waive rule: {line}")
        nodeid, marker, reason = match.groups()
        if marker == "TIMEOUT" and (reason is None or not reason.isdigit()):
            raise ValueError(f"Invalid TIMEOUT rule: {line}")
        rules.append(WaiveRule(nodeid, reason or ""))
    return rules


def _scoped_test_name(nodeid: str, stage_name: str, sm_version: int | None) -> str | None:
    """Return the test a waive entry skips on this stage, or None if it does not apply.

    Keep scope expansion in sync with parse_test_line()/parse_test_name() in
    tests/integration/defs/test_list_parser.py: the expanded name must equal
    the "<stage>/<test>" node id.
    """
    full_prefix = ""
    match = re.match(r"(full:.*?/)(.+)", nodeid)
    if match:
        full_prefix, nodeid = match.groups()
    if nodeid.startswith("unittest/"):
        nodeid = f"{_UNITTEST_PREFIX}{nodeid}]"
    name = full_prefix + nodeid
    if name.startswith("full:"):
        name = name.removeprefix("full:")
        sm_match = re.match(r"sm(\d+)/", name)
        if sm_match:
            sm = int(sm_match.group(1))
            if sm_version is None or sm != sm_version:
                return None
            name = name.replace(f"sm{sm}", stage_name)
        else:
            platform = stage_name.split("-")[0]
            if platform not in name:
                return None
            name = name.replace(platform, stage_name)
    else:
        name = f"{stage_name}/{name}"
    if not name.startswith(stage_name + "/"):
        return None
    name = name[len(stage_name) + 1 :]
    if name.startswith(_UNITTEST_PREFIX) and name.endswith("]"):
        name = name[len(_UNITTEST_PREFIX) : -1]
    return name


def _matching_rules(
    rules: Iterable[WaiveRule], stage_name: str, sm_version: int | None
) -> dict[str, WaiveRule]:
    """Index applicable rules by the test names used in this stage's reports."""
    result = {}
    for rule in rules:
        name = _scoped_test_name(rule.nodeid, stage_name, sm_version)
        if name:
            result[name] = rule
    return result


def _read_test_lists(value: str) -> list[str] | None:
    """Read os.pathsep-separated shard test lists as test names, without directives."""
    names = []
    for path in filter(None, value.split(os.pathsep)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if name := _LIST_DIRECTIVES.sub("", line).strip():
                names.append(name)
    return names or None


def _test_name(case: ET.Element, stage_name: str) -> str:
    """Recover a test's unprefixed node ID from JUnit attributes."""
    filename = case.get("file", "")
    if not filename:
        return ""
    name = parse_name(case.get("classname", ""), case.get("name", ""), filename)
    return name.removeprefix(stage_name + "/")


def _failed_cases(root: ET.Element) -> list[ET.Element]:
    """Return cases with either a pytest failure or an execution error."""
    return [
        case
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]


def _update_counts(root: ET.Element) -> None:
    """Recount suites the same way as test_rerun.merge_junit_xmls (one outcome per case)."""
    for node in reversed(list(root.iter())):
        if node.tag not in {"testsuites", "testsuite"}:
            continue
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        for case in node.iter("testcase"):
            counts["tests"] += 1
            if case.find("failure") is not None:
                counts["failures"] += 1
            elif case.find("error") is not None:
                counts["errors"] += 1
            elif case.find("skipped") is not None:
                counts["skipped"] += 1
        for attribute, value in counts.items():
            node.set(attribute, str(value))


def _latest_states(records: list[dict]) -> list[dict]:
    """Treat missing and empty scopes as the same rule, retaining its latest state."""
    latest = {}
    for record in records:
        fields = record["_source"]
        key = (fields["s_turtle_name"], fields.get("s_full_scope") or "")
        previous = latest.get(key)
        if previous is None or record["sort"] > previous["sort"]:
            latest[key] = record
        elif (
            record["sort"] == previous["sort"]
            and fields["s_status"] != (previous["_source"]["s_status"])
        ):
            raise ValueError("Conflicting OpenDB waive states have the same timestamp")
    return list(latest.values())


def query_latest_waives(test_names: list[str], branch: str) -> tuple[list[WaiveRule], dict]:
    """Read the latest waive state per test and scope from OpenDB.

    The latest state is selected before filtering for active waives, because
    filtering for Waived first would resurrect revoked rules. All pages use the
    same ingestion cutoff so updates arriving during the query are excluded.

    Args:
        test_names: Failed test names to look up.
        branch: Target branch whose waive states are read.

    Returns:
        The active exact rules with the time the tracker recorded each, and a
        snapshot of the query and its records for the audit trail.
    """
    if open_search_db.requests is None:
        raise RuntimeError("Python package 'requests' is not installed; cannot query OpenDB")
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
    wanted = set(test_names)
    records = []
    while True:
        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {"states": {"composite": composite, "aggs": {"latest": {"top_hits": latest}}}},
        }
        # Non-2xx responses are retried inside queryFromOpenSearchDB.
        response = OpenSearchDB.queryFromOpenSearchDB(body, WAIVE_PROJECT_NAME)
        if response is None:
            raise RuntimeError("OpenDB waive query failed (is OPEN_SEARCH_DB_BASE_URL set?)")
        data = response.json()
        if data.get("timed_out") is not False or data.get("_shards", {}).get("failed", 1):
            raise RuntimeError("OpenDB returned incomplete waive states")
        states = data["aggregations"]["states"]
        for bucket in states["buckets"]:
            record = bucket["latest"]["hits"]["hits"][0]
            fields = record["_source"]
            if fields["s_branch"] != branch or fields["s_turtle_name"] not in wanted:
                raise ValueError("OpenDB returned a different branch or test")
            records.append(record)
        next_after = states.get("after_key")
        if not states["buckets"] or next_after is None:
            break
        if composite.get("after") == next_after:
            raise RuntimeError("OpenDB waive pagination did not advance")
        composite = {**composite, "after": next_after}

    rules = []
    for record in _latest_states(records):
        fields = record["_source"]
        if fields["s_status"] != "Waived" or fields.get("s_match_type") != "exact":
            continue
        name = fields["s_turtle_name"]
        if fields.get("s_full_scope"):
            name = f"full:{fields['s_full_scope']}/{name}"
        reason = fields.get("s_reason") or fields.get("s_bug_url") or "OpenDB waive"
        # sort[0] is ts_created in epoch milliseconds, the minimum value when it is missing.
        recorded_ms = int((record.get("sort") or [0])[0])
        rules.append(WaiveRule(name, " ".join(str(reason).split()), recorded_ms))
    return rules, {"branch": branch, "cutoff": cutoff, "records": records}


def _detect_sm_version() -> int | None:
    """Read the local GPU compute capability, or return None when unavailable."""
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
    """Replace a failure with a skip, keeping the raw error for result collectors."""
    failures = case.findall("failure") + case.findall("error")
    original_error = {
        "waive_original_status": "FAILED",
        "waive_original_error_message": "\n".join(
            failure.get("message", "") for failure in failures
        ),
        "waive_original_error_stack_trace": "\n".join(failure.text or "" for failure in failures),
    }
    for failure in failures:
        case.remove(failure)
    properties = case.find("properties")
    if properties is None:
        properties = ET.Element("properties")
        case.insert(0, properties)
    for key, value in {
        "waive_rule": rule.nodeid,
        "waive_reason": rule.reason,
        **source,
        **original_error,
    }.items():
        ET.SubElement(properties, "property", {"name": key, "value": value})
    message = f"Skipped by new TOT waive: {rule.reason}"
    skipped = ET.Element("skipped", {"type": "tot-waive", "message": message})
    case.insert(list(case).index(properties) + 1, skipped)
    return {
        "test": _test_name(case, stage_name),
        "file": result_file.name,
        "reason": rule.reason,
        "rule": rule.nodeid,
        "recorded_ms": rule.recorded_ms,
    }


def reclassify_failures(
    result_dir: Path,
    latest_waives: list[WaiveRule],
    initial_waives: str,
    stage_name: str,
    source: dict[str, str],
    sm_version: int | None = None,
    expected_tests: Iterable[str] | None = None,
    isolated_tests: Iterable[str] | None = None,
) -> dict:
    """Replace newly waived failures with audited skips; retain all other errors.

    A new waive must be absent from the initial TOT and recorded after its
    snapshot. Every expected test needs a result, including setup/body timeouts.
    All reports are parsed before any are changed; originals are backed up with
    a .original suffix, outside Jenkins results*.xml globs.

    Args:
        result_dir: Stage result directory holding the results*.xml reports.
        latest_waives: Latest target-branch rules from OpenDB.
        initial_waives: Waive-list text of the target branch when the PR's list was
            merged, headed by "# snapshot-time: <ISO 8601>".
        stage_name: Stage name used as the pytest --test-prefix.
        source: Provenance recorded on every waived test case.
        sm_version: GPU compute capability without the dot, for "full:smNN/" scopes.
        expected_tests: Test names from the shard test lists this stage ran.
            Without them no failure is waived.
        isolated_tests: Isolated test names. "isolated_failures_waived" is true
            only when an isolated test was waived and no retained failure belongs
            to an isolated test or cannot be attributed to a test.

    Returns:
        The decision report, also written to waive_refresh/report.json.
    """
    audit_dir = result_dir / "waive_refresh"
    if (audit_dir / "report.json").exists():
        raise ValueError("Waive refresh has already been applied to this stage attempt")
    initial_rules = _matching_rules(_parse_rules(initial_waives), stage_name, sm_version)
    latest_rules = _matching_rules(latest_waives, stage_name, sm_version)
    snapshot_ms = _snapshot_ms(initial_waives)
    added_rules = {
        name: rule
        for name, rule in latest_rules.items()
        if name not in initial_rules and rule.recorded_ms > snapshot_ms
    }
    trees = {path: ET.parse(path) for path in sorted(result_dir.glob("results*.xml"))}

    recorded = set()
    timeouts = set()
    for path, tree in trees.items():
        recorded.update(filter(None, (_test_name(c, stage_name) for c in tree.iter("testcase"))))
        if path.name == _TIMEOUT_REPORT:
            cases = _failed_cases(tree.getroot())
            timeouts.update(filter(None, (_test_name(case, stage_name) for case in cases)))
    expected = set(expected_tests or ()) or None
    missing = sorted(expected - recorded) if expected is not None else []
    # Timed-out tests outside the lists mean the lists do not describe this run.
    run_complete = expected is not None and not missing and timeouts <= expected

    report = {
        "stage_name": stage_name,
        "sm_version": sm_version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "snapshot_ms": snapshot_ms,
        "added_waives": [
            {
                "test": name,
                "rule": rule.nodeid,
                "reason": rule.reason,
                "recorded_ms": rule.recorded_ms,
            }
            for name, rule in sorted(added_rules.items())
        ],
        "unfinished_tests": sorted(timeouts),
        "missing_tests": missing,
        "waived": [],
        "remaining_failures": [],
    }
    changed_paths = set()
    unattributed_failure = False
    for path, tree in trees.items():
        for case in _failed_cases(tree.getroot()):
            name = _test_name(case, stage_name)
            if path.name == "results-stage.xml":
                reason = "synthetic failure"
            elif name in initial_rules:
                reason = "already present in initial TOT"
            elif name not in latest_rules:
                reason = "not covered by a new TOT waive"
            elif name not in added_rules:
                reason = "waive recorded before the initial TOT snapshot"
            elif not run_complete:
                reason = (
                    "cannot prove every test ran: no shard test list"
                    if expected is None
                    else "cannot prove every test ran: some shard tests have no result"
                )
            else:
                entry = _apply_waive(case, added_rules[name], path, stage_name, source)
                report["waived"].append(entry)
                changed_paths.add(path)
                continue
            unattributed_failure = unattributed_failure or not name
            report["remaining_failures"].append(
                {"test": name or case.get("name", ""), "file": path.name, "reason": reason}
            )
    report["remaining_failures"].extend(
        {"test": name, "file": "", "reason": "not executed: no result in any report"}
        for name in missing
    )

    isolated = set(isolated_tests or ())
    report["isolated_failures_waived"] = (
        any(entry["test"] in isolated for entry in report["waived"])
        and not unattributed_failure
        and not any(entry["test"] in isolated for entry in report["remaining_failures"])
    )
    report["has_unfinished_failures"] = any(
        entry["file"] in {_TIMEOUT_REPORT, ""} for entry in report["remaining_failures"]
    )

    audit_dir.mkdir(parents=True, exist_ok=True)
    for path in changed_paths:
        shutil.copyfile(path, audit_dir / (path.name + ".original"))
        _update_counts(trees[path].getroot())
        temporary = path.with_suffix(".xml.tmp")
        trees[path].write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
    report["all_failures_waived"] = bool(report["waived"]) and not report["remaining_failures"]
    (audit_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(argv: list[str] | None = None) -> int:
    """Run the refresh for one stage.

    Args:
        argv: Command-line arguments; defaults to sys.argv[1:].

    Returns:
        EXIT_ALL_WAIVED or EXIT_FAILURES_RETAINED. Errors are raised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument(
        "--initial-waives", type=Path, required=True, help="initial TOT waive-list snapshot"
    )
    parser.add_argument("--sm-version", type=int)
    parser.add_argument(
        "--expected-tests",
        default="",
        help=f"{os.pathsep!r}-separated shard test lists; without them no failure is waived",
    )
    parser.add_argument(
        "--isolated-tests",
        default="",
        help="isolated test list used to attribute failures to the isolated stage",
    )
    args = parser.parse_args(argv)
    names = set()
    for path in args.result_dir.glob("results*.xml"):
        if path.name != "results-stage.xml":
            cases = _failed_cases(ET.parse(path).getroot())
            names.update(filter(None, (_test_name(case, args.stage_name) for case in cases)))
    if not names:
        print("[TOT-WAIVE] No named test failures to check.")
        return EXIT_FAILURES_RETAINED
    expected_tests = _read_test_lists(args.expected_tests)
    if expected_tests is None:
        print("[TOT-WAIVE] No shard test list to prove a complete run; keeping failures.")
        return EXIT_FAILURES_RETAINED
    initial_waives = args.initial_waives.read_text(encoding="utf-8")
    _snapshot_ms(initial_waives)
    latest_waives, snapshot = query_latest_waives(sorted(names), args.target_branch)
    report = reclassify_failures(
        args.result_dir,
        latest_waives,
        initial_waives,
        args.stage_name,
        {"branch": snapshot["branch"], "cutoff": snapshot["cutoff"], "project": WAIVE_PROJECT_NAME},
        args.sm_version if args.sm_version is not None else _detect_sm_version(),
        expected_tests,
        _read_test_lists(args.isolated_tests),
    )
    (args.result_dir / "waive_refresh" / "opendb_snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n"
    )
    for entry in report["waived"]:
        print(f"[TOT-WAIVE] {entry['test']}: skipped; {entry['reason']}")
    for entry in report["remaining_failures"]:
        print(f"[TOT-WAIVE] {entry['test']}: retained; {entry['reason']}")
    return EXIT_ALL_WAIVED if report["all_failures_waived"] else EXIT_FAILURES_RETAINED


def main(argv: list[str] | None = None) -> int:
    """Run the refresh and map expected failures to EXIT_UNAVAILABLE.

    Args:
        argv: Command-line arguments; defaults to sys.argv[1:].

    Returns:
        EXIT_ALL_WAIVED, EXIT_FAILURES_RETAINED or EXIT_UNAVAILABLE.
    """
    try:
        return run(argv)
    except (OSError, ValueError, RuntimeError, LookupError, TypeError, ET.ParseError) as error:
        traceback.print_exc()
        print(
            f"[TOT-WAIVE] Refresh unavailable, keeping the original failure: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
