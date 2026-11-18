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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import open_search_db
from open_search_db import WAIVE_PROJECT_NAME, OpenSearchDB
from test_common.waive import waive_candidates
from test_rerun import parse_name, update_junit_counts

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
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"(.+?)\s+(SKIP|XFAIL|UNSTABLE|TIMEOUT)(?:\s*\((.+?)\))?\s*(?:#.*)?", line
        )
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
            if not sm_version or sm != sm_version:
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
    classname = case.get("classname", "")
    name = parse_name(classname, case.get("name", ""), filename).removeprefix(stage_name + "/")
    if not name.startswith("unittest/") and classname.startswith(stage_name + ".unittest."):
        name = "unittest/" + name
    return name


def _waive_names(case: ET.Element, stage_name: str) -> list[str]:
    """Keep outer entries exact; only inner unit tests have parent-name fallbacks."""
    name = _test_name(case, stage_name)
    if not name:
        return []
    if name.startswith("unittest/") and case.get("file") != "test_unittests.py":
        return waive_candidates(name)
    return [name]


def _properties(node: ET.Element) -> dict[str, str]:
    return {p.get("name", ""): p.get("value", "") for p in node.findall("properties/property")}


def _read_reports(result_dir: Path) -> dict[Path, ET.ElementTree]:
    """Include MoE child reports without publishing their individual cases to Jenkins."""
    paths = set(result_dir.glob("results*.xml")) | set(
        result_dir.glob("moe-results-sub-unittests-*.xml")
    )
    return {path: ET.parse(path) for path in sorted(paths)}


def _unit_report_links(
    trees: dict[Path, ET.ElementTree], stage_name: str
) -> dict[ET.Element, Path]:
    """Link ordinary wrapper failures to complete child reports by explicit metadata."""
    by_name = {path.name: path for path in trees}
    links = {}
    for tree in trees.values():
        for case in _failed_cases(tree.getroot()):
            props = _properties(case)
            path = by_name.get(props.get("unittest_report"))
            if (
                case.get("file") != "test_unittests.py"
                or props.get("unittest_exit_code") != "1"
                or case.find("error") is not None
                or len(case.findall("failure")) != 1
                or path is None
                or not path.name.startswith(
                    ("results-sub-unittests-", "moe-results-sub-unittests-")
                )
            ):
                continue
            root = trees[path].getroot()
            suites = list(root.iter("testsuite"))
            if len(suites) != 1:
                continue
            child_props = _properties(suites[0])
            failures = _failed_cases(root)
            if (
                child_props.get("unittest_entry") == _test_name(case, stage_name)
                and props.get("unittest_report_id")
                and child_props.get("unittest_report_id") == props["unittest_report_id"]
                and child_props.get("pytest_session_complete") == "true"
                and failures
                and all(_test_name(child, stage_name) for child in failures)
            ):
                links[case] = path
    return links


def _failed_cases(root: ET.Element) -> list[ET.Element]:
    """Return cases with either a pytest failure or an execution error."""
    return [
        case
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]


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
        test_names: Rule names to look up, including inner unit-test ancestors.
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


def _jenkins_summary(report: dict) -> dict:
    """Prepare the summary and individual messages shown as Jenkins steps."""
    waived, retained = report["waived"], report["remaining_failures"]
    verdict = "all failures waived" if report["all_failures_waived"] else "failure retained"
    return {
        "summary": (
            f"{len({entry['test'] for entry in waived})} skipped, "
            f"{len({entry['test'] for entry in retained})} retained"
        ),
        "messages": [
            f"TOT waive result: {verdict}. Full report: "
            f"{report['stage_name']}/waive_refresh/report.json in the test-results artifact.",
            *(f"SKIPPED by new TOT waive: {e['test']}; {e['reason']}" for e in waived),
            *(f"FAILURE RETAINED: {e['test']}; {e['reason']}" for e in retained),
        ],
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
    *,
    trees: dict[Path, ET.ElementTree] | None = None,
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
        trees: Already parsed result reports, reused by the CLI.

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
    if trees is None:
        trees = _read_reports(result_dir)

    recorded = set()
    timeouts = set()
    for path, tree in trees.items():
        recorded.update(
            name
            for case in tree.iter("testcase")
            if (name := _test_name(case, stage_name))
            and (not name.startswith("unittest/") or case.get("file") == "test_unittests.py")
        )
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

    def matching_rule(case: ET.Element) -> tuple[WaiveRule | None, str]:
        names = _waive_names(case, stage_name)
        if any(name in initial_rules for name in names):
            return None, "already present in initial TOT"
        rule = next((latest_rules[name] for name in names if name in latest_rules), None)
        if rule is None:
            return None, "not covered by a new TOT waive"
        if rule.recorded_ms <= snapshot_ms:
            return None, "waive recorded before the initial TOT snapshot"
        return rule, ""

    links = _unit_report_links(trees, stage_name)
    child_failures = {path: _failed_cases(tree.getroot()) for path, tree in trees.items()}
    parent_rules = {path: rule for case, path in links.items() if (rule := matching_rule(case)[0])}
    resolved = {}
    failures = [(path, case) for path, cases in child_failures.items() for case in cases]
    # Resolve inner cases first, then their outer wrappers.
    for path, case in sorted(failures, key=lambda item: item[1].get("file") == "test_unittests.py"):
        name = _test_name(case, stage_name)
        rule, reason = matching_rule(case)
        if case.get("file") == "test_unittests.py":
            child_path = links.get(case)
            if child_path is None:
                rule, reason = None, "cannot prove a complete, associated inner unittest run"
            elif rule is None and all(child in resolved for child in child_failures[child_path]):
                child_rule = resolved[child_failures[child_path][0]]
                rule = WaiveRule(
                    child_rule.nodeid,
                    f"All failed inner tests waived; {child_rule.reason}",
                    child_rule.recorded_ms,
                )
        elif path in parent_rules:
            rule, reason = parent_rules[path], ""
        if path.name == "results-stage.xml":
            rule, reason = None, "synthetic failure"
        if rule is not None and not run_complete:
            rule = None
            reason = (
                "cannot prove every test ran: no shard test list"
                if expected is None
                else "cannot prove every test ran: some shard tests have no result"
            )
        if rule is not None:
            report["waived"].append(_apply_waive(case, rule, path, stage_name, source))
            changed_paths.add(path)
            resolved[case] = rule
        else:
            unattributed_failure = unattributed_failure or not name
            report["remaining_failures"].append(
                {"test": name or case.get("name", ""), "file": path.name, "reason": reason}
            )
    report["remaining_failures"].extend(
        {"test": name, "file": "", "reason": "no matching test result in any report"}
        for name in missing
    )

    isolated = set(isolated_tests or ())
    report["isolated_failures_waived"] = (
        any(entry["test"] in isolated for entry in report["waived"])
        and not unattributed_failure
        and not any(entry["test"] in isolated for entry in report["remaining_failures"])
    )
    report["has_unfinished_failures"] = any(
        entry["file"] == _TIMEOUT_REPORT for entry in report["remaining_failures"]
    )

    audit_dir.mkdir(parents=True, exist_ok=True)
    for path in changed_paths:
        shutil.copyfile(path, audit_dir / (path.name + ".original"))
        update_junit_counts(trees[path].getroot())
        temporary = path.with_suffix(".xml.tmp")
        trees[path].write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
    report["all_failures_waived"] = bool(report["waived"]) and not report["remaining_failures"]
    report["jenkins"] = _jenkins_summary(report)
    (audit_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _jenkins_arguments() -> argparse.Namespace:
    """Read Jenkins context and shard-list options without depending on Groovy parsing."""
    stage_name = os.environ["WAIVE_REFRESH_STAGE"]
    result_dir = Path(os.environ["WORKSPACE"]) / stage_name
    options = json.loads(os.environ["WAIVE_REFRESH_OPTIONS"])
    branch = os.environ.get("gitlabTargetBranch")
    if not branch:
        branch = json.loads(os.environ.get("globalVars") or "{}").get("target_branch")
    lists = options.get("expectedTestLists")
    if lists is None:
        lists = [
            options[kind] for kind in ("regular", "isolate") if options.get(kind + "Count", 0) > 0
        ]
    sm_version = options.get("smVersion")
    if sm_file := options.get("smVersionFile"):
        capability = Path(sm_file).read_text().strip()
        sm_version = (
            int(capability.replace(".", "")) if re.fullmatch(r"\d+\.\d+", capability) else 0
        )
    return argparse.Namespace(
        result_dir=result_dir,
        stage_name=stage_name,
        target_branch=branch,
        initial_waives=result_dir / "waive_refresh" / "initial_tot_waives.txt",
        sm_version=int(sm_version) if sm_version is not None else None,
        expected_tests=os.pathsep.join(lists),
        isolated_tests=options.get("isolatedTestList") or options.get("isolate", ""),
    )


def run(argv: list[str] | None = None) -> int:
    """Run the refresh for one stage.

    Args:
        argv: Command-line arguments; defaults to sys.argv[1:].

    Returns:
        EXIT_ALL_WAIVED or EXIT_FAILURES_RETAINED. Errors are raised.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jenkins", action="store_true", help="read context from Jenkins environment"
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--stage-name")
    parser.add_argument("--target-branch")
    parser.add_argument("--initial-waives", type=Path, help="initial TOT waive-list snapshot")
    parser.add_argument(
        "--sm-version", type=int, help="GPU SM; 0 means unknown, omitted detects the local GPU"
    )
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
    if args.jenkins:
        args = _jenkins_arguments()
    if not (args.result_dir and args.stage_name and args.target_branch and args.initial_waives):
        parser.error("result directory, stage name, target branch and initial waives are required")
    trees = _read_reports(args.result_dir)
    names = set()
    for path, tree in trees.items():
        if path.name != "results-stage.xml":
            cases = _failed_cases(tree.getroot())
            names.update(name for case in cases for name in _waive_names(case, args.stage_name))
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
        trees=trees,
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
    """Run the refresh and map execution errors to EXIT_UNAVAILABLE.

    Args:
        argv: Command-line arguments; defaults to sys.argv[1:].

    Returns:
        EXIT_ALL_WAIVED, EXIT_FAILURES_RETAINED or EXIT_UNAVAILABLE.
    """
    try:
        return run(argv)
    except Exception as error:
        traceback.print_exc()
        print(
            f"[TOT-WAIVE] Refresh unavailable, keeping the original failure: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
