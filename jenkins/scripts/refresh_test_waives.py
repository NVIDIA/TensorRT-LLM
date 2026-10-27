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
inputs, OpenDB or Artifactory unavailable, malformed reports, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
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

_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5
_UNITTEST_PREFIX = "test_unittests.py::test_unittests_v2["
# Trailing test-list directives, e.g. "test TIMEOUT (90) ISOLATION".
_LIST_DIRECTIVES = re.compile(r"(?:\s+(?:TIMEOUT\s*\(\s*\d+\s*\)|ISOLATION))+\s*$")
_UNFINISHED_FILES = {"results-timeout.xml", "unfinished_test.txt"}


@dataclass(frozen=True)
class WaiveRule:
    nodeid: str
    marker: str
    reason: str


def _format_rule(rule: WaiveRule) -> str:
    return f"{rule.nodeid} {rule.marker} ({rule.reason})"


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


def _scoped_test_name(nodeid: str, stage_name: str, sm_version: int | None) -> str | None:
    """Return the test a waive entry skips on this stage, or None if it does not apply.

    Mirrors parse_test_line()/parse_test_name() in
    tests/integration/defs/test_list_parser.py, which is what pytest uses to
    apply waives.txt: an entry only skips a test when its transformed name
    equals the "<stage>/<test>" node id. Keep the two in sync. (The parser
    strips "full:" with lstrip(); that only differs for scopes starting with
    a lowercase f/u/l, and no such scope exists.)
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
    result = {}
    for rule in rules:
        name = _scoped_test_name(rule.nodeid, stage_name, sm_version)
        if name:
            result[name] = rule
    return result


def _read_test_list(path: Path) -> list[str]:
    """Read a shard test list as test names, without TIMEOUT/ISOLATION directives."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = _LIST_DIRECTIVES.sub("", line).strip()
        if name:
            names.append(name)
    return names


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


def _query_opendb(body: dict) -> dict:
    """Return the decoded OpenDB response, retrying transport errors.

    Non-2xx responses are already retried inside OpenSearchDB.queryFromOpenSearchDB.
    """
    if open_search_db.requests is None:
        raise RuntimeError("Python package 'requests' is not installed; cannot query OpenDB")
    for attempt in range(_ATTEMPTS):
        try:
            response = OpenSearchDB.queryFromOpenSearchDB(body, WAIVE_PROJECT_NAME)
        except OSError:  # requests.RequestException derives from OSError
            if attempt + 1 == _ATTEMPTS:
                raise
            time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
            continue
        if response is None:
            raise RuntimeError("OpenDB waive query failed (is OPEN_SEARCH_DB_BASE_URL set?)")
        return response.json()
    raise AssertionError("unreachable")


def query_latest_waives(test_names: list[str], branch: str) -> tuple[list[WaiveRule], dict]:
    """Read the latest waive state per test and scope from OpenDB.

    The latest state is selected before filtering for active waives, because
    filtering for Waived first would resurrect revoked rules. All pages use the
    same ingestion cutoff so updates arriving during the query are excluded.
    Rules are built from the record fields directly; free-form reasons are never
    re-parsed as waive-list text.

    Args:
        test_names: Failed test names to look up.
        branch: Target branch whose waive states are read.

    Returns:
        The active exact SKIP rules, and a snapshot of the query and its records
        for the audit trail.
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
    wanted = set(test_names)
    records = []
    while True:
        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {"states": {"composite": composite, "aggs": {"latest": {"top_hits": latest}}}},
        }
        data = _query_opendb(body)
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
        rules.append(WaiveRule(name, "SKIP", " ".join(str(reason).split())))
    return rules, {"branch": branch, "cutoff": cutoff, "records": records}


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
    properties = case.find("properties")
    if properties is None:
        properties = ET.Element("properties")
        case.insert(0, properties)
    for key, value in {"waive_rule": rule.nodeid, **source}.items():
        ET.SubElement(properties, "property", {"name": key, "value": value})
    message = f"Skipped by new TOT waive: {rule.reason}"
    skipped = ET.Element("skipped", {"type": "tot-waive", "message": message})
    case.insert(list(case).index(properties) + 1, skipped)
    return {
        "test": _test_name(case, stage_name),
        "file": result_file.name,
        "reason": rule.reason,
        "rule": rule.nodeid,
    }


def reclassify_failures(
    result_dir: Path,
    latest_waives: str | Iterable[WaiveRule],
    initial_waives: str,
    stage_name: str,
    source: dict[str, str],
    sm_version: int | None = None,
    expected_tests: Iterable[str] | None = None,
    isolated_tests: Iterable[str] | None = None,
) -> dict:
    """Replace newly waived failures with audited skips; retain all other errors.

    All input reports are parsed before any are changed. Original XML files are
    retained with a .original suffix so Jenkins results*.xml globs cannot ingest
    them. Historical rerun reports remain unchanged. Waives already covering the
    stage in its original TOT snapshot cannot grant a new skip, even if their
    reason, scope spelling, or marker changes.

    A run that pytest did not finish (a crash, or a timeout during setup or the
    test body) leaves the remaining tests without results, and a setup timeout
    leaves no unfinished-test record either. Nothing is waived unless the shard
    test lists prove the run complete: every listed test has a result in the
    reports and every unfinished test is listed.

    Args:
        result_dir: Stage result directory holding results*.xml and unfinished_test.txt.
        latest_waives: Latest target-branch rules (from OpenDB) or waive-list text.
        initial_waives: Waive-list text of the target branch when the PR's list was merged.
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
    if isinstance(latest_waives, str):
        latest_text = latest_waives
        latest_rule_list = _parse_rules(latest_waives)
    else:
        latest_rule_list = list(latest_waives)
        latest_text = "".join(_format_rule(rule) + "\n" for rule in latest_rule_list)
    latest_rules = _matching_rules(latest_rule_list, stage_name, sm_version)
    initial_rules = _matching_rules(_parse_rules(initial_waives), stage_name, sm_version)
    added_rules = {
        name: rule
        for name, rule in latest_rules.items()
        if rule.marker == "SKIP" and name not in initial_rules
    }
    report_files = sorted(result_dir.glob("results*.xml"))
    trees = {path: ET.parse(path) for path in report_files}

    recorded = set()
    reported_timeouts = set()
    for path, tree in trees.items():
        for case in tree.getroot().iter("testcase"):
            name = _test_name(case, stage_name)
            if name:
                recorded.add(name)
        if path.name == "results-timeout.xml":
            reported_timeouts.update(
                _test_name(case, stage_name) for case in _failed_cases(tree.getroot())
            )
    unfinished_path = result_dir / "unfinished_test.txt"
    unfinished_listed = set()
    if unfinished_path.exists():
        unfinished_listed = {
            line.strip().removeprefix(stage_name + "/")
            for line in unfinished_path.read_text().splitlines()
            if line.strip()
        }
    unfinished = sorted((reported_timeouts | unfinished_listed) - {""})
    expected = set(expected_tests or ()) or None
    missing = sorted(expected - recorded) if expected is not None else []
    # Unfinished tests outside the lists mean the lists do not describe this run.
    run_complete = expected is not None and not missing and set(unfinished) <= expected

    report = {
        "stage_name": stage_name,
        "sm_version": sm_version,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "added_waives": [
            {"test": name, "rule": rule.nodeid, "reason": rule.reason}
            for name, rule in sorted(added_rules.items())
        ],
        "unfinished_tests": unfinished,
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
            elif name not in added_rules:
                reason = "not covered by a new TOT waive"
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
        {
            "test": name,
            "file": unfinished_path.name,
            "reason": "unfinished test missing from timeout report",
        }
        for name in sorted(unfinished_listed - reported_timeouts)
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
        entry["file"] in _UNFINISHED_FILES or not entry["file"]
        for entry in report["remaining_failures"]
    )

    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "initial_tot_waives.txt").write_text(initial_waives)
    (audit_dir / "latest_tot_waives.txt").write_text(latest_text)
    (audit_dir / "added_tot_waives.txt").write_text(
        "".join(_format_rule(rule) + "\n" for _, rule in sorted(added_rules.items()))
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


def _download_initial_waives(url: str) -> str:
    """Download the initial TOT snapshot, retrying transient Artifactory errors."""
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt + 1 == _ATTEMPTS:
                raise
        except OSError:
            if attempt + 1 == _ATTEMPTS:
                raise
        time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def _read_test_lists(value: str) -> list[str] | None:
    if not value:
        return None
    names = []
    for path in value.split(os.pathsep):
        if path:
            names.extend(_read_test_list(Path(path)))
    return names


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
    parser.add_argument("--initial-waives-url", required=True)
    parser.add_argument("--target-branch", required=True)
    parser.add_argument("--pr-commit", required=True)
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
    if (args.result_dir / "waive_refresh" / "report.json").exists():
        raise ValueError("Waive refresh has already been applied to this stage attempt")
    names = set()
    for path in args.result_dir.glob("results*.xml"):
        if path.name == "results-stage.xml":
            continue
        for case in _failed_cases(ET.parse(path).getroot()):
            name = _test_name(case, args.stage_name)
            if name:
                names.add(name)
    if not names:
        print("[TOT-WAIVE] No named test failures to check.")
        return EXIT_FAILURES_RETAINED
    if not args.target_branch or not args.pr_commit:
        raise ValueError("Missing target branch or PR commit")
    expected_suffix = f"/waives_TOT_initial_{urllib.parse.quote(args.pr_commit, safe='')}.txt"
    if not urllib.parse.urlsplit(args.initial_waives_url).path.endswith(expected_suffix):
        raise ValueError("Initial TOT waive URL belongs to a different PR commit")
    expected_tests = _read_test_lists(args.expected_tests)
    if expected_tests is None:
        print("[TOT-WAIVE] No shard test list to prove a complete run; keeping failures.")
        return EXIT_FAILURES_RETAINED
    isolated_tests = _read_test_lists(args.isolated_tests)
    initial_waives = _download_initial_waives(args.initial_waives_url)
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
        expected_tests,
        isolated_tests,
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
    except (OSError, ValueError, RuntimeError, LookupError, ET.ParseError) as error:
        traceback.print_exc()
        print(
            f"[TOT-WAIVE] Refresh unavailable, keeping the original failure: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE


if __name__ == "__main__":
    raise SystemExit(main())
