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
"""Prints a per-stage CBTS coverage liveness line to the CI console log."""

import argparse
import glob
import sqlite3
import sys
import xml.etree.ElementTree as ET


def count_covered_cases(cov_glob: str) -> int:
    """Union of distinct non-empty contexts across all matching .coverage files."""
    contexts = set()
    for path in glob.glob(cov_glob):
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                rows = con.execute("SELECT context FROM context WHERE context != ''").fetchall()
            finally:
                con.close()
        except sqlite3.DatabaseError as e:
            print(f"CBTS coverage summary: skipping unreadable {path}: {e}")
            continue
        contexts |= {r[0] for r in rows}
    return len(contexts)


def count_ran_cases(junit_glob: str) -> int:
    """Number of ``<testcase>`` elements across all matching junit XMLs."""
    ran = 0
    for path in glob.glob(junit_glob):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            print(f"CBTS coverage summary: skipping unparsable {path}: {e}")
            continue
        ran += len(root.findall(".//testcase"))
    return ran


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cov-glob", required=True, help="glob for per-process .coverage.* files")
    parser.add_argument("--junit-glob", required=True, help="glob for junit results*.xml files")
    parser.add_argument("--stage", default="", help="stage name for the log line")
    args = parser.parse_args()

    covered = count_covered_cases(args.cov_glob)
    ran = count_ran_cases(args.junit_glob)

    label = f" [{args.stage}]" if args.stage else ""
    pct = f" ({100 * covered // ran}%)" if ran else ""
    print(f"CBTS coverage{label}: {covered}/{ran} test cases produced coverage data{pct}")
    if ran > 0 and covered == 0:
        print(
            f"WARNING: CBTS coverage{label} produced ZERO coverage contexts -- "
            "capture likely broken (plugin not loaded, PYTHONPATH/env gate wrong, "
            "or source filter excluded everything)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
