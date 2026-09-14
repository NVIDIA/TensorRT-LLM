#!/usr/bin/env python3
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
"""Reject explicit CI-default TIMEOUT markers in CI test-db lists only."""

import re
import sys
from pathlib import Path

# Checked against getPytestBaseCommandLine() in jenkins/L0_Test.groovy below.
_DEFAULT_CI_TIMEOUT_MINUTES = 60


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ci_config = repo_root / "jenkins/L0_Test.groovy"
    # Recognize the current top-level Groovy function and its literal assignment.
    # Fail if that structure changes instead of silently trusting a stale default.
    source = re.sub(r"(?ms)^[ \t]*/\*.*?\*/", "", ci_config.read_text(encoding="utf-8"))
    functions = re.findall(
        r"(?ms)^def\s+getPytestBaseCommandLine\s*\([^)]*\)\s*\{(.*?)^\}",
        source,
    )
    assignments = (
        re.findall(r"(?m)^[ \t]*(?:def[ \t]+)?pytestTestTimeout[ \t]*=[ \t]*(.*)$", functions[0])
        if len(functions) == 1
        else []
    )
    timeout = (
        re.fullmatch(r"(['\"])([0-9]+)\1[ \t]*;?[ \t]*(?://[^\n]*)?", assignments[0])
        if len(assignments) == 1
        else None
    )
    if timeout is None:
        print(
            f"{ci_config}: Cannot identify a unique literal pytestTestTimeout in "
            "getPytestBaseCommandLine(); update the CI default timeout check.",
            file=sys.stderr,
        )
        return 1
    seconds = int(timeout.group(2))
    if seconds != _DEFAULT_CI_TIMEOUT_MINUTES * 60:
        print(
            f"{ci_config}: pytestTestTimeout={seconds} seconds does not match "
            f"_DEFAULT_CI_TIMEOUT_MINUTES={_DEFAULT_CI_TIMEOUT_MINUTES} minutes.",
            file=sys.stderr,
        )
        return 1

    ci_dir = repo_root / "tests/integration/test_lists/test-db"
    if not ci_dir.is_dir():
        print(f"Missing CI test-list directory: {ci_dir}", file=sys.stderr)
        return 1

    failed = False
    for path in sorted(ci_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".txt", ".yml", ".yaml"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(
                rf"\bTIMEOUT\s*\(\s*0*{_DEFAULT_CI_TIMEOUT_MINUTES}\s*\)", line.partition("#")[0]
            ):
                print(
                    f"{path}:{lineno}: Remove TIMEOUT ({_DEFAULT_CI_TIMEOUT_MINUTES}); "
                    f"{_DEFAULT_CI_TIMEOUT_MINUTES} minutes is the CI command-line default.",
                    file=sys.stderr,
                )
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
