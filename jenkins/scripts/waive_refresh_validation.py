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

"""Temporary, explicitly selected PR #21009 CPU validation fixtures."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = {
        "CPU-Generic-WaiveRefresh-Inner-1",
        "CPU-Generic-WaiveRefresh-Outer-1",
        "CPU-Generic-WaiveRefresh-Partial-1",
    }
    if args.stage not in allowed:
        parser.error("Only the three explicit validation stages are supported")
    waived = "unittest/executor/test_rpc_proxy.py"
    retained = "unittest/executor/test_waive_refresh_retained.py"
    fixture = args.root / "tests" / waived
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        "import pytest\npytestmark = pytest.mark.cpu_only\n\n"
        "def test_waive_refresh_pass():\n    pass\n\n"
        "def test_waive_refresh_a():\n    assert False, 'intentional validation failure A'\n\n"
        "def test_waive_refresh_b():\n    assert False, 'intentional validation failure B'\n"
    )
    entry = waived
    if "-Inner-" in args.stage:
        entry += ' -k "waive_refresh"'
    elif "-Partial-" in args.stage:
        (args.root / "tests" / retained).write_text(
            "import pytest\npytestmark = pytest.mark.cpu_only\n\n"
            "def test_waive_refresh_retained():\n"
            "    assert False, 'intentional uncovered validation failure'\n"
        )
        entry += " " + retained
    args.output.write_text(entry + "\n")
    print(f"[WAIVE-VALIDATION] Real unit wrapper entry: {entry}")


if __name__ == "__main__":
    main()
