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
"""Reject redundant 60-minute TIMEOUT markers in CI test-db lists only."""

import re
import sys
from pathlib import Path


def main() -> int:
    ci_dir = Path(__file__).resolve().parents[1] / "tests/integration/test_lists/test-db"
    if not ci_dir.is_dir():
        print(f"Missing CI test-list directory: {ci_dir}", file=sys.stderr)
        return 1

    failed = False
    for path in sorted(ci_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".txt", ".yml", ".yaml"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bTIMEOUT\s*\(\s*0*60\s*\)", line.partition("#")[0]):
                print(
                    f"{path}:{lineno}: Remove TIMEOUT (60); 60 minutes is the default.",
                    file=sys.stderr,
                )
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
