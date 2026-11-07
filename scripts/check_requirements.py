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
"""Check the requirement files at the repository root, which setup.py reads for wheel metadata."""

import pathlib
import re
import sys

from packaging.requirements import InvalidRequirement, Requirement

# The pip options that setup.py reads: it skips -r and -c lines, and takes the URL of -i and
# --extra-index-url only when a space separates it. It adds every other line to the dependencies.
_SKIPPED_OPTIONS = ("-r", "-c")
_INDEX_OPTION = re.compile(r"(-i|--extra-index-url)\s+\S+")


def _check_line(line: str) -> str | None:
    requirement = re.sub(r"(^|\s)#.*", "", line).strip()
    if (
        not requirement
        or requirement.startswith(_SKIPPED_OPTIONS)
        or _INDEX_OPTION.fullmatch(requirement)
    ):
        return None
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement as exc:
        return f"Not a requirement on a package: {str(exc).splitlines()[0]}"
    if parsed.url:
        return "Direct dependency on a URL instead of a released package version."
    return None


def main(argv: list[str]) -> int:
    has_violations = False
    for file_arg in argv[1:]:
        path = pathlib.Path(file_arg)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            message = _check_line(line)
            if message:
                has_violations = True
                print(f"{path}:{lineno}: {message}")

    if has_violations:
        print("\nRequirement check failed.")
        print("The tensorrt_llm wheel on PyPI is a stub that cannot install direct dependencies")
        print("(git+https, URLs, local paths), so depend on released package versions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
