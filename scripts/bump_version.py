# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set the TensorRT-LLM version in every file that records it, or check that they agree.

Exit codes:
  0  the files are at VERSION; with --check, every record matches tensorrt_llm/version.py
  1  an error; with --check, a record that does not match tensorrt_llm/version.py
  2  invalid arguments
  3  --from-version is given and tensorrt_llm/version.py is at another version
  4  --from-version is given and tensorrt_llm/version.py is already at VERSION
"""

import argparse
import re
import runpy
import sys
from pathlib import Path

EXIT_VERSION_MISMATCH = 3
EXIT_ALREADY_AT_VERSION = 4

VERSION_FILE = "tensorrt_llm/version.py"
# Every file that records the package version, with a pattern whose first group is that version.
# When a file starts recording the version, add it here.
VERSION_PATTERNS = {
    VERSION_FILE: re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE),
    "examples/constraints.txt": re.compile(r"^tensorrt_llm==(\S+)$", re.MULTILINE),
    "README.md": re.compile(r"img\.shields\.io/badge/release-([^-]+)-"),
}
# The serving config schemas record the version in their $id and title. Their generator
# updates and checks that metadata without importing TensorRT-LLM.
SCHEMA_GENERATOR = "scripts/generate_trtllm_serve_schemas.py"
SCHEMA_DIR = "tensorrt_llm/schemas"

_ROOT = Path(__file__).resolve().parents[1]
# A canonical PEP 440 version (PEP 440, Appendix B), optionally with a local segment.
_CANONICAL_VERSION = re.compile(
    r"([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?"
    r"(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?(\+[a-z0-9]+(\.[a-z0-9]+)*)?"
)


def _full_version(value: str) -> str:
    if not _CANONICAL_VERSION.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a full canonical PEP 440 version, such as 1.4.0rc5"
        )
    return value


def _replace_versions(text: str, pattern: re.Pattern, version: str) -> str:
    def replace(match: re.Match) -> str:
        start, end = match.span(1)
        return match.string[match.start() : start] + version + match.string[end : match.end()]

    return pattern.sub(replace, text)


def _sync_schemas(check: bool) -> None:
    generator = runpy.run_path(str(_ROOT / SCHEMA_GENERATOR))
    generator["sync_schema_version_metadata"](_ROOT / SCHEMA_DIR, check=check)


def read_version() -> str:
    """Return the version in tensorrt_llm/version.py."""
    match = VERSION_PATTERNS[VERSION_FILE].search((_ROOT / VERSION_FILE).read_text())
    if not match:
        raise ValueError(f"No __version__ in {VERSION_FILE}")
    return match.group(1)


def set_version(version: str, dry_run: bool = False) -> None:
    """Write the version into every file that records it.

    Files that are not in the tree are skipped, since a Docker wheel build copies only part of it.
    """
    for path, pattern in VERSION_PATTERNS.items():
        file = _ROOT / path
        if not file.is_file():
            continue
        text = file.read_text()
        new_text = _replace_versions(text, pattern, version)
        if new_text != text:
            print(f"{'Would update' if dry_run else 'Updating'} {path}")
            if not dry_run:
                file.write_text(new_text)
    if (_ROOT / SCHEMA_DIR).is_dir():
        if dry_run:
            print(f"Would sync the version metadata in {SCHEMA_DIR}")
        else:
            # The generator reads the version from tensorrt_llm/version.py, written above.
            _sync_schemas(check=False)


def check() -> list[str]:
    """Return the records that do not match the version in tensorrt_llm/version.py."""
    version = read_version()
    problems = []
    for path, pattern in VERSION_PATTERNS.items():
        file = _ROOT / path
        found = pattern.findall(file.read_text()) if file.is_file() else []
        if not found:
            problems.append(f"{path}: no version found")
        problems += [f"{path}: {other}" for other in found if other != version]
    try:
        _sync_schemas(check=True)
    except ValueError as error:
        problems.append(str(error))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "version",
        nargs="?",
        type=_full_version,
        help="full version to set, e.g. 1.4.0rc5 or 1.4.0rc5.dev2026110700",
    )
    parser.add_argument(
        "--from-version",
        type=_full_version,
        help="change the files only if tensorrt_llm/version.py is at this version",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the result without changing files"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check that every record matches the version in tensorrt_llm/version.py",
    )
    args = parser.parse_args()

    if args.check:
        if args.version or args.from_version or args.dry_run:
            parser.error("--check takes no other arguments")
        problems = check()
        if problems:
            print(f"Records that do not match {VERSION_FILE} ({read_version()}):", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "Set the version with: python3 scripts/bump_version.py <version>", file=sys.stderr
            )
            return 1
        return 0

    if not args.version:
        parser.error("the version is required without --check")
    current = read_version()
    if args.from_version and current == args.version:
        print(f"{VERSION_FILE} is already {args.version}")
        return EXIT_ALREADY_AT_VERSION
    if args.from_version and current != args.from_version:
        print(f"{VERSION_FILE} is {current}, not {args.from_version}", file=sys.stderr)
        return EXIT_VERSION_MISMATCH
    set_version(args.version, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
