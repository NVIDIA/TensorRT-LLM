"""Find bad third-party usage in cmake.

This script searches for cmake function invocations that might indicate
the addition of new third-party dependencies outside of the intended
process (3rdparty/README.md).
"""

import argparse
import collections
import logging
import os
import pathlib
import sys
from typing import Generator

logger = logging.getLogger(__name__)

IGNORE_PATTERNS = [
    ".*",  # Hidden files and directories, like .git
    # This is where we actually want third-party stuff to go
    "3rdparty/CMakeLists.txt",
    # Historical use of ExternalProject_Add that is not yet migrated to 3rdparty
    "cpp/tensorrt_llm/deep_ep/CMakeLists.txt",
    # Historical build that is not included in the wheel build and thus exempt
    # from the third-party process.
    "triton_backend/inflight_batcher_llm/*",
    "build",  # Default build directory
    "cpp/build",  # Default extension module build directory
]


class DirectoryFilter:
    """Callable filter for directories.

    This filter excludes any paths matching IGNORE_PATTERNS.
    """

    def __init__(self, parent: pathlib.Path):
        self.parent = parent

    def __call__(self, name: str) -> bool:
        path = self.parent / name
        if any(path.match(pat) for pat in IGNORE_PATTERNS):
            return False
        return True


class FileFilter:
    """Callable filter for file entries.

    In order of precedence:

    1. excludes any paths matching IGNORE_PATTERNS
    2. includes only CMakeLists.txt and *.cmake files
    """

    def __init__(self, parent: pathlib.Path):
        self.parent = parent

    def __call__(self, name: str) -> bool:
        path = self.parent / name
        if any(path.match(pat) for pat in IGNORE_PATTERNS):
            return False

        if name == "CMakeLists.txt":
            return True
        elif name.endswith(".cmake"):
            return True

        return False


def yield_sources(src_tree: pathlib.Path):
    """Perform a filesystem walk and yield any paths that should be scanned."""
    for parent, dirs, files in os.walk(src_tree):
        parent = pathlib.Path(parent)
        relpath_parent = parent.relative_to(src_tree)

        # Filter out ignored directories
        dirs[:] = sorted(filter(DirectoryFilter(relpath_parent), dirs))

        for file in sorted(filter(FileFilter(relpath_parent), files)):
            yield parent / file


ThirdpartyViolation = collections.namedtuple(
    "ThirdpartyViolation", ["srcfile", "lineno", "note", "line"]
)


def yield_potential_thirdparty(
    fullpath: pathlib.Path, relpath: pathlib.Path
) -> Generator[ThirdpartyViolation, None, None]:
    """Look for bad patterns with third-party sources.

    Look for patterns that might indicate the addition of new third-party
    sources.
    """
    with fullpath.open("r", encoding="utf-8") as infile:
        for lineno, line in enumerate(infile):
            lineno += 1  # Make line numbers 1-based

            if "FetchContent_Declare" in line:
                note = "Invalid use of FetchContent_Declare outside of 3rdparty/CMakeLists.txt"
                yield ThirdpartyViolation(relpath, lineno, note, line.strip())

            if "ExternalProject_Add" in line:
                note = "Invalid use of ExternalProject_Add outside of 3rdparty/CMakeLists.txt"
                yield ThirdpartyViolation(relpath, lineno, note, line.strip())


def check_sources(src_tree: pathlib.Path) -> int:
    """Common entry-point between main() and pytest.

    Prints any violations to stderr and returns non-zero if any violations are
    found.
    """
    violations = []
    for filepath in yield_sources(src_tree):
        for violation in yield_potential_thirdparty(filepath, filepath.relative_to(src_tree)):
            violations.append(violation)

    if not violations:
        return 0

    for violation in sorted(violations):
        sys.stderr.write(
            f"{violation.srcfile}:{violation.lineno}: {violation.note}\n"
            + f"    {violation.line}\n"
        )

    logger.error(
        "Found %d potential third-party violations. "
        "If you are trying to add a new third-party dependency, "
        "please follow the instructions in 3rdparty/cpp-thirdparty.md",
        len(violations),
    )
    return 1


def test_cmake_listfiles():
    """Test that no third-party violations are found in the source tree."""
    source_tree = pathlib.Path(__file__).parents[1]
    result = check_sources(source_tree)
    assert result == 0


def main():
    parser = argparse.ArgumentParser(description="__doc__")
    parser.add_argument(
        "--src-tree",
        default=pathlib.Path.cwd(),
        type=pathlib.Path,
        help="Path to the source tree, defaults to current directory",
    )
    args = parser.parse_args()
    result = check_sources(args.src_tree)
    sys.exit(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
