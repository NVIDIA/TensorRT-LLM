"""This script audits the .gitmodules file.

... to make sure that new git submodules are not added without following the
proper process (cpp/3rdparty/cpp-thirdparty.md)
"""

import argparse
import collections
import configparser
import logging
import pathlib
import sys

logger = logging.getLogger(__name__)

ALLOWLIST_SUBMODULES = [
    # NOTE: please do not add new sobmodules here without following the process
    # in 3rdparty/cpp-thirdparty.md. Prefer to use FetchContent or other methods
    # to avoid adding new git submodules unless absolutely necessary.
]

ThirdpartyViolation = collections.namedtuple("ThirdpartyViolation", ["section", "path", "note"])


def find_violations(config: configparser.ConfigParser) -> list[str]:
    violations = []
    for section in config.sections():
        if not section.startswith("submodule "):
            raise ValueError(f"Unexpected section in .gitmodules: {section}")

        path = config[section].get("path", "")
        if not path:
            raise ValueError(f"Missing path for submodule {section}")

        if path not in ALLOWLIST_SUBMODULES:
            violations.append(
                ThirdpartyViolation(
                    section=section,
                    path=path,
                    note="Submodule not in allowlist (see test_git_modules.py)",
                )
            )

        if not path.startswith("3rdparty/"):
            violations.append(
                ThirdpartyViolation(
                    section=section,
                    path=path,
                    note="Submodule path must be under 3rdparty/",
                )
            )
    return violations


def check_modules_file(git_modules_path: pathlib.Path) -> int:
    """Common entry-point between main() and pytest.

    Prints any violations to stderr and returns non-zero if any violations are
    found.
    """
    config = configparser.ConfigParser()
    config.read(git_modules_path)

    violations = find_violations(config)

    if violations:
        for violation in violations:
            sys.stderr.write(f"{violation.section} (path={violation.path}): {violation.note}\n")

        logger.error(
            "Found %d potential third-party violations. "
            "If you are trying to add a new third-party dependency, "
            "please follow the instructions in cpp/3rdparty/cpp-thirdparty.md",
            len(violations),
        )
        return 1
    return 0


def test_gitmodules():
    """Test that no git submodules are added to .gitmodules.

    ... without following the defined process.
    """
    git_modules_path = pathlib.Path(__file__).parents[1] / ".gitmodules"
    result = check_modules_file(git_modules_path)
    assert result == 0


def main():
    parser = argparse.ArgumentParser(description="__doc__")
    parser.add_argument(
        "--git-modules-path",
        default=pathlib.Path(".gitmodules"),
        type=pathlib.Path,
        help="Path to the .gitmodules file, defaults to .gitmodules in current directory",
    )
    args = parser.parse_args()
    result = check_modules_file(args.git_modules_path)
    sys.exit(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
