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
"""Check C++ source license headers using the shared Jenkins license policy."""

import argparse
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSIONS = {".h", ".cpp", ".cuh", ".cu", ".inl", ".map", ".py"}
_HEADER_BYTES = 4096
_APACHE_BODY = """Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


def _segments(text: str) -> list[str]:
    segments = [""]
    for line in text.split("\n"):
        line = line.strip()
        if line:
            segments[-1] += line + " "
        else:
            segments.append("")
    return [" ".join(segment.split()) for segment in segments]


def _license_header(data: bytes, suffix: str) -> str:
    text = data.decode("utf-8", errors="replace")
    if suffix == ".py":
        lines = text.splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        header = []
        for line in lines:
            if not line.startswith("#"):
                break
            header.append(line[2:] if line.startswith("# ") else line[1:])
        return "\n".join(header)

    text = text.lstrip()
    if not text.startswith("/*"):
        return ""
    end = text.find("*/")
    if end == -1:
        raise ValueError("license comment exceeds the header read limit")
    header = []
    for line in text[: end + 2].split("\n"):
        line = line.strip()
        if not line.strip("*"):
            header.append("")
        elif not line.strip("*/"):
            continue
        else:
            header.append(line[2:] if line.startswith("* ") else line)
    return "\n".join(header)


def _check_header(path: Path) -> None:
    with path.open("rb") as source:
        header = _license_header(source.read(_HEADER_BYTES), path.suffix)
    if not header:
        raise ValueError("license header is missing")
    segments = _segments(header)
    body = _segments(_APACHE_BODY)
    copyright_paragraph = header.split("\n\n", 1)[0]
    copyright_prefix = re.search(
        r"^(?:Copyright\b(?:[ \t]*(?:\(c\)|©))?(?:[ \t]+|$)|SPDX-FileCopyrightText:[ \t]*)",
        copyright_paragraph,
        re.IGNORECASE | re.MULTILINE,
    )
    if copyright_prefix is None or not copyright_paragraph[copyright_prefix.end() :].strip():
        raise ValueError("license copyright notice is missing")
    # The copyright paragraph varies; permit one trailing attribution paragraph.
    if len(segments) not in (len(body) + 1, len(body) + 2) or segments[1 : 1 + len(body)] != body:
        raise ValueError("license header does not match Apache-2.0")


def _raise_walk_error(error: OSError) -> None:
    raise error


def _scan_directory(root: Path, directory: str) -> Path:
    path = root / directory
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"scan directory is a symlink: {parent}")
        if parent == root:
            break
    normalized = Path(os.path.abspath(path))
    if not normalized.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"scan directory is missing or outside the source root: {path}")
    return normalized


def _source_paths(root: Path, directories: list[str], files: list[Path] | None) -> Iterator[Path]:
    scan_roots = [_scan_directory(root, directory) for directory in directories]
    if files is not None:
        paths: set[Path] = set()
        for filename in files:
            path = filename.absolute()
            for parent in path.parents:
                if parent.is_symlink():
                    raise ValueError(f"scan directory is a symlink: {parent}")
                if parent == root:
                    break
            path = Path(os.path.abspath(path))
            if any(path.is_relative_to(scan_root) for scan_root in scan_roots):
                paths.add(path)
        yield from sorted(paths)
        return

    for scan_root in scan_roots:
        for current, dirnames, filenames in os.walk(scan_root, onerror=_raise_walk_error):
            for dirname in sorted(dirnames):
                path = Path(current) / dirname
                if path.is_symlink():
                    raise ValueError(f"scan directory is a symlink: {path}")
            for filename in sorted(filenames):
                yield Path(current) / filename


def check_licenses(
    root: Path, config: Path, directories: list[str], *, files: list[Path] | None = None
) -> tuple[int, int, int]:
    """Return checked, exempted and failed file counts; reject invalid scan paths."""
    root = Path(os.path.abspath(root))
    policy = json.loads(config.read_text())
    if not isinstance(policy, dict) or policy.keys() - {"filters", "skip"}:
        raise ValueError("unknown license policy fields")
    filters = policy["filters"]
    skip = policy["skip"]
    if not isinstance(filters, list) or not isinstance(skip, dict):
        raise ValueError("license policy requires a filters list and a skip mapping")
    for rule in filters:
        if not isinstance(rule, dict) or rule.keys() - {
            "include_re",
            "exclude",
            "exclude_re",
            "license_name",
        }:
            raise ValueError("unknown license filter fields")
        re.compile(rule["include_re"])
    checked = warnings = failed = 0
    for path in _source_paths(root, directories, files):
        if path.suffix not in _EXTENSIONS:
            continue
        checked += 1
        relative = path.relative_to(root).as_posix()
        try:
            if path.is_symlink():
                raise ValueError("source file is a symlink")
            matches = [
                rule
                for rule in filters
                if re.search(rule["include_re"], relative)
                and relative not in rule.get("exclude", {})
                and not any(re.search(pattern, relative) for pattern in rule.get("exclude_re", []))
            ]
            if len(matches) > 1:
                raise ValueError("file matches multiple license policies")
            if matches and matches[0]["license_name"] != "apache":
                raise ValueError("unsupported license policy; expected Apache-2.0")
            try:
                _check_header(path)
            except ValueError as error:
                if relative not in skip:
                    raise
                warnings += 1
                print(f"warning {relative}: {error}")
        except (OSError, ValueError) as error:
            failed += 1
            print(f"error {relative}: {error}")
    if not checked and files is None:
        raise ValueError("no supported source files were checked")
    print(f"filecount: {checked}\nwarncount: {warnings}\nerrcount: {failed}")
    return checked, warnings, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT / "cpp")
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "jenkins/license_cpp.json")
    parser.add_argument("directories", nargs="*", default=["include", "tensorrt_llm"])
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="Check only these files, relative to the current directory",
    )
    args = parser.parse_args()
    try:
        _, _, failed = check_licenses(args.root, args.config, args.directories, files=args.files)
    except (OSError, ValueError, KeyError, TypeError, re.error) as error:
        parser.exit(1, f"License check failed: {error}\n")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
