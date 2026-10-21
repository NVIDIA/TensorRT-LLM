# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check installed wheel provenance without importing CUDA-linked modules."""

import re
from argparse import ArgumentParser
from importlib.metadata import distribution
from pathlib import Path

from packaging.version import Version


def validate_wheel_metadata(
    expected_version: str, expected_commit: str, source_version_file: Path | None = None
) -> None:
    """Check installed metadata, resolving relative versions from the build source file."""
    if not expected_version and not expected_commit:
        return
    if expected_version.startswith((".", "+")):
        if source_version_file is None:
            raise SystemExit("Relative versions require --source-version-file")
        match = re.search(r'(?m)^__version__ = "([^"]+)"$', source_version_file.read_text())
        if not match:
            raise SystemExit(f"Unable to read __version__ from {source_version_file}")
        base_version = match.group(1)
        expected_version = (
            base_version
            if base_version.endswith(expected_version)
            else base_version + expected_version
        )
    wheel = distribution("tensorrt_llm")
    if expected_version and Version(wheel.version) != Version(expected_version):
        raise SystemExit(
            f"TRT-LLM version mismatch: installed={wheel.version}, expected={expected_version}"
        )
    if expected_commit:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit):
            raise SystemExit(f"Expected a full 40-character source commit SHA: {expected_commit}")
        source_urls = []
        for entry in wheel.metadata.get_all("Project-URL", []):
            label, _, url = entry.partition(",")
            if label.strip() == "Source Commit":
                source_urls.append(url.strip().lower())
        expected_url = f"https://github.com/NVIDIA/TensorRT-LLM/commit/{expected_commit}"
        if source_urls != [expected_url.lower()]:
            raise SystemExit(
                f"TRT-LLM source commit mismatch: installed={source_urls or '<missing>'}, "
                f"expected={expected_url}"
            )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default="")
    parser.add_argument("--expected-commit", default="")
    parser.add_argument(
        "--source-version-file", type=Path, help="Build-context version.py for relative overrides"
    )
    args = parser.parse_args()
    validate_wheel_metadata(args.expected_version, args.expected_commit, args.source_version_file)
