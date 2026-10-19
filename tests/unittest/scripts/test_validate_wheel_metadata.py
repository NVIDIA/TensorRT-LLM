# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks against installed wheel metadata fixtures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

SCRIPT = Path(__file__).resolve().parents[3] / "docker/common/validate_wheel_metadata.py"
COMMIT = "abcdef01" * 5
OTHER_COMMIT = "12345678" * 5
SOURCE_URL = f"Source Commit, https://github.com/NVIDIA/TensorRT-LLM/commit/{COMMIT}"


@pytest.mark.parametrize(
    "installed_version,project_urls,expected_version,expected_commit,error",
    [
        ("1.4.0a0", [SOURCE_URL], "1.4.0a0", COMMIT, None),
        ("1.4.0a0", [SOURCE_URL], "1.4.0alpha0", COMMIT.upper(), None),
        ("1.4.0a0", [SOURCE_URL], "1.4.0a0", OTHER_COMMIT, "source commit mismatch"),
        ("1.4.0a0", [SOURCE_URL], "1.4.0a0.dev202610190100", COMMIT, "version mismatch"),
        ("1.4.0a0", [], "1.4.0a0", COMMIT, "source commit mismatch"),
        ("1.4.0a0", ["Homepage, https://example.com"], "", COMMIT, "source commit mismatch"),
        ("1.4.0a0", [SOURCE_URL, SOURCE_URL], "", COMMIT, "source commit mismatch"),
        ("1.4.0a0", [SOURCE_URL], "", COMMIT[:7], "full 40-character"),
        ("1.4.0a0", [SOURCE_URL], "", COMMIT, None),
        ("1.4.0a0", [], "1.4.0a0", "", None),
        (None, [], "", "", None),
    ],
)
def test_installed_metadata(
    tmp_path, installed_version, project_urls, expected_version, expected_commit, error
):
    if installed_version:
        metadata = tmp_path / "tensorrt_llm-1.4.0a0.dist-info" / "METADATA"
        metadata.parent.mkdir()
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: tensorrt_llm\nVersion: {installed_version}\n"
            + "".join(f"Project-URL: {url}\n" for url in project_urls)
        )
    # Any accidental package import must fail even on a GPU-capable test runner.
    package = tmp_path / "tensorrt_llm"
    package.mkdir()
    (package / "__init__.py").write_text('raise RuntimeError("must not import TRT-LLM")\n')
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--expected-version",
            expected_version,
            "--expected-commit",
            expected_commit,
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    if error:
        assert result.returncode != 0
        assert error in result.stderr
    else:
        assert result.returncode == 0, result.stderr
