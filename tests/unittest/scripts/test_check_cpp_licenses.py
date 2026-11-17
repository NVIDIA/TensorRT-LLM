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

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/check_cpp_licenses.py"
SPEC = importlib.util.spec_from_file_location("check_cpp_licenses", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
HEADER = (REPO / "cpp/include/tensorrt_llm/executor/types.h").read_text().split("*/", 1)[0] + "*/\n"


@pytest.fixture
def scan(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "cpp"
    (root / "include").mkdir(parents=True)
    (root / "tensorrt_llm").mkdir()
    config = tmp_path / "policy.json"
    policy = {
        "filters": [{"include_re": "^(include|tensorrt_llm)/", "license_name": "apache"}],
        "skip": {},
    }
    config.write_text(json.dumps(policy))
    return root, config, policy


def run_scan(scan: tuple[Path, Path, dict]) -> tuple[tuple[int, int, int], str]:
    root, config, policy = scan
    config.write_text(json.dumps(policy))
    with contextlib.redirect_stdout(io.StringIO()) as output:
        counts = checker.check_licenses(root, config, ["include", "tensorrt_llm"])
    return counts, output.getvalue()


def test_scans_both_trees_and_all_jenkins_extensions(scan: tuple[Path, Path, dict]) -> None:
    root, _, _ = scan
    for suffix in (".h", ".cpp", ".cuh", ".cu", ".inl", ".map"):
        (root / "include" / f"example{suffix}").write_text(HEADER + "int value;\n")
    # Python files under cpp/ also belong to the existing Jenkins scan.
    header = HEADER.replace("/*\n", "").replace(" */\n", "")
    python_header = "\n".join(
        "# " + line.removeprefix(" * ").removeprefix(" *") for line in header.splitlines()
    )
    (root / "tensorrt_llm/example.py").write_text(
        "#!/usr/bin/env python3\n" + python_header + "\nprint(1)\n"
    )
    (root / "include/ignored.txt").write_text("no license\n")
    assert run_scan(scan)[0] == (7, 0, 0)


@pytest.mark.parametrize(
    "mutation", ["missing", "spdx_only", "wrong_body", "too_long", "line_comment"]
)
def test_rejects_missing_or_damaged_license(scan: tuple[Path, Path, dict], mutation: str) -> None:
    root, _, _ = scan
    content = {
        "missing": "int value;\n",
        "spdx_only": "/* SPDX-License-Identifier: Apache-2.0 */\n",
        "wrong_body": HEADER.replace("WITHOUT WARRANTIES", "WITH WARRANTIES"),
        "too_long": "/*" + "x" * 4096 + "*/\n" + HEADER,
        "line_comment": "// " + HEADER,
    }[mutation]
    (root / "include/bad.h").write_text(content)
    counts, output = run_scan(scan)
    assert counts == (1, 0, 1)
    assert "error include/bad.h:" in output


def test_preserves_policy_skip_without_hiding_other_failures(scan: tuple[Path, Path, dict]) -> None:
    root, _, policy = scan
    (root / "include/external.h").write_text("external license\n")
    (root / "include/bad.h").write_text("int value;\n")
    policy["skip"]["include/external.h"] = "third-party"
    counts, output = run_scan(scan)
    assert counts == (2, 1, 1)
    assert "warning include/external.h:" in output
    assert "error include/bad.h:" in output


def test_ignores_line_wrapping_but_preserves_license_paragraphs(
    scan: tuple[Path, Path, dict],
) -> None:
    root, _, _ = scan
    path = root / "include/example.h"
    path.write_text(HEADER.replace('License");\n * you', 'License");you'))
    assert run_scan(scan)[0] == (1, 0, 0)
    path.write_text(HEADER.replace(" *\n * Unless", " * Unless"))
    assert run_scan(scan)[0] == (1, 0, 1)


def test_read_boundary_does_not_reject_utf8_after_header(scan: tuple[Path, Path, dict]) -> None:
    root, _, _ = scan
    prefix = (HEADER + "// ").encode()
    data = prefix + b"a" * (4095 - len(prefix)) + "→".encode() + b"\n"
    (root / "include/example.h").write_bytes(data)
    assert run_scan(scan)[0] == (1, 0, 0)


@pytest.mark.parametrize("exclude", ["exclude", "exclude_re"])
def test_filter_exclusions_do_not_hide_missing_headers(
    scan: tuple[Path, Path, dict], exclude: str
) -> None:
    root, _, policy = scan
    policy["filters"][0][exclude] = (
        {"include/example.h": "reason"} if exclude == "exclude" else ["example"]
    )
    path = root / "include/example.h"
    path.write_text(HEADER)
    assert run_scan(scan)[0] == (1, 0, 0)
    path.write_text("int value;\n")
    assert run_scan(scan)[0] == (1, 0, 1)


def test_multiple_matching_rules_fail(scan: tuple[Path, Path, dict]) -> None:
    root, _, policy = scan
    (root / "include/example.h").write_text(HEADER)
    policy["filters"].append({"include_re": "^include/", "license_name": "apache"})
    assert run_scan(scan)[0] == (1, 0, 1)


def test_unknown_policy_fields_fail(scan: tuple[Path, Path, dict]) -> None:
    root, config, policy = scan
    (root / "include/example.h").write_text(HEADER)
    policy["unknown_field"] = {"include/example.h": "unexpected policy field"}
    config.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="unknown license policy fields"):
        checker.check_licenses(root, config, ["include"])


def test_symlink_is_not_read(scan: tuple[Path, Path, dict]) -> None:
    root, _, _ = scan
    (root / "include/example.h").symlink_to(root / "nonexistent")
    assert run_scan(scan)[0] == (1, 0, 1)


def test_missing_or_empty_scan_fails(scan: tuple[Path, Path, dict]) -> None:
    root, config, _ = scan
    for directory in ("missing", "include"):
        with pytest.raises(ValueError):
            checker.check_licenses(root, config, [directory])


def test_cli_exit_status_and_paths_outside_cwd(
    scan: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    root, config, _ = scan
    path = root / "include/example.h"
    command = [sys.executable, str(SCRIPT), f"--root={root}", f"--config={config}"]
    for content, expected in ((HEADER, 0), ("int value;\n", 1)):
        path.write_text(content)
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == expected, result.stdout + result.stderr
        assert "filecount: 1" in result.stdout
