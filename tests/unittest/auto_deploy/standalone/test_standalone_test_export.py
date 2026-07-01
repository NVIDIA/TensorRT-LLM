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

"""Tests for exporting standalone-compatible AutoDeploy tests to LLMC."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CREATE_SCRIPT = REPO_ROOT / "examples" / "auto_deploy" / "llmc" / "create_standalone_package.py"
AD_MULTIGPU_TESTS = REPO_ROOT / "tests" / "unittest" / "auto_deploy" / "multigpu"

EXPECTED_MULTIGPU_FILES = {
    Path("custom_ops/test_dist.py"),
    Path("custom_ops/test_sharded_rmsnorm.py"),
    Path("transformations/library/conftest.py"),
    Path("transformations/library/test_apply_sharding_hints.py"),
    Path("transformations/library/test_bmm_sharding.py"),
    Path("transformations/library/test_ep_sharding.py"),
    Path("transformations/library/test_rmsnorm_sharding.py"),
    Path("transformations/library/test_sharding_num_correctness.py"),
    Path("transformations/library/test_step3p7_sharding_ir.py"),
    Path("transformations/library/test_tp_sharding.py"),
}
CANONICAL_IMPORT = "tensorrt_llm._torch.auto_deploy"
STANDALONE_IMPORT = "llmc"


@pytest.fixture(scope="module")
def generated_multigpu_tests(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the package once and return its multi-GPU test directory."""
    output_dir = tmp_path_factory.mktemp("llmc_test_export")
    subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output_dir / "tests" / "multigpu"


def test_exports_exact_multigpu_allowlist(generated_multigpu_tests: Path) -> None:
    exported_files = {
        path.relative_to(generated_multigpu_tests)
        for path in generated_multigpu_tests.rglob("*.py")
    }
    assert exported_files == EXPECTED_MULTIGPU_FILES


def test_registers_inert_threadleak_marker(generated_multigpu_tests: Path) -> None:
    pyproject = generated_multigpu_tests.parents[1] / "pyproject.toml"
    assert '"threadleak(enabled): configure thread-leak checks' in pyproject.read_text()


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_MULTIGPU_FILES, key=str), ids=str)
def test_rewrites_multigpu_auto_deploy_imports(
    generated_multigpu_tests: Path, relative_path: Path
) -> None:
    source = (AD_MULTIGPU_TESTS / relative_path).read_text()
    exported = (generated_multigpu_tests / relative_path).read_text()

    assert exported == source.replace(CANONICAL_IMPORT, STANDALONE_IMPORT)
    assert CANONICAL_IMPORT not in exported
