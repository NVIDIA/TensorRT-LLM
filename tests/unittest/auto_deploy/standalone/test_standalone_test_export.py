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
LLMC_TRTLLM_SMOKE_TEST = Path("smoke/test_ad_build_small_multi.py")

EXPECTED_MULTIGPU_FILES = {
    Path("custom_ops/test_dist.py"),
    Path("custom_ops/test_sharded_rmsnorm.py"),
    LLMC_TRTLLM_SMOKE_TEST,
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
BUILD_AND_RUN_AD_IMPORT = "from build_and_run_ad import ExperimentConfig, main"
LLMC_TRTLLM_RUNNER_IMPORT = """
if os.environ.get("TRTLLM_REDIRECT_AD_TO_LLMC") != "true":
    pytest.skip(
        "LLMC TRT-LLM redirect smoke requires TRTLLM_REDIRECT_AD_TO_LLMC=true",
        allow_module_level=True,
    )
pytest.importorskip("tensorrt_llm")
from runners.trtllm.build_and_run_llmc_trtllm import ExperimentConfig, main"""


def _expected_exported_test(source: str, relative_path: Path) -> str:
    expected = source.replace(CANONICAL_IMPORT, STANDALONE_IMPORT)
    if relative_path == LLMC_TRTLLM_SMOKE_TEST:
        expected = expected.replace("import pytest\n", "import os\n\nimport pytest\n")
        expected = expected.replace(BUILD_AND_RUN_AD_IMPORT, LLMC_TRTLLM_RUNNER_IMPORT)
    return expected


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


def test_conftest_allows_explicit_redirect_mode(generated_multigpu_tests: Path) -> None:
    conftest = generated_multigpu_tests.parent / "conftest.py"
    content = conftest.read_text()
    assert "TRTLLM_REDIRECT_AD_TO_LLMC" in content
    assert "set TRTLLM_REDIRECT_AD_TO_LLMC=true only for redirect smoke tests" in content


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_MULTIGPU_FILES, key=str), ids=str)
def test_rewrites_multigpu_auto_deploy_imports(
    generated_multigpu_tests: Path, relative_path: Path
) -> None:
    source = (AD_MULTIGPU_TESTS / relative_path).read_text()
    exported = (generated_multigpu_tests / relative_path).read_text()

    assert exported == _expected_exported_test(source, relative_path)
    assert CANONICAL_IMPORT not in exported
    if relative_path == LLMC_TRTLLM_SMOKE_TEST:
        assert BUILD_AND_RUN_AD_IMPORT not in exported
        assert "build_and_run_llmc_trtllm" in exported
        assert "TRTLLM_REDIRECT_AD_TO_LLMC" in exported
