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

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CREATE_SCRIPT = REPO_ROOT / "examples" / "auto_deploy" / "llmc" / "create_standalone_package.py"
AD_SINGLEGPU_TESTS = REPO_ROOT / "tests" / "unittest" / "auto_deploy" / "singlegpu"
AD_MULTIGPU_TESTS = REPO_ROOT / "tests" / "unittest" / "auto_deploy" / "multigpu"
AD_TORCH_TESTS = REPO_ROOT / "tests" / "unittest" / "_torch" / "auto_deploy"
LLMC_TRTLLM_SINGLEGPU_SMOKE_TEST = Path("smoke/test_ad_build_small_single.py")
LLMC_TRTLLM_MULTIGPU_SMOKE_TEST = Path("smoke/test_ad_build_small_multi.py")
LLMC_TRTLLM_RUNNER_TESTS = {
    LLMC_TRTLLM_SINGLEGPU_SMOKE_TEST,
    Path("smoke/test_ad_guided_decoding_regex.py"),
    Path("smoke/test_ad_speculative_decoding.py"),
    Path("smoke/test_ad_trtllm_sampler.py"),
    LLMC_TRTLLM_MULTIGPU_SMOKE_TEST,
}

EXPECTED_SINGLEGPU_SMOKE_FILES = {
    path for path in LLMC_TRTLLM_RUNNER_TESTS if path != LLMC_TRTLLM_MULTIGPU_SMOKE_TEST
} | {
    Path("smoke/test_ad_trtllm_bench.py"),
    Path("smoke/test_ad_trtllm_serve.py"),
    Path("smoke/test_disagg.py"),
}
EXPECTED_MULTIGPU_FILES = {
    Path("custom_ops/test_dist.py"),
    Path("custom_ops/test_sharded_rmsnorm.py"),
    LLMC_TRTLLM_MULTIGPU_SMOKE_TEST,
    Path("transformations/library/conftest.py"),
    Path("transformations/library/test_apply_sharding_hints.py"),
    Path("transformations/library/test_bmm_sharding.py"),
    Path("transformations/library/test_ep_sharding.py"),
    Path("transformations/library/test_rmsnorm_sharding.py"),
    Path("transformations/library/test_sharding_num_correctness.py"),
    Path("transformations/library/test_step3p7_sharding_ir.py"),
    Path("transformations/library/test_tp_sharding.py"),
}
EXPECTED_TORCH_UNIT_FILES = {
    Path("unit/singlegpu/models/test_gpt_oss_modeling.py"),
}
CANONICAL_IMPORT = "tensorrt_llm._torch.auto_deploy"
STANDALONE_IMPORT = "llmc"
BUILD_AND_RUN_AD_IMPORT = "from build_and_run_ad import ExperimentConfig, main"
TRTLLM_IMPORT_RE = re.compile(
    r"(?m)^(?:from|import) "
    r"(?:tensorrt_llm(?:\.|\b)|llmc\.models\.custom\.modeling_gpt_oss(?:\.|\b))"
)
LLMC_OPTIONAL_TRTLLM_GUARD = """
_trtllm_redirect_value = os.environ.get("TRTLLM_REDIRECT_AD_TO_LLMC", "").lower()
if _trtllm_redirect_value not in {"1", "true", "yes", "on"}:
    pytest.skip(
        "LLMC optional TRT-LLM tests require TRTLLM_REDIRECT_AD_TO_LLMC=true",
        allow_module_level=True,
    )
pytest.importorskip("tensorrt_llm")"""
LLMC_TRTLLM_RUNNER_IMPORT = (
    "from runners.trtllm.build_and_run_llmc_trtllm import ExperimentConfig, main"
)


def _expected_exported_test(source: str, relative_path: Path) -> str:
    expected = source.replace(CANONICAL_IMPORT, STANDALONE_IMPORT)

    def ensure_imports(before_pos: int, *imports: str) -> None:
        nonlocal expected
        prefix = expected[:before_pos]
        missing_imports = [
            import_name for import_name in imports if f"import {import_name}\n" not in prefix
        ]
        if not missing_imports:
            return
        first_import = re.search(r"(?m)^(?:import|from) ", expected)
        assert first_import is not None
        expected = (
            expected[: first_import.start()]
            + "\n".join(f"import {import_name}" for import_name in missing_imports)
            + "\n"
            + expected[first_import.start() :]
        )

    def insert_optional_trtllm_guard() -> None:
        nonlocal expected
        pytest_import = re.search(r"(?m)^import pytest\n", expected)
        assert pytest_import is not None
        expected = (
            expected[: pytest_import.end()]
            + LLMC_OPTIONAL_TRTLLM_GUARD
            + "\n"
            + expected[pytest_import.end() :]
        )

    if relative_path in LLMC_TRTLLM_RUNNER_TESTS:
        build_import_pos = expected.index(BUILD_AND_RUN_AD_IMPORT)
        ensure_imports(build_import_pos, "os", "pytest")
        insert_optional_trtllm_guard()
        expected = expected.replace(BUILD_AND_RUN_AD_IMPORT, LLMC_TRTLLM_RUNNER_IMPORT)
    else:
        trtllm_import = TRTLLM_IMPORT_RE.search(expected)
        if trtllm_import is not None:
            ensure_imports(trtllm_import.start(), "os", "pytest")
            insert_optional_trtllm_guard()
    expected = expected.replace(
        '    script_dir = Path(root_dir, "benchmarks", "cpp")\n',
        "    script_dir = Path(temp_dir)\n",
    )
    return expected


@pytest.fixture(scope="module")
def generated_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the package once and return the package directory."""
    output_dir = tmp_path_factory.mktemp("llmc_test_export")
    subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return output_dir


@pytest.fixture(scope="module")
def generated_singlegpu_tests(generated_package: Path) -> Path:
    """Return the generated single-GPU test directory."""
    return generated_package / "tests" / "singlegpu"


@pytest.fixture(scope="module")
def generated_multigpu_tests(generated_package: Path) -> Path:
    """Return the generated multi-GPU test directory."""
    return generated_package / "tests" / "multigpu"


@pytest.fixture(scope="module")
def generated_torch_unit_tests(generated_package: Path) -> Path:
    """Return the generated _torch/auto_deploy unit test directory."""
    return generated_package / "tests" / "_torch" / "auto_deploy"


def test_exports_exact_multigpu_allowlist(generated_multigpu_tests: Path) -> None:
    exported_files = {
        path.relative_to(generated_multigpu_tests)
        for path in generated_multigpu_tests.rglob("*.py")
    }
    assert exported_files == EXPECTED_MULTIGPU_FILES


def test_exports_exact_singlegpu_smoke_allowlist(generated_singlegpu_tests: Path) -> None:
    exported_smoke_files = {
        path.relative_to(generated_singlegpu_tests)
        for path in (generated_singlegpu_tests / "smoke").rglob("*.py")
    }
    assert exported_smoke_files == EXPECTED_SINGLEGPU_SMOKE_FILES


def test_exports_exact_torch_unit_allowlist(generated_torch_unit_tests: Path) -> None:
    exported_files = {
        path.relative_to(generated_torch_unit_tests)
        for path in generated_torch_unit_tests.rglob("test*.py")
    }
    assert exported_files == EXPECTED_TORCH_UNIT_FILES


def test_registers_inert_threadleak_marker(generated_multigpu_tests: Path) -> None:
    pyproject = generated_multigpu_tests.parents[1] / "pyproject.toml"
    assert '"threadleak(enabled): configure thread-leak checks' in pyproject.read_text()


def test_conftest_allows_explicit_redirect_mode(generated_multigpu_tests: Path) -> None:
    conftest = generated_multigpu_tests.parent / "conftest.py"
    content = conftest.read_text()
    assert "TRTLLM_REDIRECT_AD_TO_LLMC" in content
    assert "set TRTLLM_REDIRECT_AD_TO_LLMC=true only for optional TRT-LLM tests" in content
    assert "_package_root = os.path.dirname(_tests_dir)" in content
    assert "sys.path.insert(0, _package_root)" in content
    assert "def llm_root()" in content


def test_creates_minimal_utils_util_stub(generated_package: Path) -> None:
    util_stub = generated_package / "tests" / "utils" / "util.py"
    content = util_stub.read_text()
    assert "skip_pre_hopper" in content
    assert "torch.cuda.get_device_capability" in content


def test_does_not_add_test_guards_to_runners(generated_package: Path) -> None:
    runner = generated_package / "runners" / "trtllm" / "build_and_run_llmc_trtllm.py"
    content = runner.read_text()
    assert "pytest.skip" not in content
    assert "pytest.importorskip" not in content


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_SINGLEGPU_SMOKE_FILES, key=str), ids=str)
def test_rewrites_singlegpu_smoke_auto_deploy_imports(
    generated_singlegpu_tests: Path, relative_path: Path
) -> None:
    source = (AD_SINGLEGPU_TESTS / relative_path).read_text()
    exported = (generated_singlegpu_tests / relative_path).read_text()

    assert exported == _expected_exported_test(source, relative_path)
    assert CANONICAL_IMPORT not in exported
    if relative_path in LLMC_TRTLLM_RUNNER_TESTS:
        assert BUILD_AND_RUN_AD_IMPORT not in exported
        assert "build_and_run_llmc_trtllm" in exported
    else:
        assert "LLMC optional TRT-LLM tests require TRTLLM_REDIRECT_AD_TO_LLMC=true" in exported
    assert "TRTLLM_REDIRECT_AD_TO_LLMC" in exported


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_MULTIGPU_FILES, key=str), ids=str)
def test_rewrites_multigpu_auto_deploy_imports(
    generated_multigpu_tests: Path, relative_path: Path
) -> None:
    source = (AD_MULTIGPU_TESTS / relative_path).read_text()
    exported = (generated_multigpu_tests / relative_path).read_text()

    assert exported == _expected_exported_test(source, relative_path)
    assert CANONICAL_IMPORT not in exported
    if relative_path == LLMC_TRTLLM_MULTIGPU_SMOKE_TEST:
        assert BUILD_AND_RUN_AD_IMPORT not in exported
        assert "build_and_run_llmc_trtllm" in exported
        assert "TRTLLM_REDIRECT_AD_TO_LLMC" in exported


@pytest.mark.parametrize("relative_path", sorted(EXPECTED_TORCH_UNIT_FILES, key=str), ids=str)
def test_rewrites_torch_unit_auto_deploy_imports(
    generated_torch_unit_tests: Path, relative_path: Path
) -> None:
    source = (AD_TORCH_TESTS / relative_path).read_text()
    exported = (generated_torch_unit_tests / relative_path).read_text()

    assert exported == _expected_exported_test(source, relative_path)
    assert CANONICAL_IMPORT not in exported
    assert "LLMC optional TRT-LLM tests require TRTLLM_REDIRECT_AD_TO_LLMC=true" in exported
