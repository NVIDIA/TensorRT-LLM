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

"""Tests for exporting standalone-compatible AutoDeploy tests to Paragraf."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CREATE_SCRIPT = REPO_ROOT / "examples" / "auto_deploy" / "paragraf" / "create_standalone_package.py"
LEGACY_CREATE_SCRIPT = (
    REPO_ROOT / "examples" / "auto_deploy" / "llmc" / "create_standalone_package.py"
)
AUTODEPLOY = re.compile(r"auto_?deploy|_ad_", re.IGNORECASE)
TEST_FILE = re.compile(r"(?:^test_.*|.*_test)\.py$")
CANONICAL_IMPORT = "tensorrt_llm._torch.auto_deploy"
OPTIONAL_TRTLLM_GUARD = "Paragraf optional TRT-LLM tests require"
GENERATED_TEST_RENAMES = {
    "test_llm_api_autodeploy.py": "test_llm_api_paragraf_trtllm.py",
    "test_ad_disagg.py": "test_paragraf_trtllm_disagg.py",
    "test_ad_disagg_trtllm_serve.py": "test_paragraf_trtllm_disagg_serve.py",
    "test_ad_guided_decoding.py": "test_paragraf_trtllm_guided_decoding.py",
    "test_ad_speculative_decoding.py": "test_paragraf_trtllm_speculative_decoding.py",
    "test_ad_dist_strategies.py": "test_paragraf_trtllm_dist_strategies.py",
    "test_ad_build_small_multi.py": "test_paragraf_trtllm_build_small_multi.py",
    "test_ad_moe_op.py": "test_paragraf_trtllm_moe_op.py",
    "test_ad_executor_swa_eviction.py": "test_paragraf_trtllm_executor_swa_eviction.py",
    "test_create_ad_executor.py": "test_create_paragraf_trtllm_executor.py",
    "test_ad_build_small_single.py": "test_paragraf_trtllm_build_small_single.py",
    "test_ad_guided_decoding_regex.py": "test_paragraf_trtllm_guided_decoding_regex.py",
    "test_ad_trtllm_bench.py": "test_paragraf_trtllm_bench.py",
    "test_ad_trtllm_sampler.py": "test_paragraf_trtllm_sampler.py",
    "test_ad_trtllm_serve.py": "test_paragraf_trtllm_serve.py",
}


def _tracked_autodeploy_tests() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "tests"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        Path(path)
        for path in result.stdout.splitlines()
        if TEST_FILE.fullmatch(Path(path).name)
        and AUTODEPLOY.search(path)
        and "/standalone/" not in path
    }


def _generated_test_path(source_path: Path) -> Path:
    path = source_path.as_posix()
    legacy_prefix = "tests/unittest/auto_deploy/"
    torch_prefix = "tests/unittest/_torch/auto_deploy/"
    integration_prefix = "tests/integration/"
    if path.startswith(legacy_prefix):
        generated_path = Path(path.removeprefix(legacy_prefix))
    elif path.startswith(torch_prefix):
        generated_path = Path(path.removeprefix(torch_prefix))
        if generated_path.parts[0] == "unit":
            generated_path = Path(*generated_path.parts[1:])
    elif path.startswith(integration_prefix):
        generated_path = Path("integration") / path.removeprefix(integration_prefix)
    else:
        raise AssertionError(f"Unhandled AutoDeploy test path: {source_path}")
    return generated_path.with_name(
        GENERATED_TEST_RENAMES.get(generated_path.name, generated_path.name)
    )


@pytest.fixture(scope="module")
def generated_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the package once and return the package directory."""
    output_dir = tmp_path_factory.mktemp("paragraf_test_export")
    legacy_package_dir = output_dir / "llmc"
    legacy_package_dir.mkdir()
    (legacy_package_dir / "stale.py").touch()
    subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert legacy_package_dir.is_symlink()
    assert os.readlink(legacy_package_dir) == "paragraf"
    subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert legacy_package_dir.is_symlink()
    return output_dir


def test_generates_paragraf_package_identity(generated_package: Path) -> None:
    assert (generated_package / "paragraf" / "__init__.py").is_file()
    assert (generated_package / "llmc").is_symlink()
    pyproject = (generated_package / "pyproject.toml").read_text()
    assert 'name = "nvidia-llmc"' in pyproject
    assert 'include = ["paragraf*", "llmc"]' in pyproject
    assert 'trtllm = ["tensorrt-llm"]' in pyproject


def test_legacy_generator_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "legacy-generator-output"
    subprocess.run(
        [sys.executable, str(LEGACY_CREATE_SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (output_dir / "paragraf" / "__init__.py").is_file()
    assert (output_dir / "llmc").is_symlink()


def test_exports_every_tracked_autodeploy_test(generated_package: Path) -> None:
    generated_tests = generated_package / "tests"
    missing = {
        source_path
        for source_path in _tracked_autodeploy_tests()
        if not (generated_tests / _generated_test_path(source_path)).is_file()
    }
    assert not missing


def test_does_not_export_generator_meta_tests(generated_package: Path) -> None:
    assert not (generated_package / "tests" / "standalone").exists()


def test_registers_inert_threadleak_marker(generated_package: Path) -> None:
    pyproject = generated_package / "pyproject.toml"
    assert '"threadleak(enabled): configure thread-leak checks' in pyproject.read_text()


def test_conftest_allows_explicit_redirect_mode(generated_package: Path) -> None:
    conftest = generated_package / "tests" / "conftest.py"
    content = conftest.read_text()
    assert "TRTLLM_REDIRECT_AD_TO_PARAGRAF" in content
    assert "TRTLLM_REDIRECT_AD_TO_LLMC" in content
    assert "set TRTLLM_REDIRECT_AD_TO_PARAGRAF=true only for optional TRT-LLM tests" in content
    assert "_package_root = os.path.dirname(_tests_dir)" in content
    assert "sys.path.insert(0, _package_root)" in content
    assert "def llm_root()" in content


def test_creates_minimal_utils_util_stub(generated_package: Path) -> None:
    util_stub = generated_package / "tests" / "utils" / "util.py"
    content = util_stub.read_text()
    assert "skip_pre_hopper" in content
    assert "skip_no_hopper" in content
    assert "skip_pre_blackwell" in content
    assert "torch.cuda.get_device_capability" in content
    assert (generated_package / "tests" / "utils" / "cpp_paths.py").is_file()
    assert (generated_package / "tests" / "_torch" / "helpers.py").is_file()


def test_copies_focused_integration_support(generated_package: Path) -> None:
    integration_defs = generated_package / "tests" / "integration" / "defs"
    assert (integration_defs / "conftest.py").is_file()
    assert (integration_defs / "accuracy" / "accuracy_core.py").is_file()
    assert (integration_defs / "disaggregated" / "disagg_test_utils.py").is_file()
    assert (generated_package / "runners" / "trtllm" / "model_registry" / "models.yaml").is_file()


def test_uses_paragraf_test_layout(generated_package: Path) -> None:
    generated_tests = generated_package / "tests"
    assert not (generated_tests / "_torch" / "auto_deploy").exists()
    assert not (generated_package / "examples" / "auto_deploy").exists()
    assert not {
        path.relative_to(generated_tests)
        for path in generated_tests.rglob("*")
        if "auto_deploy" in path.relative_to(generated_tests).parts
    }
    assert (
        generated_tests / "integration" / "defs" / "accuracy" / "test_llm_api_paragraf_trtllm.py"
    ).is_file()
    assert (generated_tests / "singlegpu" / "models" / "test_gpt_oss_modeling.py").is_file()
    stale_names = {
        path.name for path in generated_tests.rglob("test*.py") if AUTODEPLOY.search(path.name)
    }
    assert not stale_names


def test_keeps_layout_rewrites_out_of_source_tests() -> None:
    llm_api_source = (
        REPO_ROOT / "tests" / "integration" / "defs" / "accuracy" / "test_llm_api_autodeploy.py"
    ).read_text()
    assert "'examples' / 'auto_deploy'" in llm_api_source
    assert '"runners" / "trtllm"' not in llm_api_source

    mrope_source = (
        REPO_ROOT
        / "tests"
        / "unittest"
        / "auto_deploy"
        / "singlegpu"
        / "transformations"
        / "library"
        / "test_mrope_delta_cache.py"
    ).read_text()
    assert "Path(__file__).resolve().parents[6]" in mrope_source
    assert '_repo_root() / "tensorrt_llm" / "_torch" / "auto_deploy"' in mrope_source

    example_configs_source = (
        REPO_ROOT
        / "tests"
        / "unittest"
        / "auto_deploy"
        / "singlegpu"
        / "utils"
        / "test_example_configs.py"
    ).read_text()
    assert '_AD_EXAMPLES_DIR = _REPO_ROOT / "examples" / "auto_deploy"' in example_configs_source


def test_rewrites_generated_test_resource_layout(generated_package: Path) -> None:
    generated_tests = generated_package / "tests"
    llm_api = (
        generated_tests / "integration" / "defs" / "accuracy" / "test_llm_api_paragraf_trtllm.py"
    ).read_text()
    runner_registry = '"runners" / "trtllm" / "model_registry"'
    assert runner_registry in llm_api
    assert f'{runner_registry} / "configs" / "nano_v3.yaml"' in llm_api
    assert f'{runner_registry} / "configs" / "super_v3.yaml"' in llm_api

    mrope = (
        generated_tests / "singlegpu" / "transformations" / "library" / "test_mrope_delta_cache.py"
    ).read_text()
    assert "Path(__file__).resolve().parents[4]" in mrope
    assert '_repo_root() / "paragraf" / "config" / "default.yaml"' in mrope
    assert runner_registry in mrope

    example_configs = (
        generated_tests / "singlegpu" / "utils" / "test_example_configs.py"
    ).read_text()
    assert (
        '_AD_EXAMPLES_DIR = _REPO_ROOT / "runners" / "trtllm" / "model_registry"' in example_configs
    )


def test_does_not_add_test_guards_to_runners(generated_package: Path) -> None:
    runner = generated_package / "runners" / "trtllm" / "build_and_run_paragraf_trtllm.py"
    content = runner.read_text()
    assert "pytest.skip" not in content
    assert "pytest.importorskip" not in content


def test_generates_legacy_runner_wrapper(generated_package: Path) -> None:
    runner = generated_package / "runners" / "trtllm" / "build_and_run_llmc_trtllm.py"
    content = runner.read_text()
    assert "build_and_run_paragraf_trtllm" in content
    assert "main()" in content


def test_rewrites_all_canonical_auto_deploy_imports(generated_package: Path) -> None:
    offenders = {
        path.relative_to(generated_package)
        for path in (generated_package / "tests").rglob("*.py")
        if CANONICAL_IMPORT in path.read_text()
    }
    assert not offenders


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("singlegpu/custom_ops/attention/test_trtllm_attention_op.py"),
        Path("singlegpu/shim/test_engine.py"),
        Path("multigpu/compile/test_bypass_captured_graphs.py"),
        Path("singlegpu/models/test_gpt_oss_modeling.py"),
        Path("integration/defs/examples/test_paragraf_trtllm_guided_decoding.py"),
    ],
    ids=str,
)
def test_guards_optional_trtllm_tests(generated_package: Path, relative_path: Path) -> None:
    content = (generated_package / "tests" / relative_path).read_text()
    assert OPTIONAL_TRTLLM_GUARD in content
    assert 'pytest.importorskip("tensorrt_llm")' in content


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("singlegpu/test_pattern_matcher.py"),
        Path("singlegpu/transformations/library/test_mrope_delta_cache.py"),
        Path("multigpu/custom_ops/test_dist.py"),
    ],
    ids=str,
)
def test_keeps_pure_standalone_tests_unguarded(
    generated_package: Path, relative_path: Path
) -> None:
    content = (generated_package / "tests" / relative_path).read_text()
    assert OPTIONAL_TRTLLM_GUARD not in content


def test_rewrites_runner_imports_in_optional_tests(generated_package: Path) -> None:
    content = (
        generated_package
        / "tests"
        / "singlegpu"
        / "smoke"
        / "test_paragraf_trtllm_build_small_single.py"
    ).read_text()
    assert "from build_and_run_ad import" not in content
    assert "from runners.trtllm.build_and_run_paragraf_trtllm import" in content
