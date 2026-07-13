#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Create a standalone paragraf package from the TensorRT-LLM source tree.

This script copies the ``tensorrt_llm/_torch/auto_deploy`` source tree and
tests into a standalone pip-installable package. The output directory can
be pushed directly to the read-only standalone repository.

Usage:
    python create_standalone_package.py [--output-dir /path/to/output]

The generated package uses ``paragraf`` as the canonical top-level Python package
name. During the compatibility window, ``llmc`` remains an import alias and
``nvidia-llmc`` remains the distribution name:

    from paragraf._compat import TRTLLM_AVAILABLE
    from paragraf.custom_ops.attention_interface import SequenceInfo

The ``auto_deploy`` source tree itself is copied verbatim — internal imports
must already be relative (enforced by the
``auto-deploy-import-discipline`` pre-commit hook), so no source rewriting
is required. Test files use absolute ``tensorrt_llm._torch.auto_deploy``
imports by design and ARE rewritten to ``paragraf`` on copy.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import textwrap

from _license_data import VENDORED_PROJECTS, generate_attributions, generate_license

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
AUTO_DEPLOY_SRC = os.path.join(REPO_ROOT, "tensorrt_llm", "_torch", "auto_deploy")
TRTLLM_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements.txt")
TRTLLM_DEV_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements-dev.txt")
TRTLLM_LICENSE = os.path.join(REPO_ROOT, "LICENSE")
TRTLLM_GITIGNORE = os.path.join(REPO_ROOT, ".gitignore")
TRTLLM_EDITORCONFIG = os.path.join(REPO_ROOT, ".editorconfig")
TRTLLM_CODE_OF_CONDUCT = os.path.join(REPO_ROOT, "CODE_OF_CONDUCT.md")
TRTLLM_SECURITY = os.path.join(REPO_ROOT, "SECURITY.md")
TRTLLM_ATTRIBUTIONS_PYTHON = os.path.join(REPO_ROOT, "ATTRIBUTIONS-Python.md")
PARAGRAF_README = os.path.join(SCRIPT_DIR, "README.md")
PARAGRAF_CONTRIBUTING = os.path.join(SCRIPT_DIR, "CONTRIBUTING.md")

# Test source directories
AD_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "unittest", "auto_deploy")
AD_TORCH_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "unittest", "_torch", "auto_deploy")
AD_INTEGRATION_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "integration", "defs")

# Example/e2e harness sources (Tier-1 e2e: build_and_run_ad.py + model registry).
# These ship with the package so the standalone install can run e2e models via
# the same entrypoint as TRT-LLM. Python is rewritten auto_deploy -> paragraf; the
# model_registry YAML is data and copied verbatim.
AD_EXAMPLES_SRC = os.path.join(REPO_ROOT, "examples", "auto_deploy")
# Source filename -> destination filename in the standalone package. The e2e
# entrypoint is renamed to make its scope explicit in the paragraf distribution.
EXAMPLE_FILES = {"build_and_run_ad.py": "build_and_run_paragraf_trtllm.py"}
EXAMPLE_DIRS = ["model_registry"]
LEGACY_RUNNER_NAME = "build_and_run_llmc_trtllm.py"
LEGACY_RUNNER_WRAPPER = textwrap.dedent("""\
    #!/usr/bin/env python3
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

    if __package__:
        from .build_and_run_paragraf_trtllm import *
    else:
        from build_and_run_paragraf_trtllm import *

    if __name__ == "__main__":
        main()
    """)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COPY_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".txt", ".md"}
EXCLUDE_PATTERNS = {"__pycache__", ".pyc", ".pyo"}

# Standalone runtime dependencies (version pins pulled from TRT-LLM requirements.txt)
STANDALONE_DEPS = [
    "torch",
    "transformers",
    "pydantic",
    "pydantic-settings",
    "triton",
    "flashinfer-python",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "omegaconf",
    "pyyaml",
    "numpy",
    "pillow",
    "einops",
    # Transitive deps that may not be pulled in by all installers/platforms
    "six",
    "importlib-metadata",
    "werkzeug",
    "StrEnum",
    "graphviz",
]

# Dev/test dependencies (version pins pulled from TRT-LLM requirements-dev.txt)
DEV_DEPS = [
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "pytest-cov",
    "pytest-mock",
    "pytest-asyncio",
    "parameterized",
    "cloudpickle",
    "mpi4py",
    "openai",
    "requests",
    "scipy",
]

# These tests validate the generator and its output. They belong to the source
# repository rather than the generated package.
SOURCE_ONLY_TEST_DIRS = {"standalone"}

# Tests in this set are copied, but only collected when the optional TensorRT-LLM
# wheel is enabled through TRTLLM_REDIRECT_AD_TO_PARAGRAF. Keeping this explicit
# also covers indirect dependencies where a rewritten ``paragraf`` import loads
# a module that depends on TensorRT-LLM.
OPTIONAL_TRTLLM_TEST_FILES = {
    # TRT-LLM kernel tests
    "test_trtllm_moe.py",
    "test_trtllm_attention_op.py",
    "test_fla_cached_gated_delta_rule.py",
    "test_flashinfer_trtllm_mla_op.py",
    "test_trtllm_mla_op.py",
    "test_fuse_trtllm_attention_quant_fp8.py",
    "test_fuse_relu2_quant_nvfp4.py",
    "test_moe_fusion.py",
    "test_trtllm_gen_diag.py",
    # Standalone flashinfer ROPE path has a known BF16 strided-interleaved mismatch.
    "test_rope_op_variants.py",
    # QKV fusion → trtllm cache insertion (TRT-LLM attention backend only)
    "test_gemm_fusion_trtllm.py",
    # Require TRT-LLM LlmArgs / runtime
    "test_eagle.py",
    "test_modeling_nemotron_h.py",
    "test_example_configs.py",
    "test_hybrid_patches.py",
    "test_captured_graph.py",
    # Require external scripts (build_and_run_ad.py)
    "test_llama4_vlm_patch.py",
    "test_mistral3_patches.py",
    # Require TRT-LLM test utils not in standalone
    "test_ad_moe_op.py",
    "test_triton_moe.py",
    # Require onnx (optional dep)
    "test_export_fp8_linear_to_onnx.py",
    # Uses hardcoded TRT-LLM repo path
    "test_mrope_delta_cache.py",
    # Depend on TRT-LLM mamba/fla kernels (relative imports beyond auto_deploy)
    "test_mamba_rms_norm.py",
    "test_triton_rms_norm.py",
    "test_fuse_rmsnorm.py",
    "test_fused_add_rms_norm.py",
    "test_fuse_l2norm.py",
    # Require TRT-LLM KVCacheManager or runtime
    "test_kv_cache.py",
    "test_torch_gated_delta_rule_cache.py",
    "test_gated_delta_rule_cache.py",
    "test_kv_cache_transformers.py",
    # trtllm attention backend (insert_cached_attention backend=trtllm) not available standalone
    "test_kv_cache_trtllm_multipool.py",
    # Require TRT-LLM CUDA causal conv / mamba kernels (ops not registered standalone)
    "test_cuda_causal_conv_cached_op.py",
    "test_triton_causal_conv_cached_op.py",
    "test_triton_mamba_cached_op.py",
    "test_flashinfer_mamba_cached_op.py",
    # Require TRT-LLM custom ops (dsv3_router_gemm_op, noaux_tc_op, etc.)
    "test_deepseek_custom.py",
    "test_glm4_moe_modeling.py",
    "test_glm4_moe_lite_modeling.py",
    "test_glm_moe_dsa_modeling.py",
    # Full-model tests hit standalone-incompatible HF cache behavior.
    "test_granite_moe_hybrid_modeling.py",
    # Imports triton_kernels, which is not a standalone dependency.
    "test_mxfp4_moe_layout.py",
    # Require TRT-LLM distributed ops (trtllm_dist_all_gather)
    "test_gather_logits_before_lm_head.py",
    # Multimodal processors depend on TensorRT-LLM multimodal request types.
    "test_gemma4_modeling.py",
    "test_qwen3_5_moe.py",
    # Hardware-specific (requires H100+ shared memory)
    "test_triton_mla_op.py",
    # Require TRT-LLM ops (noaux_tc_op) — split from test_export.py
    "test_export_glm4_moe_lite.py",
    # fuse_fp8_linear / fuse_nvfp4_linear / fuse_finegrained_fp8_linear transforms
    # live in fuse_quant.py which imports tensorrt_llm.quantization.utils.fp8_utils;
    # the module is silently skipped in standalone so the transforms aren't registered.
    "test_quant_fusion.py",
    # Imports utils.util.skip_pre_blackwell (not shipped in standalone) and exercises
    # fuse_finegrained_fp8_swiglu which depends on TRT-LLM runtime.
    "test_finegrained_fp8_swiglu.py",
    # Exercise trtllm-gen MXFP4 MoE kernels (Blackwell-only) and import the
    # prepare_trtllm_gen_moe_mxfp4_weights / utils.util helpers not in standalone.
    "test_fuse_mxfp4_moe.py",
    "test_trtllm_quant_mxfp4_trtllm_gen_moe.py",
}

# Multi-GPU tests known to run without the optional TensorRT-LLM wheel. Other
# AutoDeploy multi-GPU tests are still copied, but receive the optional-wheel
# collection guard.
PURE_STANDALONE_MULTIGPU_TEST_FILES = {
    "custom_ops/test_dist.py",
    "custom_ops/test_sharded_rmsnorm.py",
    "smoke/test_ad_build_small_multi.py",
    "transformations/library/test_apply_sharding_hints.py",
    "transformations/library/test_bmm_sharding.py",
    "transformations/library/test_ep_sharding.py",
    "transformations/library/test_rmsnorm_sharding.py",
    "transformations/library/test_sharding_num_correctness.py",
    "transformations/library/test_step3p7_sharding_ir.py",
    "transformations/library/test_tp_sharding.py",
}

# AutoDeploy integration files selected by the same classifier used by CI.
AUTODEPLOY_TEST_RE = re.compile(r"auto_?deploy|_ad_", re.IGNORECASE)
TEST_FILE_RE = re.compile(r"(?:^test_.*|.*_test)\.py$")

# Support files needed by the selected integration tests. The source integration
# conftest is intentionally not copied because it pulls in the complete TRT-LLM
# CI harness; a focused replacement is generated below.
INTEGRATION_SUPPORT_FILES = (
    "__init__.py",
    "common.py",
    "trt_test_alternative.py",
    "accuracy/__init__.py",
    "accuracy/accuracy_core.py",
    "accuracy/video_mme.py",
    "disaggregated/disagg_test_utils.py",
)
INTEGRATION_SUPPORT_DIRS = ("accuracy/references",)

# Additional source-tree test helpers imported by copied AutoDeploy tests.
TORCH_TEST_SUPPORT_FILES = (
    "tests/unittest/_torch/__init__.py",
    "tests/unittest/_torch/helpers.py",
)

# Import path rewrite: old -> new (applied to test files only).
_IMPORT_REWRITE = "tensorrt_llm._torch.auto_deploy"
_IMPORT_TARGET = "paragraf"
_BUILD_AND_RUN_AD_IMPORT = "from build_and_run_ad import ExperimentConfig, main"
_TRTLLM_IMPORT_RE = re.compile(
    r"(?m)^(?:from|import) "
    r"(?:tensorrt_llm(?:\.|\b)|paragraf\.models\.custom\.modeling_gpt_oss(?:\.|\b))"
)
_PARAGRAF_OPTIONAL_TRTLLM_GUARD = """
_trtllm_environ = __import__("os").environ
_trtllm_redirect_value = _trtllm_environ.get("TRTLLM_REDIRECT_AD_TO_PARAGRAF")
if _trtllm_redirect_value is None:
    _trtllm_redirect_value = _trtllm_environ.get("TRTLLM_REDIRECT_AD_TO_LLMC", "")
_trtllm_redirect_value = _trtllm_redirect_value.lower()
if _trtllm_redirect_value not in {"1", "true", "yes", "on"}:
    pytest.skip(
        "Paragraf optional TRT-LLM tests require TRTLLM_REDIRECT_AD_TO_PARAGRAF=true",
        allow_module_level=True,
    )
pytest.importorskip("tensorrt_llm")"""
_PARAGRAF_TRTLLM_RUNNER_IMPORT = (
    "from runners.trtllm.build_and_run_paragraf_trtllm import ExperimentConfig, main"
)

# Paths that the script owns and regenerates on every run.
# Everything else in the output directory (e.g., .git/, .github/) is preserved
# and owned by the standalone repo itself.
_MANAGED_PATHS = [
    "paragraf",
    # Remove the package directory produced before the Paragraf rename.
    "llmc",
    "tests",
    "examples/auto_deploy/model_registry",
    "runners",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "ATTRIBUTIONS-Python.md",
    "CONTRIBUTING.md",
    ".gitignore",
    ".editorconfig",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "ATTRIBUTIONS-Python.md",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _should_copy(filepath: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return False
    basename = os.path.basename(filepath)
    if ".bak." in basename or basename.endswith("~"):
        return False
    _, ext = os.path.splitext(filepath)
    return ext in COPY_EXTENSIONS


def _tracked_files_under(directory: str) -> list[str]:
    """Return tracked files below a source directory, with an archive fallback."""
    relative_directory = os.path.relpath(directory, REPO_ROOT)
    try:
        result = subprocess.run(
            ["git", "-C", REPO_ROOT, "ls-files", "--", relative_directory],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        discovered_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [directory_name for directory_name in dirs if directory_name != "__pycache__"]
            discovered_files.extend(os.path.join(root, filename) for filename in files)
        return discovered_files

    return [os.path.join(REPO_ROOT, path) for path in result.stdout.splitlines()]


def _copy_file(src_path: str, dst_path: str) -> int:
    if not os.path.isfile(src_path) or not _should_copy(src_path):
        return 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return 1


def _copy_tracked_tree(src_dir: str, dst_dir: str) -> int:
    """Copy tracked files from a tree, or all files in a source archive."""
    count = 0
    for src_path in _tracked_files_under(src_dir):
        rel_path = os.path.relpath(src_path, src_dir)
        count += _copy_file(src_path, os.path.join(dst_dir, rel_path))
    return count


def _copy_tree(src_dir: str, dst_dir: str) -> int:
    """Copy files from src_dir to dst_dir, preserving directory structure."""
    count = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for filename in files:
            src_path = os.path.join(root, filename)
            if not _should_copy(src_path):
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            count += 1
    return count


def _rewrite_imports_in_file(
    filepath: str,
    *,
    optional_trtllm_guards: bool = True,
    force_optional_trtllm_guard: bool = False,
) -> int:
    """Rewrite imports in a copied test file for standalone mode.

    Source files inside ``tensorrt_llm/_torch/auto_deploy`` already use
    relative imports (enforced by the ``auto-deploy-import-discipline``
    pre-commit hook), so no rewriting is needed for them. Tests, however,
    are written against the canonical absolute path
    ``tensorrt_llm._torch.auto_deploy`` and need to be rewritten to
    ``paragraf``. Cross-package types (e.g. ``KvCacheConfig``,
    ``ActivationType``) are sourced via ``..._torch.auto_deploy._compat``,
    so the primary rewrite handles them too.

    Returns the number of line-level changes made.
    """
    with open(filepath) as f:
        content = f.read()

    original = content
    content = content.replace(_IMPORT_REWRITE, _IMPORT_TARGET)
    content = content.replace("from auto_deploy.", "from ")
    content = re.sub(
        r"(?m)^from (test_[A-Za-z0-9_]+|disagg_test_utils) import ",
        r"from .\1 import ",
        content,
    )
    content = content.replace(
        'pytest_plugins = ["disagg_test_utils"]',
        'pytest_plugins = ["integration.defs.disaggregated.disagg_test_utils"]',
    )
    content = content.replace(
        "_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]",
        "_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]",
    )

    def ensure_imports(before_pos: int, *imports: str) -> None:
        nonlocal content
        prefix = content[:before_pos]
        missing_imports = [
            import_name for import_name in imports if f"import {import_name}\n" not in prefix
        ]
        if not missing_imports:
            return
        first_import = re.search(r"(?m)^(?:import|from) ", content)
        if first_import is None:
            raise ValueError(f"No import block found in {filepath}")
        content = (
            content[: first_import.start()]
            + "\n".join(f"import {import_name}" for import_name in missing_imports)
            + "\n"
            + content[first_import.start() :]
        )

    def insert_optional_trtllm_guard() -> None:
        nonlocal content
        if _PARAGRAF_OPTIONAL_TRTLLM_GUARD in content:
            return
        pytest_import = re.search(r"(?m)^import pytest\n", content)
        if pytest_import is None:
            raise ValueError(f"No pytest import found in {filepath}")
        content = (
            content[: pytest_import.end()]
            + _PARAGRAF_OPTIONAL_TRTLLM_GUARD
            + "\n"
            + content[pytest_import.end() :]
        )

    if optional_trtllm_guards and _BUILD_AND_RUN_AD_IMPORT in content:
        build_import_pos = content.index(_BUILD_AND_RUN_AD_IMPORT)
        ensure_imports(build_import_pos, "os", "pytest")
        insert_optional_trtllm_guard()
        content = content.replace(_BUILD_AND_RUN_AD_IMPORT, _PARAGRAF_TRTLLM_RUNNER_IMPORT)
    elif optional_trtllm_guards and force_optional_trtllm_guard:
        ensure_imports(len(content), "os", "pytest")
        insert_optional_trtllm_guard()
    elif optional_trtllm_guards:
        trtllm_import = _TRTLLM_IMPORT_RE.search(content)
        if trtllm_import is not None:
            ensure_imports(trtllm_import.start(), "os", "pytest")
            insert_optional_trtllm_guard()

    if optional_trtllm_guards:
        # The standalone package can rely on the installed trtllm-bench entrypoint,
        # but it does not ship TensorRT-LLM's source-tree benchmarks/cpp directory.
        content = content.replace(
            '    script_dir = Path(root_dir, "benchmarks", "cpp")\n',
            "    script_dir = Path(temp_dir)\n",
        )

    replacements = sum(1 for a, b in zip(original, content) if a != b)  # rough count
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        # Count actual line-level changes
        replacements = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)

    return replacements


def _requires_optional_trtllm_guard(filepath: str, tests_dir: str) -> bool:
    relative_path = os.path.relpath(filepath, tests_dir).replace("\\", "/")
    if not TEST_FILE_RE.fullmatch(os.path.basename(filepath)):
        return False
    path_parts = relative_path.split("/")
    if relative_path.startswith("integration/"):
        return True
    if "shim" in path_parts:
        return True
    if os.path.basename(filepath) in OPTIONAL_TRTLLM_TEST_FILES:
        return True
    if relative_path.startswith("multigpu/"):
        multigpu_path = relative_path.removeprefix("multigpu/")
        return multigpu_path not in PURE_STANDALONE_MULTIGPU_TEST_FILES
    return False


def _rewrite_imports_in_dir(directory: str, *, optional_trtllm_guards: bool = True) -> int:
    """Rewrite imports in all .py files in a directory tree."""
    total = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                total += _rewrite_imports_in_file(
                    os.path.join(root, filename),
                    optional_trtllm_guards=optional_trtllm_guards,
                    force_optional_trtllm_guard=(
                        optional_trtllm_guards
                        and _requires_optional_trtllm_guard(os.path.join(root, filename), directory)
                    ),
                )
    return total


def _read_pinned_versions(req_file: str) -> dict:
    """Read requirements.txt and extract package->version-spec mapping."""
    versions = {}
    if not os.path.exists(req_file):
        return versions
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            line = line.split("#")[0].strip()
            # Handle semicolons (environment markers like ; python_version >= "3.10")
            line = line.split(";")[0].strip()
            match = re.match(r"^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)(.*)", line)
            if match:
                pkg_name = match.group(1).split("[")[0].lower()
                version_spec = match.group(2).strip()
                if version_spec:
                    versions[pkg_name] = version_spec
    return versions


def _resolve_dependencies(dep_names: list, pinned: dict) -> list:
    """Resolve dependency list by adding version pins from requirements."""
    resolved = []
    for name in dep_names:
        extras = ""
        if "[" in name:
            base, extras_part = name.split("[", 1)
            extras = f"[{extras_part}"
        else:
            base = name
        version = pinned.get(base.lower(), "")
        resolved.append(f"{base}{extras}{version}")
    return resolved


# ---------------------------------------------------------------------------
# Test copying
# ---------------------------------------------------------------------------
def _copy_tests(output_dir: str) -> int:
    """Copy auto_deploy test files to the standalone package tests/ directory."""
    tests_dst = os.path.join(output_dir, "tests")
    count = 0

    # Copy every tracked legacy AutoDeploy unit-test file except the tests of
    # this generator itself. Optional TensorRT-LLM dependencies are handled by
    # collection guards after the files are copied.
    for src_path in _tracked_files_under(AD_TESTS_DIR):
        rel_path = os.path.relpath(src_path, AD_TESTS_DIR)
        if any(part in SOURCE_ONLY_TEST_DIRS for part in rel_path.split(os.sep)):
            continue
        count += _copy_file(src_path, os.path.join(tests_dst, rel_path))

    # Copy every tracked test from the newer unit-test tree. This avoids an
    # allowlist that silently misses tests added alongside new AutoDeploy code.
    for src_path in _tracked_files_under(AD_TORCH_TESTS_DIR):
        rel_path = os.path.relpath(src_path, AD_TORCH_TESTS_DIR)
        count += _copy_file(
            src_path,
            os.path.join(tests_dst, "_torch", "auto_deploy", rel_path),
        )

    # The CI classifier also finds a small number of AutoDeploy integration
    # tests outside the unit-test roots. Copy those tests and their focused
    # support modules without bringing in the complete TensorRT-LLM CI suite.
    integration_tests_root = os.path.join(REPO_ROOT, "tests", "integration")
    for src_path in _tracked_files_under(AD_INTEGRATION_TESTS_DIR):
        rel_from_tests = os.path.relpath(src_path, integration_tests_root)
        if not TEST_FILE_RE.fullmatch(os.path.basename(src_path)):
            continue
        if not AUTODEPLOY_TEST_RE.search(rel_from_tests.replace("\\", "/")):
            continue
        count += _copy_file(
            src_path,
            os.path.join(tests_dst, "integration", rel_from_tests),
        )

    for rel_path in INTEGRATION_SUPPORT_FILES:
        count += _copy_file(
            os.path.join(AD_INTEGRATION_TESTS_DIR, rel_path),
            os.path.join(tests_dst, "integration", "defs", rel_path),
        )
    for rel_path in INTEGRATION_SUPPORT_DIRS:
        count += _copy_tracked_tree(
            os.path.join(AD_INTEGRATION_TESTS_DIR, rel_path),
            os.path.join(tests_dst, "integration", "defs", rel_path),
        )

    count += _copy_tracked_tree(
        os.path.join(AD_EXAMPLES_SRC, "model_registry"),
        os.path.join(output_dir, "examples", "auto_deploy", "model_registry"),
    )

    for rel_path in TORCH_TEST_SUPPORT_FILES:
        count += _copy_file(
            os.path.join(REPO_ROOT, rel_path),
            os.path.join(tests_dst, os.path.relpath(rel_path, "tests/unittest")),
        )

    # Create conftest.py for test discovery and imports
    _create_test_package_init_files(tests_dst)
    _create_test_conftest(tests_dst)
    _create_integration_conftest(tests_dst)

    # Create a stub for test_common.llm_data (used by some model tests)
    _create_test_common_stub(tests_dst)
    _create_test_utils_stub(tests_dst)

    return count


def _create_test_package_init_files(tests_dir: str) -> None:
    """Give copied test directories stable package-qualified module names."""
    content = (
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION"
        " & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
    )
    for root, dirs, _ in os.walk(tests_dir):
        dirs[:] = [directory for directory in dirs if directory != "__pycache__"]
        if root == tests_dir:
            continue
        init_path = os.path.join(root, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write(content)


def _create_test_conftest(tests_dir: str) -> None:
    """Create a conftest.py that configures the test environment for standalone mode."""
    content = textwrap.dedent("""\
        # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        \"\"\"Conftest for standalone auto_deploy tests.\"\"\"
        import importlib.util
        import os
        import sys
        from pathlib import Path

        import pytest

        _trtllm_redirect_value = os.environ.get("TRTLLM_REDIRECT_AD_TO_PARAGRAF")
        if _trtllm_redirect_value is None:
            _trtllm_redirect_value = os.environ.get("TRTLLM_REDIRECT_AD_TO_LLMC", "")
        _allow_trtllm_redirect = _trtllm_redirect_value.lower() in {
            "1", "true", "yes", "on"
        }
        _trtllm_spec = importlib.util.find_spec("tensorrt_llm")
        if _trtllm_spec is not None and not _allow_trtllm_redirect:
            raise RuntimeError(
                "Standalone paragraf tests must not be able to import tensorrt_llm; "
                "set TRTLLM_REDIRECT_AD_TO_PARAGRAF=true only for optional TRT-LLM tests; "
                f"found {getattr(_trtllm_spec, 'origin', None)!r}"
            )

        _tests_dir = os.path.dirname(__file__)
        _package_root = os.path.dirname(_tests_dir)
        _integration_tests_dir = os.path.join(_tests_dir, "integration")

        # Add generated package/test roots to the Python path so tests can import
        # local paragraf, runners, integration helpers, and _utils_test even
        # under safe-path settings.
        sys.path.insert(0, _package_root)
        sys.path.insert(0, _tests_dir)
        sys.path.insert(0, _integration_tests_dir)
        sys.path.insert(0, os.path.join(_tests_dir, "_utils_test"))


        @pytest.fixture(scope="module")
        def llm_root():
            env_root = os.environ.get("LLM_ROOT")
            if env_root:
                return Path(env_root)
            return Path(_package_root)
    """)
    with open(os.path.join(tests_dir, "conftest.py"), "w") as f:
        f.write(content)


def _create_integration_conftest(tests_dir: str) -> None:
    """Create the focused helpers needed by copied AutoDeploy integration tests."""
    defs_dir = os.path.join(tests_dir, "integration", "defs")
    os.makedirs(defs_dir, exist_ok=True)
    content = textwrap.dedent("""\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        import os
        from pathlib import Path

        import pytest
        import torch


        def get_llm_root():
            return str(Path(__file__).resolve().parents[3])


        def llm_models_root():
            models_root = os.environ.get("LLM_MODELS_ROOT")
            if not models_root:
                pytest.skip("LLM_MODELS_ROOT is required for AutoDeploy integration tests")
            return models_root


        def get_sm_version():
            if not torch.cuda.is_available():
                return 0
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor


        def get_device_count():
            return torch.cuda.device_count()


        def get_device_memory():
            if not torch.cuda.is_available():
                return 0
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)


        def check_device_contain(keyword_list):
            if not torch.cuda.is_available():
                return False
            device_name = torch.cuda.get_device_name(0)
            return any(keyword in device_name for keyword in keyword_list)


        skip_pre_ada = pytest.mark.skipif(
            get_sm_version() < 89,
            reason="This test is not supported in pre-Ada architecture",
        )
        skip_pre_hopper = pytest.mark.skipif(
            get_sm_version() < 90,
            reason="This test is not supported in pre-Hopper architecture",
        )
        skip_pre_blackwell = pytest.mark.skipif(
            get_sm_version() < 100,
            reason="This test is not supported in pre-Blackwell architecture",
        )


        @pytest.fixture(autouse=True)
        def _apply_resource_markers(request):
            device_marker = request.node.get_closest_marker("skip_less_device")
            if device_marker and get_device_count() < device_marker.args[0]:
                pytest.skip(f"Test requires {device_marker.args[0]} GPUs")

            for memory_marker in request.node.iter_markers("skip_less_device_memory"):
                if get_device_memory() < memory_marker.args[0]:
                    pytest.skip(f"Test requires {memory_marker.args[0]} MiB of GPU memory")
    """)
    with open(os.path.join(defs_dir, "conftest.py"), "w") as f:
        f.write(content)


def _create_test_common_stub(tests_dir: str) -> None:
    """Create a stub for test_common.llm_data (provides HF model path resolution).

    In standalone mode, tests that need local model weights will be skipped
    unless LLM_MODELS_ROOT is set.
    """
    stub_dir = os.path.join(tests_dir, "test_common")
    os.makedirs(stub_dir, exist_ok=True)

    with open(os.path.join(stub_dir, "__init__.py"), "w") as f:
        f.write(
            "# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION"
            " & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
        )

    content = textwrap.dedent("""\
        # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        \"\"\"Stub for test_common.llm_data in standalone mode.\"\"\"
        import os
        from pathlib import Path
        from unittest.mock import patch

        LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT")


        def llm_models_root():
            return Path(LLM_MODELS_ROOT) if LLM_MODELS_ROOT else None


        def hf_id_to_local_model_dir(hf_id: str):
            root = llm_models_root()
            if root is None:
                return hf_id  # Fall back to HF hub download
            # Try direct match
            candidate = root / hf_id.split("/")[-1]
            if candidate.exists():
                return str(candidate)
            return hf_id


        def with_mocked_hf_download_for_single_gpu(func):
            return func  # No-op in standalone mode
    """)
    with open(os.path.join(stub_dir, "llm_data.py"), "w") as f:
        f.write(content)


def _create_test_utils_stub(tests_dir: str) -> None:
    """Create minimal TensorRT-LLM unittest utility shims used by copied tests."""
    utils_dir = os.path.join(tests_dir, "utils")
    os.makedirs(utils_dir, exist_ok=True)

    with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
        f.write(
            "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION"
            " & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
        )

    content = textwrap.dedent("""\
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

        \"\"\"Minimal unittest utility shims for standalone Paragraf tests.\"\"\"

        import pytest
        import torch


        def _sm_version() -> int:
            if not torch.cuda.is_available():
                return 0
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor


        skip_pre_hopper = pytest.mark.skipif(
            _sm_version() < 90,
            reason="This test is not supported in pre-Hopper architecture",
        )
        skip_no_hopper = pytest.mark.skipif(
            _sm_version() != 90,
            reason="This test is only supported in Hopper architecture",
        )
        skip_pre_blackwell = pytest.mark.skipif(
            _sm_version() < 100,
            reason="This test is not supported in pre-Blackwell architecture",
        )
    """)
    with open(os.path.join(utils_dir, "util.py"), "w") as f:
        f.write(content)

    cpp_paths_content = textwrap.dedent("""\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        from pathlib import Path

        import pytest


        @pytest.fixture(scope="module")
        def llm_root():
            return Path(__file__).resolve().parents[2]
    """)
    with open(os.path.join(utils_dir, "cpp_paths.py"), "w") as f:
        f.write(cpp_paths_content)

    llm_data_content = textwrap.dedent("""\
        # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        # SPDX-License-Identifier: Apache-2.0

        from test_common.llm_data import llm_models_root

        __all__ = ["llm_models_root"]
    """)
    with open(os.path.join(utils_dir, "llm_data.py"), "w") as f:
        f.write(llm_data_content)


# ---------------------------------------------------------------------------
# Example / e2e harness copying
# ---------------------------------------------------------------------------
def _copy_runners(output_dir: str) -> int:
    """Copy the Tier-1 e2e harness into the standalone package under ``runners/trtllm/``.

    ``examples/auto_deploy/build_and_run_ad.py`` is copied to
    ``runners/trtllm/build_and_run_paragraf_trtllm.py`` (see ``EXAMPLE_FILES``) together
    with its sibling ``model_registry/``. Because the script resolves the registry
    relative to its own location (``Path(__file__).parent / "model_registry"``),
    the rename + relocation are safe and ``--use-registry`` keeps working.
    ``.py`` files get the usual ``auto_deploy -> paragraf`` import rewrite (applied by
    the caller); the ``model_registry`` YAML is data, copied verbatim.
    """
    runners_dst = os.path.join(output_dir, "runners", "trtllm")
    count = 0
    for src_name, dst_name in EXAMPLE_FILES.items():
        src = os.path.join(AD_EXAMPLES_SRC, src_name)
        if os.path.isfile(src):
            os.makedirs(runners_dst, exist_ok=True)
            shutil.copy2(src, os.path.join(runners_dst, dst_name))
            count += 1
    for dname in EXAMPLE_DIRS:
        src = os.path.join(AD_EXAMPLES_SRC, dname)
        if os.path.isdir(src):
            count += _copy_tree(src, os.path.join(runners_dst, dname))
    if os.path.isdir(runners_dst):
        with open(os.path.join(runners_dst, LEGACY_RUNNER_NAME), "w") as f:
            f.write(LEGACY_RUNNER_WRAPPER)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Package generation
# ---------------------------------------------------------------------------
def _create_pyproject_toml(output_dir: str, dependencies: list, dev_dependencies: list) -> None:
    """Create a pyproject.toml for the standalone package."""
    deps_lines = "\n".join(f'    "{dep}",' for dep in dependencies)
    dev_deps_lines = "\n".join(f'    "{dep}",' for dep in dev_dependencies)

    content = (
        "[build-system]\n"
        'requires = ["setuptools>=64", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        'name = "nvidia-llmc"\n'
        'version = "0.1.0"\n'
        'description = "paragraf: standalone LLM compiler — '
        'automatic model optimization and deployment for LLM inference"\n'
        'readme = "README.md"\n'
        'license = {text = "Apache-2.0"}\n'
        'requires-python = ">=3.10"\n'
        "dependencies = [\n"
        f"{deps_lines}\n"
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        'trtllm = ["tensorrt-llm"]\n'
        "dev = [\n"
        f"{dev_deps_lines}\n"
        "]\n"
        "\n"
        "[tool.setuptools.packages.find]\n"
        'include = ["paragraf*", "llmc"]\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
        "markers = [\n"
        '    "threadleak(enabled): configure thread-leak checks (inert in standalone tests)",\n'
        '    "skip_less_device(count): require at least count GPUs",\n'
        '    "skip_less_device_memory(mib): require at least mib MiB on one GPU",\n'
        "]\n"
    )

    with open(os.path.join(output_dir, "pyproject.toml"), "w") as f:
        f.write(content)


def create_standalone_package(output_dir: str) -> None:
    """Create the standalone paragraf package at the given output directory.

    Safe to run against an existing git repository: only the managed paths
    (source, tests, and packaging files) are deleted and regenerated. The .git
    directory and any repo-specific files (e.g., .github/) are preserved.
    After running, ``git add -A && git commit`` captures all changes.
    """
    if not os.path.isdir(AUTO_DEPLOY_SRC):
        print(f"ERROR: auto_deploy source not found at {AUTO_DEPLOY_SRC}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Clean only the paths this script manages, preserving .git and other repo files
    for name in _MANAGED_PATHS:
        target = os.path.join(output_dir, name)
        if os.path.islink(target):
            os.remove(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.isfile(target):
            os.remove(target)

    print(f"Creating standalone package at: {output_dir}")

    # 1. Copy auto_deploy source as top-level `paragraf/` package. No import
    #    rewriting is needed: in-package imports are relative (enforced by
    #    the auto-deploy-import-discipline pre-commit hook).
    ad_dst = os.path.join(output_dir, "paragraf")
    count = _copy_tree(AUTO_DEPLOY_SRC, ad_dst)
    print(f"  Copied {count} source files to paragraf/")

    legacy_dst = os.path.join(output_dir, "llmc")
    os.symlink("paragraf", legacy_dst, target_is_directory=True)
    print("  Created legacy llmc -> paragraf package alias")

    # 2. Copy and rewrite tests (tests use absolute self-imports by design).
    test_count = _copy_tests(output_dir)
    rewrite_count = _rewrite_imports_in_dir(os.path.join(output_dir, "tests"))
    print(f"  Copied {test_count} test/support files ({rewrite_count} import rewrites)")

    # 2b. Copy the Tier-1 e2e harness into runners/ (build_and_run_paragraf_trtllm.py
    #     + model_registry) and rewrite its imports auto_deploy -> paragraf. YAML is
    #     left untouched.
    runner_count = _copy_runners(output_dir)
    runner_rewrites = _rewrite_imports_in_dir(
        os.path.join(output_dir, "runners"),
        optional_trtllm_guards=False,
    )
    print(
        f"  Copied {runner_count} runner files to runners/trtllm/ ({runner_rewrites} import rewrites)"
    )

    # 3. Resolve dependencies and create pyproject.toml
    pinned = _read_pinned_versions(TRTLLM_REQUIREMENTS)
    dev_pinned = _read_pinned_versions(TRTLLM_DEV_REQUIREMENTS)
    # Merge: dev_pinned has the same packages as pinned plus test-only packages
    all_pinned = {**pinned, **dev_pinned}
    dependencies = _resolve_dependencies(STANDALONE_DEPS, pinned)
    dev_dependencies = _resolve_dependencies(DEV_DEPS, all_pinned)
    _create_pyproject_toml(output_dir, dependencies, dev_dependencies)
    print(f"  Created pyproject.toml ({len(dependencies)} deps + {len(dev_dependencies)} dev deps)")

    # 4. Generate standalone LICENSE (only vendored projects in auto_deploy)
    generate_license(output_dir)
    print(f"  Generated LICENSE ({len(VENDORED_PROJECTS)} vendored projects)")

    # 5. Generate ATTRIBUTIONS-Python.md (direct dependency licenses)
    generate_attributions(output_dir, dependencies)
    print(f"  Generated ATTRIBUTIONS-Python.md ({len(dependencies)} direct deps)")

    # 6. Copy README
    if os.path.exists(PARAGRAF_README):
        shutil.copy2(PARAGRAF_README, os.path.join(output_dir, "README.md"))
        print("  Copied README.md")

    # 7. Copy CONTRIBUTING.md
    if os.path.exists(PARAGRAF_CONTRIBUTING):
        shutil.copy2(PARAGRAF_CONTRIBUTING, os.path.join(output_dir, "CONTRIBUTING.md"))
        print("  Copied CONTRIBUTING.md")

    # 8. Copy .gitignore
    if os.path.exists(TRTLLM_GITIGNORE):
        shutil.copy2(TRTLLM_GITIGNORE, os.path.join(output_dir, ".gitignore"))
        print("  Copied .gitignore")

    # 9. Copy .editorconfig
    if os.path.exists(TRTLLM_EDITORCONFIG):
        shutil.copy2(TRTLLM_EDITORCONFIG, os.path.join(output_dir, ".editorconfig"))
        print("  Copied .editorconfig")

    # 10. Copy OSS compliance files (CODE_OF_CONDUCT, SECURITY)
    for src, name in (
        (TRTLLM_CODE_OF_CONDUCT, "CODE_OF_CONDUCT.md"),
        (TRTLLM_SECURITY, "SECURITY.md"),
    ):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            print(f"  Copied {name}")

    print(f"\nStandalone package created at: {output_dir}")
    print("\nTo install:")
    print(f"  cd {output_dir}")
    print("  uv venv .venv --python 3.12")
    print("  source .venv/bin/activate")
    print("  uv pip install -e '.[dev]'")
    print("\nTo run tests:     pytest tests/")
    print("To run optional TensorRT-LLM tests:")
    print("  uv pip install -e '.[dev,trtllm]'")
    print("  TRTLLM_REDIRECT_AD_TO_PARAGRAF=true pytest tests/")
    print(
        'To verify:        python -c "from paragraf._compat import TRTLLM_AVAILABLE; print(TRTLLM_AVAILABLE)"'
    )
    print(
        "To run e2e:       python runners/trtllm/build_and_run_paragraf_trtllm.py "
        "--model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --use-registry  (needs tensorrt-llm installed)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a standalone paragraf package from TensorRT-LLM source.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "build", "paragraf_standalone"),
        help="Output directory for the standalone package (default: build/paragraf_standalone)",
    )
    args = parser.parse_args()
    create_standalone_package(os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
