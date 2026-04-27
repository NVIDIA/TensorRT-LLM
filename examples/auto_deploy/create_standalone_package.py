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

"""Create a standalone auto_deploy package from the TensorRT-LLM source tree.

This script copies auto_deploy source files and tests into a standalone
pip-installable package. The output directory can be pushed directly to the
standalone repository.

Usage:
    python create_standalone_package.py [--output-dir /path/to/output]

The generated package uses ``auto_deploy`` as the top-level package name:

    from auto_deploy._compat import TRTLLM_AVAILABLE
    from auto_deploy.custom_ops.attention_interface import SequenceInfo
"""

import argparse
import os
import re
import shutil
import sys
import textwrap

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
AUTO_DEPLOY_SRC = os.path.join(REPO_ROOT, "tensorrt_llm", "_torch", "auto_deploy")
TRTLLM_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements.txt")
TRTLLM_DEV_REQUIREMENTS = os.path.join(REPO_ROOT, "requirements-dev.txt")
TRTLLM_LICENSE = os.path.join(REPO_ROOT, "LICENSE")
TRTLLM_GITIGNORE = os.path.join(REPO_ROOT, ".gitignore")
AD_README = os.path.join(SCRIPT_DIR, "README.md")
AD_CONTRIBUTING = os.path.join(SCRIPT_DIR, "CONTRIBUTING.md")

# Test source directories
AD_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "unittest", "auto_deploy")
AD_UTILS_TEST_DIR = os.path.join(AD_TESTS_DIR, "_utils_test")
AD_TORCH_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "unittest", "_torch", "auto_deploy")

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
    "parameterized",
]

# Test directories to exclude from the standalone package (require TRT-LLM runtime)
EXCLUDE_TEST_DIRS = {"smoke", "shim", "standalone"}

# Individual test files to exclude (require TRT-LLM runtime/kernels or external scripts)
EXCLUDE_TEST_FILES = {
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
    # Require TRT-LLM CUDA causal conv / mamba kernels (ops not registered standalone)
    "test_cuda_causal_conv_cached_op.py",
    "test_triton_causal_conv_cached_op.py",
    "test_triton_mamba_cached_op.py",
    "test_flashinfer_mamba_cached_op.py",
    # Require TRT-LLM custom ops (dsv3_router_gemm_op, noaux_tc_op, etc.)
    "test_deepseek_custom.py",
    "test_glm4_moe_lite_modeling.py",
    # Require TRT-LLM distributed ops (trtllm_dist_all_gather)
    "test_gather_logits_before_lm_head.py",
    # Multimodal types are None in standalone (MultimodalInput guard)
    "test_qwen3_5_moe.py",
    # Hardware-specific (requires H100+ shared memory)
    "test_triton_mla_op.py",
    # Require TRT-LLM ops (noaux_tc_op) — split from test_export.py
    "test_export_glm4_moe_lite.py",
}

# Import path rewrite: old -> new
_IMPORT_REWRITE = "tensorrt_llm._torch.auto_deploy"
_IMPORT_TARGET = "auto_deploy"

# Paths that the script owns and regenerates on every run.
# Everything else in the output directory (e.g., .git/, .github/) is preserved.
_MANAGED_PATHS = [
    "auto_deploy",
    "tests",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    ".gitignore",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _should_copy(filepath: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return False
    _, ext = os.path.splitext(filepath)
    return ext in COPY_EXTENSIONS


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


def _rewrite_imports_in_file(filepath: str) -> int:
    """Rewrite imports in a copied file for standalone mode.

    Handles:
    - tensorrt_llm._torch.auto_deploy.X -> auto_deploy.X
    - from tensorrt_llm.llmapi.llm_args import KvCacheConfig -> from auto_deploy._compat import KvCacheConfig
    - from tensorrt_llm._torch.utils import ActivationType -> from auto_deploy._compat import ActivationType

    Returns the number of replacements made.
    """
    with open(filepath) as f:
        content = f.read()

    original = content
    # Primary rewrite: tensorrt_llm._torch.auto_deploy -> auto_deploy
    content = content.replace(_IMPORT_REWRITE, _IMPORT_TARGET)
    # Test files that import KvCacheConfig from tensorrt_llm.llmapi
    content = content.replace(
        "from tensorrt_llm.llmapi.llm_args import KvCacheConfig",
        "from auto_deploy._compat import KvCacheConfig",
    )
    # Test files that import ActivationType from tensorrt_llm._torch.utils
    content = content.replace(
        "from tensorrt_llm._torch.utils import ActivationType",
        "from auto_deploy._compat import ActivationType",
    )

    replacements = sum(1 for a, b in zip(original, content) if a != b)  # rough count
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        # Count actual line-level changes
        replacements = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)

    return replacements


def _rewrite_imports_in_dir(directory: str) -> int:
    """Rewrite imports in all .py files in a directory tree."""
    total = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                total += _rewrite_imports_in_file(os.path.join(root, filename))
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
def _should_exclude_test(filepath: str) -> bool:
    """Check if a test file should be excluded from the standalone package."""
    basename = os.path.basename(filepath)
    if basename in EXCLUDE_TEST_FILES:
        return True
    parts = filepath.replace("\\", "/").split("/")
    return any(d in EXCLUDE_TEST_DIRS for d in parts)


def _copy_tests(output_dir: str) -> int:
    """Copy auto_deploy test files to the standalone package tests/ directory."""
    tests_dst = os.path.join(output_dir, "tests")
    count = 0

    # Copy singlegpu tests (excluding TRT-LLM-only dirs/files)
    singlegpu_src = os.path.join(AD_TESTS_DIR, "singlegpu")
    if os.path.isdir(singlegpu_src):
        for root, dirs, files in os.walk(singlegpu_src):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_TEST_DIRS and d != "__pycache__"]

            for filename in files:
                src_path = os.path.join(root, filename)
                if not _should_copy(src_path) or _should_exclude_test(src_path):
                    continue
                rel_path = os.path.relpath(src_path, AD_TESTS_DIR)
                dst_path = os.path.join(tests_dst, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                count += 1

    # Copy test utilities
    if os.path.isdir(AD_UTILS_TEST_DIR):
        utils_dst = os.path.join(tests_dst, "_utils_test")
        for filename in os.listdir(AD_UTILS_TEST_DIR):
            src_path = os.path.join(AD_UTILS_TEST_DIR, filename)
            if os.path.isfile(src_path) and _should_copy(src_path):
                dst_path = os.path.join(utils_dst, filename)
                os.makedirs(utils_dst, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                count += 1

    # Create conftest.py for test discovery and imports
    _create_test_conftest(tests_dst)

    # Create a stub for test_common.llm_data (used by some model tests)
    _create_test_common_stub(tests_dst)

    return count


def _create_test_conftest(tests_dir: str) -> None:
    """Create a conftest.py that configures the test environment for standalone mode."""
    content = textwrap.dedent("""\
        \"\"\"Conftest for standalone auto_deploy tests.\"\"\"
        import sys
        import os

        # Add _utils_test to the Python path so test files can import from it
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_utils_test"))

        # Add the tests directory itself to the path for cross-test imports
        sys.path.insert(0, os.path.dirname(__file__))
    """)
    with open(os.path.join(tests_dir, "conftest.py"), "w") as f:
        f.write(content)


def _create_test_common_stub(tests_dir: str) -> None:
    """Create a stub for test_common.llm_data (provides HF model path resolution).

    In standalone mode, tests that need local model weights will be skipped
    unless LLM_MODELS_ROOT is set.
    """
    stub_dir = os.path.join(tests_dir, "test_common")
    os.makedirs(stub_dir, exist_ok=True)

    with open(os.path.join(stub_dir, "__init__.py"), "w") as f:
        f.write("")

    content = textwrap.dedent("""\
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
        'name = "llm-compiler"\n'
        'version = "0.1.0"\n'
        'description = "LLM Compiler (llmc): Automatic model optimization and deployment for LLM inference"\n'
        'readme = "README.md"\n'
        'license = {text = "Apache-2.0"}\n'
        'requires-python = ">=3.10"\n'
        "dependencies = [\n"
        f"{deps_lines}\n"
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        "dev = [\n"
        f"{dev_deps_lines}\n"
        "]\n"
        "\n"
        "[tool.setuptools.packages.find]\n"
        'include = ["auto_deploy*"]\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
    )

    with open(os.path.join(output_dir, "pyproject.toml"), "w") as f:
        f.write(content)


def create_standalone_package(output_dir: str) -> None:
    """Create the standalone auto_deploy package at the given output directory.

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
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.isfile(target):
            os.remove(target)

    print(f"Creating standalone package at: {output_dir}")

    # 1. Copy auto_deploy source as top-level `auto_deploy/` package
    ad_dst = os.path.join(output_dir, "auto_deploy")
    count = _copy_tree(AUTO_DEPLOY_SRC, ad_dst)
    print(f"  Copied {count} source files to auto_deploy/")

    # 2. Copy and rewrite tests
    test_count = _copy_tests(output_dir)
    rewrite_count = _rewrite_imports_in_dir(os.path.join(output_dir, "tests"))
    print(f"  Copied {test_count} test files to tests/ ({rewrite_count} import rewrites)")

    # 3. Resolve dependencies and create pyproject.toml
    pinned = _read_pinned_versions(TRTLLM_REQUIREMENTS)
    dev_pinned = _read_pinned_versions(TRTLLM_DEV_REQUIREMENTS)
    # Merge: dev_pinned has the same packages as pinned plus test-only packages
    all_pinned = {**pinned, **dev_pinned}
    dependencies = _resolve_dependencies(STANDALONE_DEPS, pinned)
    dev_dependencies = _resolve_dependencies(DEV_DEPS, all_pinned)
    _create_pyproject_toml(output_dir, dependencies, dev_dependencies)
    print(f"  Created pyproject.toml ({len(dependencies)} deps + {len(dev_dependencies)} dev deps)")

    # 4. Copy LICENSE
    if os.path.exists(TRTLLM_LICENSE):
        shutil.copy2(TRTLLM_LICENSE, os.path.join(output_dir, "LICENSE"))
        print("  Copied LICENSE")

    # 5. Copy README
    if os.path.exists(AD_README):
        shutil.copy2(AD_README, os.path.join(output_dir, "README.md"))
        print("  Copied README.md")

    # 6. Copy CONTRIBUTING.md
    if os.path.exists(AD_CONTRIBUTING):
        shutil.copy2(AD_CONTRIBUTING, os.path.join(output_dir, "CONTRIBUTING.md"))
        print("  Copied CONTRIBUTING.md")

    # 7. Copy .gitignore
    if os.path.exists(TRTLLM_GITIGNORE):
        shutil.copy2(TRTLLM_GITIGNORE, os.path.join(output_dir, ".gitignore"))
        print("  Copied .gitignore")

    print(f"\nStandalone package created at: {output_dir}")
    print("\nTo install:")
    print(f"  cd {output_dir}")
    print("  uv venv .venv --python 3.12")
    print("  source .venv/bin/activate")
    print("  uv pip install -e '.[dev]'")
    print("\nTo run tests:     pytest tests/")
    print(
        'To verify:        python -c "from auto_deploy._compat import TRTLLM_AVAILABLE; print(TRTLLM_AVAILABLE)"'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a standalone auto_deploy package from TensorRT-LLM source.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "build", "auto_deploy_standalone"),
        help="Output directory for the standalone package (default: build/auto_deploy_standalone)",
    )
    args = parser.parse_args()
    create_standalone_package(os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
