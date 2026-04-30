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

"""Create a standalone llmc package from the TensorRT-LLM source tree.

This script copies the ``tensorrt_llm/_torch/auto_deploy`` source tree and
tests into a standalone pip-installable package. The output directory can
be pushed directly to the read-only standalone repository.

Usage:
    python create_standalone_package.py [--output-dir /path/to/output]

The generated package uses ``llmc`` as the top-level Python package name and
``nvidia-llmc`` as the distribution name:

    from llmc._compat import TRTLLM_AVAILABLE
    from llmc.custom_ops.attention_interface import SequenceInfo

The ``auto_deploy`` source tree itself is copied verbatim — internal imports
must already be relative (enforced by the
``auto-deploy-import-discipline`` pre-commit hook), so no source rewriting
is required. Test files use absolute ``tensorrt_llm._torch.auto_deploy``
imports by design and ARE rewritten to ``llmc`` on copy.
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
LLMC_README = os.path.join(SCRIPT_DIR, "README.md")
LLMC_CONTRIBUTING = os.path.join(SCRIPT_DIR, "CONTRIBUTING.md")
LLMC_GITHUB_DIR = os.path.join(SCRIPT_DIR, ".github_for_llmc")

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

# Import path rewrite: old -> new (applied to test files only).
_IMPORT_REWRITE = "tensorrt_llm._torch.auto_deploy"
_IMPORT_TARGET = "llmc"

# Vendored projects with code actually present in auto_deploy source.
# Only these should appear in the standalone LICENSE file.
_VENDORED_PROJECTS = [
    {
        "name": "causal-conv1d",
        "url": "https://github.com/Dao-AILab/causal-conv1d",
        "copyright": "Copyright (c) 2024, Tri Dao.",
        "license_id": "BSD-3-Clause",
    },
    {
        "name": "flash-linear-attention",
        "url": "https://github.com/fla-org/flash-linear-attention",
        "copyright": "Copyright (c) 2023-2025 Songlin Yang",
        "license_id": "MIT",
    },
    {
        "name": "Mamba",
        "url": "https://github.com/state-spaces/mamba",
        "copyright": "Copyright 2023 Tri Dao, Albert Gu",
        "license_id": "Apache-2.0",
    },
    {
        "name": "SGLang",
        "url": "https://github.com/sgl-project/sglang",
        "copyright": "Copyright contributors to the SGLang project",
        "license_id": "Apache-2.0",
    },
    {
        "name": "vLLM",
        "url": "https://github.com/vllm-project/vllm",
        "copyright": "Copyright contributors to the vLLM project",
        "license_id": "Apache-2.0",
    },
    {
        "name": "Transformers",
        "url": "https://github.com/huggingface/transformers",
        "copyright": "Copyright 2018 The HuggingFace Team",
        "license_id": "Apache-2.0",
    },
]

# Direct dependency license mapping for ATTRIBUTIONS generation.
_DIRECT_DEP_LICENSES = {
    "torch": ("PyTorch", "BSD-3-Clause", "https://github.com/pytorch/pytorch"),
    "transformers": ("Transformers", "Apache-2.0", "https://github.com/huggingface/transformers"),
    "pydantic": ("Pydantic", "MIT", "https://github.com/pydantic/pydantic"),
    "pydantic-settings": (
        "pydantic-settings",
        "MIT",
        "https://github.com/pydantic/pydantic-settings",
    ),
    "triton": ("Triton", "MIT", "https://github.com/triton-lang/triton"),
    "flashinfer-python": (
        "FlashInfer",
        "Apache-2.0",
        "https://github.com/flashinfer-ai/flashinfer",
    ),
    "safetensors": ("safetensors", "Apache-2.0", "https://github.com/huggingface/safetensors"),
    "accelerate": ("Accelerate", "Apache-2.0", "https://github.com/huggingface/accelerate"),
    "huggingface-hub": (
        "huggingface-hub",
        "Apache-2.0",
        "https://github.com/huggingface/huggingface_hub",
    ),
    "omegaconf": ("OmegaConf", "BSD-3-Clause", "https://github.com/omry/omegaconf"),
    "pyyaml": ("PyYAML", "MIT", "https://github.com/yaml/pyyaml"),
    "numpy": ("NumPy", "BSD-3-Clause", "https://github.com/numpy/numpy"),
    "pillow": ("Pillow", "HPND", "https://github.com/python-pillow/Pillow"),
    "einops": ("einops", "MIT", "https://github.com/arogozhnikov/einops"),
    "six": ("six", "MIT", "https://github.com/benjaminp/six"),
    "importlib-metadata": (
        "importlib-metadata",
        "Apache-2.0",
        "https://github.com/python/importlib_metadata",
    ),
    "werkzeug": ("Werkzeug", "BSD-3-Clause", "https://github.com/pallets/werkzeug"),
    "StrEnum": ("StrEnum", "MIT", "https://github.com/irgeek/StrEnum"),
    "graphviz": ("graphviz", "MIT", "https://github.com/xflr6/graphviz"),
}

# Paths that the script owns and regenerates on every run.
# Everything else in the output directory (e.g., .git/, .github/) is preserved
# and owned by the standalone repo itself.
_MANAGED_PATHS = [
    "llmc",
    "tests",
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
    """Rewrite imports in a copied test file for standalone mode.

    Source files inside ``tensorrt_llm/_torch/auto_deploy`` already use
    relative imports (enforced by the ``auto-deploy-import-discipline``
    pre-commit hook), so no rewriting is needed for them. Tests, however,
    are written against the canonical absolute path
    ``tensorrt_llm._torch.auto_deploy`` and need to be rewritten to
    ``llmc``. Cross-package types (e.g. ``KvCacheConfig``,
    ``ActivationType``) are sourced via ``..._torch.auto_deploy._compat``,
    so the primary rewrite handles them too.

    Returns the number of line-level changes made.
    """
    with open(filepath) as f:
        content = f.read()

    original = content
    content = content.replace(_IMPORT_REWRITE, _IMPORT_TARGET)

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
# License and attribution generation
# ---------------------------------------------------------------------------

# Full license text templates keyed by SPDX identifier.
_LICENSE_TEXTS = {
    "Apache-2.0": textwrap.dedent("""\
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

        TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

        1. Definitions.

           "License" shall mean the terms and conditions for use, reproduction,
           and distribution as defined by Sections 1 through 9 of this document.

           "Licensor" shall mean the copyright owner or entity authorized by
           the copyright owner that is granting the License.

           "Legal Entity" shall mean the union of the acting entity and all
           other entities that control, are controlled by, or are under common
           control with that entity. For the purposes of this definition,
           "control" means (i) the power, direct or indirect, to cause the
           direction or management of such entity, whether by contract or
           otherwise, or (ii) ownership of fifty percent (50%) or more of the
           outstanding shares, or (iii) beneficial ownership of such entity.

           "You" (or "Your") shall mean an individual or Legal Entity
           exercising permissions granted by this License.

           "Source" form shall mean the preferred form for making modifications,
           including but not limited to software source code, documentation
           source, and configuration files.

           "Object" form shall mean any form resulting from mechanical
           transformation or translation of a Source form, including but
           not limited to compiled object code, generated documentation,
           and conversions to other media types.

           "Work" shall mean the work of authorship, whether in Source or
           Object form, made available under the License, as indicated by a
           copyright notice that is included in or attached to the work
           (an example is provided in the Appendix below).

           "Derivative Works" shall mean any work, whether in Source or Object
           form, that is based on (or derived from) the Work and for which the
           editorial revisions, annotations, elaborations, or other modifications
           represent, as a whole, an original work of authorship. For the purposes
           of this License, Derivative Works shall not include works that remain
           separable from, or merely link (or bind by name) to the interfaces of,
           the Work and Derivative Works thereof.

           "Contribution" shall mean any work of authorship, including
           the original version of the Work and any modifications or additions
           to that Work or Derivative Works thereof, that is intentionally
           submitted to Licensor for inclusion in the Work by the copyright owner
           or by an individual or Legal Entity authorized to submit on behalf of
           the copyright owner. For the purposes of this definition, "submitted"
           means any form of electronic, verbal, or written communication sent
           to the Licensor or its representatives, including but not limited to
           communication on electronic mailing lists, source code control systems,
           and issue tracking systems that are managed by, or on behalf of, the
           Licensor for the purpose of discussing and improving the Work, but
           excluding communication that is conspicuously marked or otherwise
           designated in writing by the copyright owner as "Not a Contribution."

           "Contributor" shall mean Licensor and any individual or Legal Entity
           on behalf of whom a Contribution has been received by Licensor and
           subsequently incorporated within the Work.

        2. Grant of Copyright License. Subject to the terms and conditions of
           this License, each Contributor hereby grants to You a perpetual,
           worldwide, non-exclusive, no-charge, royalty-free, irrevocable
           copyright license to reproduce, prepare Derivative Works of,
           publicly display, publicly perform, sublicense, and distribute the
           Work and such Derivative Works in Source or Object form.

        3. Grant of Patent License. Subject to the terms and conditions of
           this License, each Contributor hereby grants to You a perpetual,
           worldwide, non-exclusive, no-charge, royalty-free, irrevocable
           (except as stated in this section) patent license to make, have made,
           use, offer to sell, sell, import, and otherwise transfer the Work,
           where such license applies only to those patent claims licensable
           by such Contributor that are necessarily infringed by their
           Contribution(s) alone or by combination of their Contribution(s)
           with the Work to which such Contribution(s) was submitted. If You
           institute patent litigation against any entity (including a
           cross-claim or counterclaim in a lawsuit) alleging that the Work
           or a Contribution incorporated within the Work constitutes direct
           or contributory patent infringement, then any patent licenses
           granted to You under this License for that Work shall terminate
           as of the date such litigation is filed.

        4. Redistribution. You may reproduce and distribute copies of the
           Work or Derivative Works thereof in any medium, with or without
           modifications, and in Source or Object form, provided that You
           meet the following conditions:

           (a) You must give any other recipients of the Work or
               Derivative Works a copy of this License; and

           (b) You must cause any modified files to carry prominent notices
               stating that You changed the files; and

           (c) You must retain, in the Source form of any Derivative Works
               that You distribute, all copyright, patent, trademark, and
               attribution notices from the Source form of the Work,
               excluding those notices that do not pertain to any part of
               the Derivative Works; and

           (d) If the Work includes a "NOTICE" text file as part of its
               distribution, then any Derivative Works that You distribute must
               include a readable copy of the attribution notices contained
               within such NOTICE file, excluding those notices that do not
               pertain to any part of the Derivative Works, in at least one
               of the following places: within a NOTICE text file distributed
               as part of the Derivative Works; within the Source form or
               documentation, if provided along with the Derivative Works; or,
               within a display generated by the Derivative Works, if and
               wherever such third-party notices normally appear. The contents
               of the NOTICE file are for informational purposes only and
               do not modify the License. You may add Your own attribution
               notices within Derivative Works that You distribute, alongside
               or as an addendum to the NOTICE text from the Work, provided
               that such additional attribution notices cannot be construed
               as modifying the License.

           You may add Your own copyright statement to Your modifications and
           may provide additional or different license terms and conditions
           for use, reproduction, or distribution of Your modifications, or
           for any such Derivative Works as a whole, provided Your use,
           reproduction, and distribution of the Work otherwise complies with
           the conditions stated in this License.

        5. Submission of Contributions. Unless You explicitly state otherwise,
           any Contribution intentionally submitted for inclusion in the Work
           by You to the Licensor shall be under the terms and conditions of
           this License, without any additional terms or conditions.
           Notwithstanding the above, nothing herein shall supersede or modify
           the terms of any separate license agreement you may have executed
           with Licensor regarding such Contributions.

        6. Trademarks. This License does not grant permission to use the trade
           names, trademarks, service marks, or product names of the Licensor,
           except as required for reasonable and customary use in describing the
           origin of the Work and reproducing the content of the NOTICE file.

        7. Disclaimer of Warranty. Unless required by applicable law or
           agreed to in writing, Licensor provides the Work (and each
           Contributor provides its Contributions) on an "AS IS" BASIS,
           WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
           implied, including, without limitation, any warranties or conditions
           of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
           PARTICULAR PURPOSE. You are solely responsible for determining the
           appropriateness of using or redistributing the Work and assume any
           risks associated with Your exercise of permissions under this License.

        8. Limitation of Liability. In no event and under no legal theory,
           whether in tort (including negligence), contract, or otherwise,
           unless required by applicable law (such as deliberate and grossly
           negligent acts) or agreed to in writing, shall any Contributor be
           liable to You for damages, including any direct, indirect, special,
           incidental, or consequential damages of any character arising as a
           result of this License or out of the use or inability to use the
           Work (including but not limited to damages for loss of goodwill,
           work stoppage, computer failure or malfunction, or any and all
           other commercial damages or losses), even if such Contributor
           has been advised of the possibility of such damages.

        9. Accepting Warranty or Additional Liability. While redistributing
           the Work or Derivative Works thereof, You may choose to offer,
           and charge a fee for, acceptance of support, warranty, indemnity,
           or other liability obligations and/or rights consistent with this
           License. However, in accepting such obligations, You may act only
           on Your own behalf and on Your sole responsibility, not on behalf
           of any other Contributor, and only if You agree to indemnify,
           defend, and hold each Contributor harmless for any liability
           incurred by, or claims asserted against, such Contributor by reason
           of your accepting any such warranty or additional liability.

        END OF TERMS AND CONDITIONS
    """),
    "MIT": textwrap.dedent("""\
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """),
    "BSD-3-Clause": textwrap.dedent("""\
        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """),
}


def _generate_license(output_dir: str) -> None:
    """Generate a LICENSE file listing only vendored projects present in auto_deploy."""
    lines = []
    lines.append("Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n")
    lines.append(
        "This project is licensed under the Apache 2.0 license, whose full license text "
        "is available below.\n"
    )
    lines.append(
        "This project contains portions of code that are based on or derived from\n"
        "other open source projects, which may have different licenses whose text\n"
        "is available below.\n"
    )
    lines.append(
        "All modifications and additions to other projects are licensed under the\n"
        "Apache License 2.0 unless otherwise specified. Please refer to the individual\n"
        "file headers for specific copyright and license information.\n"
    )
    lines.append(
        "Below is a list of other projects that have portions contained by this project:\n"
    )

    for proj in _VENDORED_PROJECTS:
        lines.append("-" * 80)
        lines.append(proj["name"])
        lines.append("-" * 80)
        lines.append(f"Original Source: {proj['url']}")
        lines.append(proj["copyright"])
        lines.append(f"Licensed under the {proj['license_id']} License")
        lines.append("")

    # Append full license texts (deduplicated)
    seen = set()
    for proj in _VENDORED_PROJECTS:
        lid = proj["license_id"]
        if lid not in seen and lid in _LICENSE_TEXTS:
            seen.add(lid)
            lines.append("=" * 80)
            lines.append(f"                              {lid} LICENSE")
            lines.append("=" * 80)
            lines.append("")
            lines.append(_LICENSE_TEXTS[lid])

    with open(os.path.join(output_dir, "LICENSE"), "w") as f:
        f.write("\n".join(lines))


def _generate_attributions(output_dir: str, dependencies: list) -> None:
    """Generate ATTRIBUTIONS-Python.md listing direct dependency licenses."""
    lines = []
    lines.append("<!--")
    lines.append(
        "SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved."
    )
    lines.append("SPDX-License-Identifier: Apache-2.0")
    lines.append("-->")
    lines.append("")
    lines.append("# Third-Party Software Attributions")
    lines.append("")
    lines.append(
        "This file lists the direct runtime dependencies of this package and their licenses."
    )
    lines.append(
        "For transitive dependencies, consult `uv.lock` or run `pip-licenses` after installation."
    )
    lines.append("")

    for dep_str in sorted(dependencies, key=lambda d: d.split("[")[0].lower()):
        # Extract base name (strip version spec)
        base = re.match(r"^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)", dep_str)
        if not base:
            continue
        pkg_name = base.group(1).split("[")[0]
        info = _DIRECT_DEP_LICENSES.get(pkg_name) or _DIRECT_DEP_LICENSES.get(pkg_name.lower())
        if not info:
            lines.append(f"## {dep_str}")
            lines.append("")
            lines.append("License: Unknown")
            lines.append("")
            continue
        display_name, license_id, url = info
        lines.append(f"## {display_name}")
        lines.append("")
        lines.append(f"- **PyPI package**: `{dep_str}`")
        lines.append(f"- **License**: {license_id}")
        lines.append(f"- **Source**: {url}")
        lines.append("")

    with open(os.path.join(output_dir, "ATTRIBUTIONS-Python.md"), "w") as f:
        f.write("\n".join(lines))


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
        'description = "llmc: standalone LLM compiler — '
        'automatic model optimization and deployment for LLM inference"\n'
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
        'include = ["llmc*"]\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
    )

    with open(os.path.join(output_dir, "pyproject.toml"), "w") as f:
        f.write(content)


def create_standalone_package(output_dir: str) -> None:
    """Create the standalone llmc package at the given output directory.

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

    # 1. Copy auto_deploy source as top-level `llmc/` package. No import
    #    rewriting is needed: in-package imports are relative (enforced by
    #    the auto-deploy-import-discipline pre-commit hook).
    ad_dst = os.path.join(output_dir, "llmc")
    count = _copy_tree(AUTO_DEPLOY_SRC, ad_dst)
    print(f"  Copied {count} source files to llmc/")

    # 2. Copy and rewrite tests (tests use absolute self-imports by design).
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

    # 4. Generate standalone LICENSE (only vendored projects in auto_deploy)
    _generate_license(output_dir)
    print(f"  Generated LICENSE ({len(_VENDORED_PROJECTS)} vendored projects)")

    # 5. Generate ATTRIBUTIONS-Python.md (direct dependency licenses)
    _generate_attributions(output_dir, dependencies)
    print(f"  Generated ATTRIBUTIONS-Python.md ({len(dependencies)} direct deps)")

    # 6. Copy README
    if os.path.exists(LLMC_README):
        shutil.copy2(LLMC_README, os.path.join(output_dir, "README.md"))
        print("  Copied README.md")

    # 7. Copy CONTRIBUTING.md
    if os.path.exists(LLMC_CONTRIBUTING):
        shutil.copy2(LLMC_CONTRIBUTING, os.path.join(output_dir, "CONTRIBUTING.md"))
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

    # 11. Copy .github/ tree (issue/PR templates) from .github_for_llmc/
    if os.path.isdir(LLMC_GITHUB_DIR):
        github_dst = os.path.join(output_dir, ".github")
        github_count = _copy_tree(LLMC_GITHUB_DIR, github_dst)
        print(f"  Copied {github_count} .github/ files")

    print(f"\nStandalone package created at: {output_dir}")
    print("\nTo install:")
    print(f"  cd {output_dir}")
    print("  uv venv .venv --python 3.12")
    print("  source .venv/bin/activate")
    print("  uv pip install -e '.[dev]'")
    print("\nTo run tests:     pytest tests/")
    print(
        'To verify:        python -c "from llmc._compat import TRTLLM_AVAILABLE; print(TRTLLM_AVAILABLE)"'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a standalone llmc package from TensorRT-LLM source.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "build", "llmc_standalone"),
        help="Output directory for the standalone package (default: build/llmc_standalone)",
    )
    args = parser.parse_args()
    create_standalone_package(os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
