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

"""Production-artifact checks for the Qwen-Image-2.1 VisualGen example.

These tests intentionally avoid importing the runtime or downloading model
weights.  They keep the example script, checked-in config, documentation, and
VisualGen test-db wiring discoverable as durable production artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = REPO_ROOT / "examples" / "visual_gen" / "models" / "qwen_image_21.py"
CONFIG = REPO_ROOT / "examples" / "visual_gen" / "configs" / "qwen-image-2.1-bf16-1gpu.yaml"
DOC = REPO_ROOT / "docs" / "source" / "examples" / "visual_gen" / "qwen-image-2.1.md"
PERF_SANITY = REPO_ROOT / "tests" / "scripts" / "perf-sanity" / "visual_gen" / "qwen_image_21_blackwell.yaml"
README = REPO_ROOT / "examples" / "visual_gen" / "README.md"


PROHIBITED_DIFFUSERS_RUNTIME_IMPORTS = {
    "QwenImagePipeline",
    "QwenImageTransformer2DModel",
    "AutoencoderKLQwenImage",
    "QwenImage21Pipeline",
}


def test_qwen_image_21_production_files_are_checked_in() -> None:
    for path in (EXAMPLE, CONFIG, DOC, PERF_SANITY, README):
        assert path.is_file(), f"missing production artifact: {path.relative_to(REPO_ROOT)}"


def test_qwen_image_21_example_does_not_import_upstream_pipeline_or_transformer() -> None:
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"), filename=str(EXAMPLE))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("diffusers"):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name.split(".")[0] for alias in node.names if alias.name.startswith("diffusers"))
    assert imported_names.isdisjoint(PROHIBITED_DIFFUSERS_RUNTIME_IMPORTS), imported_names


def test_qwen_image_21_config_records_production_defaults_without_unsupported_keys() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "qwen-image-2.1" in text
    assert "Qwen/Qwen-Image-2.1" in text
    assert "default_num_inference_steps: 30" in text
    assert "container_cache_preflight.json" in text
    assert "VisualGenArgs-supported runtime fields" in text
    assert "attention_config:" in text
    assert "parallel_config:" in text
    # Keep checkpoint/download policy out of parser-facing VisualGenArgs YAML.
    assert "checkpoint_policy:" not in text
    assert "allow_download:" not in text


def test_qwen_image_21_docs_link_example_config_and_validation_artifacts() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "qwen-image-2.1-bf16-1gpu.yaml" in text
    assert "production_readiness_report.json" in text
    assert "visual_gen_lpips_score_eval.py" in text
    assert "Diffusers" in text


def test_qwen_image_21_perf_sanity_wiring_is_model_specific() -> None:
    text = PERF_SANITY.read_text(encoding="utf-8")
    lower_text = text.lower()
    assert "qwen-image-2.1" in lower_text
    assert "qwen/qwen-image-2.1" in lower_text
    assert "qwen-image-2.1-bf16-1gpu.yaml" in text
    assert "qwen_image_21" in PERF_SANITY.name
