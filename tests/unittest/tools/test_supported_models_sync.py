#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import sys
import unittest
from pathlib import Path


def _render_supported_models_markdown(repo_root: Path) -> str:
    module_path = repo_root / "tensorrt_llm/llmapi/model_support_matrix.py"
    spec = importlib.util.spec_from_file_location("tllm_model_support_matrix", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)

    # Needed for dataclasses/type evaluation during module exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.render_supported_models_markdown()


class TestSupportedModelsSync(unittest.TestCase):
    def test_supported_models_md_sync(self):
        """Ensure supported-models.md is synchronized with the generator."""
        repo_root = Path(__file__).resolve().parents[3]
        md_path = repo_root / "docs/source/models/supported-models.md"
        self.assertTrue(md_path.exists(), f"Markdown file not found: {md_path}")

        existing_content = md_path.read_text()
        generated_content = _render_supported_models_markdown(repo_root)

        self.assertEqual(
            existing_content.strip(),
            generated_content.strip(),
            "docs/source/models/supported-models.md is not synchronized with the programmatic support matrix.\n"
            "Please regenerate it (e.g. build docs, or run the generator entrypoint in docs/source/helper.py).",
        )


if __name__ == "__main__":
    unittest.main()
