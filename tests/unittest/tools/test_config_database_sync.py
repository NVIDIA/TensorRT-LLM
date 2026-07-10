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

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
EXPECTED_MODEL_METADATA = {
    "deepseek-ai/DeepSeek-R1-0528": {
        "display_name": "DeepSeek-R1",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
    },
    "nvidia/DeepSeek-R1-0528-FP4-v2": {
        "display_name": "DeepSeek-R1 (NVFP4)",
        "url": "https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2",
    },
    "openai/gpt-oss-120b": {
        "display_name": "gpt-oss-120b",
        "url": "https://huggingface.co/openai/gpt-oss-120b",
    },
}

# Dynamically load generate_config_table module without modifying sys.path
_spec = importlib.util.spec_from_file_location(
    "generate_config_table", REPO_ROOT / "scripts" / "generate_config_table.py"
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
generate_json = _module.generate_json
RecipeList = _module.RecipeList

# Dynamically load generate_config_database_tests module without modifying sys.path
_db_spec = importlib.util.spec_from_file_location(
    "generate_config_database_tests",
    REPO_ROOT / "scripts" / "generate_config_database_tests.py",
)
_db_module = importlib.util.module_from_spec(_db_spec)
sys.modules[_db_spec.name] = _db_module
_db_spec.loader.exec_module(_db_module)
generate_tests = _db_module.generate_tests
generate_server_name = _db_module.generate_server_name
TEST_LIST_PATH = _db_module.TEST_LIST_PATH
PERF_SANITY_DIR = _db_module.PERF_SANITY_DIR


class TestConfigDatabaseSync(unittest.TestCase):
    def test_config_db_json_generation(self):
        """Test that config_db.json generation matches lookup.yaml entries.

        Validates the generated JSON payload shape and that its entries correspond 1:1
        with lookup.yaml recipes, without relying on a committed JSON artifact.
        """
        self.assertIsNotNone(generate_json)
        self.assertIsNotNone(RecipeList)

        yaml_path = os.path.join(REPO_ROOT, "examples/configs/database/lookup.yaml")
        self.assertTrue(os.path.exists(yaml_path), f"YAML file not found: {yaml_path}")

        recipes = RecipeList.from_yaml(Path(yaml_path))
        expected_keys = {
            (
                r.model,
                r.gpu,
                int(r.num_gpus),
                int(r.isl),
                int(r.osl),
                int(r.concurrency),
                r.config_path,
                r.profile,
                r.validated_trtllm_commit,
                r.validated_trtllm_version,
            )
            for r in recipes
        }

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tmp:
            generate_json(yaml_path, output_file=tmp.name)
            tmp.seek(0)
            payload = json.load(tmp)

        self.assertEqual(
            payload.get("source"),
            "examples/configs/database/lookup.yaml",
            "Generated JSON 'source' field is unexpected.",
        )

        expected_models = {}
        for recipe in recipes:
            expected_models[recipe.model] = EXPECTED_MODEL_METADATA.get(
                recipe.model,
                {
                    "display_name": recipe.model,
                    "url": "",
                },
            )

        self.assertEqual(
            payload.get("models"),
            expected_models,
            "Generated JSON 'models' field is unexpected.",
        )

        entries = payload.get("entries") or []
        for e in entries:
            key = (
                e.get("model"),
                e.get("gpu"),
                int(e.get("num_gpus")),
                int(e.get("isl")),
                int(e.get("osl")),
                int(e.get("concurrency")),
                e.get("config_path"),
                e.get("profile"),
                e.get("validated_trtllm_commit"),
                e.get("validated_trtllm_version"),
            )
            self.assertIn(
                key,
                expected_keys,
                f"Generated config_db.json contains an unexpected entry key: {key}",
            )
            expected_keys.remove(key)

            cmd = e.get("command") or ""
            self.assertIn("--config", cmd)
            self.assertNotIn("extra_llm_api_options", cmd)
            self.assertIn("${TRTLLM_DIR}/", cmd)

            config_path = e.get("config_path") or ""
            self.assertTrue(config_path)

            gh = e.get("config_github_url") or ""
            raw = e.get("config_raw_url") or ""
            self.assertTrue(gh.endswith(config_path))
            self.assertTrue(raw.endswith(config_path))

        self.assertFalse(
            expected_keys,
            "Generated config_db.json is missing entries from lookup.yaml.",
        )

    def test_profile_and_validated_commit_metadata(self) -> None:
        commit = "93CB6518B6D6DBD6095748189E626DB731F44545"
        version = "1.3.0rc14"
        recipes = []
        for profile in ("latency", "balanced", "throughput"):
            recipes.append(
                {
                    "model": "example/model",
                    "arch": "ExampleForCausalLM",
                    "gpu": "B200_NVL",
                    "num_gpus": 8,
                    "isl": 1024,
                    "osl": 1024,
                    "concurrency": 256,
                    "config_path": f"examples/configs/database/example_{profile}.yaml",
                    "profile": profile,
                    "validated_trtllm_commit": commit,
                    "validated_trtllm_version": version,
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "lookup.yaml"
            output_path = Path(tmp_dir) / "config_db.json"
            yaml_path.write_text(yaml.safe_dump(recipes), encoding="utf-8")
            generate_json(yaml_path, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [entry["profile"] for entry in payload["entries"]],
            ["latency", "balanced", "throughput"],
        )
        self.assertEqual(
            {entry["performance_profile"] for entry in payload["entries"]},
            {"Min Latency", "Balanced", "Max Throughput"},
        )
        self.assertTrue(
            all(entry["validated_trtllm_commit"] == commit.lower() for entry in payload["entries"])
        )
        self.assertTrue(
            all(entry["validated_trtllm_version"] == version for entry in payload["entries"])
        )

        parsed = RecipeList.model_validate(recipes)
        server_names = [generate_server_name(recipe) for recipe in parsed]
        self.assertEqual(len(server_names), len(set(server_names)))
        self.assertTrue(server_names[0].endswith("_latency"))

    def test_recipe_metadata_validation(self) -> None:
        base = {
            "model": "example/model",
            "arch": "ExampleForCausalLM",
            "gpu": "B200_NVL",
            "num_gpus": 8,
            "isl": 1024,
            "osl": 1024,
            "concurrency": 256,
            "config_path": "examples/configs/database/example.yaml",
        }

        with self.assertRaisesRegex(ValueError, "full 40-character Git SHA"):
            RecipeList.model_validate(
                [
                    {
                        **base,
                        "validated_trtllm_commit": "deadbeef",
                        "validated_trtllm_version": "1.3.0rc14",
                    }
                ]
            )

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            RecipeList.model_validate([{**base, "validated_trtllm_commit": "a" * 40}])

        with self.assertRaisesRegex(ValueError, "profile is only allowed"):
            RecipeList.model_validate([{**base, "profile": "balanced"}])

        incomplete_conflict = [
            {**base, "config_path": "latency.yaml", "profile": "latency"},
            {**base, "config_path": "throughput.yaml", "profile": "throughput"},
        ]
        with self.assertRaisesRegex(ValueError, "exactly one latency, balanced, and throughput"):
            RecipeList.model_validate(incomplete_conflict)

    @pytest.mark.skip(reason="https://nvbugs/6337224")
    def test_config_database_tests_sync(self):
        """Test that config database test files are synchronized with lookup.yaml.

        Ensures that both the test list YAML and per-GPU config files are up-to-date.
        """
        self.assertTrue(TEST_LIST_PATH.exists(), f"Test list not found: {TEST_LIST_PATH}")

        with open(TEST_LIST_PATH) as f:
            existing_test_list = f.read()

        existing_config_files = {}
        for config_path in PERF_SANITY_DIR.glob("config_database_*.yaml"):
            with open(config_path) as f:
                existing_config_files[config_path.name] = f.read()

        # Generate to temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_config_dir = Path(tmp_dir) / "configs"
            tmp_test_list_path = Path(tmp_dir) / "test_list.yml"
            tmp_config_dir.mkdir(parents=True, exist_ok=True)

            generate_tests(tmp_test_list_path, tmp_config_dir)

            with open(tmp_test_list_path) as f:
                generated_test_list = f.read()

            self.assertEqual(
                existing_test_list.strip(),
                generated_test_list.strip(),
                f"{TEST_LIST_PATH} is not synchronized with lookup.yaml. "
                "Please run 'python3 scripts/generate_config_database_tests.py' from the repo root.",
            )

            generated_config_files = {}
            for config_path in tmp_config_dir.glob("config_database_*.yaml"):
                with open(config_path) as f:
                    generated_config_files[config_path.name] = f.read()

            # Check same set of files
            self.assertEqual(
                set(existing_config_files.keys()),
                set(generated_config_files.keys()),
                "Mismatch in config database config files. "
                "Please run 'python scripts/generate_config_database_tests.py' from the repo root.",
            )

            # Compare each config file
            for filename in existing_config_files:
                self.assertEqual(
                    existing_config_files[filename].strip(),
                    generated_config_files[filename].strip(),
                    f"{filename} is not synchronized with lookup.yaml. "
                    "Please run 'python scripts/generate_config_database_tests.py' from the repo root.",
                )


if __name__ == "__main__":
    unittest.main()
