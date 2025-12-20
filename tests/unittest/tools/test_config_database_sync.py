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
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Dynamically load generate_config_table module without modifying sys.path
_spec = importlib.util.spec_from_file_location(
    "generate_config_table", REPO_ROOT / "scripts" / "generate_config_table.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
generate_rst = _module.generate_rst

# Dynamically load generate_config_database_tests module without modifying sys.path
_db_spec = importlib.util.spec_from_file_location(
    "generate_config_database_tests",
    REPO_ROOT / "scripts" / "generate_config_database_tests.py",
)
_db_module = importlib.util.module_from_spec(_db_spec)
_db_spec.loader.exec_module(_db_module)
generate_tests = _db_module.generate_tests
TEST_LIST_PATH = _db_module.TEST_LIST_PATH
PERF_SANITY_DIR = _db_module.PERF_SANITY_DIR


class TestConfigDatabaseSync(unittest.TestCase):
    def test_config_table_sync(self):
        """Test that the config_table.rst file is synchronized with the lookup.yaml database.

        Ensures that the RST file is up-to-date with the YAML database.
        """
        if generate_rst is None:
            self.skipTest("generate_config_table not available")

        # Define paths
        yaml_path = os.path.join(REPO_ROOT, "examples/configs/database/lookup.yaml")
        rst_path = os.path.join(REPO_ROOT, "docs/source/deployment-guide/config_table.rst")

        # Ensure files exist
        self.assertTrue(os.path.exists(yaml_path), f"YAML file not found: {yaml_path}")
        self.assertTrue(os.path.exists(rst_path), f"RST file not found: {rst_path}")

        # Read existing RST content
        with open(rst_path, "r") as f:
            existing_content = f.read()

        # Generate new RST content
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            generate_rst(yaml_path, output_file=tmp.name)
            tmp.seek(0)
            generated_content = tmp.read()

        # Compare content
        self.assertEqual(
            existing_content.strip(),
            generated_content.strip(),
            "config_table.rst is not synchronized with lookup.yaml. "
            "Please run 'python3 scripts/generate_config_table.py' from the repo root to update it.",
        )

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
