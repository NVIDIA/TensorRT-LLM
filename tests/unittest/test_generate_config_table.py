import os
import sys
import tempfile
import unittest

# Add scripts directory to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# Try to import generate_rst, fail gracefully if dependencies (like yaml) are missing
from generate_config_table import generate_rst  # noqa: E402


class TestConfigTableSync(unittest.TestCase):
    def test_config_table_sync(self):
        """Test that the comprehensive_table.rst file is synchronized with the scenario_list.yaml database.

        Ensures that the RST file is up-to-date with the YAML database.
        """
        if generate_rst is None:
            self.skipTest("generate_config_table not available")

        # Define paths
        yaml_path = os.path.join(REPO_ROOT, "tensorrt_llm/configure/database/scenario_list.yaml")
        rst_path = os.path.join(REPO_ROOT, "docs/source/deployment-guide/comprehensive_table.rst")

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
            "comprehensive_table.rst is not synchronized with scenario_list.yaml. "
            "Please run 'python3 scripts/generate_config_table.py' from the repo root to update it.",
        )


if __name__ == "__main__":
    unittest.main()
