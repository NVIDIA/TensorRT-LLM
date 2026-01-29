# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for vendored triton_kernels package.

The triton_kernels package is vendored from the Triton project to provide
optimized kernels. These tests verify that the vendoring mechanism works
correctly and that our version takes precedence over any external installation.
"""

import unittest
from pathlib import Path


class TestTritonKernelsVendoring(unittest.TestCase):
    def test_triton_kernels_is_vendored_version(self):
        """Verify we're using the vendored version (has VERSION and LICENSE files)."""
        import tensorrt_llm  # noqa: F401, I001
        import triton_kernels

        triton_kernels_path = Path(triton_kernels.__file__).parent

        # VERSION file is added by our vendor script and not present in external installations
        version_file = triton_kernels_path / "VERSION"
        self.assertTrue(
            version_file.exists(),
            f"VERSION file not found at {version_file}. "
            "This suggests an external triton_kernels is being used instead of our vendored version.",
        )

        # LICENSE file should also be present for compliance
        license_file = triton_kernels_path / "LICENSE"
        self.assertTrue(license_file.exists(), f"LICENSE file not found at {license_file}.")

    def test_version_matches_requirements(self):
        """Verify vendored triton_kernels VERSION matches triton version in requirements.txt."""
        import re

        repo_root = Path(__file__).parent.parent.parent.parent

        version_file = repo_root / "triton_kernels" / "VERSION"
        vendored_version = version_file.read_text().strip().split()[0].lstrip("v")

        requirements_file = repo_root / "requirements.txt"
        requirements_text = requirements_file.read_text()

        match = re.search(r"^triton==([^\s#]+)", requirements_text, re.MULTILINE)
        self.assertIsNotNone(match, "Could not find triton version in requirements.txt")
        requirements_version = match.group(1)

        self.assertEqual(
            vendored_version,
            requirements_version,
            f"Vendored triton_kernels version ({vendored_version}) does not match "
            f"triton version in requirements.txt ({requirements_version}). "
            "To update the vendored triton_kernels, run: python scripts/vendor_triton_kernels.py "
            f"--tag v{requirements_version}",
        )


if __name__ == "__main__":
    unittest.main()
