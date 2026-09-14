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
"""Tests for vendored triton_kernels package.

The triton_kernels package is vendored from the Triton project to provide
optimized kernels. These tests verify that the vendoring mechanism works
correctly and that our version takes precedence over any external installation.
"""

import re
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

# The first specifier is the lower bound for both `triton==X` and `triton>=X,<=Y`.
# Anchored on `triton` alone so that `triton-kernels==` and `tritonclient==` do not match.
_TRITON_REQUIREMENT_RE = re.compile(r"^triton[ \t]*[<>~!=]*=[ \t]*([^\s,#]+)", re.MULTILINE)


def _triton_lower_bound(requirements_text: str) -> str | None:
    """Return the triton lower bound declared in a requirements.txt body."""
    match = _TRITON_REQUIREMENT_RE.search(requirements_text)
    return match.group(1) if match else None


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
        """Verify vendored triton_kernels VERSION matches the triton lower bound in requirements.txt.

        The requirement is either an exact pin (``triton==X``) or a range
        (``triton>=X,<=Y``); the vendored copy tracks the lower bound either way.
        """
        repo_root = Path(__file__).parent.parent.parent.parent

        version_file = repo_root / "triton_kernels" / "VERSION"
        vendored_version = version_file.read_text().strip().split()[0].lstrip("v")

        requirements_file = repo_root / "requirements.txt"
        requirements_version = _triton_lower_bound(requirements_file.read_text())

        self.assertIsNotNone(
            requirements_version, "Could not find triton version in requirements.txt"
        )

        self.assertEqual(
            vendored_version,
            requirements_version,
            f"Vendored triton_kernels version ({vendored_version}) does not match "
            f"triton version in requirements.txt ({requirements_version}). "
            "To update the vendored triton_kernels, run: python scripts/vendor_triton_kernels.py "
            f"--tag v{requirements_version}",
        )

    def test_lower_bound_parsing(self):
        """Both requirement spellings must yield the same lower bound.

        test_version_matches_requirements only ever sees whichever form
        requirements.txt currently uses, so exercise the other one here too, along
        with the neighbouring packages the pattern must not match.
        """
        for requirement in (
            "triton==3.7.0",
            "triton>=3.7.0,<=3.8.0",
            "triton >= 3.7.0, <= 3.8.0",
            "triton==3.7.0 # NOTE: also re-vendor triton_kernels",
        ):
            with self.subTest(requirement=requirement):
                self.assertEqual(_triton_lower_bound(f"blake3\n{requirement}\nxdsl\n"), "3.7.0")

        for requirement in ("triton-kernels==1.2.3", "tritonclient==2.60.0"):
            with self.subTest(requirement=requirement):
                self.assertIsNone(_triton_lower_bound(f"{requirement}\n"))


if __name__ == "__main__":
    unittest.main()
