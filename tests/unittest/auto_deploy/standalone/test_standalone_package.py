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

"""End-to-end test for the standalone auto_deploy package.

This test:
1. Runs create_standalone_package.py to generate the standalone package
   (including source, tests, and pyproject.toml)
2. Creates a venv and installs the package with dev deps
3. Verifies TRTLLM_AVAILABLE is False and core subsystems work
4. Runs the copied unit tests from the standalone package's own tests/ dir

The venv Python is run with `-I` (isolated mode) to prevent the host env's
editable TRT-LLM install from leaking in.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CREATE_SCRIPT = os.path.join(REPO_ROOT, "examples", "auto_deploy", "create_standalone_package.py")


def _find_uv():
    return shutil.which("uv")


@pytest.fixture(scope="module")
def standalone_package(tmp_path_factory):
    """Create standalone package, venv, and install."""
    base_dir = str(tmp_path_factory.mktemp("standalone_pkg"))
    pkg_dir = os.path.join(base_dir, "package")
    venv_dir = os.path.join(base_dir, "venv")

    # 1. Generate standalone package (source + tests)
    subprocess.check_call(
        [sys.executable, CREATE_SCRIPT, "--output-dir", pkg_dir],
        timeout=60,
    )

    # 2. Create venv
    uv = _find_uv()
    if uv:
        subprocess.check_call([uv, "venv", venv_dir, "--python", sys.executable], timeout=30)
        python = os.path.join(venv_dir, "bin", "python")
        pip_install = [uv, "pip", "install", "--python", python]
    else:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir], timeout=30)
        python = os.path.join(venv_dir, "bin", "python")
        pip_install = [python, "-m", "pip", "install"]

    # 3. Install the standalone package with dev deps
    subprocess.check_call(pip_install + [pkg_dir + "[dev]"], timeout=600)

    # 4. Get the venv's site-packages path
    site_packages = subprocess.check_output(
        [python, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()

    return {
        "pkg_dir": pkg_dir,
        "venv_dir": venv_dir,
        "python": python,
        "site_packages": site_packages,
    }


def _run_isolated(pkg_info, script: str, timeout=120):
    """Run Python script in the standalone venv, isolated from host TRT-LLM."""
    preamble = textwrap.dedent(f"""\
import sys
sys.path.insert(0, {pkg_info["site_packages"]!r})
""")
    full_script = preamble + textwrap.dedent(script)
    return subprocess.run(
        [pkg_info["python"], "-I", "-c", full_script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


class TestStandalonePackage:
    """Tests that verify the standalone package works end-to-end."""

    def test_trtllm_not_available(self, standalone_package):
        """After pip install, TRTLLM_AVAILABLE should be False."""
        result = _run_isolated(
            standalone_package,
            """
            from auto_deploy._compat import TRTLLM_AVAILABLE
            assert not TRTLLM_AVAILABLE, f"Got {TRTLLM_AVAILABLE}"
            print("OK")
            """,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_import_auto_deploy(self, standalone_package):
        """The full auto_deploy module should import."""
        result = _run_isolated(
            standalone_package,
            """
            import auto_deploy
            print("OK")
            """,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_ops_registered(self, standalone_package):
        """Custom ops should register under torch.ops.auto_deploy."""
        result = _run_isolated(
            standalone_package,
            """
            import torch
            import auto_deploy
            ops = [n for n in dir(torch.ops.auto_deploy) if not n.startswith('_')]
            assert len(ops) > 0, f"No ops: {ops}"
            print(f"{len(ops)} ops")
            """,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_compat_types(self, standalone_package):
        """Standalone _compat types should work."""
        result = _run_isolated(
            standalone_package,
            """
            from auto_deploy._compat import (
                ActivationType, KvCacheConfig, str_dtype_to_torch,
            )
            from auto_deploy.utils.dist_config import DistConfig
            import torch
            assert ActivationType.Silu == 4
            assert KvCacheConfig().tokens_per_block == 32
            assert DistConfig(world_size=1, rank=0, tp_size=1).tp_rank == 0
            assert str_dtype_to_torch("float16") == torch.float16
            print("OK")
            """,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_run_unit_tests(self, standalone_package):
        """Run the copied unit tests from the standalone package's tests/ dir.

        Tests have been import-rewritten to use `auto_deploy` instead of
        `tensorrt_llm._torch.auto_deploy`, so they run directly against the
        standalone package.
        """
        python = standalone_package["python"]
        pkg_dir = standalone_package["pkg_dir"]
        tests_dir = os.path.join(pkg_dir, "tests")

        if not os.path.isdir(tests_dir):
            pytest.skip("No tests directory in standalone package")

        # Pass through the host env but override PYTHONPATH to use standalone tests.
        # The venv's pip install already put auto_deploy in the venv's site-packages,
        # so `import auto_deploy` resolves there (not to the host TRT-LLM).
        standalone_env = {
            **os.environ,
            "PYTHONPATH": tests_dir + os.pathsep + os.path.join(tests_dir, "_utils_test"),
            # Override PATH to prefer venv's python/pytest
            "PATH": os.path.join(standalone_package["venv_dir"], "bin")
            + os.pathsep
            + os.environ.get("PATH", ""),
        }

        cmd = [
            python,
            "-m",
            "pytest",
            os.path.join(tests_dir, "singlegpu"),
            "-q",
            "--timeout=300",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=pkg_dir,
            env=standalone_env,
        )

        lines = result.stdout.strip().split("\n")
        summary = [lin for lin in lines if "passed" in lin or "failed" in lin or "error" in lin]
        summary_str = summary[-1] if summary else "no summary"

        # Print summary for visibility
        print(f"Standalone test results: {summary_str}")

        # Extract counts from pytest summary like "3 failed, 100 passed, 5 skipped"
        import re

        failed_match = re.search(r"(\d+) failed", summary_str)
        passed_match = re.search(r"(\d+) passed", summary_str)
        error_match = re.search(r"(\d+) error", summary_str)

        num_failed = int(failed_match.group(1)) if failed_match else 0
        num_passed = int(passed_match.group(1)) if passed_match else 0
        num_errors = int(error_match.group(1)) if error_match else 0

        # Strict: zero collection errors
        assert num_errors == 0, (
            f"Collection errors in standalone tests!\nSummary: {summary_str}\n"
            f"Stderr:\n{result.stderr[-3000:]}"
        )

        # Strict: zero test failures — if a test can't pass standalone, it must
        # be in the EXCLUDE list in create_standalone_package.py
        assert num_failed == 0, (
            f"{num_failed} test(s) failed in standalone mode!\n"
            f"Summary: {summary_str}\n"
            f"These tests should be added to EXCLUDE_TEST_FILES in "
            f"create_standalone_package.py.\n"
            f"Failed tests:\n"
            + "\n".join(lin for lin in lines if lin.startswith("FAILED"))
            + f"\n\nStderr:\n{result.stderr[-2000:]}"
        )

        # Sanity: at least some tests should have passed
        assert num_passed > 100, (
            f"Too few tests passed ({num_passed}). Something is wrong.\nSummary: {summary_str}"
        )
