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
"""Import-order tests for the deprecated ``logger`` module shim.

``logger`` is also the name under which the package re-exports the ``Logger``
singleton, so importing the shim would bind the module onto the package -- over
the singleton -- unless the re-export is pinned.  The effect is process-wide
and one-way, so every test below runs in a fresh interpreter.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cpu_only


def _run_in_fresh_interpreter(body: str, pythonpath: str = "") -> subprocess.CompletedProcess:
    # PYTHONWARNINGS is scrubbed rather than left alone: it would override the
    # default filters in the child, and one of the tests below is about what
    # those defaults do.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    if pythonpath:
        # Prepended, not assigned: an inherited PYTHONPATH may be how the child
        # finds tensorrt_llm at all.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (pythonpath, env.get("PYTHONPATH", "")) if p)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_importing_the_shim_keeps_the_package_attribute():
    """The singleton stays reachable through the package after the old import."""
    result = _run_in_fresh_interpreter(
        """
        import warnings

        import tensorrt_llm
        from tensorrt_llm.observability.logging import Logger
        from tensorrt_llm.observability.logging import logger as canonical

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import tensorrt_llm.logger

        from tensorrt_llm import logger

        assert logger is canonical, type(logger)
        assert isinstance(logger, Logger), type(logger)
        assert tensorrt_llm.logger is logger, type(tensorrt_llm.logger)
        assert callable(tensorrt_llm.logger.info)
        tensorrt_llm.logger.info("reached through the package attribute")
        """
    )
    assert result.returncode == 0, result.stderr


def test_shim_still_warns_and_re_exports_the_same_objects():
    """Pinning the re-export must not cost the shim its warning.

    This forces the filters, so it establishes that the warning is *raised* --
    exactly once, and not swallowed by the pinning in ``_bootstrap``.  Whether
    anyone would see it is the next test's question, and the two are separate on
    purpose: a shim can pass this one and warn nobody.
    """
    result = _run_in_fresh_interpreter(
        """
        import sys
        import warnings

        import tensorrt_llm  # noqa: F401
        from tensorrt_llm.observability import logging as canonical

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tensorrt_llm.logger

        raised = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(raised) == 1, [str(w.message) for w in caught]

        shim = sys.modules["tensorrt_llm.logger"]
        for name in shim.__all__:
            assert getattr(shim, name) is getattr(canonical, name), name
        """
    )
    assert result.returncode == 0, result.stderr


def test_the_warning_reaches_the_callers_the_shim_exists_for(tmp_path):
    """The warning must survive the *default* filters, from outside ``__main__``.

    The test above forces ``simplefilter("always")``, so it proves the warning is
    raised and says nothing about whether anyone sees it -- and the shim's only
    audience is code this repository cannot change.  Python ignores
    ``DeprecationWarning`` outside ``__main__``, so a shim on that category is
    raised at every one of those call sites and displayed at none; the same is
    true of a bare ``python -c "import tensorrt_llm.logger"``, where the import
    site *is* ``__main__`` and the invisible category looks fine.

    So the import here happens inside a package module, which is what a
    downstream library's call site looks like, with the filters left alone.
    """
    downstream = tmp_path / "downstream_pkg"
    downstream.mkdir()
    (downstream / "__init__.py").write_text("")
    (downstream / "consumer.py").write_text(
        "import importlib\n"
        "\n"
        "\n"
        "def use_the_old_path():\n"
        "    return importlib.import_module('tensorrt_llm.logger')\n"
    )

    result = _run_in_fresh_interpreter(
        """
        from downstream_pkg.consumer import use_the_old_path

        use_the_old_path()
        """,
        pythonpath=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "FutureWarning" in result.stderr, (
        "the shim's warning did not reach a caller importing the old path from "
        f"its own package; stderr was: {result.stderr!r}"
    )
    assert "tensorrt_llm.observability.logging" in result.stderr, result.stderr
