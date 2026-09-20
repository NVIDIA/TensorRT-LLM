# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for the k4_persistent ptx-options monkey-patch.

The patch's exception handling encodes two constraints that are invisible in
its behavior and have already round-tripped through review once:

  * The handler must stay a single ``except Exception`` clause: the
    nvidia-cutlass-dsl 4.5.0 AST preprocessor rejects tuple except handlers
    anywhere in a kernel module ("'Tuple' object has no attribute 'id'"),
    breaking every ``cute.compile`` of the file. A narrowing to
    ``except (AttributeError, ImportError)`` had to be reverted for exactly
    this (commit f79c6ef3af6).
  * Only the expected "DSL absent / API surface moved" kinds
    (``AttributeError``, ``ImportError``) may be swallowed. Anything else
    must fail the import loudly: silently skipping the patch defers the
    blow-up to the first K4 compile (ptxas over WG0's 40-register budget
    without ``--uumn``) with no trail back to the patch.

CPU-only: the import scenarios run the module top level in subprocesses (no
kernel is compiled); the parser constraint is checked structurally on the AST.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# The L0 CPU-Generic stages run `pytest -m cpu_only`, and their conftest only
# collects files containing the literal string "pytest.mark.cpu_only"; without
# this marker every test here is deselected and pytest exits 5 (no tests
# collected), which the test_unittests_v2 wrapper reports as a failure.
pytestmark = pytest.mark.cpu_only

try:
    import cutlass  # noqa: F401
    import flashinfer.gdn_kernels  # noqa: F401

    _KERNEL_DEPS_AVAILABLE = True
except ImportError:
    _KERNEL_DEPS_AVAILABLE = False

needs_kernel_deps = pytest.mark.skipif(
    not _KERNEL_DEPS_AVAILABLE, reason="requires nvidia-cutlass-dsl and flashinfer (gdn_kernels)"
)


def _k4_path() -> Path:
    """The k4_persistent.py that is actually in use (installed package if
    importable, else the source tree this test file lives in)."""
    try:
        spec = importlib.util.find_spec(
            "tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.k4_persistent"
        )
        if spec is not None and spec.origin:
            return Path(spec.origin)
    except ModuleNotFoundError:
        pass
    return (
        Path(__file__).resolve().parents[4]
        / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/"
        "k4_persistent.py"
    )


def test_no_tuple_except_handlers():
    """Guard against reintroducing ``except (A, B):`` anywhere in the module.

    Nothing else in CI compiles these kernels, so without this check the
    breakage only shows up as a runtime cute.compile failure on Blackwell.
    """
    tree = ast.parse(_k4_path().read_text())
    bad = [
        handler.lineno
        for node in ast.walk(tree)
        for handler in getattr(node, "handlers", [])
        if isinstance(handler.type, ast.Tuple)
    ]
    assert not bad, (
        f"tuple except handler(s) at line(s) {bad} of k4_persistent.py: the "
        f"nvidia-cutlass-dsl 4.5.0 AST preprocessor cannot parse these and "
        f"every cute.compile of the module fails. Use a single `except "
        f"Exception` with an isinstance re-raise in the body instead."
    )


_LOAD_K4 = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("k4_under_test", {k4!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"""

_FAKE_DSL = """
import sys, types
import cutlass.cutlass_dsl  # materialize the real parent package first
fake = types.ModuleType("cutlass.cutlass_dsl.cutlass")
{fake_body}
sys.modules["cutlass.cutlass_dsl.cutlass"] = fake
"""


def _run_import(inject: str = "") -> subprocess.CompletedProcess:
    code = _FAKE_DSL.format(fake_body=inject) if inject else ""
    code += _LOAD_K4.format(k4=str(_k4_path()))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)


@needs_kernel_deps
def test_import_applies_patch():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _LOAD_K4.format(k4=str(_k4_path()))
            + """
from cutlass.cutlass_dsl.cutlass import CuTeDSL
assert CuTeDSL._get_pipeline.__name__ == "_patched_get_pipeline", \\
    f"ptx-options patch not applied: {CuTeDSL._get_pipeline}"
""",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@needs_kernel_deps
def test_expected_patch_failure_is_soft():
    """AttributeError (API surface moved) logs a warning and continues."""
    proc = _run_import(
        inject="""
class CuTeDSL:  # no _get_pipeline -> AttributeError in the patch block
    pass
fake.CuTeDSL = CuTeDSL
"""
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@needs_kernel_deps
def test_unexpected_patch_failure_raises():
    """Anything but AttributeError/ImportError must fail the import."""
    proc = _run_import(
        inject="""
class _Meta(type):
    def __getattr__(cls, name):
        raise RuntimeError("simulated unexpected patch failure")
class CuTeDSL(metaclass=_Meta):
    pass
fake.CuTeDSL = CuTeDSL
"""
    )
    assert proc.returncode != 0, (
        "an unexpected exception in the ptx-options patch was swallowed at "
        "import; it must propagate (a silently skipped patch surfaces later "
        "as an unexplained ptxas register-budget failure)"
    )
    assert "simulated unexpected patch failure" in proc.stderr
