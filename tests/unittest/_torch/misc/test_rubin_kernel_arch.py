# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every Rubin CuTe-DSL kernel must name an arch the TMEM allocator accepts.

``self.arch`` on these kernels is consumed by
``blackwell/dense_gemm_persistent.py::_compute_num_tmem_alloc_cols``, which
forwards it to ``cutlass.utils.get_num_tmem_alloc_cols`` ->
``cutlass.cute.arch.get_max_tmem_alloc_cols``. That helper raises
``ValueError: Unsupported compute capability: <arch>`` for any name it does not
know, and it does not know the accelerated ``sm_107a`` spelling.

Because the string is only read during kernel setup, a wrong value survives
import, collection and model load, and surfaces much later as an executor
initialisation failure on every rank -- which is how it reached hardware:
``dense_bf16_gemm_persistent`` carried ``sm_107a`` while its own module
docstring and all seven sibling Rubin kernels said ``sm_107``.

This asserts the invariant over the whole package rather than that one kernel,
since the same drift is available to every kernel added later. It is a static
source scan: no GPU, no CUDA and no cutlass import required.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm

RUBIN_KERNEL_ROOT = Path(tensorrt_llm.__file__).parent / "_torch" / "cute_dsl_kernels" / "rubin"

# The base (non-accelerated) name is what the TMEM allocator recognises for
# Rubin, and matches how the Blackwell base class spells its own ("sm_100").
EXPECTED_ARCH = "sm_107"


def _iter_arch_assignments():
    """Yield (path, lineno, value) for every ``self.arch = "..."`` literal."""
    for path in sorted(RUBIN_KERNEL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "arch"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    yield path, node.lineno, node.value.value


def test_rubin_kernels_declare_a_supported_tmem_arch():
    assignments = list(_iter_arch_assignments())

    # Guard against a vacuous pass: if the scan finds nothing, the invariant is
    # untested rather than satisfied.
    assert assignments, (
        f"no 'self.arch = ...' assignment found under {RUBIN_KERNEL_ROOT}; "
        "the regression this test exists for could not be detected"
    )

    wrong = [(path, lineno, value) for path, lineno, value in assignments if value != EXPECTED_ARCH]
    assert not wrong, "Rubin kernels declare an arch the TMEM allocator rejects:\n" + "\n".join(
        f"  {path.name}:{lineno} sets {value!r}, expected {EXPECTED_ARCH!r}"
        for path, lineno, value in wrong
    )


@pytest.mark.parametrize("arch", [EXPECTED_ARCH])
def test_expected_arch_is_known_to_the_tmem_allocator(arch):
    """Pin the claim that ``EXPECTED_ARCH`` is actually accepted upstream.

    Skipped where cutlass is unavailable so the invariant above still runs.
    """
    tmem = pytest.importorskip("cutlass.cute.arch.tmem")
    get_max = getattr(tmem, "get_max_tmem_alloc_cols", None)
    if get_max is None:
        pytest.skip("cutlass build predates get_max_tmem_alloc_cols")
    # TODO: remove this guard once the CI image ships a Rubin-capable CuTe DSL.
    try:
        import cutlass.utils.rubin_helpers  # noqa: F401
    except ImportError:
        pytest.skip("installed cutlass-dsl predates Rubin support")

    assert get_max(arch) > 0
