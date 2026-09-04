# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cpu_only


def _run_python(code: str, numexpr_threads: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("NUMEXPR_MAX_THREADS", None)
    env.pop("NUMEXPR_NUM_THREADS", None)
    if numexpr_threads is not None:
        env["NUMEXPR_NUM_THREADS"] = numexpr_threads

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_skip_softmax_uses_function_local_numexpr_imports() -> None:
    result = _run_python(
        """
        import sys

        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            SkipSoftmaxFormula,
        )

        params = sys.modules[SkipSoftmaxFormula.__module__]
        assert "numexpr" not in vars(params), "NumExpr must remain a function-local import"
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("configured_threads", "expected_threads"),
    [(None, 1), ("2", 2)],
)
def test_skip_softmax_configures_numexpr_threads_at_import(
    configured_threads: str | None, expected_threads: int
) -> None:
    result = _run_python(
        f"""
        import os
        import sys

        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            SkipSoftmaxFormula,
        )

        expected_threads = {expected_threads}
        assert os.environ["NUMEXPR_NUM_THREADS"] == str(expected_threads)

        formula = SkipSoftmaxFormula(
            formula="sqrt(a + target_sparsity)", coefficients={{"a": 0.75}}
        )
        numexpr = sys.modules["numexpr"]
        assert numexpr.nthreads == expected_threads, (
            f"NumExpr initialized with {{numexpr.nthreads}} threads, "
            f"expected {{expected_threads}}"
        )
        assert numexpr.get_num_threads() == expected_threads, (
            f"NumExpr is using {{numexpr.get_num_threads()}} threads before evaluation, "
            f"expected {{expected_threads}}"
        )

        assert formula.compute_threshold_scale_factor(0.25) == 1.0
        assert numexpr.get_num_threads() == expected_threads, (
            f"NumExpr is using {{numexpr.get_num_threads()}} threads after evaluation, "
            f"expected {{expected_threads}}"
        )
        """,
        numexpr_threads=configured_threads,
    )

    assert result.returncode == 0, result.stderr
