# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only


def _run_python(code: str, numexpr_threads: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("NUMEXPR_MAX_THREADS", None)
    env.pop("NUMEXPR_NUM_THREADS", None)
    if numexpr_threads is not None:
        env["NUMEXPR_NUM_THREADS"] = numexpr_threads

    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_skip_softmax_defaults_numexpr_to_one_lazy_thread() -> None:
    result = _run_python(
        "import os; "
        "import sys; "
        "from tensorrt_llm._torch.attention_backend.sparse.skip_softmax "
        "import SkipSoftmaxFormula; "
        "assert os.environ['NUMEXPR_NUM_THREADS'] == '1'; "
        "assert 'numexpr' not in sys.modules; "
        "formula = SkipSoftmaxFormula("
        "formula='sqrt(a + target_sparsity)', coefficients={'a': 0.75}); "
        "assert 'numexpr' in sys.modules; "
        "assert formula.compute_threshold_scale_factor(0.25) == 1.0; "
        "assert sys.modules['numexpr'].get_num_threads() == 1"
    )

    assert result.returncode == 0, result.stderr


def test_skip_softmax_preserves_explicit_numexpr_threads() -> None:
    result = _run_python(
        "import os; "
        "import sys; "
        "from tensorrt_llm._torch.attention_backend.sparse.skip_softmax "
        "import SkipSoftmaxFormula; "
        "assert os.environ['NUMEXPR_NUM_THREADS'] == '2'; "
        "formula = SkipSoftmaxFormula("
        "formula='sqrt(a + target_sparsity)', coefficients={'a': 0.75}); "
        "assert sys.modules['numexpr'].get_num_threads() == 2",
        numexpr_threads="2",
    )

    assert result.returncode == 0, result.stderr
