#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the gate-GEMM unit tests in a container with no TensorRT-LLM install.

The unit tests import the kernel the normal way, which is right for CI and
useless in a bare PyTorch container: `tensorrt_llm/__init__.py` reaches for the
compiled bindings and a long tail of runtime dependencies, none of which a CuTe
DSL kernel needs, and `tests/unittest/conftest.py` imports the package too.

So pre-import the kernel through the stubbed parents from `_repo_import`, which
satisfies the test file's own import by the time pytest reads it, and skip the
conftest that would drag the package back in.

    python3 tests/microbenchmarks/minimax_m3_gate_gemm/run_bare_pytest.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from _repo_import import REPO_ROOT, import_bare  # noqa: E402

KERNEL = "tensorrt_llm._torch.cute_dsl_kernels.blackwell.minimax_m3_gate_gemm_runner"
TESTS = REPO_ROOT / "tests/unittest/_torch/modules/test_minimax_m3_gate_gemm.py"


def main() -> int:
    import_bare(KERNEL)
    return pytest.main(["--noconftest", "-p", "no:cacheprovider", str(TESTS), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
