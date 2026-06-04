# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay captured (and minimized) real-workload attention cases.

Fixtures under ``fixtures/attention_cases/`` are ``BackendCase`` specs captured
from real model runs (via ``TRTLLM_ATTN_CAPTURE_DIR``) and shrunk by
``minimize.py``. Each is run through the same ``run_case`` machinery as the
synthetic sweep, validating every supported backend against the Vanilla golden.

Skips cleanly when no fixtures are present.
"""

from pathlib import Path

import pytest
from attention_test_harness import run_case
from case_io import iter_case_specs
from test_attention_backends import _needs_skip, _sweep_id

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "attention_cases"
_CASES = list(iter_case_specs(FIXTURE_DIR)) if FIXTURE_DIR.exists() else []


@pytest.mark.skipif(not _CASES, reason="no captured cases in fixtures/attention_cases/")
@pytest.mark.parametrize("case", _CASES, ids=_sweep_id)
def test_replay(case):
    skip = _needs_skip(case)
    if skip:
        pytest.skip(skip)
    if case.is_mla:
        pytest.skip("MLA replay is handled by test_attention_mla.py")
    run_case(case)
