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

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "attention_cases"
_CASES = list(iter_case_specs(FIXTURE_DIR)) if FIXTURE_DIR.exists() else []


def _replay_id(case) -> str:
    kvd = case.kv_dtype or "samekv"
    return (
        f"h{case.num_heads}kv{case.num_kv_heads}_d{case.head_dim}_"
        f"{case.dtype}_{kvd}_nq{case.nnz_q}_nc{sum(case.num_cached_tokens)}"
    )


@pytest.mark.skipif(not _CASES, reason="no captured cases in fixtures/attention_cases/")
@pytest.mark.parametrize("case", _CASES, ids=_replay_id)
def test_replay(case):
    # run_case dispatches MLA / standard and skips unsupported backends via the
    # capability matrix (including sm-dependent dtypes), so no pre-skip is needed.
    run_case(case)
