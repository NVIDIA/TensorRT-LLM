# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import MnnvlAllReduceNorm

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def fused_norm() -> MnnvlAllReduceNorm:
    ar = MnnvlAllReduceNorm.__new__(MnnvlAllReduceNorm)
    ar.hidden = 4096
    ar._sms = 132
    ar._smem_bytes = 232448
    ar._sm_smem_bytes = 233472
    return ar


@pytest.mark.parametrize(
    ("hidden", "tokens_per_cta", "smem_bytes"),
    [(2048, 28, 115616), (4096, 14, 115616), (8192, 14, 231200)],
)
def test_shared_memory_limit(
    fused_norm: MnnvlAllReduceNorm, hidden: int, tokens_per_cta: int, smem_bytes: int
) -> None:
    fused_norm.hidden = hidden
    rows = tokens_per_cta * 2 * fused_norm._sms
    fused_norm._smem_bytes = smem_bytes
    assert fused_norm.supports_rows(rows)
    fused_norm._smem_bytes -= 1
    assert not fused_norm.supports_rows(rows)


@pytest.mark.parametrize(
    ("hidden", "rows", "supported"),
    [
        (8192, 264, True),
        (16384, 64, True),
        (16384, 67, False),
        (32768, 63, True),
        (32768, 64, False),
        (65536, 63, True),
        (65536, 64, False),
    ],
)
def test_thread_limit(
    fused_norm: MnnvlAllReduceNorm, hidden: int, rows: int, supported: bool
) -> None:
    fused_norm.hidden = hidden
    assert fused_norm.supports_rows(rows) is supported


@pytest.mark.parametrize(
    ("hidden", "tokens_per_cta", "supported"),
    [
        (2048, 28, True),
        (4096, 14, True),
        (4096, 15, False),
        (4096, 28, False),
        (8192, 8, True),
        (8192, 14, True),
    ],
)
def test_two_cta_residency(
    fused_norm: MnnvlAllReduceNorm, hidden: int, tokens_per_cta: int, supported: bool
) -> None:
    """A 512-thread CTA (hidden 4096) must fit twice in an SM's shared memory; other sizes are not bound."""
    fused_norm.hidden = hidden
    assert fused_norm.supports_rows(tokens_per_cta * 2 * fused_norm._sms) is supported


@pytest.mark.parametrize("rows", [0, -1])
def test_nonpositive_rows(fused_norm: MnnvlAllReduceNorm, rows: int) -> None:
    assert not fused_norm.supports_rows(rows)
