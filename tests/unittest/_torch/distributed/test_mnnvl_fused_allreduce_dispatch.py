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
        (16384, 67, True),  # slot transport (up to SLOT_MAX_ROWS)
        (16384, 129, True),  # two CTAs of 1024 threads per token
        (32768, 64, True),
        (65536, 64, True),
        (65536, 129, True),  # eight CTAs of 1024 threads per token
        (131072, 129, False),  # eight CTAs of 2048 threads per token
    ],
)
def test_thread_limit(
    fused_norm: MnnvlAllReduceNorm, hidden: int, rows: int, supported: bool
) -> None:
    """Above SLOT_MAX_ROWS a token's CTAs shrink to fit the SMs only while each stays within 1024 threads; a hidden
    size whose eight CTAs per token still exceed them is not taken."""
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


# (m, hidden, sms, access group) -> (block, cluster, loads per thread) of twoshotAllreduceFusionOp's rmsNormLamport
# launch, from mnnvlAllreduceKernels.cu's adjustGridConfig<true> / <false> compiled on the host (getMultiProcessorCount
# replaced by sms): every step of it (cluster shrink and regrowth, the SM cap, the non-CGA fallback, loads > 1).
_LIB_NORM_LAUNCHES = [
    ((1, 4096, 148, 1), (64, 8, 1)),
    ((18, 4096, 148, 1), (64, 8, 1)),
    ((19, 4096, 148, 1), (128, 4, 1)),
    ((64, 4096, 148, 1), (256, 2, 1)),
    ((75, 4096, 148, 1), (512, 1, 1)),
    ((200, 4096, 148, 1), (512, 1, 1)),
    ((128, 16384, 148, 1), (1024, 2, 1)),
    ((128, 65536, 148, 1), (1024, 8, 1)),
    ((1, 2560, 148, 1), (80, 4, 1)),
    ((40, 2560, 148, 1), (160, 2, 1)),
    ((1, 131072, 148, 1), (2048, 1, 8)),
    ((1, 6144, 132, 2), (96, 8, 1)),
    ((17, 7168, 132, 2), (224, 4, 1)),
    ((1, 1000, 160, 1), (125, 1, 1)),
]


@pytest.mark.parametrize(("args", "launch"), _LIB_NORM_LAUNCHES)
def test_lib_norm_tree_matches_the_library_launch(
    args: tuple[int, int, int, int], launch: tuple[int, int, int]
) -> None:
    """The one Python port of the library's norm launch shape, which both the slot / two-shot all-reduce kernels
    (``_ctas_per_token``) and the GEMM + two-shot kernel replay."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import _ctas_per_token, lib_norm_tree

    m, hidden, sms, group = args
    assert lib_norm_tree(m, hidden, sms, group=group) == launch
    if group == 1:
        assert _ctas_per_token(m, hidden, sms) == launch[1]
