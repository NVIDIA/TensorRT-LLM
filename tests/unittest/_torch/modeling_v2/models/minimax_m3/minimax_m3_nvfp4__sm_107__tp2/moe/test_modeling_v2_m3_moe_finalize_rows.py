# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Who finalizes the wide target's MoE at each generic row count: past the decode rows, up to
GENERIC_MC_FINALIZE_MAX_TOKENS, the finalize (+ the shared expert) folds into the boundary's multicast all-reduce
(M3DecoderLayer._mc_finalize_boundary); from MOE_TRITON_FINALIZE_MIN_TOKENS, ops/m3_moe_finalize finalizes it.
The two ranges meet with no gap and no overlap, and the fold declines from shapes alone past its cap.

CPU only: nothing here launches a kernel.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    m3_mc_allreduce,
)

H = 6144


def _fold_cap() -> int:
    return min(dispatch.GENERIC_MC_FINALIZE_MAX_TOKENS, m3_mc_allreduce.FINALIZE_MAX_TOKENS)


def test_the_fold_stays_inside_the_finalize_kernel() -> None:
    assert dispatch.GENERIC_MC_FINALIZE_MAX_TOKENS <= m3_mc_allreduce.FINALIZE_MAX_TOKENS


def test_the_fold_and_the_triton_finalize_meet() -> None:
    """With the fold on (MC_AR_TP2), the Triton finalize starts on the first row past it; with it off, nothing folds."""
    if not dispatch.MC_AR_TP2:
        assert dispatch.GENERIC_MC_FINALIZE_MAX_TOKENS == 0
        return
    assert dispatch.DECODE_MAX_TOKENS < _fold_cap()
    if dispatch.MOE_TRITON_FINALIZE:
        assert dispatch.MOE_TRITON_FINALIZE_MIN_TOKENS == _fold_cap() + 1


def _target():
    """A DecodeTarget whose bucket lookup and MC finalize must not be reached (the row gate declines first)."""

    def unreachable(*args, **kwargs):
        raise AssertionError("the generic MC finalize ran past its row gate")

    return SimpleNamespace(_bucket_flags=unreachable, _mc_finalize=unreachable)


@pytest.mark.parametrize("past", [1, 2, 16])
def test_the_fold_declines_past_its_cap(past: int) -> None:
    hidden = torch.empty((_fold_cap() + past, H), dtype=torch.bfloat16)
    assert dispatch.DecodeTarget._generic_mc_finalize(_target(), 3, hidden, hidden, None) is None


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_the_fold_leaves_the_decode_layers_rows(rows: int) -> None:
    rows = min(rows, dispatch.DECODE_MAX_TOKENS)
    hidden = torch.empty((rows, H), dtype=torch.bfloat16)
    assert dispatch.DecodeTarget._generic_mc_finalize(_target(), 3, hidden, hidden, None) is None
