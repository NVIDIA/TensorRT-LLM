# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_ull_cp_tail, the BF16 low-latency MoE tail that pushes the latent rows over the TP group.

On the CPU: the entry's refusals and its reference's shape. The cells (``rubin_ull_cp_tail.CELLS``)
need two or four SM107 ranks of one tray, which no CI stage has.
"""

from types import SimpleNamespace

__extra_import_path__ = ["../../.."]
import pytest
import torch
from _k31_utils import cpu_or_sm

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.ops.moe import (
    _rubin_ull_common as c,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.ops.moe.rubin_ull_cp_tail import (  # noqa: E501
    rubin_ull_cp_tail,
)


def _exact(rows, cols, seed, scale=1.0):
    """Small integers times a power of two: every FP32 sum below is exact."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(-3, 4, (rows, cols), generator=g).float() * scale).to(torch.bfloat16)


def _ws(world, rank, nbytes=1 << 23):
    return SimpleNamespace(world=world, rank=rank, bytes_per_buffer=nbytes)


@pytest.mark.cpu_only
@cpu_or_sm(107)
def test_cp_tail_guard():
    latent = torch.zeros(8, c.LATENT, dtype=torch.bfloat16)
    w = torch.zeros(896, c.LATENT, dtype=torch.bfloat16)
    rubin_ull_cp_tail.is_valid(latent, w, _ws(8, 1), (64, 8, 4, 7), c0=896)
    for bad in (
        dict(ws=_ws(8, 2), c0=896),
        dict(ws=_ws(8, 1), c0=0),
        dict(ws=_ws(8, 1, 1 << 16), c0=896),
    ):
        with pytest.raises(AssertionError):
            rubin_ull_cp_tail.is_valid(latent, w, bad["ws"], (64, 8, 4, 7), c0=bad["c0"])
    with pytest.raises(AssertionError):
        rubin_ull_cp_tail.is_valid(
            torch.zeros(33, c.LATENT, dtype=torch.bfloat16), w, _ws(8, 1), (64, 8, 4, 7), c0=896
        )
    ref = rubin_ull_cp_tail.reference(
        _exact(8, c.LATENT, 1), _exact(896, c.LATENT, 2, 0.25), None, None, c0=896
    )
    assert ref.shape == (8, 896)
