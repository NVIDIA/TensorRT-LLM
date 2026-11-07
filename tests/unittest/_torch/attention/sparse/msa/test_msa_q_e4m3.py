# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin query cast preserves finite ATen bits, including fused-QKV row strides."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 required"
)


@pytest.mark.parametrize("rows", [1, 129])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "fused_qkv_view"])
def test_query_cast_finite_bits_and_explicit_saturation(monkeypatch, rows, strided):
    heads, dim = 3, 128
    pattern = torch.tensor(
        [
            0.0,
            -0.0,
            2**-133,
            -(2**-133),
            2**-11,
            -(2**-11),
            2**-10,
            -(2**-10),
            2**-10 * (1 + 2**-7),
            -(2**-10 * (1 + 2**-7)),
            2**-9,
            -(2**-9),
            2**-6,
            -(2**-6),
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            448,
            -448,
            464,
            -464,
            480,
            -480,
            1000,
            -1000,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    width = heads * dim
    backing = torch.empty(
        (rows, width * (3 if strided else 1)), dtype=pattern.dtype, device=pattern.device
    )
    q = backing.view(rows, 3 if strided else 1, heads, dim)[:, 0]
    q.copy_(
        pattern.repeat((q.numel() + pattern.numel() - 1) // pattern.numel())[: q.numel()].view_as(q)
    )
    assert q.stride(0) == width * (3 if strided else 1)
    reference = q.to(torch.float8_e4m3fn)
    monkeypatch.setattr(msa_prefill, "_SM107_Q_E4M3_ROWS", True)
    actual = msa_prefill._q_to_e4m3(q)
    assert actual.shape == q.shape and actual.is_contiguous()
    finite = torch.isfinite(reference.float())
    assert torch.equal(actual.view(torch.uint8)[finite], reference.view(torch.uint8)[finite])
    source_nan = torch.isnan(q.float())
    overflow = q.float().abs() > 448
    assert overflow.any()
    torch.testing.assert_close(
        actual.float()[overflow], q.float().sign()[overflow] * 448, rtol=0, atol=0
    )
    assert torch.isnan(actual.float()[source_nan]).all()
    monkeypatch.setattr(msa_prefill, "_SM107_Q_E4M3_ROWS", False)
    fallback = msa_prefill._q_to_e4m3(q)
    assert torch.equal(fallback.view(torch.uint8), reference.view(torch.uint8))
