# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_qk_norm_rope_fp8 entry of the MiniMax-M3 tp8 target: the out-of-place FP8 form of the fused
per-head Gemma RMSNorm + partial NeoX RoPE against the in-place kernel followed by the E4M3 cast. V is a cast, so it
matches bit for bit; Q and K round once from fp32 where the composition rounds through bf16 first, so each element is
within one E4M3 step of it."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_qk_norm_rope_fp8 import (  # noqa: E501
    m3_qk_norm_rope_fp8 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_qk_norm_rope_fp8 requires a CUDA device"

HEAD_DIM, ROTARY_DIM, EPS, THETA = 128, 64, 1e-6, 5e6


def _args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, kv = spec["q"], spec["kv"]
    qkv = torch.randn(spec["m"], (q + 2 * kv) * HEAD_DIM, generator=g, device="cuda").to(
        torch.bfloat16
    )
    qw = (torch.randn(HEAD_DIM, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    kw = (torch.randn(HEAD_DIM, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 200000, (spec["m"],), generator=g, device="cuda", dtype=torch.int32)
    return (
        qkv,
        q,
        kv,
        kv,
        HEAD_DIM,
        ROTARY_DIM,
        EPS,
        qw,
        kw,
        THETA,
        True,
        pos,
        1.0,
        0.0,
        0.0,
        1.0,
        True,
        True,
        False,
        0,
        0,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = args[0].clone()
    with validating(op):
        out = op(*args)
    assert torch.equal(args[0], before), "the FP8 form must leave qkv untouched"
    ref = op.reference(*args)
    assert out.dtype == torch.float8_e4m3fn and out.shape == ref.shape
    v_start = (cell.spec["q"] + cell.spec["kv"]) * HEAD_DIM
    assert torch.equal(out[:, v_start:].view(torch.uint8), ref[:, v_start:].view(torch.uint8))
    assert _e4m3_steps(out[:, :v_start], ref[:, :v_start]) <= 1


def _e4m3_steps(a, b) -> int:
    """The largest distance between two E4M3 tensors in representable values (E4M3 codes order by magnitude)."""

    def signed(t):
        c = t.view(torch.uint8).int()
        # Sign-magnitude to a line: -0 and +0 both map to 0.
        return torch.where(c >= 128, 128 - c, c)

    return int((signed(a) - signed(b)).abs().max())
