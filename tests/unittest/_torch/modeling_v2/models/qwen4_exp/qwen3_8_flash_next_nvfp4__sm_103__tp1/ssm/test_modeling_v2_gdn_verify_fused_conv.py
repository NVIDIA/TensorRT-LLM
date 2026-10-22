# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `DecodeTarget._gdn_block`'s verify dispatch under the dense FP8 mode.

A captured verify step (`fused_conv=True`) runs the in_proj with the multi-token conv step as its
epilogue (`fp8_gemv_conv1d_verify`); an eager one runs `fp8_gemv` and then
`causal_conv1d_triton_update`. Both must hand the recurrence the same thing: the conv output
(contiguous, as the cached replay reads it), `a`, `b` and `z` bitwise, and leave the conv pool
and the per-step snapshot buffer bitwise the same. The recurrence is stubbed to record its
inputs (it is the same call on both sides), the epilogue to record `z`.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.gemm.fp8_gemv import (  # noqa: E501
    pack_from_bf16,
)

assert torch.cuda.is_available(), "the Gated-DeltaNet block requires a CUDA device"

DEV, BF16 = "cuda", torch.bfloat16
HIDDEN, HV, V, CONV, VD, SLOTS, ROWS = 2560, 48, 128, 10240, 6144, 461, 32


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator(device=DEV).manual_seed(0)
    w = (torch.randn(CONV + VD + 2 * HV, HIDDEN, generator=g, device=DEV) * HIDDEN**-0.5).to(BF16)
    return types.SimpleNamespace(
        in_proj=w.t(),
        in_proj8=None,
        in_proj_g=pack_from_bf16(w),
        conv_w=(torch.randn(CONV, 4, generator=g, device=DEV) * 0.5).to(BF16),
    )


def _block(layer, x, si, int_idx, conv0, win0, p, fused, monkeypatch):
    """One verify `_gdn_block` on copies of the pools; returns what the recurrence and the
    epilogue were handed, and the pools."""
    seen = {}

    def replay(core, mq_out, a_d, b_d, ssm_states, i, L, si_d, n, p_, work):
        seen.update(mq_out=mq_out.clone(), a=a_d.clone(), b=b_d.clone())
        assert mq_out.is_contiguous() and mq_out.shape == (n * p_, CONV)
        return torch.zeros(1, n * p_, HV, V, dtype=BF16, device=DEV)

    monkeypatch.setattr(M, "_replay_verify", replay)
    core = types.SimpleNamespace(
        conv_dim=CONV,
        in_proj_split=(CONV, VD, HV, HV),
        _bound=types.SimpleNamespace(
            conv={0: conv0.clone()}, ssm={0: torch.zeros(1, device=DEV)}, int_conv={0: win0.clone()}
        ),
    )
    target = object.__new__(M.DecodeTarget)
    target.core = core
    target._gdn_epilogue = lambda out, z, L, num_tokens, gemm=None: z.clone()
    n = si.shape[0]
    gdn_state = (si, True, n, p, int_idx, (None, None, False))
    z = target._gdn_block(x, 0, layer, gdn_state, cublas_mm, fused)
    return dict(seen, z=z, conv=core._bound.conv[0], win=core._bound.int_conv[0])


@pytest.mark.parametrize("n,p", [(1, 4), (2, 4), (4, 4), (8, 2), (2, 8), (5, 3)])
def test_fused_verify_matches_two_launch(layer, n, p, monkeypatch) -> None:
    g = torch.Generator(device=DEV).manual_seed(10 * n + p)
    conv0 = torch.randn(SLOTS, CONV, 3, generator=g, device=DEV).to(BF16)
    win0 = torch.randn(ROWS, p, CONV, 3, generator=g, device=DEV).to(BF16)
    x = torch.randn(n * p, HIDDEN, generator=g, device=DEV).to(BF16)
    si = torch.randperm(SLOTS, generator=g, device=DEV)[:n].to(torch.int32)
    int_idx = torch.arange(ROWS, dtype=torch.int32, device=DEV)[:n]
    eager = _block(layer, x, si, int_idx, conv0, win0, p, False, monkeypatch)
    fused = _block(layer, x, si, int_idx, conv0, win0, p, True, monkeypatch)
    for key in ("mq_out", "a", "b", "z", "conv", "win"):
        assert torch.equal(eager[key].view(torch.int16), fused[key].view(torch.int16)), key
    assert not torch.equal(fused["conv"], conv0) and not torch.equal(fused["win"], win0)
