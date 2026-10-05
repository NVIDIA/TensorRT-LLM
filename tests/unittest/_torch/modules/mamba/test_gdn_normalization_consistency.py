# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent GDN normalization must use the same epsilon convention as prefill."""

from typing import Literal

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kernel", ["fused_gating", "recurrent", "cached_replay"])
@pytest.mark.parametrize("amplitude", [0.0, 1e-4, 0.1])
@pytest.mark.parametrize("nonzero_state", [False, True])
@torch.inference_mode()
def test_gdn_recurrent_matches_l2_reference(
    kernel: Literal["fused_gating", "recurrent", "cached_replay"],
    amplitude: float,
    nonzero_state: bool,
) -> None:
    from tensorrt_llm._torch.modules.fla.cached_replay import (
        fused_recurrent_gated_delta_rule_cached_replay_update,
    )
    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(2026)
    batch, heads, value_heads, dim, slots = 2, 2, 4, 64, 4
    scale = dim**-0.5
    q = (torch.randn(batch, 1, heads, dim, device="cuda") * amplitude).bfloat16()
    k = (torch.randn_like(q.float()) * amplitude).bfloat16()
    v = torch.randn(batch, 1, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(slots, value_heads, dim, dim, device="cuda") * 0.01
    if not nonzero_state:
        state.zero_()
    original = state.clone()
    indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    a = torch.zeros(batch, 1, value_heads, device="cuda")
    b = torch.zeros_like(a)
    g = torch.full_like(a, -torch.log(torch.tensor(2.0)).item())
    beta = torch.full_like(a, 0.5)

    # Independent FP64 recurrence: epsilon is added to the squared norm,
    # matching l2norm_fwd and fused_gdn_post_conv on the prefill path.
    qr, kr = q.double(), k.double()
    qr = qr / (qr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    kr = kr / (kr.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    qr = qr[:, 0].repeat_interleave(value_heads // heads, dim=1) * scale
    kr = kr[:, 0].repeat_interleave(value_heads // heads, dim=1)
    sr = original[indices.long()].double() * 0.5
    residual = (v[:, 0].double() - (sr * kr[:, :, None, :]).sum(-1)) * 0.5
    sr += residual[:, :, :, None] * kr[:, :, None, :]
    expected = (sr * qr[:, :, None, :]).sum(-1).unsqueeze(1)

    if kernel == "fused_gating":
        actual = fused_sigmoid_gating_delta_rule_update(
            A_log=torch.zeros(value_heads, device="cuda"),
            a=a,
            dt_bias=torch.zeros(value_heads, device="cuda"),
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
    elif kernel == "recurrent":
        actual = fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state_source=state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        history = 16
        actual = fused_recurrent_gated_delta_rule_cached_replay_update(
            q,
            k,
            v,
            g,
            beta,
            state,
            indices,
            torch.zeros(slots, 2, history, value_heads, dim, device="cuda", dtype=v.dtype),
            torch.zeros(slots, 2, history, heads, dim, device="cuda", dtype=k.dtype),
            torch.zeros(slots, 2, value_heads, history, device="cuda"),
            torch.zeros(slots, 2, value_heads, history, device="cuda"),
            torch.zeros(slots, device="cuda", dtype=torch.int32),
            torch.zeros(slots, device="cuda", dtype=torch.int32),
            history_size=history,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

    # The replay kernel rounds normalized Q/K and dot products to BF16;
    # the bound allows that error while rejecting the different formula.
    torch.testing.assert_close(actual.double(), expected, rtol=0.015, atol=3e-5)
    if kernel != "cached_replay":
        torch.testing.assert_close(state[indices.long()].double(), sr, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(state[[1, 3]], original[[1, 3]], rtol=0, atol=0)
