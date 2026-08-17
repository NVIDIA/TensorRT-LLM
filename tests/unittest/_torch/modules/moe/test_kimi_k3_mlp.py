# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 fused gate_up MLP tests.

The runtime dense / shared-expert MLP (``KimiK3MLP``) runs a single fused
``gate_up_proj`` GEMM whose weight is the row-concat of the HF checkpoint's
separate ``gate_proj`` / ``up_proj`` tensors (the same concat
``KimiLinearForCausalLM.load_weights`` performs). These tests check the
fused module against an unfused reference built from the split weights:

* fused ``gate_up_proj`` output matches ``two GEMMs + torch.cat`` +
  eager ``SituAndMul`` + ``down_proj`` for a decode-shaped (1 token) and
  a prefill-shaped (500 tokens) batch, with ``situ_linear_beta`` set and
  ``None`` (the two activation code paths);
* the row-concat convention is required: swapping the halves breaks
  the numerics (mutation control).
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.kimi_k3_moe._mlp import KimiK3MLP, SituAndMul

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


class _UnfusedKimiMLP(nn.Module):
    """HF ``KimiMLP`` layout: separate gate/up GEMMs + torch.cat."""

    def __init__(self, hidden_size, intermediate_size, situ_beta, situ_linear_beta, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)

    def forward(self, x):
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))


def _make_pair(hidden_size, intermediate_size, situ_beta, situ_linear_beta, device, seed=1234):
    """Unfused reference with random split weights + fused twin via row-concat."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    ref = _UnfusedKimiMLP(hidden_size, intermediate_size, situ_beta, situ_linear_beta, dtype).to(
        device
    )
    with torch.no_grad():
        for proj in (ref.gate_proj, ref.up_proj, ref.down_proj):
            proj.weight.copy_(torch.randn_like(proj.weight, dtype=torch.float32) * 0.05)

    fused = KimiK3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        # The same row-concat KimiLinearForCausalLM.load_weights performs.
        inter = intermediate_size
        fused.gate_up_proj.weight[:inter].copy_(ref.gate_proj.weight)
        fused.gate_up_proj.weight[inter:].copy_(ref.up_proj.weight)
        fused.down_proj.weight.copy_(ref.down_proj.weight)
    return fused, ref


@requires_cuda
@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [
        (4.0, 25.0),  # Kimi K3 defaults
        (1.0, None),  # linear_beta disabled (identity up half)
    ],
    ids=["default", "no_linear_beta"],
)
@pytest.mark.parametrize("num_tokens", [1, 500], ids=lambda n: f"tokens{n}")
def test_fused_gate_up_matches_unfused_reference(num_tokens, situ_beta, situ_linear_beta):
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    fused, ref = _make_pair(hidden_size, intermediate_size, situ_beta, situ_linear_beta, device)

    torch.manual_seed(7)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    out_fused = fused(x)
    out_ref = ref(x)

    assert out_fused.shape == out_ref.shape and out_fused.dtype == out_ref.dtype
    # Identical weights and identical eager activation; the only difference
    # is one [H, 2I] GEMM vs two [H, I] GEMMs (kernel-tactic level).
    torch.testing.assert_close(out_fused, out_ref, rtol=1.6e-2, atol=1e-3)


@requires_cuda
def test_gate_up_half_swap_mutation_breaks_accuracy():
    """Up-first packing must break the numerics (guards the gate-first
    row-concat convention in load_weights)."""
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    fused, ref = _make_pair(hidden_size, intermediate_size, 4.0, 25.0, device)
    with torch.no_grad():
        swapped = torch.cat(
            [
                fused.gate_up_proj.weight[intermediate_size:],
                fused.gate_up_proj.weight[:intermediate_size],
            ],
            dim=0,
        )
        fused.gate_up_proj.weight.copy_(swapped)

    torch.manual_seed(13)
    x = torch.randn(64, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    with pytest.raises(AssertionError):
        torch.testing.assert_close(fused(x), ref(x), rtol=1.6e-2, atol=1e-3)
