# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Triton ``trtllm::situ_and_mul`` tests.

Checks the fused SiTU activation op against the eager fp32
:class:`SituAndMul` reference:

* elementwise parity across shapes (masked tails, multi-block rows),
  ``beta`` / ``linear_beta`` settings (incl. ``linear_beta=None``), bf16
  in/out, tight tolerance;
* the ``KimiK3MLP(use_fused_activation=True)`` wiring matches the eager
  module bit-for-bit at the down_proj output tolerance;
* the fake/meta registration produces the right shape/dtype (graph
  tracing contract);
* the op is CUDA-graph-capturable (no host sync, no data-dependent
  control flow).
"""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_k3_moe._mlp import KimiK3MLP, SituAndMul

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

_BETAS = [
    (4.0, 25.0),  # Kimi K3 defaults
    (2.5, 7.0),  # asymmetric non-defaults
    (1.0, None),  # linear_beta disabled (identity up half)
]
_BETA_IDS = ["default", "asymmetric", "no_linear_beta"]


@requires_cuda
@pytest.mark.parametrize("situ_beta,situ_linear_beta", _BETAS, ids=_BETA_IDS)
@pytest.mark.parametrize(
    "num_tokens,two_d",
    [(1, 256), (5, 736), (128, 2048), (500, 4096), (33, 6144)],
    ids=lambda v: str(v),
)
def test_situ_and_mul_matches_eager_reference(num_tokens, two_d, situ_beta, situ_linear_beta):
    device = torch.device("cuda")
    torch.manual_seed(7)
    x = torch.randn(num_tokens, two_d, dtype=torch.bfloat16, device=device) * 4.0

    out = torch.ops.trtllm.situ_and_mul(x, situ_beta, situ_linear_beta)
    ref = SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)(x)

    assert out.shape == (num_tokens, two_d // 2)
    assert out.dtype == torch.bfloat16
    # Both paths compute in fp32 and round once to bf16; only sub-fp32-ulp
    # transcendental differences remain, well below one bf16 ulp of slack.
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)


@requires_cuda
def test_situ_and_mul_strided_rows():
    """Row-strided input (column slice) must honor x_stride."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    full = torch.randn(64, 1024, dtype=torch.bfloat16, device=device)
    x = full[:, :512]  # row stride 1024, last dim contiguous

    out = torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0)
    ref = SituAndMul(beta=4.0, linear_beta=25.0)(x)
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("situ_beta,situ_linear_beta", _BETAS, ids=_BETA_IDS)
def test_kimi_k3_mlp_fused_activation_matches_eager(situ_beta, situ_linear_beta):
    device = torch.device("cuda")
    hidden_size, intermediate_size = 512, 384
    torch.manual_seed(3)
    eager = KimiK3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.no_grad():
        for proj in (eager.gate_up_proj, eager.down_proj):
            proj.weight.copy_(torch.randn_like(proj.weight, dtype=torch.float32) * 0.05)
    fused = KimiK3MLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        use_fused_activation=True,
        dtype=torch.bfloat16,
        device=device,
    )
    fused.load_state_dict(eager.state_dict())

    x = torch.randn(64, hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    torch.testing.assert_close(fused(x), eager(x), rtol=1.6e-2, atol=1e-3)


def test_kimi_k3_mlp_rejects_fused_flag_with_custom_activation():
    with pytest.raises(ValueError, match="use_fused_activation"):
        KimiK3MLP(
            hidden_size=64,
            intermediate_size=32,
            activation=torch.nn.Identity(),
            use_fused_activation=True,
        )


def test_situ_and_mul_fake_registration():
    """Meta-device dispatch (graph tracing) must not launch the kernel."""
    x = torch.empty(16, 512, dtype=torch.bfloat16, device="meta")
    out = torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0)
    assert out.shape == (16, 256)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "meta"


@requires_cuda
def test_situ_and_mul_cuda_graph_capture():
    """The op must be capturable and correct under CUDA-graph replay."""
    device = torch.device("cuda")
    torch.manual_seed(5)
    x = torch.randn(32, 1024, dtype=torch.bfloat16, device=device)

    # Warm up on a side stream (JIT compile outside capture).
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.trtllm.situ_and_mul(x, 4.0, 25.0)

    x2 = torch.randn(32, 1024, dtype=torch.bfloat16, device=device)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    ref = SituAndMul(beta=4.0, linear_beta=25.0)(x2)
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-5)
