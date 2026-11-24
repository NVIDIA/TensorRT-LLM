# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 situ_and_mul entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.situ_and_mul import (
    situ_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31SituMLP,
)
from tensorrt_llm._torch.modules.situ import SituAndMul as EngineSituAndMul

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)

BETA, LINEAR_BETA = 4.0, 25.0
# Both sides compute in fp32 and round once to bf16; libdevice tanh differs from torch.tanh in the
# last fp32 bits, which moves the rounded bf16 result by at most one ulp.
RTOL = 2 * torch.finfo(torch.bfloat16).eps


@pytest.mark.parametrize("cell", situ_and_mul.CELLS, ids=[c.why[:48] for c in situ_and_mul.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(situ_and_mul)
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = spec["shape"]
    # The scale of a gate/up projection output: well inside the soft caps, with a tail reaching them.
    x = (torch.randn(rows, width, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    if spec.get("saturated"):
        half = width // 2
        for row, (gate, up) in enumerate(((1e4, 1e4), (1e4, -1e4), (-1e4, 1e4), (0.0, 1e4))):
            x[row, :half], x[row, half:] = gate, up
    before = x.clone()
    out = situ_and_mul(x, spec["beta"], spec["linear_beta"])
    assert out.shape == (rows, width // 2) and out.dtype == x.dtype and out.is_contiguous()
    assert torch.equal(x, before), "x was mutated"
    ref = situ_and_mul.reference(x, spec["beta"], spec["linear_beta"])
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=1e-5)
    if spec.get("saturated"):
        expected = torch.tensor(
            [BETA * LINEAR_BETA, -BETA * LINEAR_BETA, 0.0, 0.0], device="cuda"
        ).to(torch.bfloat16)
        torch.testing.assert_close(out, expected.view(4, 1).expand(4, width // 2))


def test_row_strided_input() -> None:
    """Callers hand the op a dense [M, 2d] tensor; a row-strided view must read the same rows."""
    x = torch.randn(64, 2 * 768 + 128, device="cuda").to(torch.bfloat16)
    view = x[:, : 2 * 768]
    assert not view.is_contiguous()
    torch.testing.assert_close(
        situ_and_mul(view, BETA, LINEAR_BETA), situ_and_mul(view.contiguous(), BETA, LINEAR_BETA)
    )


def test_cuda_graph_replay() -> None:
    x = torch.randn(8, 2 * 1536, device="cuda").to(torch.bfloat16)
    situ_and_mul(x, BETA, LINEAR_BETA)  # Triton compiles on the first call, outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = situ_and_mul(x, BETA, LINEAR_BETA)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(out, situ_and_mul(x, BETA, LINEAR_BETA))


@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None])
def test_mlp_activation_runs_the_entry(linear_beta) -> None:
    """The family MLP's activation is the entry on the input's rows: the engine ``SituAndMul`` module's fused
    branch's bits (the same kernel), leading dims kept."""
    from tensorrt_llm.mapping import Mapping

    mlp = K31SituMLP(
        7168,
        768,
        dtype=torch.bfloat16,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        beta=BETA,
        linear_beta=linear_beta,
        reduce_output=False,
    )
    engine = EngineSituAndMul(beta=BETA, linear_beta=linear_beta, use_fused_activation=True)
    x = (torch.randn(3, 5, 2 * 768, device="cuda") * 4.0).to(torch.bfloat16)
    out = mlp.activation(x)
    assert out.shape == (3, 5, 768)
    assert torch.equal(out, situ_and_mul(x.reshape(-1, 2 * 768), BETA, linear_beta).view(3, 5, 768))
    assert torch.equal(out, engine(x))
