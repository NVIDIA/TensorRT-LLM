# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise oracle of the dense MLP's swigluoai C++ op (ops/gemm/m3_dense_swiglu,
trtllm::minimax_m3_sm107_tp4_swiglu_oai) against the Triton silu_and_mul kernel the dense MLP runs today
(modules/swiglu.py at limit 7, alpha 1.702, beta 1): every bf16 gate bit pattern against special and random up values,
every up bit pattern against special and random gates, random pairs, and the dense MLP's shapes at every decode row
count, eager and replayed from a CUDA graph, with and without the programmatic launch."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.silu_and_mul
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.gemm.m3_dense_swiglu import (  # noqa: E501
    ALPHA,
    BETA,
    LIMIT,
    m3_dense_swiglu,
)

assert torch.cuda.is_available(), "the dense swigluoai op requires a CUDA device"
if not hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp4_swiglu_oai"):
    pytest.skip(
        "needs the TP4 op library with minimax_m3_sm107_tp4_swiglu_oai", allow_module_level=True
    )

# The TP4 dense MLP's activation width (gate_up 3072 -> d 1536).
DENSE_D = dispatch.DENSE_DOWN[1]


def _triton(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.trtllm.silu_and_mul(
        x, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA
    )


def _op(x: torch.Tensor, pdl: bool = True) -> torch.Tensor:
    out = m3_dense_swiglu(x, LIMIT, ALPHA, BETA, pdl)
    assert out is not None, "the op declined a contiguous bf16 [rows, 2d] input"
    return out


def _row(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.cat([gate.view(1, -1), up.view(1, -1)], dim=1).contiguous()


def _all_bf16() -> torch.Tensor:
    return (
        torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    )


def _specials(n: int, seed: int) -> list:
    g = torch.Generator(device="cuda").manual_seed(seed)
    vals = [0.0, -0.0, 1.0, -1.0, 6.99, 7.0, 7.01, -7.0, -7.01, 1e-38, -1e-38, 1e-40, 88.0, -88.0, 3e38, -3e38,
            float("inf"), float("-inf"), float("nan")]  # fmt: skip
    rows = [torch.full((n,), v, dtype=torch.bfloat16, device="cuda") for v in vals]
    for scale in (0.05, 1.0, 4.0, 30.0):
        rows.append((scale * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16))
    rows.append(
        torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    return rows


def _assert_bitwise(got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    bad = (got.view(torch.int16) != ref.view(torch.int16)).nonzero().flatten()
    assert bad.numel() == 0, f"{what}: {bad.numel()} mismatches, first at {bad[:4].tolist()}"


def test_every_gate_pattern_is_bitwise_the_triton_kernel() -> None:
    gate = _all_bf16()
    for j, up in enumerate(_specials(gate.numel(), 1)):
        x = _row(gate, up)
        _assert_bitwise(_op(x), _triton(x), f"up set {j}")


def test_every_up_pattern_is_bitwise_the_triton_kernel() -> None:
    up = _all_bf16()
    for j, gate in enumerate(_specials(up.numel(), 2)):
        x = _row(gate, up)
        _assert_bitwise(_op(x), _triton(x), f"gate set {j}")


@pytest.mark.parametrize("seed", range(4))
def test_random_pairs_are_bitwise_the_triton_kernel(seed: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    n = 1 << 22

    def pattern() -> torch.Tensor:
        bits = torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        return bits.to(torch.int16).view(torch.bfloat16)

    x = _row(pattern(), pattern())
    _assert_bitwise(_op(x), _triton(x), f"seed {seed}")


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "plain"])
@pytest.mark.parametrize("rows", range(1, dispatch.DECODE_MAX_TOKENS + 1))
def test_dense_shape_eager_and_graph_are_bitwise_the_triton_kernel(rows: int, pdl: bool) -> None:
    """The TP4 dense MLP's gate_up rows (a GEMV output's range plus values past the clamps), eager and replayed from a
    CUDA graph after fresh rows are written in place, against the Triton kernel on the same rows."""
    g = torch.Generator(device="cuda").manual_seed(2000 + rows)
    x = (3.0 * torch.randn(rows, 2 * DENSE_D, generator=g, device="cuda")).to(torch.bfloat16)
    x[:, :8] = torch.tensor([8.0, -8.0, 7.0, -7.0, 0.0, -0.0, 50.0, -50.0], dtype=torch.bfloat16)
    x[:, DENSE_D : DENSE_D + 8] = x[:, :8]
    _assert_bitwise(_op(x, pdl), _triton(x), f"eager rows {rows}")
    out = torch.empty(rows, DENSE_D, dtype=torch.bfloat16, device="cuda")
    _op(x, pdl)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(_op(x, pdl))
    x.copy_((3.0 * torch.randn(rows, 2 * DENSE_D, generator=g, device="cuda")).to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(out, _triton(x), f"graph rows {rows}")


def test_the_op_declines_other_parameters_and_layouts() -> None:
    x = torch.zeros(2, 2 * DENSE_D, dtype=torch.bfloat16, device="cuda")
    assert m3_dense_swiglu(x, 8.0, ALPHA, BETA, True) is None
    assert m3_dense_swiglu(x, LIMIT, 1.0, BETA, True) is None
    assert m3_dense_swiglu(x, LIMIT, ALPHA, 0.0, True) is None
    assert m3_dense_swiglu(x.float(), LIMIT, ALPHA, BETA, True) is None
    assert m3_dense_swiglu(x[:, : 2 * DENSE_D - 2], LIMIT, ALPHA, BETA, True) is None


@pytest.mark.parametrize("cell", m3_dense_swiglu.CELLS, ids=lambda c: c.why)
def test_every_cell_is_bitwise_the_triton_kernel(cell) -> None:
    """Each declared cell (the dense MLP's activation at its decode rows) through the op, bit for bit the Triton
    kernel on gate_up rows in a GEMV output's range plus values past the clamps."""
    rows, d = cell.spec["rows"], cell.spec["d"]
    g = torch.Generator(device="cuda").manual_seed(3000 + rows)
    x = (3.0 * torch.randn(rows, 2 * d, generator=g, device="cuda")).to(torch.bfloat16)
    x[:, :8] = torch.tensor([8.0, -8.0, 7.0, -7.0, 0.0, -0.0, 50.0, -50.0], dtype=torch.bfloat16)
    x[:, d : d + 8] = x[:, :8]
    _assert_bitwise(_op(x), _triton(x), cell.why)
