# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise oracle of the mx_gemv swigluoai prologue arithmetic (ops/mx_gemv/_swiglu_oai_bf16x2) against the
Triton silu_and_mul kernel the dense MLP runs today (modules/swiglu.py, alpha 1.702, limit 7, beta 1): every bf16 gate
bit pattern against special and random up values, and every up bit pattern against special and random gates."""

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops.mx_gemv.m3_mx_gemv_kernel import (  # noqa: E501
    SWIGLU_OAI_ALPHA,
    SWIGLU_OAI_LIMIT,
    _swiglu_oai_bf16x2,
)

assert torch.cuda.is_available(), "the swigluoai prologue oracle requires a CUDA device"

THREADS = 256


@cute.kernel
def _swiglu_words_kernel(gate: cute.Tensor, up: cute.Tensor, out: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * cutlass.Int32(THREADS) + tidx
    if i < n:
        out[i] = _swiglu_oai_bf16x2(gate[i], up[i])


@cute.jit
def _swiglu_words(gate: cute.Tensor, up: cute.Tensor, out: cute.Tensor, n: cutlass.Int32):
    _swiglu_words_kernel(gate, up, out, n).launch(
        grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1)
    )


_compiled = {}


def _prologue(gate_bf16: torch.Tensor, up_bf16: torch.Tensor) -> torch.Tensor:
    """The prologue helper over flat bf16 tensors of even length (pairs packed into int32 words)."""
    g = gate_bf16.contiguous().view(torch.int32)
    u = up_bf16.contiguous().view(torch.int32)
    out = torch.empty_like(g)
    args = [from_dlpack(t, assumed_align=4) for t in (g, u, out)]
    fn = _compiled.get("f")
    if fn is None:
        fn = _compiled["f"] = cute.compile(_swiglu_words, *args, cutlass.Int32(g.numel()))
    fn(*args, cutlass.Int32(g.numel()))
    return out.view(torch.bfloat16)


def _triton(gate_bf16: torch.Tensor, up_bf16: torch.Tensor) -> torch.Tensor:
    """The dense MLP's activation today: torch.ops.trtllm.silu_and_mul on one [1, 2d] row per call site layout."""
    x = torch.cat([gate_bf16.view(1, -1), up_bf16.view(1, -1)], dim=1)
    return torch.ops.trtllm.silu_and_mul(
        x, swiglu_limit=SWIGLU_OAI_LIMIT, swiglu_alpha=SWIGLU_OAI_ALPHA, swiglu_beta=1.0
    ).view(-1)


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
        _assert_bitwise(_prologue(gate, up), _triton(gate, up), f"up set {j}")


def test_every_up_pattern_is_bitwise_the_triton_kernel() -> None:
    up = _all_bf16()
    for j, gate in enumerate(_specials(up.numel(), 2)):
        _assert_bitwise(_prologue(gate, up), _triton(gate, up), f"gate set {j}")


@pytest.mark.parametrize("seed", range(4))
def test_random_pairs_are_bitwise_the_triton_kernel(seed: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(100 + seed)
    n = 1 << 22
    gate = (
        torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    up = (
        torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    _assert_bitwise(_prologue(gate, up), _triton(gate, up), f"seed {seed}")


# The GEMV with the activation in its prologue against today's two launches (Triton silu_and_mul, then the bf16 GEMV):
# bit for bit on the dense MLP's down projection at every decode row count.
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8 import (  # noqa: E402
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops.mx_gemv import (  # noqa: E402, E501
    op as gemv,
)


@pytest.mark.parametrize("m", range(1, modeling.DECODE_MAX_TOKENS + 1))
def test_down_gemv_with_the_activation_prologue_is_bitwise_two_launches(m: int) -> None:
    n, k = modeling.DENSE_DOWN
    g = torch.Generator(device="cuda").manual_seed(1000 + m)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    wq = wq.view(n, k)
    # A gate_up GEMV output's range, plus values past the clamps.
    gate_up = (3.0 * torch.randn(m, 2 * k, generator=g, device="cuda")).to(torch.bfloat16)
    gate_up[:, :8] = torch.tensor(
        [8.0, -8.0, 7.0, -7.0, 0.0, -0.0, 50.0, -50.0], dtype=torch.bfloat16
    )
    act = torch.ops.trtllm.silu_and_mul(
        gate_up, swiglu_limit=SWIGLU_OAI_LIMIT, swiglu_alpha=SWIGLU_OAI_ALPHA, swiglu_beta=1.0
    )
    ref = gemv.m3_mx_gemv_bf16(act.contiguous(), wq, wsf, use_pdl=True)
    out = gemv.m3_mx_gemv_swiglu(gate_up, wq, wsf, use_pdl=True)
    assert ref is not None and out is not None, (
        "the dense down projection must be served at every decode row count"
    )
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
