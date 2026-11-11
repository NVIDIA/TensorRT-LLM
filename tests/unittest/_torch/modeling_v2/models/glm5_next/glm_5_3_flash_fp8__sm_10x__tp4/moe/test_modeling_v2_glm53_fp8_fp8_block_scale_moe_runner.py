# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_block_scale_moe_runner operation of the glm5_next FP8 target.

The entry owns the contract: `CELLS`, `reference`, `is_valid`. This file turns a cell into
operands -- GLM-5.3-Flash's 288-expert stack drawn directly in e4m3 / fp32, so both sides read
bit-identical values, and router logits whose top-8 cannot tie -- and holds what no cell
carries: that the cell data can see the correction bias, the clamp and the unbiased combine
weights; the transposed activation-scale hazard at T == H/128; and the guards.
"""

import zlib
from collections.abc import Callable, Iterator

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    BLOCK,
    ROUTING_DEEPSEEK_V3,
    ROUTING_RENORMALIZE,
    Fp8BlockScaleMoeRunner,
    route_deepseek_v3,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="fp8_block_scale_moe_runner is certified on sm_100"
)

DEV = "cuda"
ULP = 2.0**-8
GLM = Fp8BlockScaleMoeRunner.GLM53_FLASH_TP4


def _ulp(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max, RMS) deviation in bf16 ulp of each reference row's largest magnitude."""
    o, r = out.float(), ref.float()
    d = (o - r) / (r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * ULP)
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp element-wise, 0.75 ulp RMS, both in bf16 ulp of each row's max.

    Each element is normalized by its reference row's largest magnitude, `d = (out - ref) /
    (rowmax * 2^-8)`, and gated on `max|d|` and on `sqrt(mean(d^2))`.

    Kernel and reference read bit-identical operands, pick the same experts (the routing below
    has no near-ties) and model every rounding the kernel makes: the e4m3 requantization of
    the FC1 output and of the SwiGLU product, the bf16 FC2 rows and the bf16 combine weights.
    What is left is the order of the fp32 sums, which now and then lands an FC1 output or a
    SwiGLU product on the other side of an e4m3 rounding midpoint -- one e4m3 step in one
    intermediate element, carried through FC2 into its row -- and the exponentials of the
    SiLU and the routing sigmoid, a few fp32 ulp, which can move a bf16 combine weight by one
    ulp. Both grow the maximum with the row count and leave the RMS well under one ulp. The
    same gate holds the pre-routed path of this kernel at another geometry (H 2560, I 640,
    512 experts, sm_103) with margin, worst seen 10.0 / 0.34; it has not been measured at
    GLM-5.3-Flash's geometry on sm_100.

    Wrong references are far outside it: `test_cell_data_see_the_bias_the_clamp_and_the_weights`
    asserts that dropping the correction bias from the selection, dropping the clamp, or
    weighting by the biased scores each fails this gate on a certified cell.
    """
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    elt, rms = _ulp(out, ref)
    assert elt <= 16.0, f"{elt:.2f} ulp element-wise > 16"
    assert rms <= 0.75, f"{rms:.3f} ulp RMS > 0.75"


# ── operands ──────────────────────────────────────────────────────────────


def _draw_stack(hidden: int, inter: int, experts: int, seed: int) -> tuple[torch.Tensor, ...]:
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact.

    Codes ~ N(0, 1) in e4m3. FC1 scales in [0.04, 0.08) put FC1 outputs at about 8 RMS from
    the activations below, so GLM's limit of 10 clamps a tenth of the gates and a fifth of the
    ups; FC2 scales in [0.008, 0.016) bring the layer back to O(10). Drawn 16 experts at a time
    to cap the fp32 transient.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    w1 = torch.empty(experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
    w2 = torch.empty(experts, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
    for lo in range(0, experts, 16):
        hi = min(lo + 16, experts)
        w1[lo:hi] = torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=g)
        w2[lo:hi] = torch.randn(hi - lo, hidden, inter, device=DEV, generator=g)
    w1s = 0.04 + 0.04 * torch.rand(
        experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV, generator=g
    )
    w2s = 0.008 + 0.008 * torch.rand(
        experts, hidden // BLOCK, inter // BLOCK, device=DEV, generator=g
    )
    return w1, w1s, w2, w2s


@pytest.fixture(scope="module")
def stack() -> Iterator[Callable[..., tuple[torch.Tensor, ...]]]:
    """The expert stack for a geometry, drawn once per module and released at teardown.
    GLM's is 1.8 GB of e4m3 codes and every GLM cell reads it."""
    built: dict = {}

    def get(hidden: int, inter: int, experts: int) -> tuple[torch.Tensor, ...]:
        key = (hidden, inter, experts)
        if key not in built:
            built[key] = _draw_stack(hidden, inter, experts, seed=hidden + inter + experts)
        return built[key]

    try:
        yield get
    finally:
        built.clear()


def _routing(tokens: int, experts: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 router logits and correction bias whose top-8 cannot tie.

    Each row's biased scores `sigmoid(logit) + bias` are a random permutation of an evenly
    spaced ladder from 0.85 down to 0.15: neighbouring ranks sit 0.7 / (experts - 1), 2.4e-3
    at 288 experts, apart -- four orders of magnitude above the fp32 error of the sigmoid and
    the bias add -- so the kernel and the reference pick the same experts. The bias, uniform
    in +-0.1, moves each unbiased score `ladder - bias` by up to 40 ladder steps, so a
    selection that ignored it, or weights that kept it, would differ.
    """
    bias = (torch.rand(experts, generator=g, device=DEV) - 0.5) * 0.2
    ladder = torch.linspace(0.85, 0.15, experts, device=DEV)
    order = torch.rand(tokens, experts, generator=g, device=DEV).argsort(dim=-1)
    choice = torch.empty(tokens, experts, device=DEV).scatter_(1, order, ladder.repeat(tokens, 1))
    score = choice - bias
    logits = torch.log(score) - torch.log1p(-score)
    return logits.contiguous(), bias.contiguous()


def _args(spec: dict, stack: Callable, seed: int) -> dict:
    """The op's arguments by name, as the TRTLLM MoE backend passes them for GLM-5.3-Flash:
    router logits and bias in, routing in the kernel."""
    hidden, inter, experts = spec["hidden"], spec["inter"], spec["num_experts"]
    w1, w1s, w2, w2s = stack(hidden, inter, experts)
    g = torch.Generator(device=DEV).manual_seed(seed)
    tokens = spec["tokens"]
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the K-block-major layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    logits, bias = _routing(tokens, experts, g)
    return dict(
        routing_logits=logits,
        routing_bias=bias,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        num_experts=experts,
        top_k=spec["top_k"],
        num_fused_shared_experts=0,
        n_group=spec["n_group"],
        topk_group=spec["topk_group"],
        intermediate_size=inter,
        local_expert_offset=spec["local_expert_offset"],
        local_num_experts=spec["local_num_experts"],
        routed_scaling_factor=spec["routed_scaling_factor"],
        routing_method_type=ROUTING_DEEPSEEK_V3,
        gemm1_clamp_limit=spec["gemm1_clamp_limit"],
    )


def _seed(cell: Cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


# ── the certified cells ───────────────────────────────────────────────────

# The shapes the target calls this operation with, on top of the operation's own `CELLS`.
#
# GLM-5.3-Flash at TP4 hands this op the generation steps its 4-row decode path does not take:
# MTP-3 verifies of 4 rows per request, 2 to 64 requests, and any step of 5 rows or more.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="5 rows: the smallest step past the 4-row decode path; small-token routing",
        spec=dict(tokens=5, **GLM),
    ),
    Cell(
        why="MTP-3 verify of 2 requests: 8 rows, the small-token routing kernel's last count",
        spec=dict(tokens=8, **GLM),
    ),
    Cell(
        why="MTP-3 verify of 8 requests: 32 rows == H/128, the activation scales are square",
        spec=dict(tokens=32, **GLM),
    ),
    Cell(
        why="MTP-3 verify of 64 requests: 256 rows, the single-cluster routing kernel's last",
        spec=dict(tokens=256, **GLM),
    ),
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying, stack):  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    args = _args(cell.spec, stack, seed=_seed(cell))
    out = op(**args)
    assert out.shape == args["hidden_states"].shape and out.dtype == torch.bfloat16
    ref = op.reference(**args)
    elt, rms = _ulp(out, ref)
    print(f"  {cell.why[:44]}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _compare(out, ref)


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_cell_data_see_the_bias_the_clamp_and_the_weights(stack):
    """The cells certify the bias-steered selection, the clamp and the unbiased weights only
    if their data can tell them apart: each wrong reading fails the gate on a certified cell.
    """
    args = _args(dict(tokens=8, **GLM), stack, seed=101)
    out = op(**args)
    _compare(out, op.reference(**args))

    ids, _ = route_deepseek_v3(
        args["routing_logits"], args["routing_bias"], GLM["top_k"], GLM["routed_scaling_factor"]
    )
    choice = 0.5 * torch.tanh(0.5 * args["routing_logits"]) + 0.5 + args["routing_bias"]
    biased = choice.gather(1, ids)
    biased = biased * GLM["routed_scaling_factor"] / biased.sum(dim=-1, keepdim=True)
    pre_routed = dict(args, routing_logits=None, routing_bias=None)
    wrong = {
        "selection without the bias": dict(
            args, routing_bias=torch.zeros_like(args["routing_bias"])
        ),
        "no clamp": dict(args, gemm1_clamp_limit=None),
        "weights from the biased scores": dict(
            pre_routed, topk_ids=ids.to(torch.int32), topk_weights=biased.to(torch.bfloat16)
        ),
    }
    for name, wrong_args in wrong.items():
        elt, rms = _ulp(out, op.reference(**wrong_args))
        print(f"  {name}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
        with pytest.raises(AssertionError):
            _compare(out, op.reference(**wrong_args))


def test_a_transposed_activation_scale_is_silently_wrong(stack):
    """At T == H/128 (32 rows at H 4096) the token-major scale buffer has the K-block-major
    shape: the op takes it, `is_valid` cannot see it, and the answer is wrong. This is why the
    entry says to pass `fp8_quantize_1x128`'s scales through unchanged."""
    args = _args(dict(tokens=32, **GLM), stack, seed=103)
    scale = args["hidden_states_scale"]
    assert scale.shape[0] == scale.shape[1]
    transposed = dict(args, hidden_states_scale=scale.t().contiguous())
    with validating(op):
        out = op(**transposed)
    with pytest.raises(AssertionError):
        _compare(out, op.reference(**args))


# ── the guards ────────────────────────────────────────────────────────────


def _small_args() -> dict:
    """A tiny layer for the guards, which refuse before the kernel runs."""
    g = torch.Generator(device=DEV).manual_seed(7)
    tokens, hidden, inter, experts = 4, 256, 128, 8
    return dict(
        routing_logits=torch.randn(tokens, experts, device=DEV, generator=g),
        routing_bias=torch.zeros(experts, device=DEV),
        hidden_states=torch.randn(tokens, hidden, device=DEV, generator=g).to(torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(hidden // BLOCK, tokens, device=DEV),
        gemm1_weights=torch.zeros(experts, 2 * inter, hidden, device=DEV).to(torch.float8_e4m3fn),
        gemm1_weights_scale=torch.ones(experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV),
        gemm2_weights=torch.zeros(experts, hidden, inter, device=DEV).to(torch.float8_e4m3fn),
        gemm2_weights_scale=torch.ones(experts, hidden // BLOCK, inter // BLOCK, device=DEV),
        num_experts=experts,
        top_k=2,
        num_fused_shared_experts=0,
        n_group=1,
        topk_group=1,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=1.0,
        routing_method_type=ROUTING_DEEPSEEK_V3,
        gemm1_clamp_limit=10.0,
    )


def test_guard_refuses_deepseek_routing_without_a_bias():
    """The routing kernels read the bias of every expert with no null check: refused before
    the kernel could fault the context."""
    with pytest.raises(AssertionError, match="faults on None"):
        with validating(op):
            op(**dict(_small_args(), routing_bias=None))


def test_guard_refuses_a_bias_the_routing_method_ignores():
    with pytest.raises(AssertionError, match="ignores routing_bias"):
        with validating(op):
            op(**dict(_small_args(), routing_method_type=ROUTING_RENORMALIZE))


def test_guard_refuses_logits_alongside_topk_ids():
    a = _small_args()
    ids = torch.tensor([[0, 1]] * 4, dtype=torch.int32, device=DEV)
    wts = torch.full((4, 2), 0.5, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(AssertionError, match="ignored when topk_ids"):
        with validating(op):
            op(**dict(a, routing_bias=None, topk_ids=ids, topk_weights=wts))


def test_guard_refuses_a_strided_hidden_states():
    a = _small_args()
    wide = torch.zeros(4, 512, device=DEV).to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="hidden_states must be contiguous"):
        with validating(op):
            op(**dict(a, hidden_states=wide[:, ::2]))


def test_guard_refuses_a_window_outside_the_routing_space():
    with pytest.raises(AssertionError, match="outside the routing space"):
        with validating(op):
            op(**dict(_small_args(), local_expert_offset=8, local_num_experts=8))
