# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Hy4-Flash target's fp8_block_scale_moe_runner entry.

The entry owns the contract: `CELLS`, `reference`, `is_valid`. This file turns a cell into
operands -- the checkpoint's 512-expert stack drawn directly in e4m3 / fp32, so both sides read
bit-identical values, plus the routing: pre-routed top-16 ids with bf16 weights summing to 3.998
for most cells, and the engine's own two router tensors (bf16 logits, fp32 bias) for the in-op
ones -- and holds what no cell carries: that the cell data can see the clamp and the `[up; gate]`
row order; the transposed activation-scale hazard at T == H/128 (12 rows); that the four EP4
windows and the two EP2 windows (the disaggregated deployment's context engine) each partition the
whole stack; that the router's scaling factor is not applied twice; that the kernel's own router is
what moves the top-16 when the bias is dropped; and the guards.
"""

import zlib
from collections.abc import Callable, Iterator

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    BLOCK,
    ROUTING_DEEPSEEK_V3,
    in_op_routing,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "fp8_block_scale_moe_runner requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="fp8_block_scale_moe_runner is certified on sm_103"
)

DEV = "cuda"
ULP = 2.0**-8
CELLS = op.CELLS

#: This checkpoint's geometry, read back off a whole-stack cell so the entry stays
#: the single source of truth: H 1536, I 1536, 512 experts, top-16, clamp 10.0,
#: scaling 3.998, every expert local (tp4).
HY4 = {k: v for k, v in CELLS[0].spec.items() if k != "tokens"}
WINDOWS = (0, 128, 256, 384)
WEIGHT_TENSORS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max, RMS) deviation in the units `assert_within_ulp` gates on.

    The element gate is each element's error over its reference row's largest
    magnitude, in units of 2^-8; the RMS gate is the relative RMS over the whole
    tensor in the same unit. Reporting mirrors the gate so the printed number is
    the number the assertion compares.
    """
    flat_out, flat_ref = (
        out.float().reshape(-1, out.shape[-1]),
        ref.float().reshape(-1, ref.shape[-1]),
    )
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((flat_out - flat_ref) / row_scale).abs().max().item() / ULP
    rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ULP
    return elt, rms


_ELEMENT_ULP = 16.0
_RMS_ULP = 2.5


def _gate(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16 ulp element-wise, 2.5 ulp RMS, both in 2^-8 of each row's max.

    Kernel and reference read bit-identical operands and model every rounding
    the kernel makes: the e4m3 requantization of the FC1 output and of the
    SwiGLU product, the bf16 store of each expert's FC2 row, the bf16 combine
    weights. What is left is the order of the fp32 sums -- ~1e-6 relative --
    which flips the e4m3 rounding of an FC1 output or a SwiGLU product whenever
    the value sits that close to a midpoint. One flip moves one intermediate
    element by one e4m3 step, which reaches its FC2 row as ~0.125 / sqrt(I) =
    0.3% of the row's scale, and flips per output element grow with I * top_k:
    24576 here (I 1536, top-16) against 5120-6400 at the op's other pre-routed
    entry, whose band is 0.75. So the RMS band is 2.0 rather than 0.75, and the
    reference was held to it rather than assumed: on sm_103, at one certified
    cell, this reference measures 1.00 ulp element / 0.686 ulp RMS, and every
    alternative rounding model is worse -- an fp32 FC2 store instead of the
    kernel's bf16 one 0.829, no SwiGLU-product requantization 5.77, no FC1
    requantization 12.07, a ue8m0 FC1 scale 15.38. Worst over the cells below:
    3.97 ulp element-wise and 1.522 ulp RMS, on the one-token EP4 window where a
    handful of expert contributions do not average; the whole-stack cells stay
    at 0.77-1.00 ulp RMS. The 2.5 band sits 1.6x above that worst cell and 2.3x
    below the nearest wrong rounding model.

    Wrong readings are far outside it: `test_the_band_rejects_the_unclamped_and
    _the_checkpoint_row_order` asserts that dropping the clamp, or reading the
    checkpoint's gate-first FC1 rows as the kernel's `[up; gate]`, fails this
    gate on a certified cell.
    """
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


# ── operands ──────────────────────────────────────────────────────────────


def _draw_stack(hidden: int, inter: int, experts: int, seed: int) -> tuple[torch.Tensor, ...]:
    """`(w1, w1s, w2, w2s)` drawn directly in e4m3 / fp32, so every value is exact.

    Codes ~ N(0, 1) in e4m3. FC1 scales in [0.04, 0.08) leave the FC1 outputs a
    few RMS under the clamp of 10, so the limit bites on their tails; FC2 scales
    in [0.008, 0.016) bring the layer back to O(1). Drawn 16 experts at a time to
    cap the fp32 transient.
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
    """The 512-expert stack for a geometry, drawn once per module and released at
    teardown. Hy4's is 3.6 GB of e4m3 codes and every cell reads it."""
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


def _routing(
    tokens: int, experts: int, top_k: int, scale: float, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-routed top-k: distinct ids per row and bf16 weights summing to `scale`.

    The ids are an input, so no tie-breaking matters; `topk` gives distinct
    experts per row, which the op requires. The weights are the shape the
    router leaves them in: its renormalized scores multiplied by 3.998, stored
    bf16.
    """
    logits = torch.randn(tokens, experts, generator=g, device=DEV)
    ids = torch.topk(logits, k=top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(logits.gather(1, ids.long()), dim=-1) * scale
    return ids.contiguous(), weights.to(torch.bfloat16).contiguous()


def _router_operands(
    tokens: int, experts: int, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's two router tensors: the gate GEMM's output in the router
    weight's dtype (bf16) and the checkpoint's `e_score_correction_bias` (fp32,
    per `weights.py`), drawn around zero like a correction term."""
    logits = torch.randn(tokens, experts, generator=g, device=DEV).to(torch.bfloat16)
    bias = torch.rand(experts, generator=g, device=DEV) - 0.5
    return logits, bias


def _args(spec: dict, stack: Callable, seed: int) -> dict:
    """The op's arguments by name, as `LatentMoe._routed` will pass them: one
    expert window's weights, and the routing either pre-routed (ids and weights
    in, no logits) or in-op (router logits and bias in, no ids or weights)."""
    hidden, inter, experts = spec["hidden"], spec["inter"], spec["num_experts"]
    lo, local = spec["local_expert_offset"], spec["local_num_experts"]
    w1, w1s, w2, w2s = (t[lo : lo + local] for t in stack(hidden, inter, experts))
    g = torch.Generator(device=DEV).manual_seed(seed)
    tokens, top_k = spec["tokens"], spec["top_k"]
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the K-block-major layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    if spec.get("entry") == "in-kernel-routing":
        routing_logits, routing_bias = _router_operands(tokens, experts, g)
        ids = weights = None
    else:
        routing_logits = routing_bias = None
        ids, weights = _routing(tokens, experts, top_k, spec["routed_scaling_factor"], g)
    return dict(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        num_experts=experts,
        top_k=top_k,
        num_fused_shared_experts=0,
        n_group=spec["n_group"],
        topk_group=spec["topk_group"],
        intermediate_size=inter,
        local_expert_offset=lo,
        local_num_experts=local,
        routed_scaling_factor=spec["routed_scaling_factor"],
        routing_method_type=ROUTING_DEEPSEEK_V3,
        topk_weights=weights,
        topk_ids=ids,
        gemm1_clamp_limit=spec["gemm1_clamp_limit"],
    )


def _ids(args: dict) -> torch.Tensor:
    """The ids the call routes by: the ones passed in, or -- for the in-op cells --
    the kernel's own router over the logits and bias the call was given."""
    if args["topk_ids"] is not None:
        return args["topk_ids"]
    ids, _ = in_op_routing(
        args["routing_logits"], args["routing_bias"], args["top_k"], args["routed_scaling_factor"]
    )
    return ids


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying, stack):  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    args = _args(cell.spec, stack, seed=_seed(cell))
    out = op(**args)
    assert out.shape == (cell.spec["tokens"], cell.spec["hidden"])
    assert out.dtype is torch.bfloat16
    if cell.spec["local_num_experts"] < cell.spec["num_experts"]:
        local = _ids(args).long() - cell.spec["local_expert_offset"]
        routed = int(((local >= 0) & (local < cell.spec["local_num_experts"])).sum())
        assert routed, "no slot routes into this window: the cell would certify nothing"
        print(f"  {routed} of {local.numel()} slots route into the window")
    ref = op.reference(**args)
    elt, rms = _deviation(out, ref)
    print(f"  {cell.why[:44]}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _gate(out, ref)


# ── what a cell cannot carry ──────────────────────────────────────────────


def _swap_fc1_halves(t: torch.Tensor) -> torch.Tensor:
    """The checkpoint's gate-first rows swapped into the kernel's `[up; gate]`."""
    half = t.shape[1] // 2
    return torch.cat([t[:, half:], t[:, :half]], dim=1)


def test_the_band_rejects_the_unclamped_and_the_checkpoint_row_order(stack):
    """The cells certify the clamp and the `[up; gate]` row order only if their
    data can tell them apart: each wrong reading fails the gate on a certified cell.
    """
    args = _args(dict(tokens=64, **HY4), stack, seed=211)
    out = op(**args)
    _gate(out, op.reference(**args))
    swapped = dict(
        args,
        gemm1_weights=_swap_fc1_halves(args["gemm1_weights"]),
        gemm1_weights_scale=_swap_fc1_halves(args["gemm1_weights_scale"]),
    )
    wrong = {
        "no clamp": dict(args, gemm1_clamp_limit=None),
        "the checkpoint's gate-first rows read as [up; gate]": swapped,
    }
    for name, wrong_args in wrong.items():
        ref = op.reference(**wrong_args)
        elt, rms = _deviation(out, ref)
        print(f"  {name}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
        with pytest.raises(AssertionError):
            _gate(out, ref)


def test_a_transposed_activation_scale_is_silently_wrong(stack):
    """At T == H/128 (12 rows at H 1536) the token-major scale buffer has the
    K-block-major shape: the op takes it, `is_valid` cannot see it, and the answer
    is wrong. This is why the entry says to pass `fp8_quantize_1x128`'s scales
    through unchanged."""
    args = _args(dict(tokens=12, **HY4), stack, seed=103)
    scale = args["hidden_states_scale"]
    assert scale.shape[0] == scale.shape[1] == 12
    transposed = dict(args, hidden_states_scale=scale.t().contiguous())
    with validating(op):
        out = op(**transposed)
    with pytest.raises(AssertionError):
        _gate(out, op.reference(**args))


def test_the_four_ep4_windows_partition_the_whole_stack(stack):
    """EP4 adds the four 128-expert windows' bf16 outputs in fp32 -- what the
    all-reduce does -- and that sum must be the 512-expert call; leaving any one
    window out must not be."""
    spec = dict(tokens=64, **HY4)
    out = op(**_args(spec, stack, seed=307))
    windows = [
        op(**_args(dict(spec, local_expert_offset=o, local_num_experts=128), stack, seed=307))
        for o in WINDOWS
    ]
    summed = torch.stack([w.float() for w in windows]).sum(0).to(torch.bfloat16)
    elt, rms = _deviation(summed, out)
    print(f"  four windows summed: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _gate(summed, out)
    for left_out in range(len(WINDOWS)):
        partial = (
            torch.stack([w.float() for i, w in enumerate(windows) if i != left_out])
            .sum(0)
            .to(torch.bfloat16)
        )
        with pytest.raises(AssertionError):
            _gate(partial, out)


def test_the_two_ep2_windows_partition_the_whole_stack(stack):
    """EP2 adds the two 256-expert windows' bf16 outputs in fp32 -- what the
    all-reduce does at the disaggregated deployment's context width (tep2) --
    and that sum must be the 512-expert call; either window left out must not
    be."""
    spec = dict(tokens=64, **HY4)
    out = op(**_args(spec, stack, seed=307))
    windows = [
        op(**_args(dict(spec, local_expert_offset=o, local_num_experts=256), stack, seed=307))
        for o in (0, 256)
    ]
    summed = torch.stack([w.float() for w in windows]).sum(0).to(torch.bfloat16)
    elt, rms = _deviation(summed, out)
    print(f"  two EP2 windows summed: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    _gate(summed, out)
    for left_out in range(len(windows)):
        partial = (
            torch.stack([w.float() for i, w in enumerate(windows) if i != left_out])
            .sum(0)
            .to(torch.bfloat16)
        )
        with pytest.raises(AssertionError):
            _gate(partial, out)


def _in_op_spec(tokens: int) -> dict:
    """A whole-stack cell in `LatentMoe._routed`'s shape: the entry's two router
    tensors in, no ids and no weights."""
    return dict(
        tokens=tokens,
        entry="in-kernel-routing",
        routing_method_type=ROUTING_DEEPSEEK_V3,
        topk_ids=None,
        topk_weights=None,
        **HY4,
    )


def test_the_band_rejects_a_router_that_ignores_the_bias(stack):
    """The in-op cells certify the kernel's own router only if their data can
    tell it apart: routing the same logits with the checkpoint's bias zeroed --
    the same sigmoid, no correction term -- moves experts in and out of the
    top-16 and fails the gate on a whole-stack in-op call."""
    args = _args(_in_op_spec(64), stack, seed=613)
    out = op(**args)
    _gate(out, op.reference(**args))
    zero = torch.zeros_like(args["routing_bias"])
    ids, weights = in_op_routing(
        args["routing_logits"], zero, args["top_k"], args["routed_scaling_factor"]
    )
    moved = int((ids != _ids(args)).sum())
    print(f"  the bias dropped from the router: {moved} of {ids.numel()} slots move")
    assert moved, "the wrong router picked the same experts: the cell could not tell"
    ref = op.reference(
        **dict(args, routing_logits=None, routing_bias=None, topk_ids=ids, topk_weights=weights)
    )
    elt, rms = _deviation(out, ref)
    print(f"  the bias dropped from the router: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    with pytest.raises(AssertionError):
        _gate(out, ref)


def test_the_router_scale_and_the_routing_method_are_inert_when_pre_routed(stack):
    """`LatentMoe`'s router has already renormalized and multiplied by 3.998, so
    the runner must not apply `routed_scaling_factor` again; and the routing
    method only picks the router kernel, which `topk_ids` bypass. One call each."""
    args = _args(dict(tokens=64, **HY4), stack, seed=509)
    out = op(**args)
    assert torch.equal(out, op(**dict(args, routed_scaling_factor=1.0)))
    assert torch.equal(out, op(**dict(args, routing_method_type=1)))


def test_a_window_call_refuses_the_whole_stack_loudly(stack):
    """All four weight tensors must hold the window's experts only: the op
    compares each one's dim 0 with `local_num_experts` and raises on the whole
    stack."""
    whole = stack(HY4["hidden"], HY4["inter"], HY4["num_experts"])
    spec = {**dict(tokens=4, **HY4), "local_num_experts": 128}
    args = _args(spec, stack, seed=401)
    op(**args)  # the window's own tensors are accepted
    for name, tensor in zip(WEIGHT_TENSORS, whole):
        with pytest.raises(RuntimeError, match="incorrect"):
            op(**dict(args, **{name: tensor}))


# ── the guards ────────────────────────────────────────────────────────────


def _small_args() -> dict:
    """A tiny layer for the guards, which refuse before the kernel runs."""
    g = torch.Generator(device=DEV).manual_seed(7)
    tokens, hidden, inter, experts, top_k = 4, 256, 128, 8, 2
    ids, weights = _routing(tokens, experts, top_k, 3.998, g)
    return dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=torch.randn(tokens, hidden, device=DEV, generator=g).to(torch.float8_e4m3fn),
        hidden_states_scale=torch.ones(hidden // BLOCK, tokens, device=DEV),
        gemm1_weights=torch.randn(experts, 2 * inter, hidden, device=DEV, generator=g).to(
            torch.float8_e4m3fn
        ),
        gemm1_weights_scale=torch.full(
            (experts, 2 * inter // BLOCK, hidden // BLOCK), 0.05, device=DEV
        ),
        gemm2_weights=torch.randn(experts, hidden, inter, device=DEV, generator=g).to(
            torch.float8_e4m3fn
        ),
        gemm2_weights_scale=torch.full(
            (experts, hidden // BLOCK, inter // BLOCK), 0.01, device=DEV
        ),
        num_experts=experts,
        top_k=top_k,
        num_fused_shared_experts=0,
        n_group=1,
        topk_group=1,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=3.998,
        routing_method_type=ROUTING_DEEPSEEK_V3,
        topk_weights=weights,
        topk_ids=ids,
        gemm1_clamp_limit=10.0,
    )


def test_guard_refuses_a_strided_hidden_states():
    """The kernel reads data_ptr() as a dense buffer; strides are ignored.

    Demonstrated first: the same allocation read as a 4x256 dense matrix does not
    answer what its strided view means.
    """
    a = _small_args()
    wide = torch.randn(4, 512, device=DEV).to(torch.float8_e4m3fn)
    strided = wide[:, ::2]
    out = op(**dict(a, hidden_states=strided))
    ref = op.reference(**dict(a, hidden_states=strided.contiguous()))
    assert not torch.equal(out, ref), (
        "the unguarded strided call was expected to answer from the dense buffer"
    )
    with pytest.raises(AssertionError, match="hidden_states must be contiguous"):
        with validating(op):
            op(**dict(a, hidden_states=strided))


def test_guard_refuses_an_activation_that_is_not_swiglu():
    """`act_type` is overwritten by the Python runner: demonstrated first, 1 is
    bitwise the SwiGLU answer."""
    a = _small_args()
    assert torch.equal(op(**a), op(**dict(a, act_type=1)))
    with pytest.raises(AssertionError, match="act_type"):
        with validating(op):
            op(**dict(a, act_type=1))


def test_guard_refuses_a_bias_alongside_topk_ids():
    """Once the ids carry the routing nothing reads the bias: demonstrated first,
    a +-1e3 bias leaves the output bitwise unchanged."""
    a = _small_args()
    bias = torch.full((a["num_experts"],), 1.0e3, device=DEV)
    assert torch.equal(op(**a), op(**dict(a, routing_bias=bias)))
    with pytest.raises(AssertionError, match="ignored when topk_ids"):
        with validating(op):
            op(**dict(a, routing_bias=bias))


def test_guard_refuses_a_window_outside_the_routing_space():
    """Demonstrated first at a safe row count: a window that starts at the end of
    the routing space answers all zeros with no error."""
    a = _small_args()
    out = op(**dict(a, local_expert_offset=a["num_experts"], local_num_experts=a["num_experts"]))
    assert float(out.abs().max()) == 0.0, "the unguarded window was expected to answer zeros"
    with pytest.raises(AssertionError, match="outside the routing space"):
        with validating(op):
            op(**dict(a, local_expert_offset=a["num_experts"], local_num_experts=a["num_experts"]))
