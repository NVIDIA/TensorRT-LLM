# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_NORMQ ring boundary (DEP2CTX_NORMQ_V2 / DEP2CTX_NORMQ_STAGE), bit for bit vs the served op. GPU (SM107).

- minimax_m31_ctx_boundary_normq (every write_norm / write_f32 / fp4 flavor) equals the served
  minimax_m31_ctx_residual_rmsnorm_mxfp8(_f32) on every field it writes (norm, MXFP8, swizzled scales incl. the padded
  tile rows, NVFP4, linear scales, FP32, the in-place residual update) at 1024 / 8192 / 16384 / 32768 rows, odd row
  counts and the block-scale stress set (each 32-column block scaled by 2^k, k in [-12, 12]).
- minimax_m31_ctx_boundary_normq_out into row-strided stage buffers writes the same bytes, the stage's padded tail
  (ids -1, weights 0 in the first topk columns) and nothing else, with or without the FP32 copy (without it and with
  the norm: DEP2CTX_ROUTER_BF16S's staged call); with minimax_m31_route_stage on the router logits the
  whole stage equals noaux_tc_fp4_quantize_stage_out's.
- residual None recomputes the norm of the summed rows bit for bit (the left-out norm's materialization).
- The launch form equals the pure-Python plan's (ctx_boundary_plan.launch_form: CBNQ_P2's two 384-thread CTAs per SM
  where the thread limit holds the 768-thread kernel to one (sm_107), else that kernel's CTAs per SM, and the ring
  depth the per-block shared-memory opt-in allows).
- DEP2CTX_TOPK_FOLD: minimax_m31_ctx_boundary_normq_topk equals megamoe_topk_reduce_shared_add followed by
  minimax_m31_ctx_boundary_normq bit for bit (new residual, norm, MXFP8, scales) at 1 .. 16384 rows, PDL on / off.
- CBNQ_TOPK_DUAL: the fold's launch form equals the plan's (the dual-group kernel on sm_107, the fold's grid and
  shared memory).
"""

import importlib.util
import pathlib

import pytest
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

H = 6144
EPS = 1e-6
SENT = 0xA5
TOPK = 4
E = 128
RSF = 2.0
GS = 448.0 * 6.0 / 64.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


class _Ops:
    """The dep2ctx target's native ops, then the family's shared ones."""

    def __getattr__(self, name):
        for ops in (native_ops, family_ops()):
            if hasattr(ops, name):
                return getattr(ops, name)
        raise AttributeError(f"neither the dep2ctx target nor the family library has {name}")


_ops = _Ops()

_PLAN_PATH = (
    pathlib.Path(__file__).resolve().parents[7]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/"
    "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/ctx_boundary_plan.py"
)


def _plan():
    spec = importlib.util.spec_from_file_location("ctx_boundary_plan", _PLAN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16(rows, cols, kind, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", generator=gen)
    if kind == "stress":
        e = torch.randint(-12, 13, (rows, cols // 32, 1), device="cuda", generator=gen).float()
        x = (x.view(rows, cols // 32, 32) * torch.exp2(e)).view(rows, cols)
    return x.to(torch.bfloat16)


def _u8(t):
    return t.contiguous().view(torch.uint8).reshape(-1)


def _inputs(rows, kind, seed):
    x, r = _bf16(rows, H, kind, seed), _bf16(rows, H, "randn", seed + 1)
    w = (
        torch.rand(H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed + 2))
        + 0.5
    ).to(torch.bfloat16)
    return x, r, w


def _served(x, r, w, gs):
    """The served _f32 op: (norm, q, sf, fp4, fp4_sf, f32) with the BF16 norm written; inter after the add."""
    inter = x.clone()
    out = _ops.minimax_m31_ctx_residual_rmsnorm_mxfp8_f32(inter, r, w, EPS, gs)
    return inter, out


def _assert_fields(got, ref, write_norm, write_f32, fp4, what):
    assert torch.equal(_u8(got[1]), _u8(ref[1])), f"{what}: mxfp8"
    assert torch.equal(_u8(got[2]), _u8(ref[2])), (
        f"{what}: swizzled scales (incl. padded tile rows)"
    )
    if write_norm:
        assert torch.equal(_u8(got[0]), _u8(ref[0])), f"{what}: norm"
    else:
        assert got[0].numel() == 0
    if fp4:
        assert torch.equal(_u8(got[3]), _u8(ref[3])) and torch.equal(_u8(got[4]), _u8(ref[4])), (
            f"{what}: nvfp4"
        )
    if write_f32:
        assert torch.equal(_u8(got[5]), _u8(ref[5])), f"{what}: f32"
    else:
        assert got[5].numel() == 0


@pytest.mark.parametrize("rows", [1024, 8192, 16384, 32768])
@pytest.mark.parametrize("kind", ["randn", "stress"])
@pytest.mark.parametrize("write_norm", [True, False])
@pytest.mark.parametrize("write_f32", [True, False])
@pytest.mark.parametrize("fp4", [False, True])
def test_ring_op_equals_the_served_op(rows, kind, write_norm, write_f32, fp4):
    x, r, w = _inputs(rows, kind, rows + 11 * write_norm + 5 * write_f32 + fp4)
    gs = torch.tensor([GS], device="cuda") if fp4 else None
    ref_inter, ref = _served(x, r, w, gs)
    inter = x.clone()
    got = _ops.minimax_m31_ctx_boundary_normq(inter, r, w, EPS, write_norm, write_f32, gs)
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter)), "in-place residual update"
    _assert_fields(got, ref, write_norm, write_f32, fp4, f"rows {rows} {kind}")


@pytest.mark.parametrize("rows", [1, 31, 127, 128, 129, 1000, 4097, 8191])
@pytest.mark.parametrize("pdl", [False, True])
def test_odd_rows_and_pdl(rows, pdl):
    x, r, w = _inputs(rows, "stress", rows)
    gs = torch.tensor([GS], device="cuda")
    ref_inter, ref = _served(x, r, w, gs)
    inter = x.clone()
    got = _ops.minimax_m31_ctx_boundary_normq(inter, r, w, EPS, True, True, gs, pdl)
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter))
    _assert_fields(got, ref, True, True, True, f"rows {rows} pdl {pdl}")


def test_repeat_runs_are_identical():
    x, r, w = _inputs(8192, "stress", 3)
    gs = torch.tensor([GS], device="cuda")
    a = _ops.minimax_m31_ctx_boundary_normq(x.clone(), r, w, EPS, True, True, gs)
    b = _ops.minimax_m31_ctx_boundary_normq(x.clone(), r, w, EPS, True, True, gs)
    for u, v in zip(a, b):
        assert torch.equal(_u8(u), _u8(v))


def _stage(cap):
    """A MegaMoE-like stage of cap rows whose NVFP4 rows sit in row strides wider than the data (sentinel-filled):
    (ids, weights, act full, act view, act_sf full, act_sf view)."""
    ids = torch.full((cap, TOPK + 1), 7, dtype=torch.int32, device="cuda")
    wts = torch.full((cap, TOPK + 1), 3.0, dtype=torch.float32, device="cuda")
    act_full = torch.full((cap, H // 2 + 64), SENT, dtype=torch.uint8, device="cuda")
    sf_full = torch.full((cap, H // 16 + 16), SENT, dtype=torch.uint8, device="cuda")
    return ids, wts, act_full, act_full[:, : H // 2], sf_full, sf_full[:, : H // 16]


@pytest.mark.parametrize("rows, cap", [(8192, 8192), (8000, 8192), (1000, 1024), (16384, 16384)])
@pytest.mark.parametrize("write_norm", [True, False])
@pytest.mark.parametrize("write_f32", [True, False])
@pytest.mark.parametrize("kind", ["randn", "stress"])
def test_out_op_writes_the_stage_and_its_tail(rows, cap, write_norm, write_f32, kind):
    """write_f32 False with write_norm True is DEP2CTX_ROUTER_BF16S's staged call (flavor <add, norm, mx, !f32,
    fp4>: the router reads the BF16 norm, no FP32 copy)."""
    x, r, w = _inputs(rows, kind, rows + cap + 3 * write_f32)
    gs = torch.tensor([GS], device="cuda")
    ref_inter, ref = _served(x, r, w, gs)
    ids, wts, act_full, act, sf_full, act_sf = _stage(cap)
    norm = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda") if write_norm else None
    q = torch.empty(rows, H, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.full((ref[2].numel(),), SENT, dtype=torch.uint8, device="cuda")
    f32 = torch.empty(rows, H, dtype=torch.float32, device="cuda") if write_f32 else None
    inter = x.clone()
    _ops.minimax_m31_ctx_boundary_normq_out(
        inter, r, w, EPS, norm, q, sf, f32, gs, act[:rows], act_sf[:rows], ids, wts[:, :TOPK], TOPK
    )
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter))
    if write_norm:
        assert torch.equal(_u8(norm), _u8(ref[0]))
    assert torch.equal(_u8(q), _u8(ref[1])) and torch.equal(_u8(sf), _u8(ref[2]))
    if write_f32:
        assert torch.equal(_u8(f32), _u8(ref[5]))
    assert torch.equal(_u8(act[:rows]), _u8(ref[3].view(rows, H // 2))), "stage nvfp4 rows"
    assert torch.equal(_u8(act_sf[:rows]), _u8(ref[4].view(rows, H // 16))), "stage nvfp4 scales"
    # nothing in the row strides' slack, nothing in the tail's activation rows
    assert bool((act_full[:, H // 2 :] == SENT).all()) and bool(
        (sf_full[:, H // 16 :] == SENT).all()
    )
    assert bool((act[rows:] == SENT).all()) and bool((act_sf[rows:] == SENT).all())
    # the tail: -1 / 0 in the first TOPK columns of rows [rows, cap); live rows and the extra column untouched
    assert bool((ids[rows:, :TOPK] == -1).all()) and bool((wts[rows:, :TOPK] == 0).all())
    assert bool((ids[:rows] == 7).all()) and bool((wts[:rows] == 3.0).all())
    assert bool((ids[:, TOPK] == 7).all()) and bool((wts[:, TOPK] == 3.0).all())


@pytest.mark.parametrize("rows, cap", [(8192, 8192), (6000, 8192)])
def test_staged_boundary_plus_route_stage_equals_the_fused_route_quant(rows, cap):
    x, r, w = _inputs(rows, "stress", 77 + rows)
    gs = torch.tensor([GS], device="cuda")
    ref_inter, ref = _served(x, r, w, gs)
    logits = torch.randn(
        rows, E, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)
    )
    bias = 0.01 * torch.randn(
        E, device="cuda", generator=torch.Generator(device="cuda").manual_seed(6)
    )
    # the backend's launch: route + quantize + stage from the served normed rows
    s_ids = torch.full((cap, TOPK), 9, dtype=torch.int32, device="cuda")
    s_wts = torch.full((cap, TOPK), 9.0, dtype=torch.float32, device="cuda")
    s_act = torch.zeros(cap, H // 2, dtype=torch.uint8, device="cuda")
    s_sf = torch.zeros(cap, H // 16, dtype=torch.uint8, device="cuda")
    empty = (
        torch.empty(0, TOPK, dtype=torch.int32, device="cuda"),
        torch.empty(0, TOPK, dtype=torch.float32, device="cuda"),
        torch.empty(0, H // 2, dtype=torch.uint8, device="cuda"),
        torch.empty(0, H // 16, dtype=torch.uint8, device="cuda"),
    )
    _ops.noaux_tc_fp4_quantize_stage_out(
        logits, bias, ref[0], gs, TOPK, RSF, rows, s_ids, s_wts, s_act, s_sf, *empty
    )
    # the fold: the boundary writes NVFP4 + the tail, the route stage the live ids / weights
    b_ids = torch.full((cap, TOPK), 9, dtype=torch.int32, device="cuda")
    b_wts = torch.full((cap, TOPK), 9.0, dtype=torch.float32, device="cuda")
    b_act = torch.zeros(cap, H // 2, dtype=torch.uint8, device="cuda")
    b_sf = torch.zeros(cap, H // 16, dtype=torch.uint8, device="cuda")
    q = torch.empty(rows, H, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(ref[2].numel(), dtype=torch.uint8, device="cuda")
    f32 = torch.empty(rows, H, dtype=torch.float32, device="cuda")
    inter = x.clone()
    _ops.minimax_m31_ctx_boundary_normq_out(
        inter, r, w, EPS, None, q, sf, f32, gs, b_act[:rows], b_sf[:rows], b_ids, b_wts, TOPK
    )
    _ops.minimax_m31_route_stage(logits, bias, None, TOPK, RSF, -1, b_ids, b_wts, False)
    torch.cuda.synchronize()
    assert torch.equal(_u8(b_act[:rows]), _u8(s_act[:rows])) and torch.equal(
        _u8(b_sf[:rows]), _u8(s_sf[:rows])
    )
    assert torch.equal(_u8(b_ids), _u8(s_ids)) and torch.equal(_u8(b_wts), _u8(s_wts)), (
        "ids / weights incl. tail"
    )


@pytest.mark.parametrize("rows", [129, 8192])
def test_materialize_recomputes_the_left_out_norm(rows):
    x, r, w = _inputs(rows, "stress", 900 + rows)
    ref_inter, ref = _served(x, r, w, None)
    inter = x.clone()
    q = torch.empty(rows, H, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(ref[2].numel(), dtype=torch.uint8, device="cuda")
    _ops.minimax_m31_ctx_boundary_normq_out(inter, r, w, EPS, None, q, sf)  # the norm left out
    norm = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    _ops.minimax_m31_ctx_boundary_normq_out(inter, None, w, EPS, norm, None, None)
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter)), (
        "the norm-only form must not touch the summed rows"
    )
    assert torch.equal(_u8(norm), _u8(ref[0]))


def test_out_op_rejects_flavors_it_has_no_form_for():
    x, r, w = _inputs(256, "randn", 1)
    with pytest.raises(RuntimeError):
        _ops.minimax_m31_ctx_boundary_normq_out(x.clone(), None, w, EPS, None, None, None)
    with pytest.raises(RuntimeError):
        _ops.minimax_m31_ctx_boundary_normq_out(x.clone(), r, w, EPS, None, None, None)


def test_ops_reject_zero_width_rows():
    """A [1, 0] input is rejected by the hidden check before the row count divides by the width."""
    x = torch.empty(1, 0, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="hidden must be a multiple of 256"):
        _ops.minimax_m31_ctx_boundary_normq(x, x.clone(), w, EPS, True)
    with pytest.raises(RuntimeError, match="hidden must be a multiple of 256"):
        _ops.minimax_m31_ctx_boundary_normq_out(x, x.clone(), w, EPS, None, None, None)


def test_out_op_rejects_overlapping_stage_rows():
    """Row-strided NVFP4 / stage destinations whose rows alias (stride 0) are rejected before launch."""
    rows = 256
    x, r, w = _inputs(rows, "randn", 2)
    gs = torch.tensor([GS], device="cuda")
    q = torch.empty(rows, H, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(rows * (H // 32), dtype=torch.uint8, device="cuda")
    act = torch.zeros(rows, H // 2, dtype=torch.uint8, device="cuda")
    act_sf = torch.zeros(rows, H // 16, dtype=torch.uint8, device="cuda")
    ids = torch.zeros(rows + 8, TOPK, dtype=torch.int32, device="cuda")
    wts = torch.zeros(rows + 8, TOPK, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="non-overlapping rows"):
        _ops.minimax_m31_ctx_boundary_normq_out(
            x.clone(), r, w, EPS, None, q, sf, None, gs, act[:1].expand(rows, -1), act_sf
        )
    with pytest.raises(RuntimeError, match="non-overlapping rows"):
        _ops.minimax_m31_ctx_boundary_normq_out(
            x.clone(), r, w, EPS, None, q, sf, None, gs, act, act_sf[:1].expand(rows, -1)
        )
    with pytest.raises(RuntimeError, match="rows not overlapping"):
        _ops.minimax_m31_ctx_boundary_normq_out(
            x.clone(), r, w, EPS, None, q, sf, None, gs, act, act_sf,
            ids[:1].expand(rows + 8, -1), wts, TOPK,
        )  # fmt: skip


@pytest.mark.parametrize("rows", [1, 8192, 16384])
@pytest.mark.parametrize(
    "add, write_norm, write_f32, fp4",
    [(True, True, False, False), (True, False, True, True), (False, True, False, False)],
)
def test_launch_shape_matches_the_plan(rows, add, write_norm, write_f32, fp4):
    plan = _plan()
    t = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    ctas, grid, smem = _ops.minimax_m31_ctx_boundary_normq_shape(t, add, write_norm, write_f32, fp4)
    props = torch.cuda.get_device_properties(0)
    chunks, want_ctas, stages = plan.launch_form(
        H,
        add,
        props.max_threads_per_multi_processor,
        props.shared_memory_per_block_optin,
        props.shared_memory_per_multiprocessor,
    )
    assert ctas == want_ctas, (
        f"{ctas} CTAs per SM (registers or shared memory over the launch bound; chunks {chunks})"
    )
    assert smem == plan.smem_bytes(stages, add, H)
    assert grid == plan.grid(rows, ctas, props.multi_processor_count)
    form = _ops.minimax_m31_ctx_boundary_normq_form(t, add, write_norm, write_f32, fp4)
    assert tuple(form) == (chunks, ctas, grid, smem, stages)


def _fold_inputs(rows, kind, seed):
    """(plane [rows, 4, H], scores [rows, 4], shared, residual) at served scales: plane / shared N(0, 0.05^2), scores
    sigmoid x 2 (top-4 weights), residual N(0, 1). Stress: the first min(rows, 3) rows are finite FP32-subnormal rows
    (subnormal plane values; smallest-normal plane values times scores below 1, i.e. subnormal products; subnormal
    shared values over a zero plane; a shared + routed cancellation into the subnormal range), with zero residual in
    those columns so the new residual keeps the subnormals and an .ftz difference between the stock reduce and the
    fold shows in it; rows 1 .. 3 also take +-65504 and +-0, and row 3 (rows >= 4) +-inf."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    plane = (0.05 * torch.randn(rows, TOPK, H, device="cuda", generator=g)).to(torch.bfloat16)
    scores = torch.sigmoid(torch.randn(rows, TOPK, device="cuda", generator=g)) * RSF
    shared = (0.05 * torch.randn(rows, H, device="cuda", generator=g)).to(torch.bfloat16)
    residual = torch.randn(rows, H, device="cuda", generator=g).to(torch.bfloat16)
    if kind == "stress":
        n = min(rows, 4)
        e = torch.randint(-12, 13, (rows, H // 32, 1), device="cuda", generator=g).float()
        shared = (
            (shared.float().view(rows, H // 32, 32) * torch.exp2(e))
            .view(rows, H)
            .to(torch.bfloat16)
        )
        sub = min(rows, 3)  # finite subnormal rows
        scores[:sub] = torch.tensor([0.75, 0.5, 1.25, 0.3], device="cuda")
        plane[:sub, :, :32] = 2.0**-130  # subnormal BF16 / FP32 inputs
        shared[:sub, :32] = 0.0
        plane[:sub, :, 32:64] = 2.0**-126  # the smallest normal: subnormal products for scores < 1
        shared[:sub, 32:48] = 0.0
        shared[:sub, 48:64] = -(2.0**-125)  # shared + routed cancels into the subnormal range
        plane[:sub, :, 208:240] = 0.0
        shared[:sub, 208:240] = 3 * 2.0**-133  # subnormal shared rows over a zero plane
        residual[:sub, :64] = 0.0  # the new residual keeps the subnormals
        residual[:sub, 208:240] = 0.0
        if rows > 1:
            plane[1:n, 0, 64:128] = 65504.0
            plane[1:n, 1, 64:128] = -65504.0
            plane[1:n, :, 128:192] = -0.0
            shared[1:n, 128:192] = -0.0
            residual[1:n, 128:192] = -0.0
        if rows > 3:
            plane[3, 2, 192:200] = float("inf")
            shared[3, 200:208] = float("-inf")
    return plane, scores, shared, residual


def _subnormal_f32(t: torch.Tensor) -> torch.Tensor:
    """Elements that are FP32 subnormals (non-zero, below the smallest normal 2^-126 in magnitude)."""
    a = t.double().abs()
    return (a > 0) & (a < 2.0**-126)


@pytest.mark.parametrize("rows", [1, 7, 127, 128, 129, 1024, 1025, 4096, 8192, 12288, 16383, 16384])
@pytest.mark.parametrize("kind", ["randn", "stress"])
@pytest.mark.parametrize("pdl", [False, True])
def test_topk_fold_bitwise_equals_reduce_then_boundary(rows, kind, pdl):
    """DEP2CTX_TOPK_FOLD: the fold op equals megamoe_topk_reduce_shared_add then minimax_m31_ctx_boundary_normq on
    the new residual (the shared buffer), the BF16 norm, the MXFP8 rows and the swizzled scales incl. the padded
    tile rows; the plane, the scores and the residual are not written."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_shared_add,
    )

    plane, scores, shared, residual = _fold_inputs(rows, kind, 4000 + rows)
    if kind == "stress":
        # the stress rows really reach the FP32 subnormal range (inputs, products, the shared add)
        products = plane.double() * scores.double()[:, :, None]
        assert _subnormal_f32(plane).any() and _subnormal_f32(shared).any()
        assert _subnormal_f32(products.float()).any()
        assert _subnormal_f32((shared.double() + products.sum(1)).float()).any()
    w = (
        torch.rand(H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(rows))
        + 0.5
    ).to(torch.bfloat16)
    plane0, scores0, residual0 = plane.clone(), scores.clone(), residual.clone()
    ref_shared = shared.clone()
    megamoe_topk_reduce_shared_add(plane, scores, ref_shared, enable_pdl=pdl)
    ref = _ops.minimax_m31_ctx_boundary_normq(
        ref_shared, residual.clone(), w, EPS, True, False, None, pdl
    )
    got_shared = shared.clone()
    got = _ops.minimax_m31_ctx_boundary_normq_topk(
        got_shared, plane, scores, residual, w, EPS, True, pdl
    )
    torch.cuda.synchronize()
    what = f"rows {rows} {kind} pdl {pdl}"
    if kind == "stress":
        sub = min(rows, 3)
        assert _subnormal_f32(ref_shared[:sub]).any(), (
            f"{what}: the stock reduce flushed the FP32 subnormals (.ftz); give the fold's mul / fma / add helpers "
            ".ftz (BUILD_NOTES deviation 4)"
        )
    assert torch.equal(_u8(got_shared), _u8(ref_shared)), f"{what}: new residual"
    assert torch.equal(_u8(got[0]), _u8(ref[0])), f"{what}: norm"
    assert torch.equal(_u8(got[1]), _u8(ref[1])), f"{what}: mxfp8"
    assert torch.equal(_u8(got[2]), _u8(ref[2])), (
        f"{what}: swizzled scales (incl. padded tile rows)"
    )
    assert torch.equal(_u8(plane), _u8(plane0)) and torch.equal(_u8(scores), _u8(scores0))
    assert torch.equal(_u8(residual), _u8(residual0))


def test_topk_fold_rejects_what_it_cannot_fold():
    plane, scores, shared, residual = _fold_inputs(64, "randn", 1)
    w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    op = _ops.minimax_m31_ctx_boundary_normq_topk
    with pytest.raises(RuntimeError):  # top-8
        op(shared.clone(), plane.repeat(1, 2, 1), scores.repeat(1, 2), residual, w, EPS, True)
    with pytest.raises(RuntimeError):  # unaligned scores
        s = torch.empty(64 * TOPK + 1, device="cuda")[1:].view(64, TOPK).copy_(scores)
        op(shared.clone(), plane, s, residual, w, EPS, True)
    with pytest.raises(RuntimeError):  # rows mismatch
        op(shared[:32].clone(), plane, scores, residual[:32], w, EPS, True)


@pytest.mark.parametrize("rows", [1, 129, 4096, 16384])
@pytest.mark.parametrize("write_norm", [True, False])
def test_topk_fold_form_matches_the_plan(rows, write_norm):
    plan = _plan()
    props = torch.cuda.get_device_properties(0)
    threads = props.max_threads_per_multi_processor
    dual, ctas, grid, smem = (
        int(v)
        for v in _ops.minimax_m31_ctx_boundary_normq_topk_form(
            torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"), write_norm
        )
    )
    want_dual = plan.topk_dual_form(plan.topk_dual_choice(), H, threads)
    assert dual == int(want_dual)
    assert ctas == 1 and smem == plan.topk_smem_bytes(plan.STAGES, plan.TOPK, H)
    assert grid == plan.grid(rows, ctas, props.multi_processor_count)
