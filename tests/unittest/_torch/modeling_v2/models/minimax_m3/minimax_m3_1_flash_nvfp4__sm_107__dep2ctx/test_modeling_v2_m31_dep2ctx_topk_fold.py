# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_TOPK_FOLD on the dep2ctx target, CPU only.

- The fold kernel's shared memory (ring of [top-k plane rows | shared | residual | scores] stages, three deep) fits
  the per-block opt-in only at the base depth; the ring kernel's own sizes are unchanged.
- The fold's statement order (TopkReduce's FP32 mul, then fma in slot order, BF16, the FP32 shared add, BF16, then
  the boundary's residual add + RMSNorm) equals the stock composition (megamoe_topk_reduce_shared_add, then the
  boundary) bit for bit in an exact FP32 emulation, incl. signed zeros, FP32 subnormal inputs / products / sums and
  overflow.
- _ctx_moe_tail_add parks the deferred reduce only when forward_MoE armed the fold; forward_MoE hands it to the fold
  op (MXFP8 pair published, (normed, shared) returned, the slot disarmed), and on every decline (before the MoE, on
  the plane, or when the op rejects its arguments) the reduce runs exactly once and the stock boundary follows.
- The slot is disarmed and empty after an exception inside an armed MoE call; the kill switch is read at import.
- An op rejection turns the fold off for the process; the fold's decline / taken log stays bounded (row counts do not
  make new entries), and each decline names the first failing check.
- CBNQ_TOPK_DUAL: the dual-group form is taken on sm_107 only (thread-limited, P2-shaped hidden sizes); its O(1)
  position -> row map equals the ring producer's row order; a randomized model of its ring (owner-only refill +
  release counter, quad barriers, any TMA completion order) never lets a parity wait see a stage two phases away,
  never deadlocks, and hands every position to exactly one group.

The modeling-hook tests need the MegaMoE backend's deferred top-k reduce (``MegaMoEDeferredTopkReduce``, the routed
output of a launch that leaves its reduce to the caller); they skip on an engine without it.
"""

import fractions
import os
import random
import struct
import subprocess
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        ctx_boundary_plan as plan,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

H = 6144
K = 4
OPTIN_SM10X = 232448  # cudaDevAttrMaxSharedMemoryPerBlockOptin on sm_100 / sm_107 (227 KB)


# ---- 1. plan -------------------------------------------------------------------------------------------------------
def test_plan_topk_smem_and_depth():
    stage = (K + 2) * H * 2 + 16
    assert plan.topk_stage_bytes(K, H) == stage == 73744
    assert (
        plan.topk_smem_bytes(3, K, H) == 3 * stage + 2 * (H // 128) * 16 + H // 16 + 3 * 8 == 223176
    )
    # only the base depth fits the sm_10x opt-in (with the launcher's static bound); 4 stages do not
    assert plan.topk_fits(K, H, OPTIN_SM10X)
    assert plan.topk_smem_bytes(4, K, H) + plan.STATIC_SMEM_BOUND > OPTIN_SM10X
    assert not plan.topk_fits(K, H, 101376)  # an sm_8x-class 99 KB opt-in: the launcher declines
    # every stage piece starts 16-byte aligned (the bulk copies' destinations)
    for hidden in (256, 1024, 4096, 6144):
        assert plan.topk_stage_bytes(K, hidden) % 16 == 0 and (K * hidden * 2) % 16 == 0
    # the ring kernel's own sizes are unchanged
    assert plan.smem_bytes(plan.STAGES, True, H) == 75672


# ---- CBNQ_TOPK_DUAL plan ----------------------------------------------------------------------------------------
def test_plan_topk_dual_form_and_switch():
    on = plan.topk_dual_choice()
    assert on
    assert plan.topk_dual_form(on, H, 1024)  # sm_107: one 768-thread CTA per SM
    assert not plan.topk_dual_form(
        on, H, 2048
    )  # sm_100 / sm_103: the thread limit is not what binds
    assert not plan.topk_dual_form(False, H, 1024)
    assert not plan.topk_dual_form(
        on, 6144 - 256, 1024
    )  # 5888 % 512 != 0: no whole 32-chunk warps per half
    assert (
        plan.topk_dual_form(on, 4096, 1024) is False
    )  # 4096 / 8 = 512 threads: two CTAs fit anyway


@pytest.mark.parametrize("rows", [0, 1, 7, 33, 127, 128, 129, 1500, 4097, 12287, 16384])
@pytest.mark.parametrize("grid_size", [1, 3, 212])
def test_plan_topk_dual_pos_row_is_the_producer_order(rows, grid_size):
    g = plan.grid(rows, 1, grid_size)
    for cta in range(g):
        seq = plan.producer_rows(cta, g, rows)
        got = [plan.topk_dual_pos_row(p, cta, g, rows) for p in range(len(seq) + 2)]
        assert got[: len(seq)] == seq and got[len(seq) :] == [-1, -1]


def _dual_ring_run(rows, cta, grid_size, seed, stages=plan.STAGES):
    """One randomized schedule of the dual fold's ring for CTA ``cta``: two groups walk the quad rows (positions
    owned by p % 2), a position's owner waits on the release counter, then on the stage's mbarrier parity, reads,
    refills position p + stages (arming the next phase) and counts the release; both groups meet at each quad's
    end. TMA completions land in any order. Returns the positions each group read."""
    rng = random.Random(seed)
    seq = plan.producer_rows(cta, grid_size, rows)
    n = len(seq)
    completed = [0] * stages  # completed phases per stage
    armed = [0] * stages  # armed phases per stage
    released = [0] * stages
    for p in range(min(stages, n)):
        armed[p % stages] += 1
    # each group's program: per quad, its own positions, then the quad barrier
    quads = []
    p = 0
    for q in range(cta, plan.num_quads(rows), grid_size):
        count = sum(1 for j in range(plan.QUAD_ROWS) if plan.quad_row(q, j) < rows)
        quads.append(list(range(p, p + count)))
        p += count
    prog = []
    for g in range(plan.TOPK_DUAL_GROUPS):
        steps = []
        for qi, positions in enumerate(quads):
            steps += [("row", pos) for pos in positions if plan.topk_dual_group(pos) == g]
            steps.append(("quad", qi))
        prog.append(steps)
    pc = [0, 0]
    phase = [0, 0]  # 0 guard, 1 wait, 2 read+release
    at_quad = [None, None]
    read = [[], []]
    for _ in range(100000):
        if pc[0] == len(prog[0]) and pc[1] == len(prog[1]):
            break
        moves = []
        for s in range(stages):
            if armed[s] > completed[s]:
                moves.append(("tma", s))
        for g in range(2):
            if pc[g] == len(prog[g]):
                continue
            kind, val = prog[g][pc[g]]
            if kind == "quad":
                other = 1 - g
                if at_quad[other] == val or (
                    pc[other] < len(prog[other]) and prog[other][pc[other]] == ("quad", val)
                ):
                    moves.append(("quad", g))
                continue
            pos, s = val, val % stages
            if phase[g] == 0:
                if pos < stages or released[s] >= plan.topk_dual_release_need(pos, stages):
                    moves.append(("guard", g))
            elif phase[g] == 1:
                c = completed[s]
                # the parity wait identifies its phase only within one phase of the barrier
                assert c in (pos // stages, pos // stages + 1), (rows, cta, pos, c)
                if (
                    pos // stages
                ) % 2 != c % 2:  # try_wait.parity returns true: the phase pos // stages completed
                    moves.append(("wait", g))
            else:
                moves.append(("read", g))
        assert moves, f"deadlock rows {rows} cta {cta} pc {pc} phase {phase}"
        kind, x = rng.choice(moves)
        if kind == "tma":
            completed[x] += 1
        elif kind == "quad":
            at_quad[x] = prog[x][pc[x]][1]
            pc[x] += 1
        elif kind == "guard":
            phase[x] = 1
        elif kind == "wait":
            phase[x] = 2
        else:
            pos = prog[x][pc[x]][1]
            s = pos % stages
            assert completed[s] == pos // stages + 1  # the data of this very position
            read[x].append(pos)
            if pos + stages < n:
                assert armed[s] == completed[s]  # one phase in flight per stage
                armed[s] += 1
            released[s] += 1
            phase[x] = 0
            pc[x] += 1
    assert pc == [len(prog[0]), len(prog[1])]
    return read, n


@pytest.mark.parametrize("rows", [1, 5, 129, 1500, 16384])
def test_dual_ring_model_has_no_phase_aliasing_or_deadlock(rows):
    grid_size = plan.grid(rows, 1, 212)
    for cta in sorted({0, 1, grid_size // 2, grid_size - 1}):
        for seed in range(40):
            read, n = _dual_ring_run(rows, cta, grid_size, seed)
            assert sorted(read[0] + read[1]) == list(range(n))
            assert all(plan.topk_dual_group(p) == g for g in (0, 1) for p in read[g])


# ---- 2. statement order, exact FP32 emulation ----------------------------------------------------------------------
def _fma32(a, b, c):
    """IEEE binary32 fma(a, b, c) of float32 tensors, rounded once: the product is exact in binary64, the sum is
    rounded to odd in binary64 (TwoSum error term), then to nearest-even binary32 (53 >= 24 + 2 bits)."""
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bp = s - cd
    err = (p - bp) + (cd - (s - bp))
    bits = s.view(torch.int64)
    fix = torch.isfinite(s) & (err != 0) & ((bits & 1) == 0)
    away = (err > 0) == (s > 0)  # the exact sum's magnitude is above |s|
    bits = torch.where(fix, torch.where(away, bits + 1, bits - 1), bits)
    return bits.view(torch.float64).float()


def _round_f32(exact: fractions.Fraction) -> float:
    """The binary32 nearest to ``exact`` (ties to even), by checking the neighbours of a first guess."""
    guess = torch.tensor([float(exact)], dtype=torch.float32)
    near = [
        torch.nextafter(guess, torch.tensor([float("-inf")])),
        guess,
        torch.nextafter(guess, torch.tensor([float("inf")])),
    ]
    best = min(
        near,
        key=lambda t: (abs(fractions.Fraction(t.item()) - exact), t.view(torch.int32).item() & 1),
    )
    return best.item()


def test_fma32_emulation_is_exact():
    rng = random.Random(7)
    vals = [
        [rng.uniform(-1, 1) * 2.0 ** rng.randint(-30, 30) for _ in range(3)] for _ in range(4000)
    ]
    # a double-rounding trap, a signed zero, a subnormal result
    vals += [[1.0 + 2.0**-23, 1.0 - 2.0**-23, -1.0], [-0.0, 0.5, 0.0], [3.0, 2.0**-140, 2.0**-149]]
    t = torch.tensor(vals, dtype=torch.float32)
    got = _fma32(t[:, 0], t[:, 1], t[:, 2])
    for (a, b, c), g in zip(t.tolist(), got.tolist()):
        exact = fractions.Fraction(a) * fractions.Fraction(b) + fractions.Fraction(c)
        want = _round_f32(exact) if exact else a * b + c  # an exact zero takes IEEE's sign rule
        assert struct.pack("<f", g) == struct.pack("<f", want), (a, b, c, g, want)


def _rne_bf16(x):
    return x.to(torch.bfloat16)


def _reduce_shared_add_stock(plane, scores, shared):
    """megamoe_topk_reduce_tail.py's _reduce_add, vectorized over the row: mul for k = 0, fma for k >= 1."""
    p = plane.float()
    acc = p[:, 0] * scores[:, 0:1]
    for k in range(1, plane.shape[1]):
        acc = _fma32(p[:, k], scores[:, k : k + 1].expand_as(acc), acc)
    routed = _rne_bf16(acc)
    return _rne_bf16(shared.float() + routed.float())


def _warp_xor(v):
    """cbnqWarpReduceSum on [..., 32] lanes: v += shfl_xor(v, 16 .. 1); every lane ends with the same value."""
    idx = torch.arange(32)
    for off in (16, 8, 4, 2, 1):
        v = v + v[..., idx ^ off]
    return v


def _boundary(inter, residual, weight, eps):
    """ctxBoundaryNormQuantKernel<kAdd, kNorm> up to the BF16 norm: bf16x2 add, per-thread fma accumulate of 8
    elements, warp xor tree, 24-warp xor tree, rsqrt, (v * denom) * w -> BF16."""
    rows, hidden = inter.shape
    s = _rne_bf16(inter.float() + residual.float())
    v = s.float().view(rows, hidden // 8, 8)
    acc = torch.zeros(rows, hidden // 8)
    for i in range(8):
        acc = _fma32(v[..., i], v[..., i], acc)
    warps = hidden // 8 // 32
    per_warp = _warp_xor(acc.view(rows, warps, 32))[..., 0]
    lanes = torch.zeros(rows, 32)
    lanes[:, :warps] = per_warp
    total = _warp_xor(lanes)[:, 0]
    denom = torch.rsqrt(total / hidden + eps)
    normed = _rne_bf16(s.float() * denom[:, None] * weight.float()[None, :])
    return s, normed


def _fold(plane, scores, shared, residual, weight, eps):
    """The fold kernel's statements, element by element as ctxBoundaryTopkNormQuantSm107.cuh writes them."""
    rows, k_slots, hidden = plane.shape
    moe = torch.empty(rows, hidden, dtype=torch.bfloat16)
    for e in range(hidden):
        acc = plane[:, 0, e].float() * scores[:, 0]  # mul.rn (not fma(p, s, 0))
        for k in range(1, k_slots):
            acc = _fma32(plane[:, k, e].float(), scores[:, k], acc)
        routed = _rne_bf16(acc)
        moe[:, e] = _rne_bf16(shared[:, e].float() + routed.float())
    return _boundary(moe, residual, weight, eps)


def _inputs(rows, kind, seed, hidden=256):
    g = torch.Generator().manual_seed(seed)
    plane = (torch.randn(rows, K, hidden, generator=g) * 0.05).bfloat16()
    scores = torch.sigmoid(torch.randn(rows, K, generator=g)) * 2.0
    shared = (torch.randn(rows, hidden, generator=g) * 0.05).bfloat16()
    residual = torch.randn(rows, hidden, generator=g).bfloat16()
    weight = (torch.rand(hidden, generator=g) + 0.5).bfloat16()
    if kind == "stress":
        plane[0] = -0.0  # a -0 product sum: fma(-0, s, +0) would give +0
        shared[0] = -0.0
        residual[0] = -0.0
        if rows > 1:
            scores[1] = torch.tensor([0.75, 0.5, 1.25, 0.3])
            plane[1, :, :8] = 2.0**-130  # FP32 subnormal inputs and products
            plane[1, :, 24:32] = 2.0**-126  # the smallest normal: subnormal products for scores < 1
            shared[1, 24:28] = 3 * 2.0**-133  # subnormal shared rows
            shared[1, 28:32] = -(2.0**-125)  # shared + routed in the subnormal range
            residual[1, :32] = 0.0  # the summed rows keep the subnormals
            plane[1, 0, 8:16] = 65504.0
            plane[1, 1, 8:16] = -65504.0
            shared[1, 16:24] = 3.0e38
            plane[1, 2, 16:24] = 3.0e38  # FP32 overflow in the reduce
        if rows > 2:
            plane[2] = plane[2] * 2.0**-12
            shared[2] = shared[2] * 2.0**12
        if rows > 3:
            # FP32 absorption: 1 + 2^24 - 2^24 + 1 is 1 in slot order, 2 in any other order
            scores[3] = 1.0
            for k, v in enumerate((1.0, 2.0**24, -(2.0**24), 1.0)):
                plane[3, k, :8] = v
            shared[3, :8] = 0.0
    return plane, scores, shared, residual, weight


@pytest.mark.parametrize("rows", [1, 7, 129])
@pytest.mark.parametrize("kind", ["randn", "stress"])
def test_fold_order_equals_stock_composition_cpu(rows, kind):
    plane, scores, shared, residual, weight = _inputs(rows, kind, rows)
    eps = 1e-6
    moe = _reduce_shared_add_stock(plane, scores, shared)
    ref_sum, ref_norm = _boundary(moe, residual, weight, eps)
    got_sum, got_norm = _fold(plane, scores, shared, residual, weight, eps)
    assert torch.equal(got_sum.view(torch.int16), ref_sum.view(torch.int16))
    assert torch.equal(got_norm.view(torch.int16), ref_norm.view(torch.int16))
    if kind == "stress" and rows > 1:
        tiny = lambda t: ((t.double().abs() > 0) & (t.double().abs() < 2.0**-126)).any()  # noqa: E731
        assert tiny(plane[1]) and tiny(shared[1])
        assert tiny((plane[1].float() * scores[1][:, None]).float())
        assert tiny(_reduce_shared_add_stock(plane[1:2], scores[1:2], shared[1:2]).float())
        assert tiny(ref_sum[1].float()), "the residual add keeps the subnormals observable"
    if kind == "stress":
        # the -0 row stays -0 through the reduce, the shared add and the residual add
        assert torch.equal(
            got_sum[0].view(torch.int16), torch.full((256,), -32768, dtype=torch.int16)
        )


# ---- modeling hooks ------------------------------------------------------------------------------------------------
_ENGINE_DEFERRED = modeling.MegaMoEDeferredTopkReduce
_NO_DEFERRED = (
    "needs the MegaMoE backend's deferred top-k reduce (MegaMoEDeferredTopkReduce), "
    "which this engine does not have"
)


class _Deferred(*((_ENGINE_DEFERRED,) if _ENGINE_DEFERRED is not None else ())):
    """The engine's deferred top-k reduce with an ``add_into`` that records its calls instead of launching the reduce
    kernel."""

    calls = []

    def __init__(self, plane, scores):
        # The deferred reduce may be a frozen dataclass.
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "scores", scores)

    def add_into(self, shared):
        type(self).calls.append(shared)
        return shared


class _Slot:
    def __init__(self):
        self.published = []

    def publish(self, x, xq, xsf):
        self.published.append((x, xq, xsf))


ROWS = 1024
HS = 256  # a small hidden size keeps the CPU tensors small; the fold op is a fake


@pytest.fixture
def env(monkeypatch):
    """The ring boundary and the fold op present on an sm_107 device (fakes recording their calls), and ``_Deferred``
    recording the deferred reduce's calls."""
    if _ENGINE_DEFERRED is None:
        pytest.skip(_NO_DEFERRED)
    calls = types.SimpleNamespace(fold=[], stock=[], v2=[])
    _Deferred.calls = []

    def fold_op(shared, plane, scores, residual, weight, eps, write_norm, pdl):
        calls.fold.append((shared, plane, scores, residual, weight, eps, write_norm, pdl))
        if getattr(calls, "raise_", False):
            raise RuntimeError("minimax_m31_ctx_boundary_normq_topk: bad arguments")
        rows, hidden = shared.shape
        return (
            torch.zeros(rows, hidden, dtype=torch.bfloat16),
            torch.zeros(rows, hidden, dtype=torch.uint8),
            torch.zeros(rows * hidden // 32, dtype=torch.uint8),
        )

    def v2_op(inter, residual, weight, eps, write_norm, write_f32=False, fp4=None, pdl=False):
        calls.v2.append(inter)
        rows, hidden = inter.shape
        return (
            torch.zeros(rows, hidden, dtype=torch.bfloat16),
            torch.zeros(rows, hidden, dtype=torch.uint8),
            torch.zeros(rows * hidden // 32, dtype=torch.uint8),
        )

    slot = _Slot()
    monkeypatch.setattr(modeling, "get_sm_version", lambda: 107)
    monkeypatch.setattr(modeling, "_TOPK_FOLD", True)
    monkeypatch.setattr(modeling, "_TOPK_FOLD_STATE", {"op": fold_op})
    monkeypatch.setattr(modeling, "_TOPK_FOLD_SLOT", modeling._TopkFoldSlot())
    monkeypatch.setattr(modeling, "_NORMQ", True)
    monkeypatch.setitem(modeling._NORMQ_STATE, "op", object())
    monkeypatch.setitem(modeling._NORMQ_V2_STATE, "ops", (v2_op, object()))
    monkeypatch.setattr(modeling, "_dep2ctx_capture_blocked", lambda: False)
    monkeypatch.setattr(modeling, "_bcg_note", lambda item, x: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # eager
    monkeypatch.setattr(modeling, "_NORMQ_SLOT", slot)
    calls.slot = slot
    return calls


def _deferred(rows=ROWS, k=K, hidden=HS):
    plane = torch.randn(rows, k, hidden).bfloat16()
    scores = torch.rand(rows, k)
    return _Deferred(plane=plane, scores=scores)


class _Spec:
    def __init__(self, capture):
        self.capture = capture
        self.captured = []

    def is_layer_capture(self, layer_idx):
        return self.capture

    def maybe_capture_hidden_states(self, layer_idx, hidden_states, residual):
        self.captured.append(layer_idx)


def _layer(env, routed, feeds=True, raise_in_moe=False, fused_comm=True):
    """A decoder layer stand-in: the pre-MoE boundary is the identity, the MoE is the shared expert (zeros) plus
    _ctx_moe_tail_add of ``routed``, the stock next-layer boundary is recorded."""
    norm = types.SimpleNamespace(weight=torch.ones(HS, dtype=torch.bfloat16), variance_epsilon=1e-6)

    def moe(x, attn_metadata, lora_params=None):
        shared = torch.zeros_like(x)
        out = modeling._ctx_moe_tail_add(routed, shared)
        env.moe_out = out
        if raise_in_moe:
            raise ValueError("boom")
        return out

    moe.allreduce = None if fused_comm else object()
    moe.routed_output_is_global = fused_comm

    def stock(hidden_states, residual):
        env.stock.append((hidden_states, residual))
        return hidden_states, residual

    return types.SimpleNamespace(
        layer_idx=3,
        _normq_feeds_mxfp8=feeds,
        next_layer_layernorm=norm,
        block_sparse_moe=moe,
        _apply_pre_feed_forward_norm=lambda h, r, m=None: (h, r),
        _apply_next_layer_layernorm=stock,
    )


def _forward(layer, rows=ROWS, spec=None):
    x = torch.randn(rows, HS).bfloat16()
    residual = torch.randn(rows, HS).bfloat16()
    out = modeling.MiniMaxM3DecoderLayer.forward_MoE(
        layer, x, types.SimpleNamespace(all_rank_num_tokens=None), residual, spec_metadata=spec
    )
    return out, residual


def test_tail_add_parks_only_when_armed(env):
    deferred = _deferred()
    shared = torch.zeros(ROWS, HS).bfloat16()
    assert modeling._ctx_moe_tail_add(deferred, shared) is shared
    assert len(_Deferred.calls) == 1, "unarmed: the reduce runs in the tail add"
    modeling._TOPK_FOLD_SLOT.arm()
    shared2 = torch.zeros(ROWS, HS).bfloat16()
    assert modeling._ctx_moe_tail_add(deferred, shared2) is shared2
    assert len(_Deferred.calls) == 1, "armed: parked, not reduced"
    assert modeling._TOPK_FOLD_SLOT.take() == (deferred, shared2)
    assert not modeling._TOPK_FOLD_SLOT.armed and modeling._TOPK_FOLD_SLOT.parked is None


def test_forward_moe_takes_the_fold(env):
    deferred = _deferred()
    layer = _layer(env, deferred)
    (normed, new_residual), residual = _forward(layer)
    assert len(env.fold) == 1 and not _Deferred.calls and not env.stock and not env.v2
    shared, plane, scores, res, w, eps, write_norm, pdl = env.fold[0]
    assert shared is env.moe_out and new_residual is shared
    assert plane is deferred.plane and scores is deferred.scores
    assert torch.equal(res, residual) and w is layer.next_layer_layernorm.weight and eps == 1e-6
    assert write_norm is True and isinstance(pdl, bool)
    assert len(env.slot.published) == 1 and env.slot.published[0][0] is normed
    assert not modeling._TOPK_FOLD_SLOT.armed and modeling._TOPK_FOLD_SLOT.parked is None


def _case(name, env, monkeypatch):
    """(routed output, layer kwargs, forward kwargs, whether the reduce runs) of one decline case."""
    routed, layer_kw, fwd_kw, reduced = _deferred(), {}, {}, 1
    if name == "capture layer":
        # A capture layer declines only with TRTLLM_M31_DISABLE_DEP2CTX_TOPK_CAPTURE=1.
        monkeypatch.setattr(modeling, "_TOPK_CAPTURE", False)
        fwd_kw["spec"] = _Spec(True)
    elif name == "last layer":
        layer_kw["feeds"] = False
    elif name == "moe allreduce":
        layer_kw["fused_comm"] = False
    elif name == "rows 1023":
        fwd_kw["rows"] = 1023
        routed = _deferred(rows=1023)
    elif name == "capture blocked":
        monkeypatch.setattr(modeling, "_dep2ctx_capture_blocked", lambda: True)
    elif name == "op missing":
        monkeypatch.setitem(modeling._TOPK_FOLD_STATE, "op", None)
    elif name == "no v2":
        monkeypatch.setitem(modeling._NORMQ_V2_STATE, "ops", (None, None))
    elif name == "non-deferred":
        routed, reduced = torch.zeros(ROWS, HS).bfloat16(), 0
    elif name == "plane K 8":
        routed = _deferred(k=8)
    elif name == "scores rows":
        base = _deferred()
        routed = _Deferred(plane=base.plane, scores=torch.rand(ROWS - 1, K))
    elif name == "unaligned scores":
        base = _deferred()
        scores = torch.rand(ROWS * K + 1)[1:].view(ROWS, K)  # 4-byte offset
        routed = _Deferred(plane=base.plane, scores=scores)
    elif name == "op raises":
        env.raise_ = True
    else:
        raise AssertionError(name)
    return routed, layer_kw, fwd_kw, reduced


DECLINES = [
    "capture layer",
    "last layer",
    "moe allreduce",
    "rows 1023",
    "capture blocked",
    "op missing",
    "no v2",
    "non-deferred",
    "plane K 8",
    "unaligned scores",
    "scores rows",
    "op raises",
]


@pytest.mark.parametrize("name", DECLINES)
def test_fold_declines(env, monkeypatch, name):
    routed, layer_kw, fwd_kw, reduced = _case(name, env, monkeypatch)
    layer = _layer(env, routed, **layer_kw)
    out, residual = _forward(layer, **fwd_kw)
    assert len(_Deferred.calls) == reduced, "the deferred reduce runs exactly once on a decline"
    if reduced:
        assert _Deferred.calls[0] is env.moe_out
    assert len(env.stock) == 1 and env.stock[0][0] is env.moe_out, "the stock boundary follows"
    assert out == env.stock[0]
    assert len(env.fold) == (1 if name == "op raises" else 0)
    assert not modeling._TOPK_FOLD_SLOT.armed and modeling._TOPK_FOLD_SLOT.parked is None
    if name == "capture layer":
        assert fwd_kw["spec"].captured == [3]


def test_op_rejection_latches_the_fold_off(env):
    """A rejection is the device's or the build's (the plane checks passed): the next MoE layers decline before the
    MoE instead of calling the op again, and still reduce exactly once with the stock boundary."""
    env.raise_ = True
    for i in range(3):
        _forward(_layer(env, _deferred()))
        assert len(env.fold) == 1, "the op is called once, then latched off"
        assert len(_Deferred.calls) == i + 1 and len(env.stock) == i + 1
    assert modeling._topk_fold_decline_reason(
        _layer(env, _deferred()), torch.zeros(ROWS, HS).bfloat16(), None
    ).startswith("the fold op is not built")


def test_slot_disarmed_on_exception(env):
    layer = _layer(env, _deferred(), raise_in_moe=True)
    with pytest.raises(ValueError):
        _forward(layer)
    assert not modeling._TOPK_FOLD_SLOT.armed and modeling._TOPK_FOLD_SLOT.parked is None
    # the next (unarmed) tail add reduces as usual
    shared = torch.zeros(ROWS, HS).bfloat16()
    modeling._ctx_moe_tail_add(_deferred(), shared)
    assert len(_Deferred.calls) == 1


def test_switch_off_subprocess():
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import modeling as m; print(m._TOPK_FOLD)"
    )
    cases = {
        (): "True",
        (("TRTLLM_M31_DISABLE_DEP2CTX_TOPK_FOLD", "1"),): "False",
    }
    for extra, want in cases.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_M31_DISABLE_")}
        env.update(dict(extra))
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == want, (extra, out.stdout)


def test_fold_log_is_bounded(monkeypatch):
    monkeypatch.setattr(modeling, "_TOPK_FOLD_STATE", {})
    for rows in range(1024, 1024 + 64):
        modeling._topk_fold_log(
            "taken", f"top-k reduce + shared add + next-layer boundary (rows {rows})"
        )
        modeling._topk_fold_log("not taken", f"{rows - 1000} rows < 1024")
    logged = [k for k in modeling._TOPK_FOLD_STATE if isinstance(k, tuple) and k[0] == "log"]
    assert len(logged) == 2, logged


@pytest.mark.parametrize(
    "change, reason",
    [
        (dict(enabled=False), "switched off"),
        (dict(op_built=False), "not built"),
        (dict(normq=False), "NORMQ_V2"),
        (dict(feeds_next_layer=False), "last layer"),
        (dict(fused_comm=False), "fused-communication"),
        (dict(capture_layer=True), "capture layer"),
        (dict(x_ok=False), "2-D contiguous"),
        (dict(rows=1023), "1023 rows"),
        (dict(compiling=True), "torch.compile"),
        (dict(capture_blocked=True), "graph capture"),
    ],
)
def test_plan_fold_declines_name_the_first_failing_check(change, reason):
    ok = dict(
        enabled=True,
        op_built=True,
        normq=True,
        feeds_next_layer=True,
        fused_comm=True,
        capture_layer=False,
        x_ok=True,
        rows=8192,
        min_rows=1024,
        compiling=False,
        capture_blocked=False,
    )
    assert plan.topk_fold_decline_reason(**ok) is None
    got = plan.topk_fold_decline_reason(**dict(ok, **change))
    assert got is not None and reason in got
    assert (
        plan.topk_fold_decline_reason(**dict(ok, enabled=False, op_built=False)) == "switched off"
    )
    assert plan.topk_fold_plane_decline_reason(deferred=True, topk=4, aligned=True) is None
    assert "reduced its own" in plan.topk_fold_plane_decline_reason(
        deferred=False, topk=4, aligned=True
    )
    assert "top-8" in plan.topk_fold_plane_decline_reason(deferred=True, topk=8, aligned=True)
    assert "aligned" in plan.topk_fold_plane_decline_reason(deferred=True, topk=4, aligned=False)
