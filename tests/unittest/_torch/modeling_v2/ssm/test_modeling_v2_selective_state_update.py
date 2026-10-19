# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the selective_state_update entry (Mamba2 decode step).

Every cell is driven the way the decode target calls it: x/B/C as views of the
token-major conv output, dt / A / D / dt_bias as stride-0 broadcasts (the
kernel's tied-head path), and the pool laid out like the V2 hybrid manager's --
the layer's sub-page of every slot, so the slot stride spans all layers. The
whole underlying buffer is checked afterwards: the named sub-pages move, every
other byte stays.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.selective_state_update import (  # noqa: E501
    DSTATE,
    HEADDIM,
    NGROUPS,
    NHEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.selective_state_update import (  # noqa: E501
    selective_state_update as op,
)

__extra_import_path__ = [".."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "selective_state_update requires a CUDA device"

DEV = "cuda"
PAD_SLOT_ID = -1

#: `_gate` bands (element, relative RMS), in units of each output dtype's ULP.
Y_BAND = (2.5, 0.1)
FP16_STATE_BAND = (2.5, 0.25)
FP32_STATE_BAND = (8.0, 4.0)
#: The kernel's softplus is `log(1 + exp(v))` in fp32: dt is off by up to one
#: fp32 rounding of `1 + e^v`, absolute (measured 6.0e-8 for v < 0), and 1.3e-7
#: relative where dt is large (v up to 4). Allowed: 2^-23 + 2^-22 dt.
DT_ABS_ERR, DT_REL_ERR = 2.0**-23, 2.0**-22


def _gate(
    got: torch.Tensor,
    want: torch.Tensor,
    mag: torch.Tensor,
    band: tuple,
    unit: float,
    exposure: torch.Tensor,
    floor: float = 0.0,
) -> tuple:
    """`|got - want| <= band[0] * unit * mag + exposure + floor` elementwise, and
    the RMS of the difference, relative to the RMS of `want`, at most
    `band[1] * unit` plus the same ratio for `exposure`.

    `mag` is the reference run on |x|, |B|, |C|, |D| and |state|: the sum of the
    magnitudes of the terms each output is built from, the scale a rounding
    error lives at (a D skip cancelling C.S leaves a small output, not a small
    error). `exposure` is the output's sensitivity to dt times the kernel's dt
    error (`_dt_exposure`); `floor` is for fp16 states below the normal range,
    where a rounding is an absolute 2^-25, not a relative one.

    Derivation. Both sides run the step in fp32 and round once to the output
    dtype. Apart from fp32 association and the hardware exp (a few fp32 ULP of
    `mag`), they differ in dt: the kernel's softplus loses the low bits of
    `e^v` in `1 + e^v`, an absolute error that is a large fraction of a tiny dt
    (6% at v = -14) and is what an fp32 pool sees -- up to 3.7e-4 of `mag`
    where the increment `dt x B` dominates. Rounded to bf16 (y) or fp16 (the
    state) the two sides land on one value or one rounding apart: one spacing
    of the larger, at most 2^-7 (bf16) or 2^-10 (fp16) of it -- 2 ULP and a
    hair, banded at 2.5; measured 1.5 (y) and 2.0 (fp16 state). The RMS
    counts those flips (bitwise agreement 99.96% on y, 99.8% on the fp16
    state). An fp32 pool: 8 ULP of `mag` plus the exposure.
    """
    diff = (got.float() - want.float()).abs()
    allowed = band[0] * unit * mag.float() + exposure + floor
    worst = (diff / allowed.clamp_min(1e-30)).max().item()
    want_rms = want.float().pow(2).mean().sqrt()
    rel_rms = (diff.pow(2).mean().sqrt() / want_rms).item() / unit
    exposure_rms = (exposure.pow(2).mean().sqrt() / want_rms).item() / unit
    print(
        f"  element {worst:.3f} of its band; RMS {rel_rms:.4f} ULP "
        f"(band {band[1]} + exposure {exposure_rms:.4f})"
    )
    assert bool((diff <= allowed).all()), f"element error {worst:.3f} of the band"
    assert rel_rms <= band[1] + exposure_rms, f"relative RMS {rel_rms:.4f} ULP over its band"
    return worst, rel_rms


def _dt_exposure(args: list) -> tuple:
    """`(y, state)` exposure to the kernel's dt error: `eps(dt) * |d out / d dt|`,
    with `d S / d dt = A S0 e^(A dt) + x B` and `d y / d dt = C . d S / d dt`."""
    state, x, dt, A, B, C, _, _, dt_bias, _, slots = args[:11]
    rep = x.shape[1] // B.shape[1]
    live = (slots != PAD_SLOT_ID).view(-1, 1, 1, 1)
    s0 = torch.where(live, state[slots.long().clamp_min(0)].float().abs(), 0.0)
    v = dt.float() + dt_bias.float()
    dtf = torch.where(v <= 20.0, torch.log1p(torch.exp(v)), v)[..., None]
    a = A.float()
    xb = x.float().abs()[..., None] * B.float().abs().repeat_interleave(rep, 1)[:, :, None, :]
    state_exposure = (DT_ABS_ERR + DT_REL_ERR * dtf) * (a.abs() * s0 * torch.exp(a * dtf) + xb)
    c = C.float().abs().repeat_interleave(rep, 1)[:, :, None, :]
    return (state_exposure * c).sum(-1), state_exposure


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _case(spec: dict, seed: int) -> SimpleNamespace:
    """The layer's weights as broadcasts, one decode step's activations, a pool.

    The weights are drawn from the checkpoint's ranges: A_log in [-3.2, 6.1]
    (A down to -457), dt_bias in [-8.1, -1.3], D in [-1.13, 3.94], stored bf16.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    batch = spec["batch"]
    param_dtype = spec.get("param_dtype", torch.bfloat16)
    nheads, ngroups = spec.get("nheads", NHEADS), spec.get("ngroups", NGROUPS)

    def uniform(lo, hi):
        return torch.empty(nheads, device=DEV).uniform_(lo, hi, generator=gen)

    A = -torch.exp(uniform(-3.2, 6.1))
    dt_bias = uniform(-8.1, -1.3).to(torch.bfloat16).to(param_dtype)
    D = uniform(-1.13, 3.94).to(torch.bfloat16).to(param_dtype)

    xw, bw = nheads * HEADDIM, ngroups * DSTATE
    width = xw + 2 * bw  # x | B | C: 4608 at tep4, 18432 at dep4
    conv = F.silu(torch.randn(batch, width, device=DEV, generator=gen) * 2).to(torch.bfloat16)
    proj = torch.randn(batch, xw + width + nheads, device=DEV, generator=gen).to(torch.bfloat16)

    slots_total = spec.get("pool_slots", 40)
    layers = spec.get("pool_layers", 3)
    base = torch.empty(
        slots_total,
        layers,
        nheads,
        HEADDIM,
        DSTATE,
        device=DEV,
        dtype=spec.get("pool_dtype", torch.float16),
    )
    base.fill_(1.5)  # the other layers' sub-pages: a value the op must not touch
    pool = base[:, layers // 2]
    pool.copy_(torch.randn(pool.shape, device=DEV, generator=gen) * 3)
    if "slots" in spec:
        slots = list(spec["slots"])
    else:
        perm = torch.randperm(slots_total, generator=torch.Generator().manual_seed(seed))
        slots = perm[:batch].tolist()
    for row in spec.get("pads", ()):
        slots[row] = PAD_SLOT_ID
    return SimpleNamespace(
        base=base,
        dt_raw=proj[:, -nheads:],
        args=[
            pool,
            conv[:, :xw].view(batch, nheads, HEADDIM),
            proj[:, -nheads:][:, :, None].expand(batch, nheads, HEADDIM),
            A[:, None, None].expand(nheads, HEADDIM, DSTATE),
            conv[:, xw : xw + bw].view(batch, ngroups, DSTATE),
            conv[:, xw + bw :].view(batch, ngroups, DSTATE),
            D[:, None].expand(nheads, HEADDIM),
            None,
            dt_bias[:, None].expand(nheads, HEADDIM),
            True,
            torch.tensor(slots, dtype=torch.int32, device=DEV),
            PAD_SLOT_ID,
            torch.empty(batch, nheads, HEADDIM, dtype=torch.bfloat16, device=DEV),
        ],
    )


def _magnitude(args: list) -> tuple:
    """`reference` on the absolute values, in fp32: `_gate`'s scale."""
    mags = list(args)
    for i in (0, 1, 4, 5, 6):  # state, x, B, C, D
        mags[i] = mags[i].abs().float() if i == 0 else mags[i].abs()
    mags[12] = torch.empty((), dtype=torch.float32)  # out: only its dtype is read
    return op.reference(*mags)


def _check(case: SimpleNamespace, before: torch.Tensor) -> None:
    """`out` and the named slots against `reference`, every other byte bitwise."""
    args = case.args
    pool, slots, out = args[0], args[10], args[12]
    ref_args = list(args)
    layer = case.base.shape[1] // 2
    ref_args[0] = before[:, layer]
    ref_y, ref_pool = op.reference(*ref_args)
    mag_y, mag_pool = _magnitude(ref_args)
    exposure_y, exposure_state = _dt_exposure(ref_args)
    _gate(out, ref_y, mag_y, Y_BAND, ULP[torch.bfloat16], exposure_y)
    rows = slots != PAD_SLOT_ID
    live = slots[rows].long()
    if pool.dtype == torch.float16:
        band, unit, floor = FP16_STATE_BAND, ULP[torch.float16], 2.0**-24
    else:
        band, unit, floor = FP32_STATE_BAND, ULP[torch.float32], 0.0
    _gate(pool[live], ref_pool[live], mag_pool[live], band, unit, exposure_state[rows], floor)
    expect = before.clone()
    expect[live, layer] = case.base[live, layer]
    assert torch.equal(case.base, expect), "the op wrote outside the named slots' sub-pages"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    case = _case(cell.spec, _seed(cell.why))
    before = case.base.clone()
    returned = op(*case.args)
    assert returned is case.args[12], "returns the out it wrote"
    _check(case, before)
    # Non-vacuity for the slots past 2^31 elements (int64 slot offsets): they
    # really moved. Every cell naming such a slot is checked.
    layer = case.base.shape[1] // 2
    deep = [s for s in case.args[10].tolist() if s >= 0 and s * case.base.stride(0) > 2**31]
    if cell.spec.get("pool_slots", 0) > 85:
        assert deep, "the deep-pool cell must name slots past 2^31 elements"
    for slot in deep:
        assert not torch.equal(case.base[slot, layer], before[slot, layer]), slot


def test_tied_and_dense_operands_agree() -> None:
    """The stride-0 broadcasts take the tied-head path; dense copies of the same
    values take the general one. Same numbers, either way."""
    spec = dict(batch=7)
    tied = _case(spec, seed=21)
    dense = _case(spec, seed=21)
    for i in (2, 3, 6, 8):  # dt, A, D, dt_bias
        dense.args[i] = dense.args[i].contiguous()
    op(*tied.args)
    op(*dense.args)
    assert torch.equal(tied.args[12], dense.args[12])
    assert torch.equal(tied.base, dense.base)


def test_decode_step_replays_under_a_cuda_graph() -> None:
    """Captured once, replayed with new activations and new slots: each replay
    is bitwise the eager call on the same inputs."""
    spec = dict(batch=7)
    case = _case(spec, seed=22)
    args = case.args
    op(*args)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    gen = torch.Generator(device=DEV).manual_seed(23)
    for trial in range(2):
        fresh = _case(spec, seed=30 + trial)
        # New inputs into the captured buffers (dt is a broadcast view of its base).
        args[1].copy_(fresh.args[1])
        case.dt_raw.copy_(fresh.dt_raw)
        args[4].copy_(fresh.args[4])
        args[5].copy_(fresh.args[5])
        args[10].copy_(torch.randperm(40, generator=torch.Generator().manual_seed(trial))[:7])
        args[0].copy_(torch.randn(args[0].shape, device=DEV, generator=gen) * 3)
        eager_base = case.base.clone()
        eager_args = list(args)
        eager_args[0] = eager_base[:, eager_base.shape[1] // 2]
        eager_args[12] = torch.empty_like(args[12])
        op(*eager_args)
        graph.replay()
        assert torch.equal(args[12], eager_args[12])
        assert torch.equal(case.base, eager_base)


def test_decode_step_issues_no_host_sync() -> None:
    case = _case(dict(batch=32), seed=24)
    op(*case.args)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        op(*case.args)
    finally:
        torch.cuda.set_sync_debug_mode("default")


#: Even heads first, then odd: position j holds a head h with h % G == j // (H / G).
_INTERLEAVED = torch.arange(NHEADS).view(NHEADS // NGROUPS, NGROUPS).t().reshape(-1)


@pytest.mark.parametrize("wrong", ["no_softplus", "no_dt_bias", "no_d_skip", "groups_interleaved"])
def test_band_rejects_a_wrong_reference(wrong: str) -> None:
    """The gate is tight enough to see a plausible misreading of the step."""
    case = _case(dict(batch=7), seed=25)
    before = case.base.clone()
    op(*case.args)
    args = list(case.args)
    layer = case.base.shape[1] // 2
    args[0] = before[:, layer]
    bad = list(args)
    if wrong == "no_softplus":
        bad[9] = False
    elif wrong == "no_dt_bias":
        bad[8] = torch.zeros_like(args[8])
    elif wrong == "no_d_skip":
        bad[6] = torch.zeros_like(args[6])
    elif wrong == "groups_interleaved":
        order = _INTERLEAVED
        bad[0] = args[0][:, order]
        bad[1] = args[1][:, order]
        bad[2] = args[2][:, order]
        bad[3] = args[3][order]
        bad[6] = args[6][order]
        bad[8] = args[8][order]
    ref_y, ref_pool = op.reference(*bad)
    if wrong == "groups_interleaved":
        inverse = torch.argsort(_INTERLEAVED)
        ref_y, ref_pool = ref_y[:, inverse], ref_pool[:, inverse]
    mag_y, _ = _magnitude(args)
    exposure_y, _ = _dt_exposure(args)
    with pytest.raises(AssertionError):
        _gate(case.args[12], ref_y, mag_y, Y_BAND, ULP[torch.bfloat16], exposure_y)


_REJECTED = ("d_per_head", "dt_bias_per_head", "dt_per_head", "a_per_head", "out_none")


@pytest.mark.parametrize("case_name", _REJECTED)
def test_op_rejects_loudly_what_is_valid_does_not_restate(case_name: str) -> None:
    """The per-head `[64]` forms the model stores are refused, not misread."""
    case = _case(dict(batch=7), seed=26)
    args = case.args
    expected = AssertionError  # the wrapper's shape asserts
    if case_name == "d_per_head":
        args[6] = args[6][:, 0].contiguous()
    elif case_name == "dt_bias_per_head":
        args[8] = args[8][:, 0].contiguous()
    elif case_name == "dt_per_head":
        args[2] = args[2][:, :, 0]
    elif case_name == "a_per_head":
        args[3] = args[3][:, 0, 0].contiguous()
    elif case_name == "out_none":
        args[12] = None
        expected = AttributeError
    with pytest.raises(expected):
        with validating(op):
            op(*args)


def _guard_case(hazard: str) -> SimpleNamespace:
    """A fresh case exhibiting one silent violation, on a 6-slot view of a
    larger buffer: the slots past the view are real memory the caller does
    not consider part of the pool."""
    case = _case(dict(batch=3, pool_slots=8, slots=(0, 1, 2)), seed=27)
    case.view_slots = 6
    case.args[0] = case.args[0][: case.view_slots]
    slots = case.args[10]
    if hazard == "duplicate_slot":
        slots[2] = 0
    elif hazard == "slot_outside_view":
        slots[2] = 7
    elif hazard == "pad_id_is_a_slot":
        case.args[11] = 1  # row 1 names slot 1, the "pad" id
    return case


@pytest.mark.parametrize("hazard", ["duplicate_slot", "slot_outside_view", "pad_id_is_a_slot"])
def test_guard_refuses_what_the_op_answers_silently_wrong(hazard: str) -> None:
    """First the hazard, unguarded: the op returns without raising and one row's
    update is lost or lands outside the pool. Then the guard refuses it."""
    case = _guard_case(hazard)
    layer = case.base.shape[1] // 2
    before = case.base.clone()
    op(*case.args)
    moved = [
        s
        for s in range(case.base.shape[0])
        if not torch.equal(case.base[s, layer], before[s, layer])
    ]
    if hazard == "duplicate_slot":
        assert moved == [0, 1], "two rows into slot 0: one update is all that remains"
    elif hazard == "slot_outside_view":
        assert 7 in moved, "the write landed past the 6-slot view"
    elif hazard == "pad_id_is_a_slot":
        assert 1 not in moved, "row 1's real slot was skipped as padding"

    with pytest.raises(AssertionError):
        with validating(op):
            op(*_guard_case(hazard).args)
