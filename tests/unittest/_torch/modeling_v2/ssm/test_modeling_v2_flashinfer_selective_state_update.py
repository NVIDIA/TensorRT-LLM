# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_selective_state_update entry (Mamba2 decode step, FlashInfer).

Every cell is driven the way the nemotron_h BF16 targets call it: x/B/C as views of the token-major
conv output, dt / A / D / dt_bias as stride-0 broadcasts (the only form the kernel takes), D and
dt_bias in bf16 as the checkpoint stores them, and the pool laid out like the V2 hybrid manager's --
the layer's sub-page of every slot, so the slot stride spans all layers. The whole underlying buffer
is checked afterwards: the named sub-pages move, every other byte stays.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.flashinfer_selective_state_update import (  # noqa: E501
    DSTATE,
    ENGINE_SLOT_LAYERS,
    ENGINE_SLOTS_256,
    HEADDIM,
    NGROUPS,
    NHEADS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.flashinfer_selective_state_update import (  # noqa: E501
    flashinfer_selective_state_update as op,
)

__extra_import_path__ = [".."]
from _validating import ULP, certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_selective_state_update requires a CUDA device"

DEV = "cuda"
PAD_SLOT_ID = -1

#: `_gate` bands (element, relative RMS), in units of each output dtype's ULP: the Triton entry's,
#: with the intrinsics' own error carried by the exposure below rather than by the band.
Y_BAND = (2.5, 0.1)
FP32_STATE_BAND = (8.0, 4.0)
#: The kernel's softplus is `__logf(1 + __expf(v))`. From the documented bounds of the intrinsics:
#: `__logf` is within 2^-21.41 absolute for an argument in [0.5, 2] (v <= 0) and 3 ULP elsewhere;
#: `__expf(v)` within 2 + floor(1.173 |v|) ULP, which reaches dt scaled by e^v / (1 + e^v) <= dt;
#: the sum `1 + e^v` rounds once (2^-24). So |dt error| <= 2^-21.41 + 2^-24 + (5 + 1.173 |v|) ULP
#: of dt. The decay `__expf(A dt)` is within 2 + floor(1.173 |A dt|) ULP of itself.
DT_ABS_ERR = 2.0**-21.41 + 2.0**-24
U32 = 2.0**-23


def _gate(
    got: torch.Tensor,
    want: torch.Tensor,
    mag: torch.Tensor,
    band: tuple,
    unit: float,
    exposure: torch.Tensor,
) -> tuple:
    """`|got - want| <= band[0] * unit * mag + exposure` elementwise, and the RMS of the
    difference, relative to the RMS of `want`, at most `band[1] * unit` plus the same ratio for
    `exposure`.

    `mag` is the reference run on |x|, |B|, |C|, |D| and |state|: the sum of the magnitudes of the
    terms each output is built from, the scale a rounding error lives at. `exposure` is what the
    intrinsics' error can move the output by (`_exposure`). Both sides run the step in fp32 and
    round once; apart from the intrinsics they differ by fp32 association, a few fp32 ULP of `mag`
    -- in bf16 `y` one rounding flip at most (2.5 ULP), in the fp32 state 8 ULP of `mag`.
    """
    diff = (got.float() - want.float()).abs()
    allowed = band[0] * unit * mag.float() + exposure
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


def _exposure(args: list) -> tuple:
    """`(y, state)` exposure to the intrinsics' error: the dt error times `|d S / d dt| =
    |A S0 e^(A dt)| + |x B|`, plus the decay's own error on `|S0| e^(A dt)`; `y` sees the state's
    through `|C|`."""
    state, x, dt, A, B, C, _, _, dt_bias, _, slots = args[:11]
    rep = x.shape[1] // B.shape[1]
    live = (slots != PAD_SLOT_ID).view(-1, 1, 1, 1)
    s0 = torch.where(live, state[slots.long().clamp_min(0)].float().abs(), 0.0)
    v = dt.float() + dt_bias.float()
    dtf = torch.where(v <= 20.0, torch.log1p(torch.exp(v)), v)
    dt_err = (DT_ABS_ERR + dtf * (5.0 + 1.173 * v.abs()) * U32)[..., None]
    a = A.float()
    adt = a * dtf[..., None]
    decayed = s0 * torch.exp(adt)
    xb = x.float().abs()[..., None] * B.float().abs().repeat_interleave(rep, 1)[:, :, None, :]
    state_exposure = dt_err * (a.abs() * decayed + xb) + decayed * (2.0 + 1.173 * adt.abs()) * U32
    c = C.float().abs().repeat_interleave(rep, 1)[:, :, None, :]
    return (state_exposure * c).sum(-1), state_exposure


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _case(spec: dict, seed: int) -> SimpleNamespace:
    """The layer's weights as broadcasts, one decode step's activations, a pool.

    The weights are drawn from the checkpoint's ranges: A_log in [-3.2, 6.1] (A down to -457),
    dt_bias in [-8.1, -1.3], D in [-1.13, 3.94], stored bf16.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    batch = spec["batch"]

    def uniform(lo, hi):
        return torch.empty(NHEADS, device=DEV).uniform_(lo, hi, generator=gen)

    A = -torch.exp(uniform(-3.2, 6.1))
    dt_bias = uniform(-8.1, -1.3).to(torch.bfloat16)
    D = uniform(-1.13, 3.94).to(torch.bfloat16)

    xw, bw = NHEADS * HEADDIM, NGROUPS * DSTATE
    width = xw + 2 * bw  # x | B | C: 2304 at the targets' rank
    conv = F.silu(torch.randn(batch, width, device=DEV, generator=gen) * 2).to(torch.bfloat16)
    proj = torch.randn(batch, xw + width + NHEADS, device=DEV, generator=gen).to(torch.bfloat16)

    slots_total = spec.get("pool_slots", ENGINE_SLOTS_256)
    layers = spec.get("pool_layers", ENGINE_SLOT_LAYERS)
    base = torch.empty(slots_total, layers, NHEADS, HEADDIM, DSTATE, device=DEV)
    base.fill_(1.5)  # the other layers' sub-pages: a value the op must not touch
    layer = layers // 2
    pool = base[:, layer]
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
        layer=layer,
        dt_raw=proj[:, -NHEADS:],
        args=[
            pool,
            conv[:, :xw].view(batch, NHEADS, HEADDIM),
            proj[:, -NHEADS:][:, :, None].expand(batch, NHEADS, HEADDIM),
            A[:, None, None].expand(NHEADS, HEADDIM, DSTATE),
            conv[:, xw : xw + bw].view(batch, NGROUPS, DSTATE),
            conv[:, xw + bw :].view(batch, NGROUPS, DSTATE),
            D[:, None].expand(NHEADS, HEADDIM),
            None,
            dt_bias[:, None].expand(NHEADS, HEADDIM),
            True,
            torch.tensor(slots, dtype=torch.int32, device=DEV),
            PAD_SLOT_ID,
            torch.empty(batch, NHEADS, HEADDIM, dtype=torch.bfloat16, device=DEV),
        ],
    )


def _magnitude(args: list) -> tuple:
    """`reference` on the absolute values, in fp32: `_gate`'s scale."""
    mags = list(args)
    for i in (0, 1, 4, 5, 6):  # state, x, B, C, D
        mags[i] = mags[i].abs()
    mags[12] = torch.empty((), dtype=torch.float32)  # out: only its dtype is read
    return op.reference(*mags)


def _check(case: SimpleNamespace, before: torch.Tensor) -> None:
    """`out` and the named slots against `reference`, every other byte bitwise."""
    args = case.args
    pool, slots, out = args[0], args[10], args[12]
    ref_args = list(args)
    ref_args[0] = before[:, case.layer]
    ref_y, ref_pool = op.reference(*ref_args)
    mag_y, mag_pool = _magnitude(ref_args)
    exposure_y, exposure_state = _exposure(ref_args)
    _gate(out, ref_y, mag_y, Y_BAND, ULP[torch.bfloat16], exposure_y)
    rows = slots != PAD_SLOT_ID
    live = slots[rows].long()
    _gate(
        pool[live],
        ref_pool[live],
        mag_pool[live],
        FP32_STATE_BAND,
        ULP[torch.float32],
        exposure_state[rows],
    )
    expect = before.clone()
    expect[live, case.layer] = case.base[live, case.layer]
    assert torch.equal(case.base, expect), "the op wrote outside the named slots' sub-pages"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    case = _case(cell.spec, _seed(cell.why))
    before = case.base.clone()
    returned = op(*case.args)
    assert returned is case.args[12], "returns the out it wrote"
    _check(case, before)
    # Non-vacuity for the slots past 2^31 elements (int64 slot offsets): they really moved.
    deep = [s for s in case.args[10].tolist() if s >= 0 and s * case.base.stride(0) > 2**31]
    if cell.spec.get("pool_slots", ENGINE_SLOTS_256) == ENGINE_SLOTS_256:
        assert deep, "a cell at the 271-slot pool must name slots past 2^31 elements"
    for slot in deep:
        assert not torch.equal(case.base[slot, case.layer], before[slot, case.layer]), slot


def test_decode_step_replays_under_a_cuda_graph() -> None:
    """Captured once, replayed with new activations and new slots: each replay is bitwise the
    eager call on the same inputs."""
    spec = dict(batch=64, pool_slots=80, pool_layers=3)
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
        args[10].copy_(torch.randperm(80, generator=torch.Generator().manual_seed(trial))[:64])
        args[0].copy_(torch.randn(args[0].shape, device=DEV, generator=gen) * 3)
        eager_base = case.base.clone()
        eager_args = list(args)
        eager_args[0] = eager_base[:, case.layer]
        eager_args[12] = torch.empty_like(args[12])
        op(*eager_args)
        graph.replay()
        assert torch.equal(args[12], eager_args[12])
        assert torch.equal(case.base, eager_base)


def test_decode_step_issues_no_host_sync() -> None:
    case = _case(dict(batch=64, pool_slots=80, pool_layers=3), seed=24)
    op(*case.args)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        op(*case.args)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("wrong", ["no_softplus", "no_dt_bias", "no_d_skip"])
def test_band_rejects_a_wrong_reference(wrong: str) -> None:
    """The gate is tight enough to see a plausible misreading of the step."""
    case = _case(dict(batch=32, pool_slots=40, pool_layers=3), seed=25)
    before = case.base.clone()
    op(*case.args)
    args = list(case.args)
    args[0] = before[:, case.layer]
    bad = list(args)
    if wrong == "no_softplus":
        bad[9] = False
    elif wrong == "no_dt_bias":
        bad[8] = torch.zeros_like(args[8])
    elif wrong == "no_d_skip":
        bad[6] = torch.zeros_like(args[6])
    ref_y, _ = op.reference(*bad)
    mag_y, _ = _magnitude(args)
    exposure_y, _ = _exposure(args)
    with pytest.raises(AssertionError):
        _gate(case.args[12], ref_y, mag_y, Y_BAND, ULP[torch.bfloat16], exposure_y)


_REJECTED = ("d_fp32", "dt_bias_fp32", "d_per_head", "d_dense", "a_per_head", "x_64b_aligned")


@pytest.mark.parametrize("case_name", _REJECTED)
def test_op_rejects_loudly_what_is_valid_does_not_restate(case_name: str) -> None:
    """The forms the model also stores -- fp32 D and dt_bias (the Triton entry's), per-head `[32]`
    vectors, dense copies -- and x/B/C rows that start only 64-byte aligned (a mixed step's
    generation rows after an odd number of context rows) are refused by the kernel's own checks,
    not misread."""
    case = _case(dict(batch=32, pool_slots=40, pool_layers=3), seed=26)
    args = case.args
    if case_name == "x_64b_aligned":
        # The same rows one 64-byte step into a wider buffer: every row stride stays legal.
        wide = torch.zeros(
            32, 2 * NHEADS * HEADDIM + 2 * DSTATE + 32, dtype=torch.bfloat16, device=DEV
        )
        x = wide[:, 32 : 32 + NHEADS * HEADDIM]
        assert x.data_ptr() % 128 == 64
        x.copy_(args[1].reshape(32, -1))
        args[1] = x.view(32, NHEADS, HEADDIM)
    if case_name == "d_fp32":
        args[6] = args[6].float()
    elif case_name == "dt_bias_fp32":
        args[8] = args[8].float()
    elif case_name == "d_per_head":
        args[6] = args[6][:, 0].contiguous()
    elif case_name == "d_dense":
        args[6] = args[6].contiguous()
    elif case_name == "a_per_head":
        args[3] = args[3][:, 0, 0].contiguous()
    with pytest.raises(Exception) as raised:
        with validating(op):
            op(*args)
    assert not isinstance(raised.value, AssertionError), "refused by the guard, not the kernel"


def _guard_case(hazard: str) -> SimpleNamespace:
    """A fresh case exhibiting one silent violation, on a 34-slot view of a larger buffer: the
    slots past the view are real memory the caller does not consider part of the pool."""
    case = _case(dict(batch=32, pool_slots=36, pool_layers=3, slots=tuple(range(32))), seed=27)
    case.view_slots = 34
    case.args[0] = case.args[0][: case.view_slots]
    slots = case.args[10]
    if hazard == "duplicate_slot":
        slots[2] = 0
    elif hazard == "slot_outside_view":
        slots[2] = 35
    elif hazard == "pad_id_is_a_slot":
        case.args[11] = 1  # row 1 names slot 1, the "pad" id
    return case


@pytest.mark.parametrize("hazard", ["duplicate_slot", "slot_outside_view", "pad_id_is_a_slot"])
def test_guard_refuses_what_the_op_answers_silently_wrong(hazard: str) -> None:
    """First the hazard, unguarded: the op returns without raising and one row's update is lost.
    Then the guard refuses it."""
    case = _guard_case(hazard)
    before = case.base.clone()
    op(*case.args)
    moved = [
        s
        for s in range(case.base.shape[0])
        if not torch.equal(case.base[s, case.layer], before[s, case.layer])
    ]
    if hazard == "duplicate_slot":
        assert 2 not in moved and 0 in moved, "two rows into slot 0: row 2's slot never moves"
    elif hazard == "slot_outside_view":
        # The vertical kernel reaches the pool through a tensor map bounded by the view: a slot
        # past it is read as zeros and its store is dropped, so row 2's update goes nowhere --
        # not past the view, as the Triton kernel's does, but just as silently.
        assert 35 not in moved and 2 not in moved, "row 2's update landed somewhere"
    elif hazard == "pad_id_is_a_slot":
        assert 1 not in moved, "row 1's real slot was skipped as padding"

    with pytest.raises(AssertionError):
        with validating(op):
            op(*_guard_case(hazard).args)
