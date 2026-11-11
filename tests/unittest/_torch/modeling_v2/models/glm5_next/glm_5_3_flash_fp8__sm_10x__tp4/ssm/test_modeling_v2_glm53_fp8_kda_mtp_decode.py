# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_mtp_decode operation of the glm5_next FP8 target.

Inputs as the KDA layer hands them over in a verify step: x_q / x_k / x_v column sections of the fused
``[q | k | v]`` projection's packed rows (1 + num_spec per request), the forget and output gates the two halves of
the batched ``[f_b; g_b]`` GEMM's output, beta the H-wide tail of the ``[f_a | g_a | b]`` projection's rows. Every
pool (state, replay conv caches, k / g / v / beta caches) is a view into wider per-slot records, as the hybrid
cache manager lays them out, and the accepted-draft counts are the first int32 of a 16-byte per-slot record. The
replay caches hold what a previous verify would have left: unit-norm keys, log-decays in (lower_bound, 0), SiLU
values and sigmoid betas. The reference is fp32 torch, token by token.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.ssm.kda_mtp_decode import (  # noqa: E501
    kda_mtp_decode as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 -- fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

#: GLM-5.3-Flash's KDA at TP4: 64 heads over 4 ranks, K = V = 128, conv width 4 (a 3-tap window).
HEADS, HEAD_DIM, WINDOW = 16, 128, 3
DIM = HEADS * HEAD_DIM
#: gate_lower_bound, the q scale K ** -0.5 and rms_norm_eps (the output norm's epsilon).
LOWER_BOUND, SCALE, ONORM_EPS = -5.0, HEAD_DIM**-0.5, 1e-5
#: The [f_a | g_a | b] projection's row width at TP4: beta is its last HEADS columns.
BFA_WIDTH = 2 * HEAD_DIM + HEADS
#: Elements past each slot's row in its record (the other layers' rows in the real manager).
SLOT_PAD = 1024
STATE_ROW = HEADS * HEAD_DIM * HEAD_DIM
#: The committed fp32 state's band, as a fraction of the used rows' largest entry (see the cells test).
STATE_BAND = 1e-4

# The MTP-3 verify steps the shipped target runs, by target directory under `models/`: GLM-5.3-Flash's KDA at
# TP4, 1 + 3 tokens per request, CUDA-graph batches up to 64 requests, previous rounds accepting 0..3 drafts;
# the pool bf16 (the default recurrent-state dtype) or fp32.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="MTP-3 verify of 1 request, 2 drafts pending, bf16 pool: the cluster-pair kernel",
        spec=dict(batch=1, num_spec=3, pending=(2,), state_dtype=torch.bfloat16, pool=8),
    ),
    Cell(
        why="MTP-3 verify of 8 requests, 0..3 drafts pending, bf16 pool: one CTA per (head, request)",
        spec=dict(
            batch=8,
            num_spec=3,
            pending=(0, 1, 2, 3, 3, 2, 1, 0),
            state_dtype=torch.bfloat16,
            pool=12,
        ),
    ),
    Cell(
        why="MTP-3 verify of 64 requests, 0..3 drafts pending, bf16 pool: the two-CTA-per-SM kernel",
        spec=dict(
            batch=64,
            num_spec=3,
            pending=tuple(n % 4 for n in range(64)),
            state_dtype=torch.bfloat16,
            pool=72,
        ),
    ),
    Cell(
        why="MTP-3 verify of 8 requests on an fp32 pool (mamba_ssm_cache_dtype float32)",
        spec=dict(
            batch=8,
            num_spec=3,
            pending=(3, 0, 2, 1, 1, 3, 0, 2),
            state_dtype=torch.float32,
            pool=12,
        ),
    ),
)


def _heads(cols: torch.Tensor) -> torch.Tensor:
    """``[T, H * 128]`` columns as the op's ``[1, T, H, 128]`` view (the row stride kept)."""
    return cols.unflatten(-1, (HEADS, HEAD_DIM)).unsqueeze(0)


def _records(pool: int, width: int, dtype: torch.dtype, values: torch.Tensor) -> torch.Tensor:
    """A ``[pool, width + SLOT_PAD]`` record buffer whose first ``width`` columns hold ``values``."""
    record = torch.full((pool, width + SLOT_PAD), 7.0, dtype=dtype, device="cuda")
    record[:, :width] = values.reshape(pool, width).to(dtype)
    return record


def _build(spec: dict, seed: int):
    """``(records, slots, args)``: the per-slot record buffers, the batch's slots and the op's arguments."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch, num_spec, pool = spec["batch"], spec["num_spec"], spec["pool"]
    tokens = batch * (1 + num_spec)
    cols = WINDOW + num_spec

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    qkv = rnd(tokens, 3 * DIM).to(torch.bfloat16)
    gates = rnd(2, tokens, DIM).to(torch.bfloat16)
    bfa = rnd(tokens, BFA_WIDTH).to(torch.bfloat16)
    taps = (
        rnd(3, DIM, WINDOW + 1, scale=0.3).to(torch.bfloat16).float()
    )  # checkpoint bf16 taps, held fp32
    a_log = torch.empty(HEADS, device="cuda").uniform_(1, 16, generator=gen).log()
    dt_bias = torch.empty(DIM, device="cuda").uniform_(
        math.log(1e-3), math.log(1e-1), generator=gen
    )
    onorm_weight = 1 + 0.1 * rnd(HEAD_DIM)

    conv_records = [
        _records(pool, cols * DIM, torch.bfloat16, rnd(pool, cols, DIM, scale=0.5))
        for _ in range(3)
    ]
    convs = [r[:, : cols * DIM].view(pool, cols, DIM).transpose(1, 2) for r in conv_records]
    state_record = _records(pool, STATE_ROW, spec["state_dtype"], rnd(pool, STATE_ROW, scale=0.05))
    state = state_record[:, :STATE_ROW].view(pool, HEADS, HEAD_DIM, HEAD_DIM)
    keys = rnd(pool, num_spec, HEADS, HEAD_DIM)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    cache_values = (
        (keys, torch.bfloat16, DIM),
        (LOWER_BOUND * torch.sigmoid(rnd(pool, num_spec, DIM)), torch.float32, DIM),
        (torch.nn.functional.silu(rnd(pool, num_spec, DIM)), torch.bfloat16, DIM),
        (torch.sigmoid(rnd(pool, num_spec, HEADS)), torch.bfloat16, HEADS),
    )
    cache_records = [
        _records(pool, num_spec * width, dtype, values) for values, dtype, width in cache_values
    ]
    caches = [
        r[:, : num_spec * width].view(pool, num_spec, width)
        for r, (_, _, width) in zip(cache_records, cache_values)
    ]
    count_record = torch.zeros(pool, 4, dtype=torch.int32, device="cuda")
    num_accepted = count_record[:, 0]
    slots = torch.randperm(pool, generator=gen, device="cuda")[:batch].to(torch.int32)
    num_accepted[slots.long()] = torch.tensor(spec["pending"], dtype=torch.int32, device="cuda")
    args = (
        _heads(qkv[:, :DIM]),
        _heads(qkv[:, DIM : 2 * DIM]),
        _heads(qkv[:, 2 * DIM :]),
        *(taps[i].contiguous() for i in range(3)),
        *convs,
        _heads(gates[0]),
        bfa[:, 2 * HEAD_DIM :].unsqueeze(0),
        a_log,
        dt_bias,
        state,
        *caches,
        slots,
        num_accepted,
        num_spec,
        LOWER_BOUND,
        SCALE,
        _heads(gates[1]),
        onorm_weight,
        ONORM_EPS,
    )
    records = [*conv_records, state_record, *cache_records, count_record]
    return records, slots, args


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    """The output, the committed state, the conv windows and the replay caches against the fp32 reference;
    unused slots, record padding and the accepted counts untouched.

    Tolerances. The recurrence runs on tensor cores in 3xTF32 (within about 2e-6 of fp32); gates, SiLU, norms
    and sigmoids use fast exp / rcp / rsqrt (about 1e-6 relative). The output is the gated RMS norm of the raw
    output, which the kernel first rounds to bf16 (2^-9 relative, carried through the norm), then rounded once
    to bf16 on both sides: at most 1.25 bf16 ulps apart, i.e. 2.5 units of 2^-8 of a head row's largest
    magnitude; ``element_ulp=4`` leaves 1.5 units for the rest, and the relative RMS of the three roundings is
    about 0.7 of 2^-8 (``rms_ulp=1.5``). The committed state: an fp32 pool within 1e-4 of the used rows'
    largest entry (50 times the 3xTF32 error); a bf16 pool is rounded to nearest on both sides from values
    that close, so entries differ by at most one bf16 ulp (``rtol`` 2^-7) plus that fp32 band. The replay k / v
    / beta caches are bf16 roundings of values within about 1e-6 relative, so at most one ulp apart
    (``rtol`` 2^-7, ``atol`` 1e-5 for entries near zero); the fp32 log-decay cache is ``lower_bound`` times an
    approximate sigmoid, within about 5e-6 absolute (``atol`` 1e-5). The conv caches are raw-input copies: bit
    equality.
    """
    certifying(op)
    records, slots, args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = [r.clone() for r in records]
    x_v, state = args[2], args[13]
    want = op.reference(*args)
    out = op(*args)
    assert out.shape == x_v.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert_within_ulp(
        out.reshape(-1, HEAD_DIM), want[0].reshape(-1, HEAD_DIM), element_ulp=4, rms_ulp=1.5
    )
    rows = slots.long()
    got_state, want_state = state[rows].float(), want[1][rows].float()
    fp32_band = STATE_BAND * want_state.abs().max().item()
    if state.dtype == torch.float32:
        torch.testing.assert_close(got_state, want_state, rtol=0.0, atol=fp32_band)
    else:
        torch.testing.assert_close(got_state, want_state, rtol=2.0**-7, atol=fp32_band)
    for got, expected in zip(args[6:9], want[2:5]):
        assert torch.equal(got, expected), "the conv caches are raw-input copies"
    k_cache, g_cache, v_cache, beta_cache = args[14:18]
    for got, expected in ((k_cache, want[5]), (v_cache, want[7]), (beta_cache, want[8])):
        torch.testing.assert_close(got.float(), expected.float(), rtol=2.0**-7, atol=1e-5)
    torch.testing.assert_close(g_cache, want[6], rtol=0.0, atol=1e-5)
    untouched = [s for s in range(state.shape[0]) if s not in slots.tolist()]
    for record, old in zip(records, before):
        assert torch.equal(record[untouched], old[untouched]), "the op wrote an unused slot"
    padding = [r.shape[1] - SLOT_PAD for r in records[:-1]]
    for record, old, width in zip(records, before, padding):
        assert torch.equal(record[:, width:], old[:, width:]), "the op wrote past a slot's row"
    assert torch.equal(records[-1], before[-1]), (
        "the accepted counts are the runtime's, not the op's"
    )


def test_guard_refuses_a_count_past_num_spec() -> None:
    """A count above num_spec is clamped by the kernel: the request silently replays fewer drafts than the
    runtime accepted."""
    _, slots, args = _build(
        dict(batch=2, num_spec=3, pending=(0, 1), state_dtype=torch.float32, pool=4), seed=31
    )
    args[19][slots[1].long()] = 4
    with pytest.raises(AssertionError, match="num_accepted_tokens"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_shared_slot() -> None:
    """Two requests on one slot both replay, commit and cache into it in one launch: a race."""
    _, _, args = _build(
        dict(batch=2, num_spec=3, pending=(0, 1), state_dtype=torch.float32, pool=4), seed=32
    )
    args = list(args)
    args[18] = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="each slot once"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_misaligned_operand() -> None:
    """The kernels take every token, weight and pool operand as 16-byte aligned without checking: a gate
    starting one element into its buffer would be read from the wrong address."""
    _, _, args = _build(
        dict(batch=2, num_spec=3, pending=(0, 1), state_dtype=torch.float32, pool=4), seed=33
    )
    args = list(args)
    g = args[9]
    buffer = torch.empty(g.numel() + 1, dtype=g.dtype, device="cuda")
    shifted = buffer[1:].view(g.shape)
    shifted.copy_(g)
    args[9] = shifted
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(op):
            op(*args)
