# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_decode catalog entry.

Inputs as the KDA layer hands them over: x_q / x_k / x_v column sections of the fused ``[q | k | v]``
projection's rows, the forget and output gates the two halves of the batched ``[f_b; g_b]`` GEMM's output, beta
the H-wide tail of the ``[f_a | g_a | b]`` projection's rows, the conv sections views of one packed conv pool.
Two layouts: ``indexed``, the pools' rows by slot (an fp32 recurrent pool), each pool a view into wider per-slot
records as the hybrid cache manager lays it out; ``dense``, no slot indices, request b on row b of fp32 staging
copies (how a bf16 pool's rows are decoded, the pool being bf16 by default). The reference is fp32 torch.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_decode import kda_decode as op
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 -- needs the path declared above
from _validating import (  # noqa: E402,F401 -- fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

#: GLM-5.3-Flash's KDA at TP4: 64 heads over 4 ranks, K = V = 128, conv width 4.
HEADS, HEAD_DIM, WIDTH = 16, 128, 4
DIM = HEADS * HEAD_DIM
#: gate_lower_bound, the q scale K ** -0.5 and rms_norm_eps (the output norm's epsilon).
LOWER_BOUND, SCALE, ONORM_EPS = -5.0, HEAD_DIM**-0.5, 1e-5
#: The [f_a | g_a | b] projection's row width at TP4: beta is its last HEADS columns.
BFA_WIDTH = 2 * HEAD_DIM + HEADS
#: Elements past each slot's row in its record (the other layers' rows in the real manager).
SLOT_PAD = 4096
STATE_ROW = HEADS * HEAD_DIM * HEAD_DIM
CONV_ROW = 3 * DIM * (WIDTH - 1)
#: The fp32 state's band, as a fraction of the used rows' largest entry (see the cells test).
STATE_BAND = 1e-4

# The decode steps the shipped target runs, by target directory under `models/`: GLM-5.3-Flash's KDA at TP4,
# CUDA-graph batches up to 64 requests.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4": (
        Cell(
            why="1 request on an fp32 pool by slot: the four-CTA cluster kernel",
            spec=dict(batch=1, layout="indexed", pool=72),
        ),
        Cell(
            why="8 requests on an fp32 pool by slot: the legacy compact-heads kernel",
            spec=dict(batch=8, layout="indexed", pool=72),
        ),
        Cell(
            why="64 requests on an fp32 pool by slot: the two-stage bulk-copy kernel",
            spec=dict(batch=64, layout="indexed", pool=72),
        ),
        Cell(
            why="1 request on staged rows of a bf16 pool (no slot indices)",
            spec=dict(batch=1, layout="dense"),
        ),
        Cell(
            why="8 requests on staged rows of a bf16 pool (no slot indices)",
            spec=dict(batch=8, layout="dense"),
        ),
        Cell(
            why="64 requests on staged rows of a bf16 pool (no slot indices)",
            spec=dict(batch=64, layout="dense"),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _heads(cols: torch.Tensor) -> torch.Tensor:
    """``[B, H * 128]`` columns as the op's ``[1, B, H, 128]`` view (the row stride kept)."""
    return cols.unflatten(-1, (HEADS, HEAD_DIM)).unsqueeze(0)


def _build(spec: dict, seed: int):
    """``(records, args)``: the pools' backing storage and the op's arguments."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = spec["batch"]

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    qkv = rnd(batch, 3 * DIM).to(torch.bfloat16)
    gates = rnd(2, batch, DIM).to(torch.bfloat16)
    bfa = rnd(batch, BFA_WIDTH).to(torch.bfloat16)
    taps = rnd(3, DIM, WIDTH, scale=0.3).to(torch.bfloat16)
    w_t = [taps[i].t().contiguous() for i in range(3)]
    zeros = torch.zeros(DIM, dtype=torch.bfloat16, device="cuda")
    a_log = torch.empty(HEADS, device="cuda").uniform_(1, 16, generator=gen).log()
    dt_bias = torch.empty(DIM, device="cuda").uniform_(
        math.log(1e-3), math.log(1e-1), generator=gen
    )
    onorm_weight = 1 + 0.1 * rnd(HEAD_DIM)
    if spec["layout"] == "indexed":
        pool = spec["pool"]
        conv_record = rnd(pool, CONV_ROW + SLOT_PAD, scale=0.5).to(torch.bfloat16)
        state_record = rnd(pool, STATE_ROW + SLOT_PAD, scale=0.05)
        conv = conv_record[:, :CONV_ROW].view(pool, 3 * DIM, WIDTH - 1)
        state = state_record[:, :STATE_ROW].view(pool, HEADS, HEAD_DIM, HEAD_DIM)
        slots = torch.randperm(pool, generator=gen, device="cuda")[:batch].to(torch.int32)
        records = [conv_record, state_record]
    else:
        conv = rnd(batch, 3 * DIM, WIDTH - 1, scale=0.5).to(torch.bfloat16)
        state = rnd(batch, HEADS, HEAD_DIM, HEAD_DIM, scale=0.05)
        slots = None
        records = [conv, state]
    output = torch.empty(batch, 1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    args = (
        _heads(qkv[:, :DIM]),
        _heads(qkv[:, DIM : 2 * DIM]),
        _heads(qkv[:, 2 * DIM :]),
        *w_t,
        zeros,
        zeros,
        zeros,
        conv[:, :DIM],
        conv[:, DIM : 2 * DIM],
        conv[:, 2 * DIM :],
        a_log,
        _heads(gates[0]),
        dt_bias,
        bfa[:, 2 * HEAD_DIM :].unsqueeze(0),
        _heads(gates[1]),
        onorm_weight,
        slots,
        state,
        LOWER_BOUND,
        SCALE,
        ONORM_EPS,
        output,
    )
    return records, args


def _assert_state_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    scale = ref.abs().max().clamp_min(1e-6)
    err = ((got - ref).abs().max() / scale).item()
    assert err <= STATE_BAND, f"state error {err:.3e} of its largest entry exceeds {STATE_BAND:.0e}"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    """The output, the used state rows and conv windows against the fp32 reference; nothing else written.

    Tolerances. Every product and sum is fp32 on both sides (bf16 inputs, exact bf16 x bf16 products); the
    kernel differs only by its fast exp / rcp / rsqrt (a few fp32 ulps each, about 1e-6 relative) and its
    reduction order. The output is the gated RMS norm, rounded once to bf16 on both sides, so outputs differ
    by at most one bf16 ulp plus about 1e-5 relative: per 128-wide head row that is at most 2 units of 2^-8 of
    the row's largest magnitude (one ulp at the row maximum is 2^-7 of it); ``element_ulp=4`` doubles that,
    and the relative RMS of two independent half-ulp roundings is about 0.6 of 2^-8 (``rms_ulp=1``). The fp32
    state rows carry the approximate gate and key through one step: about 1e-6 of their largest entry,
    ``STATE_BAND`` = 1e-4. The conv windows take the raw bf16 inputs: bit equality.
    """
    certifying(op)
    records, args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = [r.clone() for r in records]
    sections, slots, state, output = args[9:12], args[18], args[19], args[-1]
    want_out, want_state, *want_sections = op.reference(*args)
    assert op(*args) is output
    assert_within_ulp(
        output.reshape(-1, HEAD_DIM), want_out.reshape(-1, HEAD_DIM), element_ulp=4, rms_ulp=1
    )
    rows = torch.arange(output.shape[0], device="cuda") if slots is None else slots.long()
    _assert_state_close(state[rows], want_state[rows])
    for got, want in zip(sections, want_sections):
        assert torch.equal(got, want), "the conv windows are raw-input copies"
    if slots is not None:
        untouched = [s for s in range(state.shape[0]) if s not in slots.tolist()]
        for record, old in zip(records, before):
            assert torch.equal(record[untouched], old[untouched]), "the op wrote an unused slot"
        for record, old, row in zip(records, before, (CONV_ROW, STATE_ROW)):
            assert torch.equal(record[:, row:], old[:, row:]), "the op wrote past a slot's row"


def test_cuda_graph_replay() -> None:
    """Captured once, the step replays rewritten inputs and slots to the bits of eager steps on a copy of the
    pools (nothing allocates or syncs inside the op)."""
    records, args = _build(dict(batch=8, layout="indexed", pool=16), seed=21)
    copies = [r.clone() for r in records]
    copy_args = list(args)
    pool = copies[0].shape[0]
    copy_args[9:12] = [
        copies[0][:, :CONV_ROW].view(pool, 3 * DIM, WIDTH - 1)[:, i * DIM : (i + 1) * DIM]
        for i in range(3)
    ]
    copy_args[19] = copies[1][:, :STATE_ROW].view(pool, HEADS, HEAD_DIM, HEAD_DIM)
    copy_args[-1] = torch.empty_like(args[-1])
    op(*args)  # warm-up step, mirrored on the copies
    op(*copy_args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    inputs = (
        args[0],
        args[13],
        args[15],
        args[16],
    )  # x_q (a view of the qkv rows), g, beta, onorm_g
    slots = args[18]
    gen = torch.Generator(device="cuda").manual_seed(22)
    for _ in range(3):
        for t in inputs:
            t.copy_(torch.randn(t.shape, generator=gen, device="cuda").to(t.dtype))
        slots.copy_(
            torch.randperm(pool, generator=gen, device="cuda")[: slots.numel()].to(torch.int32)
        )
        graph.replay()
        copy_args[-1] = torch.empty_like(args[-1])
        want = op(*copy_args)
        assert torch.equal(args[-1], want)
    assert all(torch.equal(r, c) for r, c in zip(records, copies))


def test_guard_refuses_a_shared_slot() -> None:
    """Two requests on one slot both read and write its state and conv window in one launch: a race."""
    _, args = _build(dict(batch=2, layout="indexed", pool=4), seed=23)
    args = list(args)
    args[18] = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="each row once"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_slot_past_the_pool() -> None:
    """The kernel indexes the state and conv pools with the slot unchecked."""
    _, args = _build(dict(batch=1, layout="indexed", pool=4), seed=24)
    args = list(args)
    args[18] = torch.tensor([4], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="lie in the pool"):
        with validating(op):
            op(*args)
