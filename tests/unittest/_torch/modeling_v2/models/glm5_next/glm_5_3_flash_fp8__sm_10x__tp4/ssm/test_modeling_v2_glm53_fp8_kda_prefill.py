# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_prefill operation of the glm5_next FP8 target.

Inputs as the KDA layer hands them over: q / k / v the prefill conv's contiguous outputs, the forget gate the
first half of the batched ``[f_b; g_b]`` GEMM's output, beta the H-wide tail of the ``[f_a | g_a | b]``
projection's rows (row stride 2 * 128 + H), the state pool a view into wider per-slot records (as the hybrid
cache manager lays it out), fresh rows zeroed by the caller, and the scheduler's planned ``seq_order`` /
``cu_chunks``. The reference is the fp32 token-by-token recurrence.
"""

import itertools
import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.ssm.kda_prefill import (  # noqa: E501
    kda_prefill as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

HEAD_DIM = 128
#: GLM-5.3-Flash's linear_attn_config gate_lower_bound, and the q scale K ** -0.5.
LOWER_BOUND, SCALE = -5.0, HEAD_DIM**-0.5
#: Elements past each slot's state in its record (the other layers' states in the real manager).
SLOT_PAD = 4096
#: Bands against the fp32 recurrence, as fractions of the compared tensor's largest magnitude (see `_bands`).
OUT_BAND, STATE_BAND = 2e-2, 1e-2

# The prefill steps the shipped target runs, by target directory under `models/`: GLM-5.3-Flash's KDA at TP4
# (16 of 64 heads), the auto route it takes, the scheduler's planned metadata, --max_num_tokens 32768 and
# 13,312-token prompts; the pool bf16 (the default recurrent-state dtype) or fp32.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="one 13,312-token prompt into a fresh slot, bf16 pool",
        spec=dict(
            heads=16,
            seq_lens=(13312,),
            slots=(5,),
            fresh=(True,),
            state_dtype=torch.bfloat16,
            mode="auto",
            planned=True,
            pool=8,
        ),
    ),
    Cell(
        why="a 32,768-token step: two whole prompts and a 6,144-token first chunk",
        spec=dict(
            heads=16,
            seq_lens=(13312, 13312, 6144),
            slots=(3, 0, 6),
            fresh=(True, True, True),
            state_dtype=torch.bfloat16,
            mode="auto",
            planned=True,
            pool=8,
        ),
    ),
    Cell(
        why="the next step: the 7,168-token continuation from its slot's state, two fresh prompts",
        spec=dict(
            heads=16,
            seq_lens=(7168, 13312, 12288),
            slots=(6, 1, 4),
            fresh=(False, True, True),
            state_dtype=torch.bfloat16,
            mode="auto",
            planned=True,
            pool=8,
        ),
    ),
    Cell(
        why="a continuation and a fresh prompt on an fp32 pool (mamba_ssm_cache_dtype float32)",
        spec=dict(
            heads=16,
            seq_lens=(4096, 2048),
            slots=(2, 7),
            fresh=(False, True),
            state_dtype=torch.float32,
            mode="auto",
            planned=True,
            pool=8,
        ),
    ),
)


def _schedule(lens) -> tuple[torch.Tensor, torch.Tensor]:
    """``(seq_order, cu_chunks)`` as the scheduler's metadata builds them: rows of one int32 buffer."""
    n = len(lens)
    order = sorted(range(n), key=lambda i: (-lens[i], i))
    chunks = [0, *itertools.accumulate((length + 15) // 16 for length in lens)]
    schedule = torch.zeros(2, n + 1, dtype=torch.int32, device="cuda")
    schedule[0, :n] = torch.tensor(order, dtype=torch.int32, device="cuda")
    schedule[1] = torch.tensor(chunks, dtype=torch.int32, device="cuda")
    return schedule[0, :n], schedule[1]


def _build(spec: dict, seed: int):
    """``(record, args)``: the state pool's per-slot records and the op's arguments.

    The value distributions are the Kimi K3 entry's (unit q / k / v / gate / beta logits, A_log = log U(1, 16),
    dt_bias = U(log 1e-3, log 1e-1), continuing states of spread 0.1), whose bands this test keeps.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads, lens, pool = spec["heads"], spec["seq_lens"], spec["pool"]
    tokens = sum(lens)
    shape = (1, tokens, heads, HEAD_DIM)
    q, k, v = (
        torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    gates = torch.randn(2, tokens, heads * HEAD_DIM, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    g = gates[0].view(shape)
    bfa = torch.randn(tokens, 2 * HEAD_DIM + heads, generator=gen, device="cuda").to(torch.bfloat16)
    beta = bfa[:, 2 * HEAD_DIM :].unsqueeze(0)
    A_log = torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen).log()
    dt_bias = torch.empty(heads * HEAD_DIM, device="cuda").uniform_(
        math.log(1e-3), math.log(1e-1), generator=gen
    )
    row = heads * HEAD_DIM * HEAD_DIM
    record = (torch.randn(pool, row + SLOT_PAD, generator=gen, device="cuda") * 0.1).to(
        spec["state_dtype"]
    )
    state_pool = record[:, :row].view(pool, heads, HEAD_DIM, HEAD_DIM)
    for slot, fresh in zip(spec["slots"], spec["fresh"]):
        if fresh:
            state_pool[slot].zero_()  # the caller's reset_recurrent_state_rows
    state_indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int64, device="cuda")
    seq_order = cu_chunks = None
    max_seqlen = 0
    if spec["planned"]:
        seq_order, cu_chunks = _schedule(lens)
        max_seqlen = max(lens)
    args = (
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        state_indices,
        cu_seqlens,
        A_log,
        dt_bias,
        SCALE,
        LOWER_BOUND,
        spec["mode"],
        max_seqlen,
        seq_order,
        cu_chunks,
        0,
    )
    return record, args


def _assert_scaled(got: torch.Tensor, ref: torch.Tensor, band: float) -> None:
    scale = ref.float().abs().max().clamp_min(1e-6)
    err = ((got.float() - ref.float()).abs().max() / scale).item()
    assert err <= band, f"max error {err:.3e} of the largest magnitude exceeds {band:.3e}"


def _bands(state_dtype: torch.dtype) -> tuple[float, float]:
    """``(output band, state band)`` as fractions of the compared tensor's largest magnitude.

    The kernel runs 16-token chunks on tensor cores: q, k, the decayed keys, the state and each chunk's update
    enter the MMAs rounded to bf16 (2^-9 relative per operand), with fp32 accumulation and the state carried in
    fp32 across the call. Each output and update term thus carries about 2^-8 relative error; over 128 keys of
    mixed sign it stays a few 2^-9 of the output's scale, and the delta rule's projection ``I - beta k k^T``
    keeps re-writing the state along each new key, so the state's error does not grow with length. The bands
    are the ones the Kimi K3 entry holds for this kernel on the same value distributions: 2e-2 of each
    sequence's largest output, 1e-2 of a slot's largest fp32 state entry. A bf16 pool also rounds every final
    entry once, at most one bf16 ulp (2^-7 of the entry), which the state band adds.
    """
    return OUT_BAND, STATE_BAND + (2.0**-7 if state_dtype == torch.bfloat16 else 0.0)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    """Each sequence's output and its slot's final state within the bands of `_bands`; every other slot and
    all record padding bit-identical."""
    certifying(op)
    record, args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = record.clone()
    v, state_pool, state_indices, cu_seqlens = args[2], args[5], args[6], args[7]
    o_ref, pool_ref = op.reference(*args)
    o = op(*args)
    assert o.shape == v.shape and o.dtype == torch.bfloat16 and o.is_contiguous()
    out_band, state_band = _bands(state_pool.dtype)
    bounds = cu_seqlens.tolist()
    slots = state_indices.tolist()
    for seq, slot in enumerate(slots):
        lo, hi = bounds[seq], bounds[seq + 1]
        if hi > lo:
            _assert_scaled(o[0, lo:hi], o_ref[0, lo:hi], out_band)
        _assert_scaled(state_pool[slot], pool_ref[slot], state_band)
    untouched = [s for s in range(state_pool.shape[0]) if s not in slots]
    assert torch.equal(record[untouched], before[untouched]), "the op wrote an unselected slot"
    row = state_pool[0].numel()
    assert torch.equal(record[:, row:], before[:, row:]), "the op wrote past a slot's state"


def _small(**overrides):
    spec = dict(
        heads=16,
        seq_lens=(37, 100),
        slots=(0, 1),
        fresh=(False, False),
        state_dtype=torch.float32,
        mode="engine",
        planned=True,
        pool=4,
    )
    spec.update(overrides)
    return _build(spec, seed=9)[1]


def test_guard_refuses_a_shared_slot() -> None:
    """Two sequences on one slot each read and write it in one launch: a race, no error."""
    args = _small(slots=(1, 1))
    with pytest.raises(AssertionError, match="unique slot"):
        with validating(op):
            op(*args)


def test_guard_refuses_offsets_that_miss_tokens() -> None:
    """Offsets ending short of the packed tokens leave the tail rows of ``o`` unwritten."""
    args = list(_small())
    args[7] = torch.tensor([0, 37, 136], dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="partition"):
        with validating(op):
            op(*args)


@pytest.mark.parametrize("bad", ("order", "chunks"))
def test_guard_refuses_stale_planned_metadata(bad) -> None:
    """The planned schedule replaces the kernel's own: a stale one (lengths changed, schedule not refreshed)
    is taken as is."""
    args = list(_small())
    if bad == "order":
        args[14] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        match = "descending"
    else:
        args[15] = torch.tensor([0, 3, 9], dtype=torch.int32, device="cuda")
        match = "prefix-sum"
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(*args)
