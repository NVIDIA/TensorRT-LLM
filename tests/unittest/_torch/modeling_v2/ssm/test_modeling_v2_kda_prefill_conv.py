# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_prefill_conv catalog entry.

The conv pool is a view into wider per-slot records, as the hybrid cache manager lays it out, so a write past a
slot's window shows up in the record's padding; the source is the token-major projection GEMM's output.
"""

import itertools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.kda_prefill_conv import (
    kda_prefill_conv as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 -- needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

HEAD_DIM, WIDTH = 128, 4
#: Elements past each slot's [3D, 3] window in its record (the other layers' windows in the real manager).
SLOT_PAD = 1024

# The prefill steps the shipped target runs, by target directory under `models/`: GLM-5.3-Flash's KDA at TP4
# (16 of 64 heads, [q | k | v] = 6144 channels), with --max_num_tokens 32768 and 13,312-token prompts.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4": (
        Cell(
            why="one 13,312-token prompt into a fresh slot",
            spec=dict(heads=16, seq_lens=(13312,), slots=(5,), has_init=(False,), pool=8),
        ),
        Cell(
            why="a 32,768-token step: two whole 13,312-token prompts and a 6,144-token first chunk",
            spec=dict(
                heads=16,
                seq_lens=(13312, 13312, 6144),
                slots=(3, 0, 6),
                has_init=(False, False, False),
                pool=8,
            ),
        ),
        Cell(
            why="the next step: the 7,168-token continuation from its slot's window, two fresh prompts",
            spec=dict(
                heads=16,
                seq_lens=(7168, 13312, 12288),
                slots=(6, 1, 4),
                has_init=(True, False, False),
                pool=8,
            ),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec: dict, seed: int):
    """``(record, args)``: the conv pool's per-slot records and the op's arguments.

    The taps (std 0.5 over four unit inputs) give SiLU arguments of about unit spread: both signs of the
    kernel's fast-SiLU range in every cell, and in the long cells also its cancellation range (x < -4, which
    the kernel recomputes exactly).
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads, lens, pool = spec["heads"], spec["seq_lens"], spec["pool"]
    channels = 3 * heads * HEAD_DIM
    tokens = sum(lens)
    width = spec.get("src_width", channels)
    src = torch.randn(tokens, width, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(channels, WIDTH, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    window = channels * (WIDTH - 1)
    record = torch.randn(pool, window + SLOT_PAD, generator=gen, device="cuda").to(torch.bfloat16)
    conv_state = record[:, :window].view(pool, channels, WIDTH - 1)
    starts = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda")
    slots = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    has_init = torch.tensor(spec["has_init"], dtype=torch.bool, device="cuda")
    return record, (src, weight, conv_state, starts, slots, has_init, max(lens))


def _assert_conv_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The taps accumulate in fp32 on both sides (the kernel with exactly halved weights), so the outputs differ
    by the SiLU and the two bf16 roundings. The kernel's SiLU is ``h + h * tanh.approx(h)`` with ``h = x / 2``
    for x >= -4: tanh.approx has relative error at most 2^-10.987, which is at most 2^-11 of the output for
    x >= 0 and at most ``2 * tanh(2) * 2^-10.987 < 9.5e-4`` absolute for -4 <= x < 0 (smaller x is recomputed
    exactly). The two roundings to bf16 put the results at most one ulp (2^-7 relative) apart.
    ``rtol`` = 2^-7 + 2^-10 (the ulp plus twice the relative SiLU bound), ``atol`` = 2^-9 (twice the absolute
    SiLU bound)."""
    torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-7 + 2.0**-10, atol=2.0**-9)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    """q / k / v within the SiLU band; the conv pool bit-identical to the reference (its windows are raw-input
    copies, the other slots untouched); no write past a slot's window."""
    certifying(op)
    record, args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = record.clone()
    src, conv_state = args[0], args[2]
    src_before = src.clone()
    q_ref, k_ref, v_ref, state_ref = op.reference(*args)
    q, k, v = op(*args)
    for out, ref in ((q, q_ref), (k, k_ref), (v, v_ref)):
        assert out.shape == ref.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
        _assert_conv_close(out, ref)
    assert torch.equal(conv_state, state_ref)
    window = conv_state[0].numel()
    assert torch.equal(record[:, window:], before[:, window:]), "the op wrote past a slot's window"
    assert torch.equal(src, src_before), "the op mutated its source"


def test_guard_refuses_a_shared_slot() -> None:
    """Two sequences on one slot each read and write its window in one launch: a race, no error."""
    _, args = _build(
        dict(heads=16, seq_lens=(40, 9), slots=(2, 2), has_init=(True, True), pool=4), seed=8
    )
    with pytest.raises(AssertionError, match="unique"):
        with validating(op):
            op(*args)


def test_guard_refuses_offsets_short_of_the_tokens() -> None:
    """Rows past ``starts[-1]`` belong to no sequence: the op leaves them uninitialized and returns them."""
    _, args = _build(
        dict(heads=16, seq_lens=(40, 9), slots=(0, 1), has_init=(False, True), pool=4), seed=9
    )
    short = torch.tensor([0, 40, 48], dtype=torch.int32, device="cuda")
    args = (*args[:3], short, *args[4:])
    with pytest.raises(AssertionError, match="partition"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_slot_outside_the_pool() -> None:
    """Slot -1 is the kernel's pass-through: it copies the raw inputs out unconvolved."""
    _, args = _build(dict(heads=16, seq_lens=(16,), slots=(0,), has_init=(False,), pool=4), seed=10)
    args = (*args[:4], torch.tensor([-1], dtype=torch.int32, device="cuda"), *args[5:])
    with pytest.raises(AssertionError, match="lie in the pool"):
        with validating(op):
            op(*args)
