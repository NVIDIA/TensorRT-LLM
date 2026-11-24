# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 get_paged_mqa_logits_metadata entry: the schedule is integers,
compared exactly, at the device's SM count (the logits kernels' CTA count)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (  # noqa: E501
    SPLIT_KV,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (
    get_paged_mqa_logits_metadata as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _lengths(spec, seed: int) -> torch.Tensor:
    """int32 CUDA [rows, next_n]: each row's KV length repeated over its query tokens, as the
    metadata's 2D length buffer holds it; atom rows staggered as the DSL split's are."""
    gen = torch.Generator().manual_seed(seed)
    lo, hi = spec["lens"]
    atoms = spec["stagger"]
    if atoms:
        requests = spec["rows"] // atoms
        lens = torch.randint(lo, hi + 1, (requests,), generator=gen, dtype=torch.int32)
        offsets = torch.tensor(inputs.atom_ctx_offsets(5, atoms), dtype=torch.int32)
        lens = (lens[:, None] - offsets).reshape(-1)
    else:
        lens = torch.randint(lo, hi + 1, (spec["rows"],), generator=gen, dtype=torch.int32)
    return lens[:, None].expand(-1, spec["next_n"]).contiguous().cuda()


def _build(spec, seed: int) -> dict:
    sms = inputs.num_sms()
    return dict(
        context_lens=_lengths(spec, seed),
        block_kv=spec["block_kv"],
        num_sms=sms,
        out=torch.full((sms + 1, 2), -7, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert out is args["out"]
    assert torch.equal(out, ref), f"schedule differs:\n{out.tolist()}\nexpected\n{ref.tolist()}"


@pytest.mark.parametrize("next_n", [1, 8])
def test_the_starts_tile_every_split_once(next_n):
    """Independent of the reference: the CTA starts are non-decreasing positions in the list of
    (request, split) units, from the first unit to one past the last."""
    lens = [3_000, 70_000, 129, 40_000]
    sms = inputs.num_sms()
    context_lens = torch.tensor(lens, dtype=torch.int32)[:, None].expand(-1, next_n)
    out = op(
        context_lens.contiguous().cuda(),
        64,
        sms,
        torch.empty(sms + 1, 2, dtype=torch.int32, device="cuda"),
    ).tolist()
    units = [(r * next_n, s) for r, n in enumerate(lens) for s in range(-(-n // SPLIT_KV))]
    position = {unit: i for i, unit in enumerate(units)}
    starts = [position.get(tuple(row), len(units)) for row in out]
    assert starts[0] == 0 and starts[-1] == len(units)
    assert all(a <= b for a, b in zip(starts, starts[1:]))
    assert out[-1] == [len(lens) * next_n, 0]


@pytest.mark.parametrize("block_kv", [32, 128])
def test_block_kv_does_not_move_the_schedule(block_kv):
    cell = op.CELLS[4]
    args = _build(cell.spec, inputs.seed_of(cell))
    expected = op(**args).clone()
    again = op(**dict(args, block_kv=block_kv, out=torch.empty_like(args["out"])))
    torch.cuda.synchronize()
    assert torch.equal(again, expected)


def test_guard_refuses_another_sm_count():
    args = _build(op.CELLS[0].spec, seed=1)
    sms = args["num_sms"] - 1
    with pytest.raises(AssertionError, match="SM count"):
        with validating(op):
            op(**dict(args, num_sms=sms, out=torch.empty(sms + 1, 2, dtype=torch.int32).cuda()))


def test_guard_refuses_a_wider_out():
    args = _build(op.CELLS[0].spec, seed=2)
    with pytest.raises(AssertionError, match="int32"):
        with validating(op):
            op(**dict(args, out=args["out"].long()))
