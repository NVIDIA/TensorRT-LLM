# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_shard_accept entry (greedy verify from this rank's logits shard)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    SHARD_ROWS,
    exchange_tokens,
    local_argmax,
    make_keys,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    m3_shard_accept as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402
    craft_drafts,
    gen,
    head_workspace,
    num_sms,
    plant_peers,
    randn,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_shard_accept requires a CUDA device"


def _case(spec, seed, edit_logits=None):
    g = gen(seed)
    nreq, d, vs, world, rank = (
        spec["nreq"],
        spec["draft_len"],
        spec["vs"],
        spec["world"],
        spec["rank"],
    )
    m = nreq * (d + 1)
    if spec["strided"]:
        logits = randn((m, vs + 72), g, 3.0)[:, :vs]
    else:
        logits = randn((m, vs), g, 3.0)
    if edit_logits is not None:
        edit_logits(logits)
    uc, keys, counters = head_workspace(world, slot_rows=SHARD_ROWS)
    own = logits.float().max(dim=1).values
    plant_peers(
        uc, counters, own, rank, world, vs, g, slot_rows=SHARD_ROWS, ties=spec.get("ties", False)
    )
    offset = rank * vs
    best, idx = local_argmax(logits.float(), offset)
    tokens = exchange_tokens(make_keys(best, idx), uc, counters, rank, world, SHARD_ROWS)
    draft = craft_drafts(tokens, nreq, d)
    batch_idx = torch.arange(nreq, dtype=torch.int32, device="cuda")
    acc = torch.full((nreq, d + 1), -7, dtype=torch.int32, device="cuda")
    num = torch.full((nreq,), -7, dtype=torch.int32, device="cuda")
    gather = torch.full((nreq,), -7, dtype=torch.int64, device="cuda")
    next_new = torch.full((nreq, d + 1), -7, dtype=torch.int32, device="cuda")
    force_total = 3 if spec["force"] == "total" else 0
    force_vec = (torch.randint(1, d + 2, (nreq,), generator=g, device="cuda", dtype=torch.int32)
                 if spec["force"] == "vec" else None)  # fmt: skip
    return (logits, offset, keys, counters, uc, uc.data_ptr(), rank, world, draft, batch_idx, acc, num, gather,
            force_total, next_new, num_sms(), force_vec)  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bit-exact: the op reduces BF16 logits without arithmetic, ties (within the shard and across ranks) go to
    the lowest global index on both sides."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    assert torch.equal(args[10], ref["accepted"])
    assert torch.equal(args[11], ref["num_accepted"])
    assert torch.equal(args[12], ref["gather_ids"])
    assert torch.equal(args[14], ref["next_new"])
    assert args[3].tolist() == [0, 1] and not bool(args[2].any())


def test_guard_refuses_nan_logits() -> None:
    args = list(_case(op.CELLS[0].spec, 1))
    args[0][1, 7] = float("nan")
    with pytest.raises(AssertionError, match="NaN"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_batch_index_past_the_requests() -> None:
    args = list(_case(op.CELLS[0].spec, 2))
    args[9] = args[9] + 1
    with pytest.raises(AssertionError, match="batch_idx"):
        with validating(op):
            op(*args)


def _signed_zero_maxima(logits: torch.Tensor) -> None:
    """Every row's maximum is zero, reached as -0.0 at a lower index and +0.0 at a higher one."""
    logits.fill_(-1.0)
    logits[:, 3] = -0.0
    logits[:, 9] = 0.0


@pytest.mark.parametrize("ties", [False, True], ids=["peer_rows", "peer_ties"])
def test_signed_zero_maxima_take_the_lower_index(ties) -> None:
    """-0.0 and +0.0 compare equal, so the lower index wins as with torch.argmax (also across ranks)."""
    spec = dict(op.CELLS[1].spec, ties=ties)
    args = _case(spec, seed_of(f"signed-zero-{ties}"), edit_logits=_signed_zero_maxima)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    assert torch.equal(args[10], ref["accepted"])
    assert torch.equal(args[11], ref["num_accepted"])
    assert torch.equal(args[12], ref["gather_ids"])
    assert torch.equal(args[14], ref["next_new"])
