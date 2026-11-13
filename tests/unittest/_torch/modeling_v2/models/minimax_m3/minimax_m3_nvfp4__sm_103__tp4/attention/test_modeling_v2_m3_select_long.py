# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_select_long entry (top-16 MSA block selection for long decode rows)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_select_long import (  # noqa: E501
    TOPK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_select_long import (  # noqa: E501
    m3_select_long as op,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    select_blocks_from_maxscore,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401 — needs the path declared above

assert torch.cuda.is_available(), "m3_select_long requires a CUDA device"

# Valid-block counts that land a row on each of the kernel's paths: <= 16 (all picked), the 2-, 4- and 9-key
# widths, past the row (clamped), and zero (a CUDA-graph padding row).
_MIXED_VALID = (0, 5, 16, 17, 200, 2048, 2049, 4096, 4097, 8193, 9000, 1, 300, 6000, 129, 3000)


def _scores(spec, g):
    nb, tq = spec["num_blocks"], spec["total_q"]
    s = torch.randn(spec["num_kv_heads"], nb, tq, generator=g, device="cuda", dtype=torch.float32)
    kind = spec["scores"]
    if kind == "ties":
        # Even tokens: a flat row with ten winners, so thousands of keys tie at the 16th-largest score (the
        # re-read path). Odd tokens: scores on a coarse grid, so ties are broken by block index.
        s[:, :, 0::2] = 0.25
        winners = torch.randint(0, nb, (spec["num_kv_heads"], 10, tq), generator=g, device="cuda")
        s[:, :, 0::2].scatter_(1, winners[:, :, 0::2], 3.0)
        s[:, :, 1::2] = (s[:, :, 1::2] * 2).round() / 2
    elif kind == "nonfinite":
        s[:, 7::50, :] = float("inf")
        s[:, 11::70, :] = float("nan")
        s[:, 3::40, :] = -0.0
        s[:, 4::40, :] = 0.0
        # Odd tokens: almost everything -inf, so picks run past the finite scores.
        s[:, :, 1::2] = float("-inf")
        s[:, 0:5, 1::2] = 1.0
    return s


def _valid(spec, g):
    tq, nb = spec["total_q"], spec["num_blocks"]
    if spec["valid"] == "mixed":
        vals = [_MIXED_VALID[i % len(_MIXED_VALID)] for i in range(tq)]
    else:
        vals = (nb - torch.randint(0, 12, (tq,), generator=g, device="cuda")).tolist()
        vals[0] = nb
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _case(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = _scores(spec, g)
    nvb = _valid(spec, g)
    out = torch.empty(
        spec["num_kv_heads"], spec["total_q"], TOPK, dtype=torch.int32, device="cuda"
    ).permute(1, 0, 2)
    return scores, nvb, out, spec["init_blocks"], spec["local_blocks"], False


def _mass_tie_rows(scores, nvb, init_blocks, local_blocks):
    """[total_q, num_kv_heads] mask of rows where more than 1024 valid blocks share the 16th-largest score."""
    num_kv_heads, num_blocks, total_q = scores.shape
    blocks = torch.arange(num_blocks, device=scores.device)
    raw = nvb[:total_q].long()
    valid = raw.clamp(0, num_blocks)
    forced = scores.permute(2, 0, 1).clone()  # [total_q, heads, blocks]
    forced = torch.where(blocks < init_blocks, 1.0e30, forced)
    forced = torch.where(blocks >= (raw - local_blocks).clamp_min(0)[:, None, None], 1.0e29, forced)
    in_row = (blocks[None, :] < valid[:, None])[:, None, :]
    ranked = torch.where(in_row, forced, float("-inf")).sort(dim=-1, descending=True).values
    kth = ranked[..., min(TOPK, num_blocks) - 1 : min(TOPK, num_blocks)]
    return ((forced == kth) & in_row).sum(dim=-1) > 1024


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    args = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    out = op(*args)
    ref = op.reference(*args)

    # Selection is an exact claim: the same blocks or a different attention.
    assert out.dtype is torch.int32
    assert torch.equal(out, ref), "selected blocks differ from the reference"
    # And it is the in-tree selector's answer, bit for bit (the op the target falls back to) -- except on rows
    # where more than 1024 blocks tie at the 16th-largest score: there the in-tree op does not break the tie to
    # the smallest blocks, while this op and the reference do.
    scores, nvb, _, init_blocks, local_blocks, _ = args
    in_tree = select_blocks_from_maxscore(
        scores,
        topk=TOPK,
        n_valid_blocks=nvb,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        head_major_output=True,
    )
    mass_tie = _mass_tie_rows(scores, nvb, init_blocks, local_blocks)
    assert torch.equal(out[~mass_tie], in_tree[~mass_tie]), (
        "selected blocks differ from the in-tree minimax_m3_select_blocks"
    )


def test_pdl_launch_selects_the_same() -> None:
    spec = op.CELLS[2].spec
    scores, nvb, out, init_blocks, local_blocks, _ = _case(spec, seed=1)
    pdl_out = torch.empty_like(out)
    op(scores, nvb, out, init_blocks, local_blocks, False)
    op(scores, nvb, pdl_out, init_blocks, local_blocks, True)
    assert torch.equal(out, pdl_out)


def test_early_trigger_selects_the_same() -> None:
    """Releasing the dependent grid right after the select's own wait changes only when the next kernel may launch."""
    spec = op.CELLS[2].spec
    scores, nvb, out, init_blocks, local_blocks, _ = _case(spec, seed=2)
    early_out = torch.empty_like(out)
    op(scores, nvb, out, init_blocks, local_blocks, True, False)
    op(scores, nvb, early_out, init_blocks, local_blocks, True, True)
    assert torch.equal(out, early_out)


def test_op_refuses_a_row_past_its_width() -> None:
    scores = torch.zeros(1, 9217, 1, device="cuda")
    nvb = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(1, 1, TOPK, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="at most 9216"):
        op(scores, nvb, out, 0, 1, False)


def test_guard_refuses_negative_forcing() -> None:
    args = list(_case(op.CELLS[0].spec, seed=2))
    args[4] = -1
    with pytest.raises(AssertionError, match="non-negative"):
        with validating(op):
            op(*args)


def test_guard_refuses_out_aliasing_scores() -> None:
    scores, nvb, _, init_blocks, local_blocks, _ = _case(op.CELLS[0].spec, seed=3)
    aliased = (
        scores.view(torch.int32)
        .reshape(-1)[: scores.shape[2] * TOPK]
        .view(scores.shape[2], 1, TOPK)
    )
    with pytest.raises(AssertionError, match="alias"):
        with validating(op):
            op(scores, nvb, aliased, init_blocks, local_blocks, False)
