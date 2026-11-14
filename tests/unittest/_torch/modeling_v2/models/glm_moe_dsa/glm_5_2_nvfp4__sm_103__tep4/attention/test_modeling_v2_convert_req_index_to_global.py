# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's convert_req_index_to_global entry."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "convert_req_index_to_global requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact: this is integer address arithmetic, and an off-by-one row is a
    different token's KV."""
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out, ref), (
        f"{int((out != ref).sum())} of {out.numel()} pool rows differ from the reference"
    )


def _build(spec, seed):
    """`(req_id, block_table, token_indices, block_size, topk, stride_factor, layer_id)`.

    Each request contributes `seq_rows[s]` query rows; row `j` of request `s`
    sees positions `[0, kv_lens[s] - seq_rows[s] + j + 1)`, and its top-k row
    holds up to `topk` distinct positions from that range in no order, padded
    with -1 -- the shape the indexer's top-k hands over. Pages are a random
    permutation of a pool twice the needed size; unused table entries are -1.
    """
    g = torch.Generator().manual_seed(seed)
    bs, topk = spec["block_size"], spec["topk"]
    pages_needed = [math.ceil(n / bs) for n in spec["kv_lens"]]
    width = max(pages_needed)
    pad = spec.get("pad_block_table", False)
    if pad:
        # Narrower than the longest request needs, so its tail positions have no
        # table entry, and one in-range entry is an unused page.
        width = max(2, width // 2)
    num_pages = 2 * sum(pages_needed) + 2
    pool = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = torch.full((len(spec["kv_lens"]), width), -1, dtype=torch.int32)
    used = 0
    for s, need in enumerate(pages_needed):
        n = min(need, width)
        table[s, :n] = pool[used : used + n]
        used += n
    if pad:
        table[0, 1] = -1

    req, rows = [], []
    for s, (r_count, kv) in enumerate(zip(spec["seq_rows"], spec["kv_lens"])):
        for j in range(r_count):
            end = kv - r_count + j + 1
            row = torch.full((topk,), -1, dtype=torch.int32)
            pick = torch.randperm(end, generator=g)[:topk].to(torch.int32)
            row[: pick.numel()] = pick
            rows.append(row)
            req.append(s)
    return (
        torch.tensor(req, dtype=torch.int32, device="cuda"),
        table.cuda(),
        torch.stack(rows).cuda(),
        bs,
        topk,
        spec["pool_layers"] * bs,
        spec["layer_id"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    snap = [a.clone() for a in args[:3]]
    with validating(op):
        out = op(*args)
    for a, s in zip(args[:3], snap):
        assert torch.equal(a, s), "an input tensor was mutated"
    _compare(out, op.reference(*args))
    # The padding has to survive as -1, not as a row computed from -1.
    assert bool((out[args[2] < 0] == -1).all())
    if cell.spec.get("pad_block_table"):
        assert bool((out[args[2] >= 0] == -1).any()), "the cell drove no unresolvable position"


def _small():
    req = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")
    tok = torch.tensor(
        [[0, 33, -1, 5], [1, 2, 63, -1], [40, 0, -1, -1], [31, 32, 7, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    return req, table, tok


def test_hand_worked_rows() -> None:
    """Four rows worked out by hand, independent of `reference`."""
    req, table, tok = _small()
    out = op(req, table, tok, 32, 4, 64, 1)
    # row 0, req 0: p=0 -> page 3: 3*64 + 0 + 32 = 224; p=33 -> page 1: 64 + 1 + 32 = 97
    assert out[0].tolist() == [224, 97, -1, 229]
    # row 1, req 1: pages [0, 2]; p=63 -> page 2, offset 31: 128 + 31 + 32 = 191
    assert out[1].tolist() == [33, 34, 191, -1]
    _compare(out, op.reference(req, table, tok, 32, 4, 64, 1))


def test_guard_refuses_a_column_count_mismatch() -> None:
    """Wider spills into the next row; narrower leaves columns uninitialized."""
    req, table, tok = _small()
    for bad in (2, 8):
        with pytest.raises(AssertionError, match="num_topk_tokens must equal"):
            with validating(op):
                op(req, table, tok, 32, bad, 64, 0)


def test_guard_refuses_a_req_id_length_mismatch() -> None:
    """The kernel's row count is req_id's, not token_indices'."""
    req, table, tok = _small()
    for bad in (req[:2].contiguous(), torch.cat([req, req])):
        with pytest.raises(AssertionError, match="exactly one entry per row"):
            with validating(op):
                op(bad, table, tok, 32, 4, 64, 0)


def test_guard_refuses_an_int32_overflow() -> None:
    req, table, tok = _small()
    with pytest.raises(AssertionError, match="does not fit int32"):
        with validating(op):
            op(req, table, tok, 32, 4, 2**30, 0)
