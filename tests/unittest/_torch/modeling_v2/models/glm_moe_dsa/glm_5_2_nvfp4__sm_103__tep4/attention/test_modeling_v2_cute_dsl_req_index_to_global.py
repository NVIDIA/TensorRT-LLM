# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_req_index_to_global entry."""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_req_index_to_global import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_req_index_to_global import (  # noqa: E402, E501
    cute_dsl_req_index_to_global as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

BLOCK = 32

_COMPILED: dict = {}


def _compiled(topk: int, max_blocks: int):
    key = (topk, max_blocks)
    if key not in _COMPILED:
        _COMPILED[key] = build(torch.cuda.current_device(), topk=topk, max_blocks=max_blocks)
    return _COMPILED[key]


def _build(spec, seed):
    """`(req_id, block_table, token_indices, block_size, stride_factor)` for one cell.

    Request `s` contributes `seq_rows[s]` query rows; row `j` sees positions
    `[0, kv_lens[s] - seq_rows[s] + j + 1)`, and its top-k row holds up to
    `topk` distinct positions from that range in no order, padded with -1 --
    the shape the indexer's top-k hands over. The block table is `max_blocks`
    wide (the compile key); pages are a random permutation of a pool twice the
    needed size and unused entries are -1. With `pad_block_table`, one
    in-range entry is an unused page too.
    """
    g = torch.Generator().manual_seed(seed)
    topk, width = spec["topk"], spec["max_blocks"]
    pages_needed = [math.ceil(n / BLOCK) for n in spec["kv_lens"]]
    num_pages = 2 * sum(pages_needed) + 2
    pool = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = torch.full((len(spec["kv_lens"]), width), -1, dtype=torch.int32)
    used = 0
    for s, need in enumerate(pages_needed):
        n = min(need, width)
        table[s, :n] = pool[used : used + n]
        used += n
    if spec.get("pad_block_table"):
        assert max(pages_needed) > width, "the cell must hold positions past the table"
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
        BLOCK,
        spec["pool_layers"] * BLOCK,
    )


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact: integer address arithmetic, where an off-by-one row is another token's KV."""
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out, ref), (
        f"{int((out != ref).sum())} of {out.numel()} pool rows differ from the reference"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    compiled = _compiled(spec["topk"], spec["max_blocks"])
    snap = [a.clone() for a in args[:3]]
    with validating(op):
        out = op(*args, spec["layer_id"], compiled=compiled)
    for a, s in zip(args[:3], snap):
        assert torch.equal(a, s), "an input tensor was mutated"
    _compare(out, op.reference(*args, spec["layer_id"], compiled=compiled))
    # The padding has to survive as -1, not as a row computed from -1.
    assert bool((out[args[2] < 0] == -1).all())
    if spec.get("pad_block_table"):
        assert bool((out[args[2] >= 0] == -1).any()), "the cell drove no unresolvable position"


def test_hand_worked_rows() -> None:
    """Rows worked out by hand, independent of `reference`."""
    req = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    table = torch.full((2, 8), -1, dtype=torch.int32)
    table[0, :2] = torch.tensor([3, 1])
    table[1, :3] = torch.tensor([0, 2, 5])
    tok = torch.full((2, 2048), -1, dtype=torch.int32)
    tok[0, :4] = torch.tensor([0, 33, 300, 5])
    tok[1, :3] = torch.tensor([63, 64, 1])
    out = op(req, table.cuda(), tok.cuda(), 32, 64, 1, compiled=_compiled(2048, 8))
    # req 0: p=0 -> page 3: 3*64 + 0 + 32 = 224; p=33 -> page 1: 64 + 1 + 32 = 97;
    # p=300 -> block 9, past the 8-wide table: -1; p=5 -> 192 + 5 + 32 = 229
    assert out[0, :5].tolist() == [224, 97, -1, 229, -1]
    # req 1: p=63 -> page 2, offset 31: 128 + 31 + 32 = 191; p=64 -> page 5: 320 + 32 = 352
    assert out[1, :4].tolist() == [191, 352, 33, -1]


def _small():
    return _build(
        dict(topk=2048, pool_layers=79, seq_rows=[2], kv_lens=[3000], max_blocks=512), seed=7
    )


def test_guard_refuses_a_block_table_of_another_width() -> None:
    """The launcher reads the table at the compiled row stride."""
    req, table, tok, bs, stride = _small()
    with pytest.raises(AssertionError, match="blocks wide"):
        with validating(op):
            op(req, table[:, :256].contiguous(), tok, bs, stride, 0, compiled=_compiled(2048, 512))


def test_guard_refuses_a_selection_of_another_width() -> None:
    """2304 columns pass the wrapper's multiple-of-256 check; the launcher walks 2048."""
    req, table, tok, bs, stride = _small()
    wide = torch.cat([tok, torch.full((2, 256), -1, dtype=torch.int32, device="cuda")], 1)
    with pytest.raises(AssertionError, match="compiled for 2048"):
        with validating(op):
            op(req, table, wide, bs, stride, 0, compiled=_compiled(2048, 512))


def test_guard_refuses_an_int32_overflow() -> None:
    req, table, tok, bs, _ = _small()
    with pytest.raises(AssertionError, match="does not fit int32"):
        with validating(op):
            op(req, table, tok, bs, 2**30, 0, compiled=_compiled(2048, 512))
