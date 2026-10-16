# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_req_index_to_base entry."""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_req_index_to_global as _global_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_req_index_to_base import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_req_index_to_base import (  # noqa: E402, E501
    cute_dsl_req_index_to_base as op,
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

    Same construction as the per-layer entry's test: causal top-k rows padded
    with -1, a `max_blocks`-wide table over a shuffled page pool, unused
    entries -1, and with `pad_block_table` one in-range unused page as well.
    """
    g = torch.Generator().manual_seed(seed)
    topk, width = spec["topk"], spec["max_blocks"]
    pages_needed = [math.ceil(n / BLOCK) for n in spec["kv_lens"]]
    pool = torch.randperm(2 * sum(pages_needed) + 2, generator=g).to(torch.int32)
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
            row = torch.full((topk,), -1, dtype=torch.int32)
            pick = torch.randperm(kv - r_count + j + 1, generator=g)[:topk].to(torch.int32)
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
        out = op(*args, compiled=compiled)
    for a, s in zip(args[:3], snap):
        assert torch.equal(a, s), "an input tensor was mutated"
    _compare(out, op.reference(*args, compiled=compiled))
    assert bool((out[args[2] < 0] == -1).all())
    if spec.get("pad_block_table"):
        assert bool((out[args[2] >= 0] == -1).any()), "the cell drove no unresolvable position"


@pytest.mark.parametrize("layer", [0, 1, 40, 78])
def test_base_plus_layer_offset_is_the_global_row(layer) -> None:
    """What the decode kernel relies on: base + layer * 32 == the per-layer entry's row."""
    spec = dict(topk=2048, pool_layers=79, seq_rows=[2] * 4, kv_lens=[3000, 64, 9000, 2049])
    spec["max_blocks"] = 512
    req, table, tok, bs, stride = _build(spec, seed=31)
    base = op(req, table, tok, bs, stride, compiled=_compiled(2048, 512))
    glob_compiled = _global_mod.build(torch.cuda.current_device(), topk=2048, max_blocks=512)
    glob = _global_mod.cute_dsl_req_index_to_global(
        req, table, tok, bs, stride, layer, compiled=glob_compiled
    )
    live = base >= 0
    assert torch.equal(live, glob >= 0), "the two entries disagree on which slots resolve"
    assert torch.equal(base[live] + layer * BLOCK, glob[live])


def test_guard_refuses_a_block_table_of_another_width() -> None:
    spec = dict(topk=2048, pool_layers=79, seq_rows=[2], kv_lens=[3000], max_blocks=512)
    req, table, tok, bs, stride = _build(spec, seed=7)
    with pytest.raises(AssertionError, match="blocks wide"):
        with validating(op):
            op(req, table[:, :256].contiguous(), tok, bs, stride, compiled=_compiled(2048, 512))
