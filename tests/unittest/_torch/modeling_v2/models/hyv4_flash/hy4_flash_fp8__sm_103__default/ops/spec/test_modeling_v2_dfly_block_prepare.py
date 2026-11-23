# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's DFly2 draft-block prepare entries: integer outputs, exact equality."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
    PAD_SLOT_ID,
    PAGE_ROWS,
    TARGET_TOPK,
    DraftBlock,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_indices as indices,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    dfly_block_prepare as prepare,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "dfly_block_prepare requires a CUDA device"
assert get_sm_version() == 103, (
    f"the hyv4_flash target's entries are certified on sm_103, not {get_sm_version()}"
)

PREPARE_CELLS = prepare.CELLS
INDICES_CELLS = indices.CELLS

DEV = "cuda"
VOCAB = 120832


def _prepare_args(spec: dict, seed: int) -> dict:
    """A target batch of `prefix` context tokens then one block per request, disjoint page ranges.

    With raw page ids the first request starts at page 0 (a real page in TRT-LLM);
    `page_scale`/`layer_offset` give the caller's `page * scale + layer_offset` table form.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    gens = spec["gens"]
    width = spec.get("block_width", BLOCK_WIDTH)
    prefix, offset = spec.get("prefix", 0), spec.get("pos_offset", 0)
    pages, block_size = spec.get("pages", 256), spec.get("block_size", PAGE_ROWS)
    tokens = prefix + gens * width
    positions = torch.arange(tokens, device=DEV, dtype=torch.int64) + offset
    query_start_loc = torch.tensor(
        [prefix + i * width for i in range(gens + 1)], device=DEV, dtype=torch.int64
    )
    accepted = spec["accepted"]
    if accepted == "all":
        num_accepted = torch.full((gens,), width - 1, device=DEV, dtype=torch.int32)
    elif accepted == "mixed":
        num_accepted = torch.randint(0, width, (gens,), generator=g, device=DEV, dtype=torch.int32)
    elif accepted == "mixed-with-zero":
        num_accepted = torch.randint(1, width, (gens,), generator=g, device=DEV, dtype=torch.int32)
        num_accepted[gens // 2] = 0
    elif accepted == "zero":
        num_accepted = torch.zeros(gens, device=DEV, dtype=torch.int32)
    else:
        raise ValueError(accepted)
    bonus = torch.randint(0, VOCAB, (gens,), generator=g, device=DEV, dtype=torch.int64)
    used = (offset + tokens + block_size - 1) // block_size + 1
    assert used <= pages, (used, pages)
    page_scale, layer_offset = spec.get("page_scale", 1), spec.get("layer_offset", 0)
    table = torch.zeros(gens, pages, dtype=torch.int32, device=DEV)
    page_ids = torch.arange(used, dtype=torch.int32, device=DEV)
    for i in range(gens):
        table[i, :used] = (page_ids + i * used) * page_scale + layer_offset
    return dict(
        bonus=bonus,
        num_accepted=num_accepted,
        query_start_loc=query_start_loc,
        target_positions=positions,
        block_table=table,
        max_model_len=spec.get("max_model_len", 5120),
        mask_token_id=MASK_TOKEN_ID,
        block_width=width,
        block_size=block_size,
    )


def _indices_args(spec: dict, seed: int) -> dict:
    """Distinct pool rows as picks, a state wider than the batch, block slots above every pick."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    layers, gens, states = spec["layers"], spec["gens"], spec["states"]
    width = spec.get("block_width", BLOCK_WIDTH)
    valid = spec["valid"]
    ranks = torch.rand(layers, states, TARGET_TOPK, generator=g, device=DEV).argsort(-1)
    history = torch.full((layers, states, TARGET_TOPK), PAD_SLOT_ID, dtype=torch.int32, device=DEV)
    if valid > 0:
        history[..., :valid] = ranks[..., :valid].to(torch.int32)
    state_rows = torch.randperm(states, generator=g, device=DEV)[:gens].to(torch.int64)
    pool = 1 << 18
    block_slots = (pool + torch.arange(gens * width, device=DEV)).reshape(gens, width)
    block_slots = block_slots.to(torch.int32)
    if spec["active"] == "all":
        active = torch.ones(gens, dtype=torch.bool, device=DEV)
    elif spec["active"] == "mixed":
        active = torch.arange(gens, device=DEV) % 2 == 0
    else:
        raise ValueError(spec["active"])
    # Inactive requests carry pad slots, as DflyBlockPrepare writes them.
    block_slots[~active] = PAD_SLOT_ID
    out_width = spec.get("out_width", TARGET_TOPK + width)
    return dict(
        history=history,
        state_rows=state_rows,
        block_slots=block_slots,
        active=active,
        packed=torch.empty(layers, gens * width, out_width, dtype=torch.int32, device=DEV),
        lengths=torch.empty(layers, gens * width, dtype=torch.int32, device=DEV),
    )


@pytest.mark.parametrize("cell", PREPARE_CELLS, ids=[c.why[:44] for c in PREPARE_CELLS])
def test_block_prepare_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture
    certifying(prepare)
    args = _prepare_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = prepare(**args)
    ref = prepare.reference(**args)
    for name in DraftBlock._fields:
        a, b = getattr(out, name), getattr(ref, name)
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
        assert torch.equal(a, b), f"{name} differs in {int((a != b).sum())} of {a.numel()} elements"


@pytest.mark.parametrize("cell", INDICES_CELLS, ids=[c.why[:44] for c in INDICES_CELLS])
def test_block_indices_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture
    certifying(indices)
    args = _indices_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = _indices_args(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    indices(**args)
    indices.reference(**ref)
    for name in ("packed", "lengths"):
        differing = int((args[name] != ref[name]).sum())
        assert differing == 0, f"{name} differs in {differing} of {args[name].numel()} elements"
