# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for k31_dflash2_noise_block and k31_draft_agreed_tokens: every cell bitwise against its
reference (the engine's noise block over ``F.embedding``; the agreement's conversion and
concatenation), and each launch captured in a CUDA graph and replayed on new inputs."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_glue import (  # noqa: E501
    HIDDEN,
    VOCAB,
    k31_dflash2_noise_block,
    k31_draft_agreed_tokens,
)
from tensorrt_llm._torch.speculative.dflash import dflash_noise_block_embedding
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107), reason="the K3.1 drafter glue runs on sm_103 and sm_107"
)

MASK_ID = 163838


def _table(g):
    table = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").bfloat16()
    # -0.0, infinities and NaN payloads copy as bits.
    table[MASK_ID, :4] = torch.tensor([-0.0, float("inf"), float("-inf"), float("nan")])
    table.view(torch.int16)[0, :2] = torch.tensor([0x7FC1, 0x0001], dtype=torch.int16)
    return table


def _bonus(spec, g):
    n = spec["n"]
    bonus = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int64)
    if n >= 2:
        bonus[0], bonus[-1] = 0, VOCAB - 1
    if spec.get("bonus_is_mask"):
        bonus[1] = MASK_ID
    return bonus


@pytest.fixture(scope="module")
def table():
    return _table(torch.Generator(device="cuda").manual_seed(11))


@pytest.mark.parametrize(
    "cell", k31_dflash2_noise_block.CELLS, ids=[c.why[:48] for c in k31_dflash2_noise_block.CELLS]
)
def test_noise_block_cells(cell, table, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    bonus = _bonus(spec, g)
    certifying(k31_dflash2_noise_block)
    got = k31_dflash2_noise_block(table, bonus, MASK_ID, spec["block"])
    want = k31_dflash2_noise_block.reference(table, bonus, MASK_ID, spec["block"])
    engine = dflash_noise_block_embedding(
        lambda ids: torch.nn.functional.embedding(ids, table), bonus, MASK_ID, spec["block"]
    )
    assert torch.equal(want.view(torch.int16), engine.view(torch.int16)), "reference vs engine"
    assert got.shape == want.shape == (spec["n"], spec["block"], HIDDEN)
    assert got.is_contiguous()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_noise_block_graph(table) -> None:
    """Captured once, replayed on new bonus ids: the reference of each."""
    g = torch.Generator(device="cuda").manual_seed(5)
    bonus = torch.randint(0, VOCAB, (2,), generator=g, device="cuda")
    k31_dflash2_noise_block(table, bonus, MASK_ID, 8)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = k31_dflash2_noise_block(table, bonus, MASK_ID, 8)
    for _ in range(3):
        bonus.copy_(torch.randint(0, VOCAB, (2,), generator=g, device="cuda"))
        graph.replay()
        want = k31_dflash2_noise_block.reference(table, bonus, MASK_ID, 8)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))


def _agreement(spec, g):
    rows, width = spec["rows"], spec["width"]
    reduced = torch.zeros(rows + 3, width, device="cuda")[:rows]
    reduced[:, :7] = torch.randint(0, VOCAB, (rows, 7), generator=g, device="cuda").float()
    reduced[0, 0] = VOCAB - 1
    reduced[:, 7:] = 3.5  # columns past the drafts are not read
    new_dtype = spec.get("new_dtype", torch.int32)
    new_tokens = torch.randint(0, VOCAB, (rows, 8), generator=g, device="cuda").to(new_dtype)
    return reduced, new_tokens


@pytest.mark.parametrize(
    "cell", k31_draft_agreed_tokens.CELLS, ids=[c.why[:48] for c in k31_draft_agreed_tokens.CELLS]
)
def test_agreed_tokens_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    reduced, new_tokens = _agreement(cell.spec, g)
    certifying(k31_draft_agreed_tokens)
    drafts, next_new = k31_draft_agreed_tokens(reduced, 7, torch.int32, new_tokens)
    want_drafts, want_new = k31_draft_agreed_tokens.reference(reduced, 7, torch.int32, new_tokens)
    assert drafts.dtype == want_drafts.dtype and next_new.dtype == want_new.dtype
    assert torch.equal(drafts, want_drafts) and torch.equal(next_new, want_new)
    assert int(drafts[0, 0]) == VOCAB - 1


def test_agreed_tokens_graph() -> None:
    g = torch.Generator(device="cuda").manual_seed(9)
    reduced, new_tokens = _agreement(dict(rows=2, width=2048), g)
    k31_draft_agreed_tokens(reduced, 7, torch.int32, new_tokens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        drafts, next_new = k31_draft_agreed_tokens(reduced, 7, torch.int32, new_tokens)
    for _ in range(3):
        fresh, fresh_new = _agreement(dict(rows=2, width=2048), g)
        reduced.copy_(fresh)
        new_tokens.copy_(fresh_new)
        graph.replay()
        want_drafts, want_new = k31_draft_agreed_tokens.reference(
            reduced, 7, torch.int32, new_tokens
        )
        assert torch.equal(drafts, want_drafts) and torch.equal(next_new, want_new)
