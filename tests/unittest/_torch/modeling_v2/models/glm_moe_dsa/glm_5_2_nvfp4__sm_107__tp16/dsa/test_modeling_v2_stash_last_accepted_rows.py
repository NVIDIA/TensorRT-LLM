# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM stash_last_accepted_rows entry: a row gather, compared exactly.

The destination is either its own [requests, 2048] buffer or rows of a wider shared buffer whose
other rows must stay as they were.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa.stash_last_accepted_rows import (  # noqa: E501
    stash_last_accepted_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "stash_last_accepted_rows requires a CUDA device"


def _build(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    requests, next_n, n_cols = spec["requests"], spec["next_n"], spec["n_cols"]
    gen_topk = torch.randint(-1, 1 << 20, (requests * next_n, n_cols), generator=gen)
    shared = torch.full((requests + 3, n_cols), -7, dtype=torch.int32, device="cuda")
    dst = shared[:requests] if spec["dst"] == "wide" else torch.full_like(shared[:requests], -7)
    args = dict(
        gen_topk=gen_topk.to(torch.int32).cuda(),
        num_accepted=torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda"),
        dst=dst,
        next_n=next_n,
    )
    return args, shared


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, shared = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert out is args["dst"]
    assert torch.equal(out, ref)
    if cell.spec["dst"] == "wide":
        assert (shared[cell.spec["requests"] :] == -7).all(), "rows past the requests moved"


def test_rows_follow_the_accepted_counts_in_place_under_replay():
    """Captured once, replayed with new counts: the counts are read on the device."""
    args, _ = _build(op.CELLS[0].spec, seed=71)
    op(**args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)
    for accepted in (1, 6, 8):
        args["num_accepted"].fill_(accepted)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(args["dst"][0], args["gen_topk"][accepted - 1])


def test_guard_refuses_a_topk_of_another_row_count():
    args, _ = _build(op.CELLS[0].spec, seed=72)
    with pytest.raises(AssertionError, match="gen_topk must be"):
        with validating(op):
            op(**dict(args, gen_topk=args["gen_topk"][:4]))


def test_guard_refuses_a_strided_destination():
    args, _ = _build(op.CELLS[0].spec, seed=73)
    wide = torch.zeros(1, 2 * 2048, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="unit column strides"):
        with validating(op):
            op(**dict(args, dst=wide[:, ::2]))
