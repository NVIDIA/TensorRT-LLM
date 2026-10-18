# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_path_walk`` (fast path ``draft_head``).

Gated on its cells with ``torch.equal`` against its reference and against the
built-in walk + ``fill_(-inf)`` / ``scatter_`` + ``argmax`` over the vocabulary.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_path_walk as walk_mod,
)
from tensorrt_llm._torch.models.modeling_dflash import dflash2_walk_candidate_paths

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import TOP_K, VOCAB, draft_path_walk, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"


def _walk_inputs(spec, seed):
    requests, positions = spec["requests"], spec["positions"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(VOCAB, generator=g, device="cuda")[:TOP_K]
            for _ in range(requests * positions)
        ]
    ).view(requests, positions, TOP_K)
    if spec["scores"] == "ties":
        unary = torch.zeros(requests, positions, TOP_K, device="cuda")
        edges = (
            torch.randint(
                0, 3, (requests, positions, TOP_K, TOP_K), generator=g, device="cuda"
            ).float()
            / 4
        )
    else:
        unary = torch.randn(requests, positions, TOP_K, generator=g, device="cuda") * 4
        edges = torch.randn(requests, positions, TOP_K, TOP_K, generator=g, device="cuda")
    if spec["scores"] == "special":
        unary[0, 0, 3] = float("nan")
        unary[0, 1, :] = float("-inf")
        if positions > 2:
            unary[0, 2, 5] = float("inf")
            edges[0, 2, :, 9] = float("inf")
    return edges.contiguous(), unary.contiguous(), ids.contiguous()


def _builtin_tokens(edges, unary, ids):
    realized = dflash2_walk_candidate_paths(ids, unary[:, :, None] + edges)
    block = torch.full((*unary.shape[:2], VOCAB), float("-inf"), device="cuda")
    block.scatter_(-1, ids, realized)
    return torch.argmax(block.reshape(-1, VOCAB), dim=-1).to(torch.int32).view(unary.shape[:2])


@pytest.mark.parametrize(
    "cell", walk_mod.DraftPathWalk.CELLS, ids=[c.why[:48] for c in walk_mod.DraftPathWalk.CELLS]
)
def test_path_walk_cells(cell, certifying):  # noqa: F811
    certifying(draft_path_walk)
    edges, unary, ids = _walk_inputs(cell.spec, seed_of(cell.why))
    out = draft_path_walk(edges, unary, ids)
    assert torch.equal(out, draft_path_walk.reference(edges, unary, ids))
    assert torch.equal(out, _builtin_tokens(edges, unary, ids))


def test_path_walk_guards_refuse_what_the_kernel_cannot_take():
    with validating(draft_path_walk):
        with pytest.raises(ValueError):  # int32 candidate ids
            draft_path_walk(
                torch.zeros(1, 7, 16, 16, device="cuda"),
                torch.zeros(1, 7, 16, device="cuda"),
                torch.zeros(1, 7, 16, dtype=torch.int32, device="cuda"),
            )
