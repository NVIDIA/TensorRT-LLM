# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_decode_gvr entries (deterministic self-sampling GVR).

Each row's selected set must equal the reference's exactly (ties at the K-th score to the smaller
column); a second call must repeat the output, order included. The pool-row entry's rows must equal
its reference's bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E501
    indexer_topk_decode_gvr as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E501
    indexer_topk_decode_gvr_paged as paged_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS`` and of ``paged_op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="batch")[:2]
_PAGED_CELLS = sample_cells(paged_op.CELLS, size_key="batch")[:2]


_SPLIT_KV = 256


def _scores(n: int, data: str, gen: torch.Generator) -> torch.Tensor:
    if data == "ties":
        return torch.randint(0, 64, (n,), generator=gen, device="cuda").float()
    # Indexer-like scores: a broad negative bulk with a heavy positive tail.
    bulk = torch.randn(n, generator=gen, device="cuda") * 1.5 - 2.0
    tail = torch.rand(n, generator=gen, device="cuda") < 0.02
    spikes = torch.rand(n, generator=gen, device="cuda") * 6.0
    return torch.where(tail, bulk.abs() + spikes, bulk)


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch, next_n, k = spec["batch"], spec["next_n"], spec["topk"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
    width = spec["max_seq_len"]
    stride = -(-width // _SPLIT_KV) * _SPLIT_KV + spec["row_pad"]
    rows = batch * next_n
    base = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    for r in range(rows):
        n = int(lens[r // next_n]) - next_n + r % next_n + 1
        base[r, :n] = _scores(n, spec["data"], gen)
    return dict(
        logits=base[:, :width],
        kv_lens=lens,
        indices=torch.full((rows, k), -5, dtype=torch.int32, device="cuda"),
        next_n=next_n,
        max_seq_len=width,
    )


def _sorted(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(_sorted(out), _sorted(ref))
    # Deterministic: the same output, order included, on a second call.
    args["indices"].fill_(-5)
    op(**args)
    assert torch.equal(args["indices"], out)


# ---------------------------------------------------------------------------------------------
# indexer_topk_decode_gvr_paged: the decode rows as ordered KV-pool rows
# ---------------------------------------------------------------------------------------------

_BLOCK = 64
_SCALE = 24
_LAYER = 7


def _build_paged(spec, seed: int) -> dict:
    args = _build(spec, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    batch = spec["batch"]
    width = -(-spec["context_lens"][1] // _BLOCK) + 1
    if spec.get("holes"):
        width = width // 2
    full = torch.randint(
        0, 50_000, (batch + 3, width), generator=gen, dtype=torch.int32, device="cuda"
    )
    if spec.get("holes"):
        full.masked_fill_(torch.rand(full.shape, generator=gen, device="cuda") < 0.05, -1)
    return dict(
        args,
        block_table=full[3:],
        block_size=_BLOCK,
        stride_factor=_SCALE * _BLOCK,
        layer_id=_LAYER,
    )


@pytest.mark.parametrize("cell", _PAGED_CELLS, ids=[c.why[:48] for c in _PAGED_CELLS])
def test_paged_certified_cells(cell, certifying):  # noqa: F811
    certifying(paged_op)
    args = _build_paged(cell.spec, inputs.seed_of(cell))
    paged_op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    paged_op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
