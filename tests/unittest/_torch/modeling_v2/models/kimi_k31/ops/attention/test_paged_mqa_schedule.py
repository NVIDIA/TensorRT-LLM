# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 paged_mqa_schedule entry: integers, compared exactly with the
reference and with DeepGEMM's schedule (get_paged_mqa_logits_metadata) it replaces."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (  # noqa: E501
    get_paged_mqa_logits_metadata,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.paged_mqa_schedule import (  # noqa: E501
    paged_mqa_schedule as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _lengths(spec, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    lo, hi = spec["lens"]
    atoms = spec["stagger"]
    if atoms:
        requests = spec["rows"] // atoms
        lens = torch.randint(lo, hi + 1, (requests,), generator=gen, dtype=torch.int32)
        offsets = torch.tensor(inputs.atom_ctx_offsets(5, atoms), dtype=torch.int32)
        lens = (lens[:, None] - offsets).reshape(-1)
    else:
        lens = torch.randint(lo, hi + 1, (spec["rows"],), generator=gen, dtype=torch.int32)
    return lens[:, None].expand(-1, spec["next_n"]).contiguous().cuda()


def _build(spec, seed: int) -> dict:
    sms = inputs.num_sms()
    return dict(
        context_lens=_lengths(spec, seed),
        block_kv=spec["block_kv"],
        num_sms=sms,
        out=torch.full((sms + 1, 2), -7, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    ref = op.reference(**args)
    out = op(**args)
    deep_gemm = get_paged_mqa_logits_metadata(**dict(args, out=torch.empty_like(args["out"])))
    torch.cuda.synchronize()
    assert out is args["out"]
    assert torch.equal(out, ref), f"schedule differs:\n{out.tolist()}\nexpected\n{ref.tolist()}"
    assert torch.equal(out, deep_gemm)


def test_strided_lengths_and_graph_replay():
    """A column of the metadata's 2D length buffer (row stride > 1) under CUDA-graph replay."""
    sms = inputs.num_sms()
    lens2d = torch.randint(1000, 200_000, (32, 8), dtype=torch.int32, device="cuda")
    view = lens2d[:, :1]
    out = torch.zeros(sms + 1, 2, dtype=torch.int32, device="cuda")
    expected = get_paged_mqa_logits_metadata(view.contiguous(), 64, sms, torch.empty_like(out))
    graph = torch.cuda.CUDAGraph()
    op(view, 64, sms, out)
    torch.cuda.synchronize()
    out.zero_()
    with torch.cuda.graph(graph):
        op(view, 64, sms, out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


def test_guard_refuses_too_many_rows():
    sms = inputs.num_sms()
    lens = torch.ones(op_rows := 16385, 1, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="schedule rows"):
        with validating(op):
            op(lens, 64, sms, torch.empty(sms + 1, 2, dtype=torch.int32, device="cuda"))
    assert op_rows > 16384
