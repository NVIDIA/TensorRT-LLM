# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_draft_input entry (embedding + both RMSNorms + concat in one launch), against its reference and
against the in-tree drafter's modules."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops import (  # noqa: E501
    m3_spec_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_draft_input import (  # noqa: E501
    m3_draft_input as op,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402 — needs the path declared above
    EPS,
    HIDDEN,
    VOCAB,
    assert_bf16_close,
    gen,
    randn,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_draft_input requires a CUDA device"


def _case(spec, seed):
    g = gen(seed)
    m, h = spec["m"], spec["h"]
    table = randn((VOCAB, h), g)
    ids = torch.randint(0, VOCAB, (m,), generator=g, device="cuda", dtype=torch.int32)
    if spec["invalid_id"]:
        ids[0] = -1
    hidden = randn((4 * m if spec["gather"] else m, h), g)
    rows = torch.arange(m, device="cuda", dtype=torch.int64) * 4 + 2 if spec["gather"] else None
    w_e = (1.0 + randn((h,), g, 0.1).float()).to(torch.bfloat16)
    w_h = (1.0 + randn((h,), g, 0.1).float()).to(torch.bfloat16)
    out = torch.full((m, 2 * h), 7.0, dtype=torch.bfloat16, device="cuda")
    residual = torch.full((m, h), 7.0, dtype=torch.bfloat16, device="cuda")
    return (ids, table, hidden, rows, w_e, w_h, EPS, EPS, out, residual, True)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Normalized halves within one bf16 ulp per element on at most 0.1% of the elements (fp32 statistic in a
    different summation order, rsqrtf); the gathered residual and the zero embedding of an invalid id exact."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    out, residual = op(*args)
    ref_out, ref_res = op.reference(*args)
    torch.cuda.synchronize()
    assert_bf16_close(out, ref_out)
    assert torch.equal(residual, ref_res)
    if cell.spec["invalid_id"]:
        assert not bool(out[0, : cell.spec["h"]].float().any())


def test_guard_refuses_rows_past_hidden() -> None:
    gather = next(c for c in op.CELLS if c.spec["gather"])
    args = list(_case(gather.spec, 1))
    args[3] = args[3] + 100
    with pytest.raises(AssertionError, match="rows must index hidden"):
        with validating(op):
            op(*args)


@pytest.mark.parametrize("m,gather", [(4, False), (32, False), (8, True)], ids=str)
def test_the_in_tree_draft_input(m, gather) -> None:
    """Against the in-tree drafter's own modules: the Embedding lookup, then the layer's input_layernorm and
    hidden_norm (in-tree RMSNorm) and the concat of Eagle3DecoderLayer.forward. Within one bf16 ulp on at most 0.1% of
    the elements (the RMS statistic summed in another order); the residual (the carried hidden rows) exact."""
    g = gen(seed_of(f"in-tree draft input {m} {gather}"))
    h = HIDDEN
    embed = Embedding(VOCAB, h, dtype=torch.bfloat16).cuda()
    embed.weight.data = randn((VOCAB, h), g)
    norms = []
    for _ in range(2):
        norm = RMSNorm(hidden_size=h, eps=EPS, dtype=torch.bfloat16).cuda()
        norm.weight.data = (1.0 + randn((h,), g, 0.1).float()).to(torch.bfloat16)
        norms.append(norm)
    input_layernorm, hidden_norm = norms
    ids = torch.randint(0, VOCAB, (m,), generator=g, device="cuda", dtype=torch.int32)
    hidden = randn((4 * m if gather else m, h), g, 2.0)
    rows = torch.arange(m, device="cuda", dtype=torch.int64) * 4 + 3 if gather else None
    carried = hidden[rows] if gather else hidden
    stock = torch.cat([input_layernorm(embed(ids)), hidden_norm(carried)], dim=-1)
    out, residual = m3_spec_tail.draft_input(
        ids, embed.weight, hidden, rows, input_layernorm.weight, EPS, hidden_norm.weight, EPS
    )
    torch.cuda.synchronize()
    assert_bf16_close(out, stock)
    assert torch.equal(residual, carried)
