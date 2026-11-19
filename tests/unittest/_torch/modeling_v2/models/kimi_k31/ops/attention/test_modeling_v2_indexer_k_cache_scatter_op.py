# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_k_cache_scatter_op entry (index keys into the paged cache).

The cache is random bytes; the step's keys go to distinct random slots. The op and the reference
each write a copy of the whole backing buffer, page-stride padding included, and the two copies
must be byte-identical: every byte the op should write, and none it should not.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_k_cache_scatter_op import (  # noqa: E501
    indexer_k_cache_scatter_op as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="num_tokens")[:2]


def _build(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    page, pages, n = spec["tokens_per_block"], spec["num_pages"], spec["num_tokens"]
    rows = spec["slot_rows"] or n
    base, view = inputs.raw_index_cache(pages, page, gen, spec["page_stride_pad"])
    fp8_slots, scale_slots = inputs.slot_mappings(rows, pages, page, gen)
    k, scales = inputs.keys(rows, "randn", gen)
    args = dict(
        k_fp8=k.cuda(),
        # The indexer's fp32 [T, 1] scales, viewed as int32 as it passes them.
        k_scale=scales.reshape(-1, 1).view(torch.int32).cuda(),
        k_cache=view,
        slot_mapping_fp8=fp8_slots,
        slot_mapping_scale=scale_slots,
        num_tokens=n,
    )
    return base, args


def _rebase(base: torch.Tensor, args: dict) -> tuple[torch.Tensor, dict]:
    """A copy of the backing buffer and the same args pointing into it."""
    copy = base.clone()
    page_bytes = args["k_cache"].shape[1] * args["k_cache"].shape[3]
    view = copy[:, :page_bytes].view(args["k_cache"].shape)
    return copy, dict(args, k_cache=view)


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    base, args = _build(cell.spec, inputs.seed_of(cell))
    got, op_args = _rebase(base, args)
    want, ref_args = _rebase(base, args)
    op(**op_args)
    op.reference(**ref_args)
    torch.cuda.synchronize()
    assert torch.equal(got, want), f"{int((got != want).sum())} bytes differ"
    assert not torch.equal(got, base), "nothing was written"
