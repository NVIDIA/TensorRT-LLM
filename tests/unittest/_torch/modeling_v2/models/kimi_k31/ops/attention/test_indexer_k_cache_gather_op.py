# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_k_cache_gather_op entry (a prefill chunk's index keys out of
the paged cache).

The cache is random bytes; the full-KV slot mappings name distinct random slots and the chunk is a
window of them. Byte copies, so the op must equal the reference bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_k_cache_gather_op import (  # noqa: E501
    indexer_k_cache_gather_op as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    page, pages = spec["tokens_per_block"], spec["num_pages"]
    start, n = spec["k_token_start"], spec["num_tokens"]
    _, view = inputs.raw_index_cache(pages, page, gen, spec["page_stride_pad"])
    fp8_slots, scale_slots = inputs.slot_mappings(start + n + 7, pages, page, gen)
    return dict(
        k_cache=view,
        slot_mapping_fp8=fp8_slots,
        slot_mapping_scale=scale_slots,
        k_token_start=start,
        num_tokens=n,
        head_dim=inputs.HEAD_DIM,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    k_fp8, k_scale = op(**args)
    ref_fp8, ref_scale = op.reference(**args)
    torch.cuda.synchronize()
    n = args["num_tokens"]
    assert k_fp8.shape == (n, inputs.HEAD_DIM) and k_fp8.dtype == torch.float8_e4m3fn
    assert k_scale.shape == (n, 1) and k_scale.dtype == torch.float32
    assert torch.equal(k_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
    assert torch.equal(k_scale.view(torch.int32), ref_scale.view(torch.int32))


def test_guard_refuses_a_slot_past_the_cache():
    args = _build(op.CELLS[0].spec, seed=9)
    fp8 = args["slot_mapping_fp8"].clone()
    fp8[0] = args["k_cache"].numel()
    with pytest.raises(AssertionError, match="past the cache"):
        with validating(op):
            op(**dict(args, slot_mapping_fp8=fp8))
