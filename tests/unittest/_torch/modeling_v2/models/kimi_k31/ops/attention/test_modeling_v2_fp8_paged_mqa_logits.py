# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fp8_paged_mqa_logits entry (DeepGEMM paged logits).

Same cache and comparison as the CuTe DSL entry's test; the context lengths are the 2-D [B, n]
form and the schedule is built from them, as the K3.1 DSA metadata does for this kernel.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fp8_paged_mqa_logits import (  # noqa: E501
    fp8_paged_mqa_logits as op,
)
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    kv_cache, table = inputs.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"]
    )
    lens_2d = lens[:, None].expand(batch, next_n).contiguous().cuda()
    schedule = get_paged_mqa_logits_metadata(lens_2d, inputs.SCHEDULE_BLOCK_KV, inputs.num_sms())
    q = inputs.queries(batch * next_n, spec["data"], gen).view(
        batch, next_n, inputs.NUM_INDEX_HEADS, inputs.HEAD_DIM
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        weights=inputs.weights(batch * next_n, spec["data"], gen),
        context_lens=lens_2d,
        block_table=table,
        schedule_meta=schedule,
        max_context_len=spec["max_context_len"],
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    if cell.spec["data"] == "int":
        assert torch.equal(out[causal], ref[causal])
    else:
        inputs.check_paged_logits(
            out, ref, args, args["q"].shape[1], args["context_lens"][:, -1].tolist()
        )
