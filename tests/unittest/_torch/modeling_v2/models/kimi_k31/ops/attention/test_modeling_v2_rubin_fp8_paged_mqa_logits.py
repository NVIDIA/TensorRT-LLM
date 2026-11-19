# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_fp8_paged_mqa_logits entry (family FP8 paged MQA logits, Rubin).

The keys live on random physical pages of a [pages, P, 1, 132] cache view; the schedule comes from
DeepGEMM's schedule builder. Only each row's causal keys are compared. Each sampled cell is gated against
the entry's reference: bit for bit on integer data, within the fp32 accumulation band otherwise.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_fp8_paged_mqa_logits import (  # noqa: E501
    rubin_fp8_paged_mqa_logits as op,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the Rubin family entries are certified on sm_107",
)

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


_SENTINEL = 12345.0
_ROW_ALIGN = 256


def _out(rows: int, max_context_len: int, row_stride: int | None = None) -> torch.Tensor:
    """A sentinel-filled caller buffer with one spare row, and the [rows, max_context_len] view."""
    stride = -(-max_context_len // _ROW_ALIGN) * _ROW_ALIGN if row_stride is None else row_stride
    base = torch.full((rows + 1, stride), _SENTINEL, dtype=torch.float32, device="cuda")
    return base[:rows, :max_context_len]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    kv_cache, table = inputs.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"]
    )
    schedule = get_paged_mqa_logits_metadata(
        lens.reshape(-1, 1).cuda(), inputs.SCHEDULE_BLOCK_KV, inputs.num_sms()
    )
    q = inputs.queries(batch * next_n, spec["data"], gen).view(
        batch, next_n, inputs.NUM_INDEX_HEADS, inputs.HEAD_DIM
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        weights=inputs.weights(batch * next_n, spec["data"], gen),
        context_lens=lens.cuda(),
        block_table=table,
        schedule_meta=schedule,
        max_context_len=spec["max_context_len"],
        kv_evict_first=spec["kv_evict_first"],
        out=_out(batch * next_n, spec["max_context_len"]),
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    first = op(**args).clone()
    flipped = op(**dict(args, kv_evict_first=not args["kv_evict_first"], out=_out(*first.shape)))
    ref = op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    assert torch.equal(first[causal], flipped[causal]), "the L2 policy changed the logits"
    if cell.spec["data"] == "int":
        assert torch.equal(first[causal], ref[causal])
        return
    inputs.check_paged_logits(first, ref, args, args["q"].shape[1], args["context_lens"])
