# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 cute_dsl_fp8_paged_mqa_logits_pdl entry (the PDL launch of the CuTe
DSL paged logits).

The keys live on random physical pages of a [pages, P, 1, 132] cache view; the schedule comes from
DeepGEMM's schedule builder over the cell's atom rows, as the K3.1 DSA metadata builds it. Only
each row's causal keys are compared with the entry's reference.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (
    cute_dsl_fp8_paged_mqa_logits_pdl as pdl_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (
    logits_row_stride,
)
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``pdl_op.CELLS``.
_CELLS = sample_cells(pdl_op.CELLS, size_key="batch")[:2]


_SENTINEL = 12345.0


def _out(rows: int, max_context_len: int) -> torch.Tensor:
    """A sentinel-filled caller buffer and the [rows, max_context_len] view the op takes."""
    stride = logits_row_stride(max_context_len)
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
    atoms = spec["num_next_n_atoms"]
    if atoms > 1:
        offsets = torch.tensor(inputs.atom_ctx_offsets(next_n, atoms), dtype=torch.int32)
        schedule_lens = (lens[:, None] - offsets).reshape(-1, 1)
    else:
        schedule_lens = lens.reshape(-1, 1)
    schedule = get_paged_mqa_logits_metadata(
        schedule_lens.cuda(), inputs.SCHEDULE_BLOCK_KV, inputs.num_sms()
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
        num_next_n_atoms=atoms,
        atom=spec["atom"],
        out=_out(batch * next_n, spec["max_context_len"]),
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_pdl_certified_cells(cell, certifying):  # noqa: F811
    certifying(pdl_op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = pdl_op(**args)
    ref = pdl_op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    if cell.spec["data"] == "int":
        assert torch.equal(out[causal], ref[causal])
    else:
        inputs.check_paged_logits(out, ref, args, args["q"].shape[1], args["context_lens"])
