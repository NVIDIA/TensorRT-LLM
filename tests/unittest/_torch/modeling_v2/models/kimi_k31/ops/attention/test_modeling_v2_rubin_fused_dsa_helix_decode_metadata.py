# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_fused_dsa_helix_decode_metadata entry: integer outputs, compared exactly
against the eager chain with the Helix ownership rewrite (the entry's reference).

The outputs are slices of longer sentinel-filled buffers, as the metadata's persistent buffers are
sliced to the step: the live extent must match and the tail must stay untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_fused_dsa_helix_decode_metadata import (  # noqa: E501
    rubin_fused_dsa_helix_decode_metadata as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(107)

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


_TAIL = 29
_OUTPUTS = (
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
    "kv_lens_2d",
    "kv_lens_expanded",
)


def _build(spec, seed: int, rank: int = 3) -> tuple[dict, dict]:
    """(arguments, the full buffers the outputs are slices of) of CP rank ``rank``."""
    gen = torch.Generator().manual_seed(seed)
    batch, next_n, tpb, cp = spec["batch"], spec["next_n"], spec["tokens_per_block"], spec["cp"]
    lo, hi = spec["kv_lens"]
    seq_lens = torch.full((batch,), next_n, dtype=torch.int32)
    global_kv = torch.randint(max(lo, next_n), hi + 1, (batch,), generator=gen, dtype=torch.int32)
    # Page g of a sequence lives on rank g % cp; this rank holds its pages back to back.
    pos = global_kv.long().unsqueeze(1) - next_n + torch.arange(next_n)
    owner = (pos // tpb) % cp
    local = (pos // (tpb * cp)) * tpb + pos % tpb
    local_slots = torch.where(owner == rank, local, -1).reshape(-1).int()
    local_kv = ((global_kv.long() - 1) // (tpb * cp) * tpb + tpb).int()
    bounds = torch.randint(0, 2, (batch * next_n,), generator=gen).int() * (
        local_kv.repeat_interleave(next_n)
    )
    width = int(local_kv.max()) // tpb + 2
    table = (torch.randint(0, 40_000, (batch + 2, width), generator=gen) * 3).int()
    num_tokens = batch * next_n
    full = dict(
        req_idx_per_token=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int32),
        slot_mapping_fp8=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int64),
        slot_mapping_scale=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int64),
        gen_kv_indptr=torch.full((batch + 1 + _TAIL,), -7, dtype=torch.int64),
        gen_cached_token_indptr=torch.full((batch + 1 + _TAIL,), -7, dtype=torch.int64),
        kv_lens_2d=torch.full((batch + 2, next_n + 1), -7, dtype=torch.int32),
        kv_lens_expanded=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int32),
    )
    full = {k: v.cuda() for k, v in full.items()}
    args = dict(
        seq_lens=seq_lens.cuda(),
        kv_lens=local_kv.cuda(),
        block_offsets=table.cuda(),
        helix_local_slots=local_slots.cuda(),
        helix_kv_bounds=bounds.cuda(),
        req_idx_per_token=full["req_idx_per_token"],
        slot_mapping_fp8=full["slot_mapping_fp8"],
        slot_mapping_scale=full["slot_mapping_scale"],
        gen_kv_indptr=full["gen_kv_indptr"],
        gen_cached_token_indptr=full["gen_cached_token_indptr"],
        kv_lens_2d=full["kv_lens_2d"],
        kv_lens_expanded=full["kv_lens_expanded"],
        num_tokens=num_tokens,
        width=next_n,
        tokens_per_block=tpb,
        index_head_dim=inputs.HEAD_DIM,
        quant_block_size=128,
        data_bytes_per_token=inputs.HEAD_DIM,
    )
    return args, full


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, full = _build(cell.spec, inputs.seed_of(cell))
    full2 = {k: torch.full_like(v, -7) for k, v in full.items()}
    ref_args = dict(args, **full2)
    op(**args)
    op.reference(**ref_args)
    torch.cuda.synchronize()
    for name in _OUTPUTS:
        assert torch.equal(full[name], full2[name]), f"{name} differs from the eager chain"
