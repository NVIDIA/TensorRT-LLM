# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_dsa_decode_metadata entry: integer outputs, compared exactly
against the eager chain it replaces.

The outputs are slices of longer sentinel-filled buffers, as the metadata's persistent buffers
are sliced to the step: the live extent must match the reference and the tail must stay
untouched.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.fused_dsa_decode_metadata import (
    fused_dsa_decode_metadata as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: K3.1's index keys: 128 FP8 bytes per token, one fp32 scale per 128 values.
INDEX_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128

_TAIL = 37
_OUTPUTS = (
    "req_idx_per_token",
    "slot_mapping_fp8",
    "slot_mapping_scale",
    "gen_kv_indptr",
    "gen_cached_token_indptr",
)


def _build(spec, seed: int) -> tuple[dict, dict]:
    """(arguments, the full buffers the outputs are slices of)."""
    gen = torch.Generator().manual_seed(seed)
    batch, next_n, tpb = spec["batch"], spec["next_n"], spec["tokens_per_block"]
    lo, hi = spec["kv_lens"]
    seq_lens = torch.full((batch,), next_n, dtype=torch.int32)
    kv_lens = torch.randint(max(lo, next_n), hi + 1, (batch,), generator=gen, dtype=torch.int32)
    width = -(-int(kv_lens.max()) // tpb) + spec["table_slack"]
    # The index-key cache's block offsets: pool page ids scaled to index-key pages, the first
    # rows of a larger table.
    table = torch.randint(0, spec["pool_pages"], (batch + 2, width), generator=gen) * 3
    num_tokens = batch * next_n
    full = dict(
        req_idx_per_token=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int32),
        slot_mapping_fp8=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int64),
        slot_mapping_scale=torch.full((num_tokens + _TAIL,), -7, dtype=torch.int64),
        gen_kv_indptr=torch.full((batch + 1 + _TAIL,), -7, dtype=torch.int64),
        gen_cached_token_indptr=torch.full((batch + 1 + _TAIL,), -7, dtype=torch.int64),
    )
    full = {k: v.cuda() for k, v in full.items()}
    args = dict(
        seq_lens=seq_lens.cuda(),
        kv_lens=kv_lens.cuda(),
        block_offsets=table.int().cuda()[:batch],
        req_idx_per_token=full["req_idx_per_token"][:num_tokens],
        slot_mapping_fp8=full["slot_mapping_fp8"][:num_tokens],
        slot_mapping_scale=full["slot_mapping_scale"][:num_tokens],
        gen_kv_indptr=full["gen_kv_indptr"][: batch + 1],
        gen_cached_token_indptr=full["gen_cached_token_indptr"][: batch + 1],
        num_tokens=num_tokens,
        max_query_len=spec["max_query_len"],
        tokens_per_block=tpb,
        index_head_dim=INDEX_HEAD_DIM,
        quant_block_size=QUANT_BLOCK_SIZE,
        data_bytes_per_token=INDEX_HEAD_DIM,
    )
    return args, full


def _fresh_outputs(args: dict, full: dict) -> tuple[dict, dict]:
    """The same arguments over a second, identical set of sentinel buffers."""
    full2 = {k: torch.full_like(v, -7) for k, v in full.items()}
    args2 = dict(args)
    for name in _OUTPUTS:
        args2[name] = full2[name][: args[name].shape[0]]
    return args2, full2


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, full = _build(cell.spec, inputs.seed_of(cell))
    ref_args, ref_full = _fresh_outputs(args, full)
    op(**args)
    op.reference(**ref_args)
    torch.cuda.synchronize()
    for name in _OUTPUTS:
        assert torch.equal(full[name], ref_full[name]), f"{name} differs from the eager chain"
        tail = full[name][args[name].shape[0] :]
        assert bool((tail == -7).all()), f"{name} written past its view"


def test_slot_mappings_address_the_index_key_page_layout():
    """Independent of the reference: each token's key bytes sit at its position in its page,
    its scale after the page's P keys."""
    spec = op.CELLS[6].spec
    args, _ = _build(spec, seed=9)
    op(**args)
    torch.cuda.synchronize()
    tpb, d = spec["tokens_per_block"], INDEX_HEAD_DIM
    page_bytes = tpb * (d + 4)
    fp8, scale = args["slot_mapping_fp8"].long(), args["slot_mapping_scale"].long()
    seq = spec["next_n"]
    req = torch.arange(args["num_tokens"], device="cuda") // seq
    pos = args["kv_lens"].long()[req] - seq + torch.arange(args["num_tokens"], device="cuda") % seq
    page = args["block_offsets"].long()[req, pos // tpb]
    assert torch.equal(fp8 // page_bytes, page) and torch.equal(fp8 % page_bytes, pos % tpb * d)
    assert torch.equal(scale - fp8, (tpb - pos % tpb) * d + pos % tpb * 4)
    assert torch.equal(args["req_idx_per_token"].long(), req)


def test_guard_refuses_a_token_count_off_the_step():
    args, _ = _build(op.CELLS[1].spec, seed=1)
    with pytest.raises(AssertionError, match="num_tokens"):
        with validating(op):
            op(**dict(args, num_tokens=args["num_tokens"] - 1))


def test_guard_refuses_a_request_past_the_lane_count():
    args, _ = _build(op.CELLS[5].spec, seed=2)
    with pytest.raises(AssertionError, match="max_query_len"):
        with validating(op):
            op(**dict(args, max_query_len=4))


def test_guard_refuses_a_context_past_the_table():
    args, _ = _build(op.CELLS[1].spec, seed=3)
    pages = -(-int(args["kv_lens"].max()) // args["tokens_per_block"])
    with pytest.raises(AssertionError, match="block table row"):
        with validating(op):
            op(**dict(args, block_offsets=args["block_offsets"][:, : pages - 1]))
