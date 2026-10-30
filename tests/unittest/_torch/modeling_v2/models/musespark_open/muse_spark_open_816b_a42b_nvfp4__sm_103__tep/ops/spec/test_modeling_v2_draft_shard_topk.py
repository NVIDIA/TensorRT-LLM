# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_shard_topk`` (fast path ``draft_head``).

Gated on its cells with ``torch.equal`` against its torch reference, in both
output layouts (the int32 keys the all-gather takes and the fp32 halves the
fabric all-reduce takes), with and without a drafter logit transform's
canonical keys.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_shard_topk as shard_mod,
)

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import (  # noqa: E402
    MUSESPARK,
    draft_head,
    draft_shard_topk,
    logits,
    seed_of,
    transform,
)
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"

CELLS = shard_mod.DraftShardTopK.CELLS
KINDS = ("gemm3", "special_nan", "straddle_up", "straddle_down", "ladder", "saturated")


def _call_args(spec, x, rank=3, num_ranks=8):
    kwargs = {}
    if spec["halves"]:
        kwargs.update(rank=rank, num_ranks=num_ranks)
    if spec["transform"] is not None:
        kwargs.update(draft_head.canon_kwargs(transform(MUSESPARK)))
    return kwargs


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_shard_topk_cells(cell, kind, certifying):  # noqa: F811
    certifying(draft_shard_topk)
    spec = cell.spec
    x = logits(
        kind, spec["rows"], spec["shard"], seed_of(cell.why + kind), shard=spec["shard"] // 2
    )
    kwargs = _call_args(spec, x)
    out = draft_shard_topk(x, **kwargs)
    ref = draft_shard_topk.reference(x, **kwargs)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out, ref), f"{(out != ref).sum().item()} keys differ"


def test_without_a_table_the_keys_are_the_plain_ones():
    """``canon=None`` is the table-free kernel; a table changes only keys outside its band."""
    t = transform(MUSESPARK)
    x = logits("gemm100", 7, 25256, 11)
    plain = draft_shard_topk(x)
    assert torch.equal(plain, draft_shard_topk.reference(x))
    inside = x.clamp(-500, 500)
    assert torch.equal(
        draft_shard_topk(inside, **draft_head.canon_kwargs(t)), draft_shard_topk(inside)
    )
    straddle = logits("straddle_up", 7, 25256, 12)
    assert not torch.equal(
        draft_shard_topk(straddle, **draft_head.canon_kwargs(t)), draft_shard_topk(straddle)
    )


def test_shard_topk_reads_a_strided_row_view():
    t = transform(MUSESPARK)
    full = logits("gemm400", 7, 202048, 5)
    view = full[:, 3 * 25256 : 4 * 25256]
    assert view.stride(0) == 202048
    kwargs = draft_head.canon_kwargs(t)
    assert torch.equal(
        draft_shard_topk(view, **kwargs), draft_shard_topk(view.contiguous(), **kwargs)
    )


def test_shard_topk_guards_refuse_what_the_kernel_cannot_take():
    t = transform(MUSESPARK)
    x = torch.zeros(2, 1024, device="cuda", dtype=torch.bfloat16)
    with validating(draft_shard_topk):
        with pytest.raises(ValueError):  # wider than the 16-bit column field
            draft_shard_topk(torch.zeros(2, 70000, device="cuda", dtype=torch.bfloat16))
        with pytest.raises(ValueError):  # not bf16
            draft_shard_topk(torch.zeros(2, 1024, device="cuda", dtype=torch.float32))
        with pytest.raises(ValueError):  # a table that is not the identity on its band
            draft_shard_topk(x, canon=t.canon, canon_band=(0, 0xFFFF))
        with pytest.raises(ValueError):  # a table of the wrong dtype
            draft_shard_topk(x, canon=t.canon.long(), canon_band=t.band)
        with pytest.raises(ValueError):  # a table that does not keep the order
            draft_shard_topk(x, canon=t.canon.flip(0).contiguous(), canon_band=(0x8000, 0x8000))
        draft_shard_topk(x, **draft_head.canon_kwargs(t))
