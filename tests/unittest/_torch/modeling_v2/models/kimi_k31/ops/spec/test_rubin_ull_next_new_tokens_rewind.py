# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's draft_next_new_tokens_rewind entry: the next input tokens and the
rewound kv_lens bitwise against the reference (integer ops)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_next_new_tokens_rewind import (  # noqa: E501
    draft_next_new_tokens_rewind as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31_pro_nvfp4__sm_107__helix8": tuple(
        Cell(
            why=f"helix8 b{gens} generation, K {k}",
            spec=dict(contexts=0, gens=gens, draft_len=k, dtypes=("int32", "int32")),
        )
        for gens, k in ((2, 5), (4, 5), (9, 5), (11, 5), (13, 5), (16, 5), (9, 1), (9, 8))
    ),
}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    nc, ng, k = spec["contexts"], spec["gens"], spec["draft_len"]
    batch = nc + ng
    acc_dtype, draft_dtype = (getattr(torch, d) for d in spec["dtypes"])
    accepted = torch.randint(0, 160000, (batch + 3, k + 1), generator=g).to(acc_dtype)
    num_accepted = torch.ones(batch + 3, dtype=torch.int32)
    num_accepted[nc:batch] = torch.randint(1, k + 2, (ng,), generator=g).to(torch.int32)
    drafts = torch.randint(0, 160000, (batch, k), generator=g).to(draft_dtype)
    kv_lens = torch.randint(20, 5000, (batch + 4,), generator=g).to(torch.int32)
    rewind = torch.randint(0, k + 1, (ng,), generator=g).to(torch.int32)
    if spec.get("clamp"):
        kv_lens[nc : nc + ng] = torch.randint(0, 4, (ng,), generator=g).to(torch.int32)
        rewind[:] = 7
    return (
        accepted[:batch].cuda(),
        drafts.cuda(),
        num_accepted.cuda(),
        kv_lens.cuda(),
        rewind.cuda(),
        nc,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    want, kv_want = op.reference(*args)
    with validating(op):
        got = op(*args)
    torch.cuda.synchronize()
    assert got.dtype == want.dtype and torch.equal(got, want)
    assert torch.equal(args[3], kv_want)


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    args = list(_build(CELLS[1].spec, seed=1))
    bad = args[2].clone()
    bad[0] = args[0].shape[1] + 1
    with pytest.raises(ValueError, match="accepted count"):
        op.is_valid(*args[:2], bad, *args[3:])
    with pytest.raises(ValueError, match="kv_lens"):
        op.is_valid(*args[:3], args[3][:2], *args[4:])
