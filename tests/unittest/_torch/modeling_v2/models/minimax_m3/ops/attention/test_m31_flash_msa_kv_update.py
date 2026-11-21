# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m31_msa_kv_update entry of the MiniMax-M3.1-Flash sm_107 targets (the one-launch MSA
KV-length correction through the family native op)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.m31_msa_kv_update import (  # noqa: E501
    TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.m31_msa_kv_update import (  # noqa: E501
    msa_kv_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402, F401 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

#: Spare capacity past the live rows, as the metadata's buffers have: the op must leave it alone.
_SPARE = 3
_STALE = -12345
# Every target certifies the same cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


def _case(spec, seed):
    """The metadata's buffers for one step, with stale values in every output so a skipped write shows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch, qlen = spec["batch"], spec["query_len"]
    total_q = batch * qlen
    staged = torch.randint(qlen, 5000, (batch,), generator=g, dtype=torch.int32)
    accept = spec["accept"]
    if accept == "full":
        kv = staged.clone()
    elif accept == "page_edge":
        staged = torch.tensor([130, 257, 2 * TOKENS_PER_BLOCK + qlen][:batch], dtype=torch.int32)
        kv = staged - torch.tensor([0, 1, qlen - 1][:batch], dtype=torch.int32)
    else:
        kv = staged - torch.randint(0, qlen, (batch,), generator=g, dtype=torch.int32)
        if accept == "padded":
            kv[batch // 2 :] = 0
            staged[batch // 2 :] = 0
    ncols = (5000 + qlen) // TOKENS_PER_BLOCK + 2
    table = (
        torch.randperm(batch * ncols + 7, generator=g)[: batch * ncols]
        .to(torch.int32)
        .reshape(batch, ncols)
    )
    dev = dict(device="cuda", dtype=torch.int32)
    return (
        torch.cat([kv, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.cat([staged, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.full((batch + _SPARE,), qlen, **dev),
        torch.arange(batch, **dev).repeat_interleave(qlen),
        torch.arange(qlen, **dev).repeat(batch),
        table.cuda(),
        TOKENS_PER_BLOCK,
        batch,
        total_q,
        torch.full((batch + _SPARE,), _STALE, **dev),
        torch.full((batch + 1 + _SPARE,), _STALE, **dev) if spec["cu_kv"] else None,
        torch.full((total_q + _SPARE,), _STALE, **dev),
        torch.full((total_q + _SPARE,), _STALE, **dev) if spec["n_valid"] else None,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Integer bookkeeping: exact, and nothing past the live rows is touched."""
    args = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    op(*args)
    batch, total_q = args[7], args[8]
    for got, want, live in zip(args[9:], op.reference(*args), (batch, batch + 1, total_q, total_q)):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.equal(got[:live], want)
            assert bool((got[live:] == _STALE).all())
