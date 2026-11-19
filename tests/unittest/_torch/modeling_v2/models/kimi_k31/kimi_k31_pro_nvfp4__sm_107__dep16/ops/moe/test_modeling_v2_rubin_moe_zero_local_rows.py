# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_moe_zero_local_rows, the DEP16 context's zero of the rows the all-to-all received.

On SM107: the DEP16 context's cell bit for bit against the reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.moe.rubin_moe_zero_local_rows import (  # noqa: E501
    rubin_moe_zero_local_rows,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 -- path above

gpu = on_sm(107)
# The median cell of the sample: the DEP16 context, 16 sources of 1024 tokens.
_CELLS = sample_cells(rubin_moe_zero_local_rows.CELLS, size_key="slots")[1:2]


def _received(counts, slots, total, top_k, offset, local, seed, device="cpu"):
    """``(ids, weights)`` of an all-to-all's receive slots, ``[len(counts) * slots, top_k]``.

    Source ``s`` sends ``counts[s]`` tokens, packed at the front of its ``slots``: ``top_k``
    distinct experts each (global ids), at least one of the window ``[offset, offset + local)``.
    Every id of the slots past them is -1, as the dispatch's sanitize leaves them, with weight 0.
    """
    g = torch.Generator().manual_seed(seed)
    n, rows = len(counts), sum(counts)
    logits = torch.randn(rows, total, generator=g)
    ids = torch.topk(logits, top_k, dim=-1).indices
    need = ~((ids >= offset) & (ids < offset + local)).any(1)
    pick = offset + torch.randint(0, local, (rows,), generator=g)
    col = torch.randint(0, top_k, (rows,), generator=g)
    ids[need, col[need]] = pick[need]
    weights = torch.rand(rows, top_k, generator=g)
    weights = weights / weights.sum(-1, keepdim=True)
    slot_ids = torch.full((n * slots, top_k), -1, dtype=torch.int32)
    slot_w = torch.zeros(n * slots, top_k)
    at = 0
    for s, c in enumerate(counts):
        slot_ids[s * slots : s * slots + c] = ids[at : at + c].to(torch.int32)
        slot_w[s * slots : s * slots + c] = weights[at : at + c]
        at += c
    return slot_ids.to(device), slot_w.to(device)


def _received_rows(counts, slots, device="cpu"):
    """The bool mask of the slots a source filled: the front ``counts[s]`` of each block."""
    pos = torch.arange(len(counts) * slots, device=device)
    return pos % slots < torch.tensor(counts, device=device).repeat_interleave(slots)


@gpu
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:40] for c in _CELLS])
def test_zero_local_rows_cells(cell, certifying):  # noqa: F811
    """Bit for bit: the received rows +0, every other row's random bits kept."""
    certifying(rubin_moe_zero_local_rows)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    counts = spec["counts"]
    ids, _ = _received(
        counts,
        spec["slots"],
        spec["num_experts"],
        spec["top_k"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        seed,
        "cuda",
    )
    g = torch.Generator(device="cuda").manual_seed(seed)
    bits = torch.randint(
        -32768, 32767, (ids.shape[0], spec["hidden"]), generator=g, device="cuda"
    ).to(torch.int16)
    out = bits.view(torch.bfloat16).clone()
    args = (ids, spec["local_expert_offset"], spec["local_num_experts"])
    ref = rubin_moe_zero_local_rows.reference(out, *args)
    rubin_moe_zero_local_rows(out, *args)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    live = _received_rows(counts, spec["slots"], "cuda")
    assert not out[live].view(torch.int16).any()
