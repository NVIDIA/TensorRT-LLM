# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the key-sharded decode Top-K entries rubin_dsa_key_shard_pack / _merge.

GPU (sm_107): every rank of a group on one GPU over plain local buffers (``key_shard_workspace``),
packed then merged in group order, as the multi-rank run does over MNNVL: each rank's rows, counts
and mask against the merge reference, bit for bit; the exact fallback rows are counted in the
state; a second step reuses the workspace (the epoch advances).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.kernels.attention.rubin_dsa_key_shard import (  # noqa: E501
    KEY_SHARD_LOCAL_K,
    key_shard_bytes_needed,
    key_shard_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_merge as merge_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.attention.rubin_dsa_key_shard import (  # noqa: E501
    rubin_dsa_key_shard_pack as pack_op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

on_sm107 = on_sm(107)

#: A boundary and a typical cell of ``merge_op.CELLS``, and a typical one of ``pack_op.CELLS``.
_CELLS = (
    sample_cells(merge_op.CELLS, size_key="rows")[:2]
    + sample_cells(pack_op.CELLS, size_key="rows")[1:2]
)


def _group(spec, gen):
    """Every rank's local logits ``[rows, width]`` and row lengths; distinct scores unless ``ties``."""
    world, rows = spec["world"], spec["rows"]
    lo, hi = spec["row_lens"]
    width = -(-hi // 256) * 256
    lens = [
        torch.randint(lo, hi + 1, (rows,), generator=gen, dtype=torch.int32) for _ in range(world)
    ]
    skewed = range(int(spec.get("skew", 0)))
    for w in skewed:
        # Rank 0 (ranks 0 .. skew - 1) holds the best keys of every row: more than its 512 make
        # the global Top-K.
        lens[w] = torch.full((rows,), hi, dtype=torch.int32)
    total = world * rows * width
    if spec.get("ties"):
        vals = torch.randint(-4, 5, (total,), generator=gen).float()
    else:
        vals = (torch.randperm(total, generator=gen).float() + 1) / (total + 1)
    logits = vals.view(world, rows, width).clone()
    for w in skewed:
        logits[w] += 2.0
    return logits, lens, width


def _local_topk(logits, lens, k):
    rows = logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r])
        order = torch.sort(-logits[r, :n].double(), stable=True).indices[:k]
        out[r, : order.numel()] = order.to(torch.int32)
    return out


# --- GPU: every rank on one GPU --------------------------------------------------------------


def _run_group(spec, logits, lens, width, workspaces, bounds):
    rows, top_k = spec["rows"], spec["top_k"]
    for r, ws in enumerate(workspaces):
        ws.local_logits[:rows, :width].copy_(logits[r].cuda())
        local = torch.full((rows, ws.local_k), -1, dtype=torch.int32)
        local.copy_(_local_topk(logits[r], lens[r], ws.local_k))
        ws.local_topk[:rows].copy_(local.cuda())
        pack_op(ws.local_topk[:rows], lens[r].cuda(), ws)
    results = []
    for r, ws in enumerate(workspaces):
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device="cuda")
        counts = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
        mask = torch.zeros(rows, dtype=torch.bool, device="cuda") if bounds is not None else None
        merge_op(
            out,
            counts,
            None if bounds is None else bounds.cuda(),
            mask,
            ws,
            failing_only=spec.get("failing_only", False),
        )
        results.append((out.cpu(), counts.cpu(), None if mask is None else mask.cpu()))
    torch.cuda.synchronize()
    return results


def _group_workspaces(spec, width):
    world, rows, top_k = spec["world"], spec["rows"], spec["top_k"]
    need = key_shard_bytes_needed(rows, top_k, world, KEY_SHARD_LOCAL_K, width)
    base = torch.zeros(world, -(-need // 256) * 256, dtype=torch.uint8, device="cuda")
    bufs = [base[r] for r in range(world)]
    return [
        key_shard_workspace(bufs, r, rows, top_k, width, max_spin=1 << 20) for r in range(world)
    ]


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=lambda c: c.why[:48])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(merge_op)
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["rows"] + spec["world"])
    logits, lens, width = _group(spec, gen)
    world, rows, top_k = spec["world"], spec["rows"], spec["top_k"]
    workspaces = _group_workspaces(spec, width)
    bounds = None
    if spec.get("bounds", True):
        bounds = torch.randint(0, 3, (rows,), generator=gen, dtype=torch.int32)
    for step in range(2):
        got = _run_group(spec, logits, lens, width, workspaces, bounds)
        for r in range(world):
            ref_rows, ref_counts, ref_mask = merge_op.reference(
                list(logits), lens, r, top_k, bounds
            )
            out, counts, mask = got[r]
            assert torch.equal(out, ref_rows), f"rank {r} step {step}: rows"
            assert torch.equal(counts, ref_counts), f"rank {r} step {step}: counts"
            if bounds is not None:
                assert torch.equal(mask, ref_mask), f"rank {r} step {step}: mask"
        logits = logits.flip(-1)  # a new step's scores through the same workspace
    # The workspace state words: [merges, ticket, fallback rows, spin timeouts].
    stats = workspaces[0].state.tolist()
    assert stats[3] == 0, "a merge waited out its spin bound"
    if spec.get("skew"):
        assert stats[2] > 0, "the skewed rows take the exact fallback"
