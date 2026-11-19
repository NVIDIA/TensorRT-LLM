# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_selector_draft entry: a sample of its cells bitwise against the torch
path."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_draft import (  # noqa: E501
    k31_dflash2_selector_draft as op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import RANK, VOCAB, selector_codebooks  # noqa: E402
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(
    103, 107, reason="the K3.1 DFlash2 selector entry is certified on sm_103 and sm_107"
)


def _temperatures(kind, n, g):
    if kind == "hot":
        return 0.5 + 2.5 * torch.rand(n, generator=g, device="cuda")
    if kind == "edges":
        threshold = torch.tensor(GREEDY_TEMPERATURE_THRESHOLD, dtype=torch.float32)
        values = torch.stack(
            [
                threshold,
                torch.nextafter(threshold, torch.tensor(1.0)),
                torch.tensor(1e-5),
                torch.tensor(1e-6),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(2e-5),
            ]
        )
        return values.repeat(n)[:n].cuda()
    temps = 1.5 * torch.rand(n, generator=g, device="cuda")
    temps[::3] = 0.0
    return temps


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots, rows = spec["n"], spec["slots"], spec["rows"]
    pred_cb, succ_cb = selector_codebooks(spec["codebook"])
    candidate_ids = torch.randint(0, VOCAB, (n, slots, 32), generator=g, device="cuda")
    unary = 10.0 + 2.0 * torch.randn(n, slots, 32, generator=g, device="cuda")
    unary = unary.sort(dim=-1, descending=True).values
    if spec["scores"] == "ties":
        # Repeated candidates with equal logits score identically: the argmax and the
        # cumulative sums meet exact ties.
        candidate_ids[..., 16:] = candidate_ids[..., :16]
        unary[..., 16:] = unary[..., :16]
        unary[..., 1] = unary[..., 0]
        candidate_ids[..., 1] = candidate_ids[..., 0]
    unary = unary.to(spec["unary"])
    gate = torch.randn(n, slots, RANK, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    seed_t = torch.randint(0, 2**62, (n,), generator=g, device="cuda")
    offset = torch.randint(0, 2**40, (n,), generator=g, device="cuda")
    temps = _temperatures(spec["temps"], n, g)
    store = (
        torch.zeros(rows, slots, 32, dtype=torch.int64, device="cuda"),
        torch.zeros(rows, slots, 32, dtype=torch.float32, device="cuda"),
        torch.full((rows, slots), -1, dtype=torch.int64, device="cuda"),
    )
    slot_ids = torch.randperm(rows, generator=g, device="cuda")[:n]
    return [
        pred_cb,
        succ_cb,
        candidate_ids,
        unary,
        gate,
        anchor,
        seed_t,
        offset,
        temps,
        GREEDY_TEMPERATURE_THRESHOLD,
        *store,
        slot_ids,
    ]


def _with_fresh_store(args):
    out = list(args)
    out[10:13] = [t.clone() for t in args[10:13]]
    return out


CELLS = sample_cells(op.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    got_args = _with_fresh_store(args)
    ref_args = _with_fresh_store(args)
    assert torch.equal(op(*got_args), op.reference(*ref_args))
    for got, ref in zip(got_args[10:13], ref_args[10:13]):
        assert torch.equal(got, ref)
