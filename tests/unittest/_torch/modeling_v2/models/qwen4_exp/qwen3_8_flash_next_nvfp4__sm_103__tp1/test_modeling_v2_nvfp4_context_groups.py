# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the target's context pass over an NVFP4 pool, a group of requests at a time.

The context rows read a bf16 scratch of their requests' pages (`_nvfp4_context_plan`), filled
and attended one group of consecutive requests at a time, so it holds the largest group's
pages rather than the step's. Under a budget of a few pages, with one request longer than
the budget, the rows equal, bitwise, one group over the whole step -- for a prefill step's
rows and for a mixed step's one kept row a request. The prefill requests' rows come in
multiples of the row-tile pass's 10-row groups, so a launch per group splits none of them and
every row takes the same pass as in the one launch (a split group's rows go to the per-row
pass, which rounds differently).
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

assert torch.cuda.is_available(), "the NVFP4 context pass requires a CUDA device"

DEV = "cuda"
TPB, HKV, D, QH, RATIO, TOPK = 32, 2, 256, 24, 4, 2048
# (cached, new) tokens a context request: 10, 30, 1, 47 and 4 pages.
SEGMENTS = ((0, 300), (900, 40), (20, 10), (1400, 100), (64, 60))
BUDGET_PAGES = 32  # request 3's 47 pages form a group of their own


def _step(one_row_per_request: bool, seed: int = 5):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    kv_lens = [s + n for s, n in SEGMENTS]
    pages = [math.ceil(k / TPB) for k in kv_lens]
    pool_pages = sum(pages) + 3
    shape = (pool_pages, 2, HKV, TPB)
    data = torch.randint(0, 256, (*shape, D // 2), generator=gen, device=DEV).to(torch.uint8)
    scales = torch.randint(0x08, 0x31, (*shape, D // 16), generator=gen, device=DEV)
    scales = scales.to(torch.uint8)
    slots = 2 * len(SEGMENTS)
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))
    slot_of = slot_of[: len(SEGMENTS)].to(DEV, torch.int32)
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, max(pages)), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), pages):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    new = torch.tensor([n for _, n in SEGMENTS], dtype=torch.int32)
    md = SimpleNamespace(
        num_contexts=len(SEGMENTS),
        kv_lens_runtime=torch.tensor(kv_lens, dtype=torch.int32),
        kv_lens_cuda_runtime=torch.tensor(kv_lens, dtype=torch.int32, device=DEV),
        seq_lens=new,
        seq_lens_cuda=new.to(DEV),
    )
    # The step's rows: a request's new tokens, or the last of them alone.
    kept = [1 if one_row_per_request else n for _, n in SEGMENTS]
    logical = torch.cat(
        [torch.arange(s + n - k, s + n, device=DEV) for (s, n), k in zip(SEGMENTS, kept)]
    )
    rows = logical.numel()
    # Every visible token, as the expansion lays out a selection under its top-k.
    column = torch.arange(TOPK + RATIO - 1, device=DEV)[None, :]
    selected = torch.where(column <= logical[:, None], column, -1).to(torch.int32)
    core = SimpleNamespace(
        heads_kv=HKV,
        head_dim=D,
        q_width=QH * D,
        kv_scale_quant_orig=torch.ones(3, dtype=torch.float32, device=DEV),
        softmax_scale=D**-0.5,
        compress_ratio=RATIO,
    )
    kv = SimpleNamespace(k=data[:, 0], v=data[:, 1], k_scale=scales[:, 0], v_scale=scales[:, 1])
    return dict(
        core=core,
        kv=kv,
        md=md,
        block_table=block_table,
        req_all=slot_of.repeat_interleave(new.to(DEV).long()),
        req_rows=slot_of.repeat_interleave(torch.tensor(kept, device=DEV)),
        q=torch.randn(rows, QH, D, generator=gen, device=DEV).to(torch.bfloat16),
        selected=selected,
        logical=logical.to(torch.int32),
        rows=rows,
    )


def _context_pass(step, one_row_per_request):
    plan = M._nvfp4_context_plan(
        step["core"],
        step["md"],
        step["req_all"],
        step["block_table"],
        TPB,
        one_row_per_request=one_row_per_request,
    )
    out = torch.empty(step["rows"], QH * D, dtype=torch.bfloat16, device=DEV)
    M._sparse_gqa_nvfp4_context(
        step["core"],
        step["kv"],
        plan,
        step["q"],
        step["block_table"],
        step["selected"],
        step["req_rows"],
        step["logical"],
        TPB,
        step["rows"],
        out,
    )
    return plan, out


@pytest.mark.parametrize("one_row_per_request", [False, True], ids=["prefill", "mixed_kept"])
def test_grouped_context_pass_equals_one_group(one_row_per_request, monkeypatch) -> None:
    step = _step(one_row_per_request)
    whole, want = _context_pass(step, one_row_per_request)
    assert len(whole[0]) == 1 and whole[2].shape[0] == 92

    page_bytes = 2 * HKV * TPB * D * 2
    monkeypatch.setattr(M, "_NVFP4_CONTEXT_SCRATCH_BYTES", BUDGET_PAGES * page_bytes)
    (groups, _, k_scratch, _), got = _context_pass(step, one_row_per_request)
    ends = [1, 2, 3, 4, 5] if one_row_per_request else [300, 340, 350, 450, 510]
    # Requests [0], [1, 2], [3] (over the budget, alone), [4].
    assert [(g[0], g[1]) for g in groups] == [
        (0, ends[0]),
        (ends[0], ends[2]),
        (ends[2], ends[3]),
        (ends[3], ends[4]),
    ]
    assert k_scratch.shape[0] == 47
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
