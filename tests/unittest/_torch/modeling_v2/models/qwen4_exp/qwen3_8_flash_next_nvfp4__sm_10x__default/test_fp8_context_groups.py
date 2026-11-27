# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefill context pass over an fp8 pool, a group of requests at a time.

Over an fp8 pool the row-tile pass widens its requests' pages into a bf16 scratch, and a launch
whose scratch would pass `_FP8_ROWTILE_MAX_SCRATCH` declines to the per-row kernel. A step whose
context requests together hold more pages than that runs them in groups of consecutive requests
that each fit (`_fp8_context_groups`), so only a request longer than the scratch on its own
takes the per-row kernel.

The grouping is not bitwise the one launch: the row tiles span a group's rows, so a different
grouping tiles them differently. Each group's rows are held to the band `qsa_paged_sparse_gqa`
and the row-tile pass are held to against the fp32 reference, which the per-row kernel the
groups replace meets too.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa import (
    qsa_paged_sparse_gqa,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs on a CUDA device")
DEV = "cuda"

TPB, D, RATIO, TOPK = 32, 256, 4, 2048
QH, HKV = 6, 1
# (cached, new) tokens a context request: 10, 35 and 47 pages.
SEGMENTS = ((0, 300), (900, 200), (1400, 100))
# Requests 0 and 1 (45 pages) fit one scratch, request 2 (47 pages) one of its own.
BUDGET_PAGES = 46


def _page_bytes(heads_kv: int = HKV) -> int:
    return 2 * heads_kv * TPB * D * 2  # K and V, bf16


def _metadata(segments, device="cpu"):
    kv_lens = [s + n for s, n in segments]
    new = [n for _, n in segments]
    return SimpleNamespace(
        num_contexts=len(segments),
        kv_lens_runtime=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(new, dtype=torch.int32, device=device),
    )


def _core():
    return SimpleNamespace(
        heads_kv=HKV,
        head_dim=D,
        q_width=QH * D,
        softmax_scale=D**-0.5,
        compress_ratio=RATIO,
    )


@pytest.mark.cpu_only
def test_a_step_within_one_scratch_is_one_launch(monkeypatch) -> None:
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", 92 * _page_bytes())
    md = _metadata(SEGMENTS)
    assert M._fp8_context_pages(md, TPB) == 92
    assert M._fp8_context_groups(_core(), md, TPB, 92) is None


@pytest.mark.cpu_only
def test_a_step_past_one_scratch_runs_in_groups(monkeypatch) -> None:
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", BUDGET_PAGES * _page_bytes())
    md = _metadata(SEGMENTS)
    groups = M._fp8_context_groups(_core(), md, TPB, M._fp8_context_pages(md, TPB))
    # (first row, end row, pages): requests [0, 1], then [2].
    assert groups == [(0, 500, 45), (500, 600, 47)]


@pytest.mark.cpu_only
def test_a_request_longer_than_the_scratch_is_a_group_of_its_own(monkeypatch) -> None:
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", 20 * _page_bytes())
    md = _metadata(SEGMENTS)
    groups = M._fp8_context_groups(_core(), md, TPB, 92)
    assert groups == [(0, 300, 10), (300, 500, 35), (500, 600, 47)]


@pytest.mark.cpu_only
def test_one_request_is_never_grouped(monkeypatch) -> None:
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", _page_bytes())
    md = _metadata(SEGMENTS[2:])
    assert M._fp8_context_groups(_core(), md, TPB, 47) is None


def _fp8_step(seed: int = 3):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    kv_lens = [s + n for s, n in SEGMENTS]
    pages = [math.ceil(k / TPB) for k in kv_lens]
    pool_pages = sum(pages) + 5
    shape = (pool_pages, HKV, TPB, D)
    k = (torch.randn(shape, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(shape, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    slots = 2 * len(SEGMENTS)
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))
    slot_of = slot_of[: len(SEGMENTS)].to(DEV, torch.int32)
    page_ids = torch.randperm(pool_pages, generator=gen, device=DEV).to(torch.int32)
    block_table = torch.full((slots, max(pages)), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), pages):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    new = torch.tensor([n for _, n in SEGMENTS], dtype=torch.long, device=DEV)
    logical = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in SEGMENTS])
    # Every visible token, as the expansion lays out a selection under its top-k: each
    # request's history fits the top-k, so a row attends its whole causal prefix.
    column = torch.arange(TOPK + RATIO - 1, device=DEV)[None, :]
    selected = torch.where(column <= logical[:, None], column, -1).to(torch.int32)
    return dict(
        kv=SimpleNamespace(k=k, v=v, k_scale=None, v_scale=None),
        block_table=block_table,
        req_idx=slot_of.repeat_interleave(new),
        q=torch.randn(logical.numel(), QH, D, generator=gen, device=DEV).to(torch.bfloat16),
        selected=selected,
        logical=logical.to(torch.int32),
    )


def _call(step, lo, hi, pages):
    return M._sparse_gqa(
        _core(),
        step["kv"],
        step["q"][lo:hi],
        step["block_table"],
        step["selected"][lo:hi],
        step["req_idx"][lo:hi],
        TPB,
        step["logical"][lo:hi],
        pages,
    ).reshape(hi - lo, QH, D)


@_cuda
def test_grouped_fp8_context_pass_is_within_the_band(monkeypatch) -> None:
    # Every group takes the row-tile pass; the scratch holds 46 pages.
    monkeypatch.setattr(M, "_ROWTILE_MIN_ROWS", 64)
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", BUDGET_PAGES * _page_bytes())
    step = _fp8_step()
    md = _metadata(SEGMENTS)
    num_rows = step["q"].shape[0]
    pages = M._fp8_context_pages(md, TPB)
    groups = M._fp8_context_groups(_core(), md, TPB, pages)
    assert groups == [(0, 500, 45), (500, 600, 47)]
    # One launch over the step's 92 pages declines to the per-row kernel: what the step ran
    # before it was grouped.
    assert (
        M._sparse_gqa_rowtile(
            _core(),
            step["kv"],
            step["q"],
            step["block_table"],
            step["selected"],
            step["req_idx"],
            TPB,
            step["logical"],
            pages,
        )
        is None
    )
    # The second group (47 pages) is past the scratch on its own too, so only the first takes
    # the row-tile pass; both answer within the band.
    grouped = torch.cat([_call(step, lo, hi, n) for lo, hi, n in groups])
    per_row = _call(step, 0, num_rows, pages)
    want = qsa_paged_sparse_gqa.reference(
        q=step["q"],
        k_cache=step["kv"].k,
        v_cache=step["kv"].v,
        block_table=step["block_table"],
        selected_tokens=step["selected"],
        request_indices=step["req_idx"],
        tokens_per_block=TPB,
        softmax_scale=D**-0.5,
        query_positions=step["logical"],
        compress_ratio=RATIO,
    )
    # `qsa_paged_sparse_gqa`'s band, which the row-tile pass shares: 4 ulp of each (row,
    # head)'s magnitude, 2 ulp relative RMS, against the fp32 reference rounded once.
    assert_within_ulp(grouped, want, element_ulp=4.0, rms_ulp=2.0)
    assert_within_ulp(per_row, want, element_ulp=4.0, rms_ulp=2.0)
    # The rows past the first group are the per-row kernel's, bitwise.
    assert torch.equal(grouped[500:], _call(step, 500, num_rows, None))


@_cuda
def test_each_group_takes_the_row_tile_pass(monkeypatch) -> None:
    """With room for 47 pages every group fits its scratch: each launch takes the row-tile pass
    (a non-None `_sparse_gqa_rowtile`), where the one launch over 92 pages would not."""
    monkeypatch.setattr(M, "_ROWTILE_MIN_ROWS", 64)
    monkeypatch.setattr(M, "_FP8_ROWTILE_MAX_SCRATCH", 47 * _page_bytes())
    step = _fp8_step()
    md = _metadata(SEGMENTS)
    groups = M._fp8_context_groups(_core(), md, TPB, M._fp8_context_pages(md, TPB))
    assert groups == [(0, 500, 45), (500, 600, 47)]
    for lo, hi, n in groups:
        out = M._sparse_gqa_rowtile(
            _core(),
            step["kv"],
            step["q"][lo:hi],
            step["block_table"],
            step["selected"][lo:hi],
            step["req_idx"][lo:hi],
            TPB,
            step["logical"][lo:hi],
            n,
        )
        assert out is not None, (lo, hi, n)
