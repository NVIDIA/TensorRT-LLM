# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's attention-path launch levers, all default off.

* **Switches** (CPU): `TEP_INDEX_SPLIT_ENV` is off when unset, refuses other spellings, and warns
  and does nothing at a layout without the tep4 fast forms.
* **Index launch shape** (CPU, recorder): under `TEP_INDEX_SPLIT_ENV` a trunk call of `p` = 4
  rows a request outside the prefetch rows asks the visible split for one request slot a row
  tile; a draft pass (`p` = 1) and an unset switch ask nothing.
* **Bitwise** (GPU): the visible split with one slot a row tile and another program count equals
  the default launch bit for bit at 3 and 4 requests of four verify rows.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

_SM103 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)
needs_sm103 = pytest.mark.skipif(not _SM103, reason="SM103 (GB300) kernels")


# ---- switches (CPU) ----


@pytest.mark.cpu_only
def test_index_split_is_off_unless_set(monkeypatch):
    monkeypatch.delenv(M.TEP_INDEX_SPLIT_ENV, raising=False)
    assert M._tep_index_split(SimpleNamespace(tep_fast=True)) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("value, expected", [("32", 32), ("0", 0)])
def test_index_split_parses(monkeypatch, value, expected):
    monkeypatch.setenv(M.TEP_INDEX_SPLIT_ENV, value)
    assert M._tep_index_split(SimpleNamespace(tep_fast=True)) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["65", "wide"])
def test_index_split_refuses_other_spellings(monkeypatch, value):
    monkeypatch.setenv(M.TEP_INDEX_SPLIT_ENV, value)
    with pytest.raises(ValueError):
        M._tep_index_split(SimpleNamespace(tep_fast=True))


@pytest.mark.cpu_only
def test_index_split_warns_at_another_layout(monkeypatch):
    warned = []
    monkeypatch.setenv(M.TEP_INDEX_SPLIT_ENV, "16")
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._tep_index_split(SimpleNamespace(tep_fast=False)) == 0
    assert len(warned) == 1 and M.TEP_INDEX_SPLIT_ENV in warned[0]


# ---- index launch shape (CPU, recorder) ----


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "split, rows, p, expected",
    [
        (32, 16, 4, (32, 1)),  # 4 requests of four verify rows: one slot a row tile
        (32, 12, 4, (32, 1)),
        (32, 4, 4, None),  # one request of four rows: one row tile keeps the default
        (32, 4, 1, None),  # a draft pass: one row a request keeps the default
        (0, 16, 4, None),  # switch off
    ],
)
def test_index_split_shape(monkeypatch, split, rows, p, expected):
    calls = []

    def rec(**kw):
        calls.append(kw)
        return "scores"

    monkeypatch.setattr(M, "qsa_paged_index_scores", rec)
    core = SimpleNamespace(tep_fast=True, tep_index_split=split, compress_ratio=4)
    q = torch.empty(rows, 4, 128, dtype=torch.bfloat16)
    M._decode_index_scores(core, q, None, None, None, None, 32, p)
    assert calls[0].get("split_shape") == expected


# ---- bitwise (GPU) ----


@needs_sm103
@pytest.mark.parametrize("requests", [3, 4])
@pytest.mark.parametrize("pps", [8, 32, 64])
def test_index_split_is_bitwise(requests, pps):
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (  # noqa: E501
        qsa_paged_index_scores,
    )

    dev, p, tpb, cr, width, pages = "cuda", 4, 32, 4, 1200, 4096
    g = torch.Generator(device=dev).manual_seed(7 + requests)
    rows = requests * p
    cache = torch.randn(pages, tpb, 1, 128, generator=g, device=dev).to(torch.bfloat16)
    bt = torch.stack(
        [torch.randperm(pages, generator=g, device=dev)[:width] for _ in range(requests)]
    )
    bt = bt.to(torch.int32).contiguous()
    pos = torch.tensor(
        [30000 + 1777 * r + j for r in range(requests) for j in range(p)], device=dev
    )
    req = torch.arange(requests, dtype=torch.int32, device=dev).repeat_interleave(p)
    q = torch.randn(rows, 4, 128, generator=g, device=dev).to(torch.bfloat16)
    args = (q, cache, bt, pos, req, tpb, cr, True, False)
    ref = qsa_paged_index_scores(*args, use_pdl=False, wide_split=True)
    out = qsa_paged_index_scores(*args, use_pdl=False, wide_split=True, split_shape=(pps, 1))
    for r, v in enumerate(((pos + 1) // cr).tolist()):
        assert torch.equal(out[r, :v].view(torch.int32), ref[r, :v].view(torch.int32))
