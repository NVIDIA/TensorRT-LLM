# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target's DFlash drafter paths over each layout of the drafter's context.

* ``dflash_window_attention`` and ``dflash_window_attention_wide`` (at each device's split
  geometry) over pages are bitwise their dense calls, the appended block included.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash_window_attention_wide as wide_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    WINDOW,
    dflash_window_attention,
)

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _randn(gen, *shape):
    return torch.randn(*shape, device="cuda", generator=gen).bfloat16()


@pytest.mark.parametrize("append", [False, True], ids=["read", "append"])
@pytest.mark.parametrize("sms", [None, *sorted(wide_ops.WIDE_GEOMETRY_BY_SMS)])
def test_the_window_ops_over_pages_are_their_dense_calls(sms, append, monkeypatch):
    """``sms``: the wide op at that device's split geometry, run on the device at hand; None:
    the single-request op. Eight requests, a TP4 rank's heads, the served window, PDL."""
    op = dflash_window_attention
    if sms is not None:
        monkeypatch.setattr(wide_ops, "_device_sms", lambda index: sms)
        op = wide_ops.dflash_window_attention_wide
    gen = torch.Generator(device="cuda").manual_seed(0)
    # Windows starting inside a page; blocks inside one, straddling two or starting one.
    ctx = [5_000, 13, 1_020, 2_047, 300, 64, 1, 3_333]
    batch, m, hq, hkv, d, page, capacity = len(ctx), 8, 16, 2, 128, 32, 5_024
    slots, per_slot = batch + 1, capacity // page
    k_arena, v_arena = _randn(gen, slots, capacity, hkv, d), _randn(gen, slots, capacity, hkv, d)
    q, new = _randn(gen, batch, m, hq, d) * 2, {}
    if append:
        new = dict(k_new=_randn(gen, batch * m, hkv, d), v_new=_randn(gen, batch * m, hkv, d))
    # The slots' columns on scattered pages of a [pages, K/V, Hkv, page, D] pool.
    table = torch.randperm(slots * per_slot + 3, device="cuda", generator=gen)
    table = table[: slots * per_slot].view(slots, per_slot).int()
    pool = _randn(gen, slots * per_slot + 3, 2, hkv, page, d)
    k_pages, v_pages = pool[:, 0].transpose(1, 2), pool[:, 1].transpose(1, 2)
    cols = torch.arange(capacity, device="cuda")
    rows = table[:, cols // page].long(), cols % page
    k_pages[rows], v_pages[rows] = k_arena, v_arena
    # Request b on slot slots - 1 - b, so its table row is not its batch index.
    seqlens = torch.tensor(ctx, dtype=torch.int32, device="cuda") + m
    call = (seqlens, (slots - 1 - torch.arange(batch, device="cuda")).int(), WINDOW, None, True)
    dense = op(q, k_arena, v_arena, *call, **new)
    with validating(op):
        paged = op(q, k_pages, v_pages, *call, page_table=table, **new)
    assert torch.equal(paged, dense)
    # The blocks went to the requests' pages as to their slots, and nothing else moved.
    assert torch.equal(k_pages[rows], k_arena) and torch.equal(v_pages[rows], v_arena)
