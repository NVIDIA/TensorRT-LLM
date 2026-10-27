# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target's DFlash drafter paths over each layout of the drafter's context.

* ``dflash_window_attention`` and ``dflash_window_attention_wide`` (at each device's split
  geometry) over pages are bitwise their dense calls, the appended block included.
* ``context_kv_fork``'s commit empties the dummy slot, as the worker's own commit does.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash_window_attention_wide as wide_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    WINDOW,
    dflash_window_attention,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

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


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "sm107_windowed_commit"])
def test_the_fork_commit_empties_the_dummy_slot(windowed):
    """The dummy slot, which CUDA-graph padding rows share, is empty after the commit, the DFlash
    worker's rule: over the dense arena, and after the sm107 windowed commit's own launch."""
    worker = object.__new__(DFlashWorker)
    worker._ctx_k_buf = torch.zeros(4, 5, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    worker._ctx_v_buf = torch.zeros_like(worker._ctx_k_buf)
    worker._ctx_len = torch.tensor([5, 0, 0, 6], device="cuda")
    # A windowed arena without drafter_window_compact: each slot starts at a base.
    worker._ctx_base = torch.tensor([2, 0, 0, 3], device="cuda") if windowed else None
    worker._ctx_keep, worker._max_ctx, worker._dummy_slot = (48 if windowed else 0), 64, 3
    worker._ctx_paged, worker._ctx_block_tables = False, None
    fork = context_kv_fork.ContextKvFork(worker, None, context_kv_fork.POINT)
    main = torch.cuda.current_stream()
    launched = context_kv_fork.Launched(main, None, 2, 7, context_kv_fork.POINT, None)
    # A request on slot 0 accepts 3 rows, a padding row on the dummy slot 8.
    accepted = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    num_ctx = fork.commit(launched, torch.tensor([0, 3], device="cuda"), accepted)
    fork.join(launched)
    assert worker._ctx_len.tolist() == [8, 0, 0, 0]
    # The lengths the drafter attends: physical over the windowed arena.
    assert num_ctx.tolist() == ([6, 0] if windowed else [8, 0])
