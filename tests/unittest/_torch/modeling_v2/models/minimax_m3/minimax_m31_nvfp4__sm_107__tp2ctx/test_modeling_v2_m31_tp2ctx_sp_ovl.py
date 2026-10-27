# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP_OVL plumbing: the scale-factor ranges of the two row halves and the NORMQ slot's pending-push handling.

The row-split fused QKV calls are checked byte for byte against the full-step call by the SP_OVL microbench (T 16384 /
8000), and in-model by the kill-switch gate (ids, digests, KV equal).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)


def _ws(rank: int, hidden: int = 6144):
    ws = object.__new__(tp2ctx_ipc.Tp2ctxIpcWorkspace)
    ws.rank, ws.peer, ws.hidden = rank, 1 - rank, hidden
    ws._row_bytes = {"sf_swizzled": (hidden // 32 + 3) // 4 * 4}
    return ws


@pytest.mark.parametrize("rows", [16384, 16000, 15616, 8000, 5632])
def test_own_and_peer_scale_ranges_tile_the_step(rows):
    for rank in (0, 1):
        ws = _ws(rank)
        own, peer = ws.own_sf_slice(rows), ws.peer_sf_slice(rows)
        lo, hi = sorted([own, peer], key=lambda s: s.start)
        per = ws._row_bytes["sf_swizzled"]
        assert lo.start == 0 and lo.stop == hi.start
        assert hi.stop == (rows + 127) // 128 * 128 * per
        assert peer == _ws(1 - rank).own_sf_slice(rows)  # the peer's own range


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the slot records the producing CUDA stream"
)
def test_take_without_a_pending_push_is_unchanged():
    x = torch.zeros(4, 8, device="cuda")
    xq, xsf = torch.ones(4, 8, device="cuda"), torch.ones(4, device="cuda")
    slot = m._NormqSlot()
    slot.publish(x, xq, xsf)
    got = slot.take_ovl(x)
    assert got[0] is xq and got[1] is xsf and got[2] is None and slot.pending is None
    slot.publish(x, xq, xsf)
    got = slot.take(x)
    assert got[0] is xq and got[1] is xsf and slot.pending is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="events need a CUDA device")
def test_take_waits_for_a_pending_push_and_take_ovl_hands_it_over():
    x = torch.zeros(256, 8, device="cuda")
    xq, xsf = torch.ones(256, 8, device="cuda"), torch.ones(64, device="cuda", dtype=torch.uint8)
    event = torch.cuda.Event()
    event.record()
    ovl = m._SpOvl(event, (0, 128), (128, 128), slice(0, 32), slice(32, 64))
    slot = m._NormqSlot()
    slot.publish(x, xq, xsf, ovl)
    assert slot.take_ovl(x)[2] is ovl
    slot.publish(x, xq, xsf, ovl)
    got = slot.take(x)
    assert got[0] is xq and len(got) == 2


def test_switch_and_threshold():
    import inspect

    src = inspect.getsource(m)
    assert 'm31_disabled("TP2CTX_SP_OVL")' in src
    assert (
        m._SP_OVL_MIN_ROWS == 12288
    )  # T 16384 / 16000 / 15616 overlap; 8000 measured flat (MB 845158)
    assert SimpleNamespace  # noqa: B018 - keep the import used if the CUDA test is skipped
