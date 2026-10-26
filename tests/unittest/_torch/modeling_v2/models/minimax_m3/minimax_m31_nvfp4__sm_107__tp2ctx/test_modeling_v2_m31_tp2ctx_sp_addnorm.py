# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP_ADDNORM plumbing: the owned slot-row addresses and the engagement / switch lines.

The fused kernel itself is checked byte for byte against GEMM-AR's peer add + NORMQ out= ops on two GPUs by its
microbench oracle, and in-model by the kill-switch gate (ids, digests, KV equal).
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)


def _ws(rank: int, hidden: int = 6144):
    ws = object.__new__(tp2ctx_ipc.Tp2ctxIpcWorkspace)
    ws.rank, ws.peer, ws.hidden = rank, 1 - rank, hidden
    ws._mem = {"slot": SimpleNamespace(local_ptr=1 << 40, peer_ptrs={0: 3 << 40, 1: 5 << 40})}
    return ws


@pytest.mark.parametrize(
    "rows, rank, first", [(16384, 0, 0), (16384, 1, 8192), (8000, 1, 3968), (8000, 0, 0)]
)
def test_owned_slot_ptrs_address_the_owned_rows_of_both_slots(rows, rank, first):
    ws = _ws(rank)
    assert ws.own_rows(rows)[0] == first
    local, peer = ws.owned_slot_ptrs(rows)
    off = first * ws.hidden * 2
    assert local == ws._mem["slot"].local_ptr + off
    assert peer == ws._mem["slot"].peer_ptrs[1 - rank] + off
    assert off % 16 == 0  # the fused op's 16-byte row loads


def test_engagement_line_once_per_kind_and_rows(monkeypatch):
    lines = []
    monkeypatch.setattr(m.logger, "info", lambda msg: lines.append(msg))
    monkeypatch.setattr(m, "_SP_STATE", {})
    for _ in range(3):
        m._addnorm_note(16384, True)
        m._addnorm_note(16384, False)
    assert lines == [
        "[M3.1 tp2ctx] TP2CTX_SP_ADDNORM taken: pre-MoE rows 16384 (peer add fused into NORMQ)",
        "[M3.1 tp2ctx] TP2CTX_SP_ADDNORM taken: boundary rows 16384 (peer add fused into NORMQ)",
    ]


def test_switch_is_the_documented_item_id():
    import inspect

    src = inspect.getsource(m)
    assert 'm31_disabled("TP2CTX_SP_ADDNORM")' in src
    assert "TRTLLM_M31_DISABLE_TP2CTX_SP_ADDNORM=1" in src
