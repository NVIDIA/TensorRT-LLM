# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2AGG_SP on CPU stand-ins: the band, the owned-row split and its 128-row scale-factor tiles (the tp2ctx_ipc
workspace's split_row / own_sf_slice), the "logits" gather kind, the default-on dispatch
(TRTLLM_M31_DISABLE_TP2AGG_SP=1 or TRTLLM_M31_DISABLE_AGG_SP=1 turns it off), the residual unshard, the router /
finalize reads of the gathered products and the SP_OVL hand-off. The collectives and the bitwise boundary are gated
on GPU (det + TRTLLM_M31_TP2CTX_SP_CHECK)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling as m,
)

H = 6144


def test_default_on_with_both_kill_switches() -> None:
    assert m.TP2AGG_SP == (not m.m31_disabled("TP2AGG_SP") and not m.m31_disabled("AGG_SP"))


@pytest.mark.parametrize("on", [False, True])
def test_band(monkeypatch, on) -> None:
    monkeypatch.setattr(m, "TP2AGG_SP", on)
    assert not m._sp_supports(m.SP_MIN_ROWS - 1)
    assert m._sp_supports(m.SP_MIN_ROWS) == on
    assert m._sp_supports(m.SP_MAX_ROWS) == on
    assert not m._sp_supports(m.SP_MAX_ROWS + 1)


@pytest.mark.parametrize("rows", [4097, 5461, 5462, 8192, 12288, 12697, 16383, 16384])
def test_owned_rows_tile_the_step(rows) -> None:
    (o0, c0), (o1, c1) = m._sp_own_rows(rows, 0), m._sp_own_rows(rows, 1)
    assert o0 == 0 and o1 == c0 and o1 + c1 == rows
    assert c0 % m.SP_SF_TILE_ROWS == 0 and abs(c0 - rows / 2) <= m.SP_SF_TILE_ROWS / 2
    if rows % 256 == 0:
        assert c0 == c1 == rows // 2


@pytest.mark.parametrize("rows", [5462, 12697, 16384])
def test_sf_bytes_are_whole_tiles_and_cover_the_padded_step(rows) -> None:
    per = (H // 32 + 3) // 4 * 4
    a, b = m._sp_own_sf_bytes(rows, 0, H), m._sp_own_sf_bytes(rows, 1, H)
    rows_pad = (rows + 127) // 128 * 128
    assert a[0] == 0 and a[1] == b[0] and b[1] == rows_pad * per
    assert a[1] % (128 * per) == 0


@pytest.mark.parametrize("rows", [5462, 12697])
def test_logits_kind_is_512_bytes_per_owned_row(rows) -> None:
    a, b = m._sp_logits_bytes(rows, 0), m._sp_logits_bytes(rows, 1)
    assert m.SP_LOGITS_ROW_BYTES == 128 * torch.tensor([], dtype=torch.float32).element_size()
    assert a[0] == 0 and a[1] == b[0] and b[1] == rows * 512


class _FakeWs:
    """CPU stand-in for tp2ctx_ipc.Tp2ctxIpcWorkspace: the same row / scale ownership, local views only."""

    def __init__(self, rank, max_tokens=16384, hidden=8):
        self.rank, self.max_tokens, self.hidden = rank, max_tokens, hidden
        self.views = {}
        self.pushed = []

    def supports(self, rows, *, nccl_only=True):
        return 256 <= rows <= self.max_tokens

    def own_rows(self, rows):
        return m._sp_own_rows(rows, self.rank)

    def own_sf_slice(self, rows):
        return slice(*m._sp_own_sf_bytes(rows, self.rank, self.hidden * 32))

    def gather_view(self, kind, rows):
        shapes = {
            "logits": ((self.max_tokens, 128), torch.float32),
            "bf16": ((self.max_tokens, self.hidden), torch.bfloat16),
        }
        shape, dtype = shapes.get(kind, ((self.max_tokens, self.hidden), torch.uint8))
        if kind not in self.views:
            self.views[kind] = torch.zeros(shape, dtype=dtype)
        return self.views[kind][:rows]

    def push_owned(self, kinds, rows):
        self.pushed.append((tuple(kinds), rows))


@pytest.fixture
def sp_on(monkeypatch):
    monkeypatch.setattr(m, "TP2AGG_SP", True)
    monkeypatch.setattr(m, "is_torch_compiling", lambda: False)
    monkeypatch.setattr(m, "is_in_breakable_cuda_graph", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    state = {}
    monkeypatch.setattr(m, "_SP_STATE", state)
    return state


def test_no_workspace_keeps_todays_boundary(sp_on) -> None:
    assert m._sp_ws(torch.zeros(8192, 4)) is None


@pytest.mark.parametrize("rank", [0, 1])
def test_ws_dispatch_is_the_band(sp_on, rank) -> None:
    sp_on["ws"] = _FakeWs(rank)
    assert m._sp_ws(torch.zeros(8192, 4)) is sp_on["ws"]
    assert m._sp_ws(torch.zeros(m.SP_MIN_ROWS - 1, 4)) is None
    assert m._sp_ws(torch.zeros(8192)) is None


def test_ws_dispatch_off_when_capturing(sp_on, monkeypatch) -> None:
    sp_on["ws"] = _FakeWs(0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert m._sp_ws(torch.zeros(8192, 4)) is None


@pytest.mark.parametrize("rank", [0, 1])
def test_logits_put_writes_only_the_owned_rows(sp_on, rank) -> None:
    ws, rows = _FakeWs(rank), 12697
    first, count = ws.own_rows(rows)
    owned = torch.randn(count, 128)
    view = m._sp_put(ws, "logits", rows, owned)
    assert view.dtype == torch.float32 and view.shape == (rows, 128)
    assert torch.equal(view[first : first + count], owned)
    others = torch.cat([view[:first], view[first + count :]])
    assert int(others.count_nonzero()) == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_unshard_gathers_an_owned_residual(sp_on, rank) -> None:
    ws = sp_on["ws"] = _FakeWs(rank)
    rows = 8192
    first, count = ws.own_rows(rows)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    owned = torch.randn(count, 8).to(torch.bfloat16)
    full = m._sp_unshard(owned, x)
    assert full.shape == (rows, 8) and torch.equal(full[first : first + count], owned)
    assert ws.pushed == [(("bf16",), rows)]
    # a full residual (or one outside an SP step) is left alone
    assert m._sp_unshard(full, x) is full
    small = owned[:7]
    assert m._sp_unshard(small, x) is small and ws.pushed == [(("bf16",), rows)]


def test_router_and_finalize_read_the_gathered_products(sp_on) -> None:
    keyed = torch.zeros(8192, 8, dtype=torch.bfloat16)
    logits = torch.randn(8192, 128)
    m._SP_LOGITS.publish(keyed, logits, None)
    assert m._router_logits(None, 3, keyed) is logits
    assert m._SP_LOGITS.pending is None
    sp_on["keyed"] = keyed.data_ptr()
    assert m._sp_keyed(keyed) and not m._sp_keyed(keyed.clone())


def test_ovl_take_matches_the_keyed_view(sp_on) -> None:
    class _Ev:
        pass

    keyed = torch.zeros(12288, 8, dtype=torch.bfloat16)
    ovl = m._SpOvl(_Ev(), (0, 6144), (6144, 6144), slice(0, 1), slice(1, 2))
    sp_on["ovl"] = (keyed.data_ptr(), ovl)
    assert m._sp_ovl_take(keyed) is ovl
    assert sp_on["ovl"] is None and m._sp_ovl_take(keyed) is None


@pytest.mark.parametrize(
    "fused,decode,tuned,allowed",
    [
        (True, False, False, True),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_placeholder_requires_prequant_consumers(monkeypatch, fused, decode, tuned, allowed):
    core = SimpleNamespace(fused_projection=fused, mxfp8=None)
    mode = SimpleNamespace(decode_graph=decode, tuned_quantize_gemm=tuned)
    workspace = object()
    monkeypatch.setattr(m, "_mx_mode", lambda _: mode)
    monkeypatch.setattr(m, "_sp_ws", lambda _: workspace)
    assert m._sp_consumer_ws(core, None) is (workspace if allowed else None)


@pytest.mark.parametrize("rows", [4097, 5461, 5462, 8192, 12288, 16384, 16385])
def test_agg_band_override_preserves_workspace_capacity(monkeypatch, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc

    monkeypatch.setattr(tp2ctx_ipc, "_nccl_fallback_bytes", lambda: 64 * 1024 * 1024)
    workspace = SimpleNamespace(max_tokens=16384, hidden=6144)
    supports = tp2ctx_ipc.Tp2ctxIpcWorkspace.supports
    assert supports(workspace, rows, nccl_only=False) == (rows <= 16384)
    assert supports(workspace, rows) == (5462 <= rows <= 16384)
