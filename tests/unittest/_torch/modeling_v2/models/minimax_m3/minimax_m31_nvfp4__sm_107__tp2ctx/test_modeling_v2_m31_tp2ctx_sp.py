# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP (sequence-parallel prefill boundaries): band, row ownership, the workspace probe and the boundary plumbing.

The boundaries run on GEMM-AR's IPC workspace (catalog/comm/tp2ctx_ipc.py, its own UT + 2-rank gate); these tests cover
the fork's side on CPU stand-ins. In-model correctness is gate-a (full forward bitwise vs the kill switch).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)


@pytest.mark.parametrize(
    "rows, ok",
    [
        (5461, False),
        (5632, True),
        (8192, True),
        (8000, True),  # A8K: the 128-row split (workspace API v3); supports(T) has the final say
        (16384, True),
        (16640, False),
        (2048, False),
    ],
)
def test_band(monkeypatch, rows, ok):
    monkeypatch.setattr(m, "_SP", True)
    assert m._sp_supports(rows) is ok
    monkeypatch.setattr(m, "_SP", False)
    assert m._sp_supports(rows) is False  # kill switch


@pytest.mark.parametrize(
    "rows, split", [(5632, 2816), (8192, 4096), (16384, 8192), (8000, 3968), (15616, 7808)]
)
def test_owned_rows_split_on_a_scale_factor_tile(rows, split):
    (o0, n0), (o1, n1) = m._sp_own_rows(rows, 0), m._sp_own_rows(rows, 1)
    assert (o0, o0 + n0, o1, o1 + n1) == (0, split, split, rows)
    assert (
        n0 % 128 == 0 and o1 % 128 == 0
    )  # MXFP8 swizzled scale factors: whole 128-row tiles per half


def test_workspace_probe_logs_once_and_falls_back(monkeypatch):
    lines = []
    monkeypatch.setattr(m.logger, "info", lambda msg: lines.append(msg))
    monkeypatch.setattr(m, "_SP_STATE", {})
    monkeypatch.setattr(m, "_SP", True)
    mapping = SimpleNamespace(tp_size=4)
    assert m._sp_workspace(mapping) is None
    assert m._sp_workspace(mapping) is None
    assert lines == [
        "[M3.1 tp2ctx] TP2CTX_SP sequence-parallel boundaries: not taken: tp_size 4 != 2"
    ]


# TP2CTX_SP boundaries on GEMM-AR's IPC workspace (in-model correctness: gate-a, full forward vs the kill switch).

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


class _FakeWs:
    """CPU stand-in for Tp2ctxIpcWorkspace's row bookkeeping and views (rank 1 of 2, hidden 64)."""

    hidden, sf_cols = 64, 4

    def __init__(self, rows):
        self.rank, self.views = 1, {}
        shapes = {
            "mxfp8": ((rows, self.hidden), torch.uint8),
            "mxfp8_sf": ((rows * self.sf_cols,), torch.uint8),
            "nvfp4": ((rows, self.hidden // 2), torch.uint8),
            "nvfp4_sf": ((rows, self.hidden // 16), torch.uint8),
            "topk_ids": ((rows, 4), torch.int32),
            "topk_w": ((rows, 4), torch.float32),
            "bf16": ((rows, self.hidden), torch.bfloat16),
        }
        for kind, (shape, dtype) in shapes.items():
            self.views[kind] = torch.zeros(shape, dtype=dtype)
        self.slot = torch.zeros((rows, self.hidden), dtype=torch.bfloat16)
        self.pushed = []

    def own_rows(self, rows):
        return self.rank * (rows // 2), rows // 2

    def own_sf_slice(self, rows):
        rows_pad = (rows + 127) // 128 * 128
        first = self.rank * (rows // 2)
        return slice(first * self.sf_cols, rows_pad * self.sf_cols)

    def gather_view(self, kind, rows):
        return self.views[kind]

    def input_slot(self, rows):
        return self.slot

    def push_owned(self, kinds, rows):
        self.pushed.append(tuple(kinds))


def test_put_places_owned_rows_and_scale_factor_bytes():
    rows = 256
    ws = _FakeWs(rows)
    q = torch.full((128, 64), 7, dtype=torch.uint8)
    m._sp_put(ws, "mxfp8", rows, q)
    assert int(ws.views["mxfp8"][:128].sum()) == 0 and bool((ws.views["mxfp8"][128:] == 7).all())
    sf = torch.full((128 * 4,), 3, dtype=torch.uint8)
    m._sp_put(ws, "mxfp8_sf", rows, sf)
    assert int(ws.views["mxfp8_sf"][: 128 * 4].sum()) == 0
    assert bool((ws.views["mxfp8_sf"][128 * 4 :] == 3).all())
    ids = torch.arange(128 * 4, dtype=torch.int32).view(128, 4)
    m._sp_put(ws, "topk_ids", rows, ids)
    assert torch.equal(ws.views["topk_ids"][128:], ids)
    with pytest.raises(RuntimeError, match="scale factors"):
        m._sp_put(ws, "mxfp8_sf", rows, torch.zeros(100, dtype=torch.uint8))


def test_unshard_passes_full_residuals_through(monkeypatch):
    monkeypatch.setattr(m, "_SP_STATE", {})
    r = torch.zeros(8, 64, dtype=torch.bfloat16)
    assert m._sp_unshard(r, torch.zeros(16, 64)) is r  # no workspace
    monkeypatch.setattr(m, "_SP_STATE", {"ws": _FakeWs(16)})
    assert m._sp_unshard(r, torch.zeros(8, 64)) is r  # already full
    assert m._sp_unshard(r, torch.zeros(20, 64)) is r  # not this rank's owned count (10)


def test_unshard_gathers_an_owned_residual(monkeypatch):
    ws = _FakeWs(16)
    monkeypatch.setattr(m, "_SP_STATE", {"ws": ws})
    monkeypatch.setattr(m.logger, "info", lambda msg: None)
    owned = torch.ones(8, 64, dtype=torch.bfloat16)
    full = m._sp_unshard(owned, torch.zeros(16, 64, dtype=torch.bfloat16))
    assert full.shape == (16, 64) and bool((full[8:] == 1).all())
    assert ws.pushed == [("bf16",)]
    assert full.data_ptr() != ws.views["bf16"].data_ptr()  # cloned: the next gather reuses the view


def test_sp_ws_needs_the_workspace_and_the_band(monkeypatch):
    monkeypatch.setattr(m, "_SP", True)
    monkeypatch.setattr(m, "_SP_STATE", {})
    assert m._sp_ws(torch.empty(8192, 8)) is None
    ws = SimpleNamespace(supports=lambda t: True)
    monkeypatch.setattr(m, "_SP_STATE", {"ws": ws})
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: False)
    assert m._sp_ws(torch.empty(8192, 8)) is ws
    assert m._sp_ws(torch.empty(2048, 8)) is None  # below the band
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: True)
    assert m._sp_ws(torch.empty(8192, 8)) is None  # eager only


def _moe_with(routing, backend):
    experts = SimpleNamespace(routing_method=routing, backend=backend)
    return SimpleNamespace(experts=experts, num_experts=8)


def test_routed_call_uses_the_gathered_routing_and_the_slot():
    rows = 16
    ws = _FakeWs(rows)
    base = SimpleNamespace(
        top_k=4, experts_per_token=4, apply=lambda *a: pytest.fail("base routing ran")
    )
    backend = SimpleNamespace(moe_output_buffer=None)
    moe = _moe_with(base, backend)
    ids = torch.zeros(rows, 4, dtype=torch.int32)
    w = torch.ones(rows, 4)
    seen = {}

    def run(logits):
        seen["buffer"] = backend.moe_output_buffer
        seen["ids"], seen["w"] = moe.experts.routing_method.apply(logits)
        seen["k"] = moe.experts.routing_method.experts_per_token  # delegated
        return "out"

    params = SimpleNamespace(enable_allreduce=False)
    assert m._sp_routed_call(moe, (None, ids, w, ws, rows), params, run) == "out"
    assert seen["buffer"] is ws.slot and seen["k"] == 4
    assert torch.equal(seen["ids"], ids) and torch.equal(seen["w"], w)
    assert moe.experts.routing_method is base and backend.moe_output_buffer is None
    with pytest.raises(pytest.fail.Exception):  # the instance's own apply is back
        base.apply(None)


class _RoutingModule(torch.nn.Module):
    """A routing method registered as a submodule, as the experts hold it (838959: assigning a non-module failed)."""

    top_k = 4

    def apply(self, router_logits, input_ids=None):  # noqa: A003 - the routing-method API
        raise AssertionError("the module's own routing ran")


def test_routed_call_shadows_a_module_routing_method_and_restores_it():
    rows = 16
    ws = _FakeWs(rows)
    experts = torch.nn.Module()
    experts.routing_method = _RoutingModule()
    experts.backend = SimpleNamespace(moe_output_buffer=None)
    moe = SimpleNamespace(experts=experts, num_experts=8)
    ids = torch.arange(rows * 4, dtype=torch.int32).view(rows, 4)
    w = torch.ones(rows, 4)

    def run(logits):
        return experts.routing_method.apply(logits)

    got_ids, got_w = m._sp_routed_call(moe, (None, ids, w, ws, rows), None, run)
    assert torch.equal(got_ids, ids) and torch.equal(got_w, w)
    assert "apply" not in vars(experts.routing_method)  # the class method is back
    with pytest.raises(AssertionError, match="own routing ran"):
        experts.routing_method.apply(None)


def test_routed_call_raises_when_routing_is_bypassed():
    rows = 16
    ws = _FakeWs(rows)
    moe = _moe_with(SimpleNamespace(), SimpleNamespace(moe_output_buffer=None))
    ids = torch.zeros(rows, 4, dtype=torch.int32)
    w = torch.ones(rows, 4)
    with pytest.raises(RuntimeError, match="did not route"):
        m._sp_routed_call(moe, (None, ids, w, ws, rows), None, lambda logits: "out")
    assert moe.experts.backend.moe_output_buffer is None


class _TrtllmGenLike:
    """A routed-experts backend that routes the router logits in its kernel (no routing_method.apply)."""


def test_premoe_boundary_declines_experts_that_route_in_their_kernel(monkeypatch):
    # The disagg CTX role with moe backend TRTLLM + EP 2: the SP pre-MoE boundary must not hand these experts a gathered
    # top-k (it would never be read), so the boundary declines before any workspace or routing work.
    moe = SimpleNamespace(
        sp_routes_through_apply=False, experts=SimpleNamespace(backend=_TrtllmGenLike())
    )
    monkeypatch.setattr(m, "_SP_STATE", {})
    hidden = torch.empty(8192, 8)
    for _ in range(2):
        assert m._sp_boundary(None, None, None, hidden, None, None, moe) is None
    assert m._SP_STATE["premoe_declined"] == {"_TrtllmGenLike"}  # one not-taken line per backend


def test_only_apply_routed_experts_take_the_premoe_boundary():
    src = (_TARGET / "modeling.py").read_text()
    assert "self.sp_routes_through_apply = isinstance(" in src
    body = src[src.index("def _sp_boundary(") : src.index("def _sp_oproj_target(")]
    assert body.index("not moe.sp_routes_through_apply") < body.index("moe.fp4_input_scale(rows)")


def test_boundary_wiring():
    src = (_TARGET / "modeling.py").read_text()
    assert "_sp_init(self._m31_probe_mapping)" in src  # workspace after weight load
    assert "_sp_boundary(ws, op, allreduce, hidden_states, residual, norm, moe)" in src
    # The fallback boundary, both all-reduce paths, both plain norms, the unwired return and both capture layers read
    # the full residual.
    assert src.count("_sp_unshard(residual, hidden_states)") == 8
    assert "return _sp_routed_call(self, sp, final_all_reduce_params, _run)" in src
    assert 'os.environ.get("TRTLLM_M31_TP2CTX_SP_CHECK", "0") == "1"' in src


def test_zero_copy_wiring():
    src = (_TARGET / "modeling.py").read_text()
    # NORMQ out=: the products land in the gather views at the owned rows (GEMM-AR 838961), else the copy path.
    assert '"minimax_m31_ctx_residual_rmsnorm_mxfp8_out"' in src
    assert "gsf[ws.own_sf_slice(rows)]," in src  # API v3: unequal halves (A8K 8000 = 3968 / 4032)
    # o_proj gemm_out: a one-shot slot target around the attention, cleared after it.
    assert "sp_oproj = _sp_oproj_target(self, hidden_states)" in src
    assert "self.self_attn.o_proj._sp_out = None" in src
    assert 'getattr(torch.ops.trtllm, "minimax_m31_mxfp8_cublaslt_gemm_out", None)' in src


def test_oproj_target_needs_gemm_out_and_an_sp_step(monkeypatch):
    o_proj = SimpleNamespace()
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=o_proj), pre_feed_forward_fusion=True)
    ws = _FakeWs(8192)
    ws.supports = lambda t: True
    monkeypatch.setattr(m, "_SP", True)
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(m, "_SP_STATE", {"ws": ws})
    x = torch.empty(8192, 8)
    assert m._sp_oproj_target(layer, x) is False  # no gemm_out op
    monkeypatch.setattr(m, "_SP_STATE", {"ws": ws, "gemm_out": object()})
    assert m._sp_oproj_target(layer, torch.empty(2048, 8)) is False  # not an SP step
    assert m._sp_oproj_target(layer, x) is True and o_proj._sp_out is ws.slot
