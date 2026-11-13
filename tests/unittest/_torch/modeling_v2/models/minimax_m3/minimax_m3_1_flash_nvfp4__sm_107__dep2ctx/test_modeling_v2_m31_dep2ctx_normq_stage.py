# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_NORMQ modeling hooks on the dep2ctx target, CPU only (fakes for the backend and the ops).

- The kill switches TRTLLM_M31_DISABLE_DEP2CTX_NORMQ_V2 / _NORMQ_STAGE / _NORMQ_STAGE_DROP_NORM are read at import,
  default ON.
- MiniMaxM3MoE.normq_stage_plan engages on a MegaMoE-local seat (a stand-in backend with the seat API) at the
  scheduler's bucket (the backend sees the chunk max while it decides, and gets its hint back), and declines on the
  backend's reason, the scheduler, an ADP cut, a stage layout the boundary op cannot write, or the switch; the BF16
  norm is left out only when every reader takes a quantized copy.

- _normq_stage_seat routes the live rows and seats (key, result, on-miss); a decline at the router materializes the
  left-out norm instead.
- The dropped-norm registry writes the norm once (norm-only boundary op on the summed rows) and the MoE call clears it.
"""

import os
import subprocess
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

H = 6144
TOPK = 4
E = 128


class _Calls(list):
    def __call__(self, *args, **kwargs):
        self.append((args, kwargs))


class _Backend:
    """The MegaMoE-local seat API (ctx_boundary_plan.SEAT_API) over CPU buffers of one bucket."""

    def __init__(self, bucket=8192, decline=None, sf_stride_pad=0):
        self.decline = decline
        self.bucket = bucket
        self.seen_hint = []
        self._prestaged_route_quant = None
        self._last_staged_T = {}
        self._active_launch_max_tokens = "hint"
        self.fc31_input_scale = torch.tensor([2.0])
        self.routing_method = types.SimpleNamespace(
            e_score_correction_bias=torch.zeros(E),
            routing_impl=types.SimpleNamespace(top_k=TOPK, routed_scaling_factor=2.0),
            experts_per_token=TOPK,
        )
        sf = torch.zeros(bucket, H // 16 + sf_stride_pad, dtype=torch.uint8)
        self.bufs = types.SimpleNamespace(
            activation=torch.zeros(bucket, H // 2, dtype=torch.uint8),
            activation_sf=sf[:, : H // 16],
            topk_idx_local=torch.zeros(bucket, TOPK, dtype=torch.int32),
            topk_weights=torch.zeros(bucket, TOPK, dtype=torch.float32),
        )
        self.pad_masks = []

    def _route_quant_decline_reason(self, x, logits):
        self.seen_hint.append(self._active_launch_max_tokens)
        return self.decline

    def _select_launch_max_tokens(self, chunk_max):
        return self.bucket

    def _acquire_buffers(self, *, top_k, hidden, device, output_dtype, launch_max_T):
        assert (top_k, hidden, launch_max_T) == (TOPK, H, self.bucket)
        return self.bufs

    def _take_pad_mask(self, rows):
        self.pad_masks.append(rows)
        return None


def _moe(backend, may_fuse=True, moe_max=16384, shared=None, ep_rank=0):
    experts = types.SimpleNamespace(
        backend=backend,
        scheduler=types.SimpleNamespace(_may_fuse_route_quant=lambda aux: may_fuse),
        mapping=types.SimpleNamespace(moe_ep_rank=ep_rank),
        moe_max_num_tokens=moe_max,
    )
    return types.SimpleNamespace(experts=experts, shared_experts=shared)


@pytest.fixture
def ops(monkeypatch):
    """The boundary / route-stage / f32 ops present on an sm_107 device (fakes recording their calls)."""
    calls = types.SimpleNamespace(alloc=_Calls(), out=_Calls(), route=_Calls(), f32=_Calls())
    monkeypatch.setattr(modeling, "get_sm_version", lambda: 107)
    monkeypatch.setitem(modeling._NORMQ_V2_STATE, "ops", (calls.alloc, calls.out))
    monkeypatch.setitem(modeling._NORMQ_V2_STATE, "route_stage", calls.route)
    monkeypatch.setitem(modeling._NORMQ_F32_STATE, "op", calls.f32)
    monkeypatch.setattr(modeling, "_NORMQ_STAGE", True)
    monkeypatch.setattr(modeling, "_NORMQ_STAGE_DROP_NORM", True)
    return calls


def _plan(moe, rows=8192, artn=None):
    x = torch.zeros(rows, H, dtype=torch.bfloat16)
    return modeling.MiniMaxM3MoE.normq_stage_plan(moe, x, artn)


def test_switches_default_on_and_read_at_import():
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx "
        "import modeling as m; print(m._NORMQ_V2, m._NORMQ_STAGE, m._NORMQ_STAGE_DROP_NORM)"
    )
    cases = {
        (): "True True True",
        ("DEP2CTX_NORMQ_V2",): "False True True",
        ("DEP2CTX_NORMQ_STAGE",): "True False True",
        ("DEP2CTX_NORMQ_STAGE_DROP_NORM",): "True True False",
    }
    for off, want in cases.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_M31_DISABLE_")}
        env.update({f"TRTLLM_M31_DISABLE_{name}": "1" for name in off})
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == want, (off, out.stdout)


def test_stage_engages_at_the_schedulers_bucket(ops):
    backend = _Backend(bucket=8192)
    stage = _plan(_moe(backend), rows=8000, artn=[8000, 7000])
    assert stage is not None and stage.rows == 8000 and stage.launch_max_t == 8192
    assert stage.chunk_max == 8000 and backend.seen_hint == [8000]
    assert backend._active_launch_max_tokens == "hint", "the backend's hint is restored"
    assert stage.drop_norm, "no shared expert: the router and the stage are the only readers"
    ids, wts, act, act_sf = stage.result
    assert (
        ids.shape == (8000, TOPK)
        and act.shape == (8000, H // 2)
        and act_sf.shape == (8000, H // 16)
    )
    assert stage.ids.shape[0] == 8192, "the tail is written through the bucket's first dimension"
    assert ops.out == [] and ops.route == [], "planning launches nothing"


@pytest.mark.parametrize(
    "backend_kw, moe_kw, artn, rows",
    [
        (dict(decline="autotuning"), {}, None, 8192),
        ({}, dict(may_fuse=False), None, 8192),
        (
            {},
            {},
            [8192, 8192],
            8000,
        ),  # an ADP padding cut: the scheduler's x is shorter than the boundary's rows
        ({}, dict(moe_max=4096), None, 8192),  # chunked
        (
            dict(sf_stride_pad=8),
            {},
            None,
            8192,
        ),  # scale rows not 16-byte strided: the op cannot store them
    ],
)
def test_stage_declines(ops, backend_kw, moe_kw, artn, rows):
    backend = _Backend(**backend_kw)
    assert _plan(_moe(backend, **moe_kw), rows=rows, artn=artn) is None
    assert backend._active_launch_max_tokens == "hint"


def test_stage_switch_off(ops, monkeypatch):
    monkeypatch.setattr(modeling, "_NORMQ_STAGE", False)
    assert _plan(_moe(_Backend())) is None


def test_stage_needs_the_ops(ops, monkeypatch):
    monkeypatch.setitem(modeling._NORMQ_V2_STATE, "route_stage", None)
    assert _plan(_moe(_Backend())) is None


def test_norm_kept_when_the_shared_expert_reads_bf16(ops, monkeypatch):
    monkeypatch.setattr(modeling, "_takes_published_mxfp8", lambda mlp: False)
    stage = _plan(_moe(_Backend(), shared=object()))
    assert stage is not None and not stage.drop_norm
    monkeypatch.setattr(modeling, "_takes_published_mxfp8", lambda mlp: True)
    assert _plan(_moe(_Backend(), shared=object())).drop_norm
    monkeypatch.setattr(modeling, "_NORMQ_STAGE_DROP_NORM", False)
    assert not _plan(_moe(_Backend(), shared=object())).drop_norm


def test_seat_routes_the_live_rows_and_seats(ops):
    backend = _Backend()
    moe = _moe(backend)
    stage = _plan(moe, rows=8192)
    x = torch.zeros(8192, H, dtype=torch.bfloat16)
    logits = torch.zeros(8192, E)
    modeling.MiniMaxM3MoE._normq_stage_seat(moe, stage, x, logits)
    ((args, _),) = ops.route
    assert (
        args[0] is logits
        and args[3] == TOPK
        and args[5] == -1
        and args[6] is stage.ids
        and args[7] is stage.weights
    )
    assert backend.pad_masks == [8192]
    key, result, on_miss = backend._prestaged_route_quant
    assert (
        key == modeling._route_quant_key(x, logits) and result is stage.result and callable(on_miss)
    )
    assert backend._last_staged_T == {8192: 8192}


def test_a_decline_at_the_router_materializes_the_norm(ops, monkeypatch):
    backend = _Backend()
    moe = _moe(backend)
    stage = _plan(moe)
    backend.decline = "routing method"
    seen = _Calls()
    monkeypatch.setattr(modeling._NORMQ_DROPPED, "materialize", seen)
    x = torch.zeros(8192, H, dtype=torch.bfloat16)
    modeling.MiniMaxM3MoE._normq_stage_seat(moe, stage, x, torch.zeros(8192, E))
    assert ops.route == [] and backend._prestaged_route_quant is None
    assert len(seen) == 1 and seen[0][0][0] is x


class _Event:
    def __init__(self):
        self.recorded = 0

    def record(self, stream=None):
        self.recorded += 1


class _Stream:
    def __init__(self):
        self.waited = []

    def wait_event(self, ev):
        self.waited.append(ev)


def test_dropped_norm_registry(ops, monkeypatch):
    stream = _Stream()
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    reg = modeling._DroppedNorms()
    normed, summed = torch.zeros(8, H, dtype=torch.bfloat16), torch.ones(8, H, dtype=torch.bfloat16)
    norm = types.SimpleNamespace(weight=torch.ones(H, dtype=torch.bfloat16), variance_epsilon=1e-6)
    reg.add(normed, summed, norm)
    reg.materialize(torch.zeros(8, H, dtype=torch.bfloat16))  # not a left-out norm: nothing
    assert ops.out == []
    reg.materialize(normed)
    ((args, _),) = ops.out
    assert args[0] is summed and args[1] is None and args[4] is normed and args[5:] == (None, None)
    reg.materialize(normed)  # a second reader waits for the first write instead of writing again
    assert len(ops.out) == 1 and len(stream.waited) == 1
    reg.clear()
    reg.materialize(normed)
    assert len(ops.out) == 1


def test_the_moe_call_clears_the_left_out_norm(ops, monkeypatch):
    seen = _Calls()
    monkeypatch.setattr(modeling._NORMQ_DROPPED, "materialize", seen)
    cleared = _Calls()
    monkeypatch.setattr(modeling._NORMQ_DROPPED, "clear", cleared)
    hs = torch.zeros(4, H, dtype=torch.bfloat16)
    stage = types.SimpleNamespace(normed=hs)
    forwards = _Calls()
    fake = types.SimpleNamespace(_normq_stage=stage, _forward=lambda *a: forwards(*a) or "out")
    assert modeling.MiniMaxM3MoE.forward(fake, hs, None, None, None) == "out"
    assert (
        forwards[0][0][-1] is stage
        and fake._normq_stage is None
        and len(cleared) == 1
        and seen == []
    )
    # LoRA: the stage is not used and the norm is written first
    fake._normq_stage = stage
    assert modeling.MiniMaxM3MoE.forward(fake, hs, None, None, {"lora": 1}) == "out"
    assert forwards[1][0][-1] is None and len(seen) == 1 and len(cleared) == 2
