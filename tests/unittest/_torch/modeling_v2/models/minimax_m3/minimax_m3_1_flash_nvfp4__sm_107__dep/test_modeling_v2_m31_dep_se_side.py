# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SE_SIDE (se_side) on stub layers, unbuilt MegaMoE backends and a stub partition. CPU only.

The install is all or nothing across ranks and declines on the knob, a missing or wrong partition, a layer the tail
does not serve, and an explicit or disabled co-run band. Where it engages, the backends take its co-run band, and
the tail's served MoE forward puts the shared expert on the side stream at the buckets above 64 up to 128 rows,
serially for a rank below 33 rows, and keeps DEP_CORUN_BAND's aux stream at 33..64.
On the side stream the shared expert's hand-off GEMMs take the pinned table's own (2-CTA-cluster) entries.
"""

import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    corun_band,
    merge_mxfp8,
    se_side,
    shared_mxfp8,
    tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl,
)
from tensorrt_llm._torch.utils import EventType

pytestmark = pytest.mark.cpu_only

LADDER = (32, 64, 128)
SIDE_STREAM = object()
AUX_STREAM = object()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (
        plan.SHARED_CORUN_BAND_ENV,
        plan.SHARED_CORUN_BAND_DISABLE_ENV,
        plan.SHARED_SERIAL_DISABLE_ENV,
        plan.SHARED_SERIAL_MAX_ROWS_ENV,
        plan.SHARED_SERIAL_MIN_EP_SIZE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(se_side, "DEP_SE_SIDE", True)
    monkeypatch.setattr(se_side, "_LOGGED", set())


def _part(num_sms=212, main=184, overlap=False):
    main_sms = frozenset(range(main))
    side_sms = frozenset(range(main - (1 if overlap else 0), num_sms))
    return types.SimpleNamespace(
        num_sms=num_sms,
        main_sms=main_sms,
        side_sms=side_sms,
        disjoint=not (main_sms & side_sms),
        side_stream=SIDE_STREAM,
        describe=lambda: f"main {len(main_sms)} / side {len(side_sms)} of {num_sms} SMs"
        + ("" if not (main_sms & side_sms) else " (OVERLAPPING)"),
    )


@pytest.fixture
def partition(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import green_partition

    state = {"part": _part()}

    def get(device, main_sms):
        assert main_sms == se_side.MAIN_SMS
        if isinstance(state["part"], Exception):
            raise state["part"]
        return state["part"]

    monkeypatch.setattr(green_partition, "get", get)
    return state


def _backend(ep_size=8):
    backend = TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(TrtllmCutedslMegaMoeLocalNvfp4Impl)
    backend._maxt_buckets = list(LADDER)
    backend.tactic_autotune = False
    backend.ep_size = ep_size
    backend._lever_environ = plan.lever_environ(os.environ)
    return backend


def _layer(backend, shared=True):
    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(backend=backend), shared_experts=object() if shared else None
    )
    return types.SimpleNamespace(block_sparse_moe=moe)


def _install(layers, serves=lambda moe: True, all_ok=True):
    return se_side.install(layers, serves, device=0, all_ranks=lambda ok: ok and all_ok)


def test_the_constants_match_the_measured_split_and_the_pinned_table():
    assert se_side.MAIN_SMS == 184 and se_side.NUM_SMS == 212 and se_side.EP_SIZE == 8
    assert se_side.SIDE_BAND == (64, 128) and se_side.CORUN_BAND == (32, 128)
    assert (
        se_side.CORUN_BAND[0] == corun_band.DEFAULT_BAND[0]
        and se_side.SIDE_BAND[0] == corun_band.DEFAULT_BAND[1]
    )
    assert se_side.MIN_SIDE_ROWS == pin.MIN_ROWS and se_side.SIDE_PIN_OVERRIDES == {}


def test_install_marks_every_moe_block_and_widens_the_band(partition):
    backends = [_backend(), _backend()]
    layers = [_layer(b) for b in backends] + [types.SimpleNamespace()]
    count, reason, part = _install(layers)
    assert (count, reason) == (2, None) and part is partition["part"]
    assert all(layer.block_sparse_moe._dep_se_side_stream is SIDE_STREAM for layer in layers[:2])
    assert corun_band.install(layers, se_side.CORUN_BAND) == 2
    for b in backends:
        assert b.shared_corun_band() == (32, 128)
        assert [b.shared_expert_serial(r) for r in (20, 50, 96, 128)] == [True, False, False, False]
    assert "on 2 MoE layers at buckets 65..128 rows" in se_side.engagement(count, reason, part)


def test_one_rank_declining_keeps_every_rank_off(partition):
    layer = _layer(_backend())
    count, reason, part = _install([layer], all_ok=False)
    assert (count, reason, part) == (0, "another rank declined", None)
    assert not hasattr(layer.block_sparse_moe, "_dep_se_side_stream")


def test_the_knob_installs_nothing_but_still_joins_the_agreement(partition, monkeypatch):
    monkeypatch.setattr(se_side, "DEP_SE_SIDE", False)
    calls = []
    count, reason, _ = se_side.install(
        [_layer(_backend())], lambda m: True, device=0, all_ranks=lambda ok: calls.append(ok) or ok
    )
    # A disabled rank still joins the agreement, contributing False, so a mixed switch cannot hang the others.
    assert count == 0 and "TRTLLM_M31_DISABLE_DEP_SE_SIDE=1" in reason and calls == [False]


@pytest.mark.parametrize(
    "part, needle",
    [
        (RuntimeError("cuDevSmResourceSplit failed"), "the green partition failed"),
        (_part(num_sms=148, main=120), "a 148-SM device"),
        (_part(overlap=True), "OVERLAPPING"),
        (_part(main=180), "main 180"),
    ],
)
def test_a_partition_this_rank_cannot_use_declines(partition, part, needle):
    partition["part"] = part
    seen = []
    count, reason, _ = se_side.install(
        [_layer(_backend())], lambda m: True, device=0, all_ranks=lambda ok: seen.append(ok) or ok
    )
    assert count == 0 and needle in reason and seen == [False]


def test_a_layer_the_tail_does_not_serve_declines(partition):
    count, reason, _ = _install([_layer(_backend())], serves=lambda moe: False)
    assert count == 0 and "DEP_TAIL" in reason


@pytest.mark.parametrize(
    "layer, needle",
    [
        (lambda: _layer(_backend(), shared=False), "no shared expert"),
        (lambda: _layer(_backend(ep_size=16)), "8-way MegaMoE"),
        (lambda: _layer(object()), "8-way MegaMoE"),
    ],
)
def test_a_layer_without_the_shape_declines(partition, layer, needle):
    count, reason, _ = _install([layer()])
    assert count == 0 and needle in reason


@pytest.mark.parametrize(
    "env, needle",
    [
        (plan.SHARED_CORUN_BAND_ENV, "explicit co-run band"),
        (plan.SHARED_CORUN_BAND_DISABLE_ENV, "=1"),
    ],
)
def test_an_explicit_or_disabled_band_declines(partition, monkeypatch, env, needle):
    monkeypatch.setenv(env, "64:128" if env == plan.SHARED_CORUN_BAND_ENV else "1")
    count, reason, _ = _install([_layer(_backend())])
    assert count == 0 and needle in reason


def test_placement_by_bucket_and_rank_rows(partition):
    backend = _backend()
    layer = _layer(backend)
    _install([layer])
    corun_band.install([layer], se_side.CORUN_BAND)
    moe = layer.block_sparse_moe
    expect = {
        (20, 20): None,  # bucket 32: serial
        (50, 50): None,  # bucket 64: DEP_CORUN_BAND's aux stream
        (64, 40): None,
        (65, 65): SIDE_STREAM,
        (96, 40): SIDE_STREAM,
        (128, 128): SIDE_STREAM,
        (128, 33): SIDE_STREAM,
        (128, 32): se_side.SERIAL,
        (96, 0): se_side.SERIAL,
        (300, 128): None,  # above the top bucket: chunks at bucket 64
    }
    for (max_rows, rows), want in expect.items():
        assert se_side.placement(moe, backend, max_rows, rows) is want, (max_rows, rows)
    assert se_side.placement(moe, backend, None, 10) is None
    plain = _layer(_backend()).block_sparse_moe
    assert se_side.placement(plain, plain.experts.backend, 96, 96) is None


class _Callable:
    def __init__(self, fn, backend):
        self._fn, self.backend = fn, backend

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _served_moe(backend):
    """A MoE block shaped like MiniMaxM3MoE for tail._routed_and_shared, recording where the shared expert ran."""
    ran = []

    def shared(hidden_states, lora_params=None):
        ran.append(("shared", se_side.on_side()))
        return torch.zeros_like(hidden_states)

    def experts(hidden_states, router_logits, **kwargs):
        ran.append(("routed", se_side.on_side()))
        return torch.ones_like(hidden_states)

    moe = types.SimpleNamespace(
        experts=_Callable(experts, backend),
        gate=lambda x: x,
        shared_experts=shared,
        shared_expert_serial=True,
        route_beside_shared=False,
        shared_corun_pinnable=False,
        aux_stream=AUX_STREAM,
        event_dict={EventType.Main: "ev_main", EventType.MoeShared: "ev_shared"},
    )
    return moe, ran


@pytest.mark.parametrize(
    "max_rows, rows, want",
    [(96, 96, "side"), (128, 40, "side"), (96, 20, "serial"), (50, 50, "aux"), (20, 20, "serial")],
)
def test_the_served_moe_forward_takes_the_placement(partition, monkeypatch, max_rows, rows, want):
    backend = _backend()
    backend.request_deferred_topk_reduce = lambda *a: "stub backend"
    backend.cancel_deferred_topk_reduce = lambda: None
    moe, ran = _served_moe(backend)
    _install([types.SimpleNamespace(block_sparse_moe=moe)])
    corun_band.install([types.SimpleNamespace(block_sparse_moe=moe)], se_side.CORUN_BAND)
    streams = []

    def parallel(fn_main, fn_aux, ev_main, ev_aux, aux_stream, disable_on_compile=False):
        streams.append(aux_stream)
        return fn_main(), fn_aux()

    monkeypatch.setattr(tail, "maybe_execute_in_parallel", parallel)
    hidden = torch.zeros(rows, 8, dtype=torch.bfloat16)
    meta = types.SimpleNamespace(all_rank_num_tokens=[max_rows] + [1] * 7)
    routed, shared = tail._routed_and_shared(moe, hidden, meta, None)
    assert routed.shape == shared.shape == hidden.shape
    if want == "side":
        assert streams == [SIDE_STREAM] and ("shared", True) in ran and ("routed", False) in ran
    elif want == "aux":
        assert streams == [AUX_STREAM] and ("shared", False) in ran
    else:
        assert streams == [] and ran == [("routed", False), ("shared", False)]


def test_on_the_side_the_shared_gemms_take_the_tables_own_entries(monkeypatch):
    seen = []

    def apply(self, module, input, bias):
        seen.append((pin._CALLER.get(), pin._OVERRIDES.get()))
        return input

    monkeypatch.setattr(merge_mxfp8.DepMergeMxfp8LinearMethod, "apply", apply)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", True)
    for cls, own in (
        (shared_mxfp8.DepSharedGateUpMxfp8LinearMethod, None),
        (shared_mxfp8.DepSharedDownMxfp8LinearMethod, None),
    ):
        method = cls.__new__(cls)
        seen.clear()
        method.apply(None, "x", None)
        with se_side.side_calls():
            method.apply(None, "x", None)
        assert method.apply(None, "x", None) == "x"
        assert seen[0] == (f"DEP_SHARED_PIN {cls.consumer}", own)
        assert seen[1] == (f"DEP_SHARED_PIN {cls.consumer} (DEP_SE_SIDE)", {})
        assert seen[2] == seen[0]
    # The table entries the side stream then takes at 33..64 and 65..128 rows launch on 28 SMs: at most 2-CTA clusters.
    for n, k in ((6144, 6144), (6144, 3072)):
        for bucket in (64, 128):
            mn, cluster = pin.TACTICS[(n, k, bucket)][:2]
            assert cluster[0] * cluster[1] <= 2, (n, k, bucket, cluster)
