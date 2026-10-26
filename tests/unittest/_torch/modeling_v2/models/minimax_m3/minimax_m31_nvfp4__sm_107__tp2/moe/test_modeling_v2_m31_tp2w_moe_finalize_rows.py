# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2W_MOE_FINALIZE (TP2 WIDE F2): who finalizes the tp2 target's MoE at each row count. From
MOE_TRITON_FINALIZE_MIN_TOKENS (the first row past the MC finalize's 48), a finalized MoE forward in one runner call
takes the runner's FC2 output unfinalized and kernels/m3_moe_finalize adds it up with the shared expert; the MC finalize
keeps 33..48, the baseline <= 32, a forward that returns its parts (the boundary folds the add) or pushes keeps the
runner's finalize, and the kill switch restores the runner's finalize everywhere.

CPU only: the runner, the shared expert and the finalize kernel are stubs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)

H, TOP_K = 64, 4


def test_the_rows_meet_the_mc_finalize() -> None:
    assert modeling.MOE_TRITON_FINALIZE_MIN_TOKENS == 49
    assert modeling.MOE_TRITON_FINALIZE_MIN_TOKENS == modeling.DECODE_MAX_TOKENS + 1
    if modeling.TP2W_MC_FINALIZE:
        assert modeling.MOE_TRITON_FINALIZE_MIN_TOKENS == modeling.MC_FINALIZE_MAX_TOKENS + 1


def _core(max_rows: int = 8192):
    return SimpleNamespace(
        moe_events=[(None, None)] * 4,
        aux_stream_moe=None,
        moe_max_num_tokens=max_rows,
        engine=lambda *a, **k: None,
    )


def _parts(rows: int):
    fc2 = torch.randn(rows * TOP_K + 8, H).to(torch.bfloat16)
    scales = torch.rand(rows * TOP_K).to(torch.bfloat16)  # flat, as the runner returns them
    perm = torch.randperm(rows * TOP_K).to(torch.int32)
    return [fc2, scales, perm]


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def finalize(fc2, weights, idx, shared=None):
        seen.append((weights.shape, idx.dtype, shared is not None))
        acc = torch.zeros(weights.shape[0], fc2.shape[1])
        for k in range(weights.shape[1]):
            acc += weights[:, k : k + 1].float() * fc2[idx[:, k].long()].float()
        out = acc.to(torch.bfloat16)
        return out if shared is None else (shared.float() + out.float()).to(torch.bfloat16)

    monkeypatch.setattr(modeling, "m3_moe_finalize", finalize)
    monkeypatch.setattr(modeling, "flashinfer_mxfp8_autotune_active", lambda: False)
    return seen


def _decode(rows: int, max_rows: int = 8192):
    shared = torch.randn(rows, H).to(torch.bfloat16)
    parts = _parts(rows)
    routed_calls = []

    def routed(i, h, do_finalize):
        routed_calls.append(do_finalize)
        return parts if not do_finalize else torch.zeros(rows, H, dtype=torch.bfloat16)

    target = SimpleNamespace(
        core=_core(max_rows),
        _routed=routed,
        _swiglu_mlp=lambda h, prefix, low_m, push=None: (shared.clone(), False),
    )
    return target, shared, routed_calls


@pytest.mark.parametrize("rows", [49, 64, 192])
def test_decode_target_finalizes_the_wide_rows(calls, rows: int) -> None:
    target, shared, routed_calls = _decode(rows)
    h = torch.zeros(rows, H, dtype=torch.bfloat16)
    out, routed, engine_pushed, shared_pushed = modeling.DecodeTarget._moe(
        target, 3, h, return_parts=False
    )
    assert routed_calls == [False] and len(calls) == 1
    assert calls[0] == ((rows, TOP_K), torch.int32, True)
    assert routed is None and not engine_pushed and not shared_pushed
    assert out.shape == (rows, H)


@pytest.mark.parametrize(
    "rows, return_parts, max_rows",
    [
        (48, False, 8192),  # the MC finalize / baseline rows
        (32, False, 8192),
        (64, True, 8192),  # the boundary folds shared + routed (multicast all-reduce)
        (64, False, 32),  # more rows than one runner call
    ],
)
def test_decode_target_keeps_the_runner_finalize(calls, rows, return_parts, max_rows) -> None:
    target, shared, routed_calls = _decode(rows, max_rows)
    h = torch.zeros(rows, H, dtype=torch.bfloat16)
    modeling.DecodeTarget._moe(target, 3, h, return_parts=return_parts)
    assert routed_calls == [True] and calls == []


def test_kill_switch_keeps_the_runner_finalize(calls, monkeypatch) -> None:
    monkeypatch.setattr(modeling, "TP2W_MOE_FINALIZE", False)
    target, shared, routed_calls = _decode(64)
    modeling.DecodeTarget._moe(
        target, 3, torch.zeros(64, H, dtype=torch.bfloat16), return_parts=False
    )
    assert routed_calls == [True] and calls == []


def test_the_unfinalized_call_keeps_its_parts(calls) -> None:
    """The MC finalize's own do_finalize=False call still gets the raw parts."""
    target, shared, routed_calls = _decode(64)
    out, *_ = modeling.DecodeTarget._moe(
        target, 3, torch.zeros(64, H, dtype=torch.bfloat16), return_parts=False, do_finalize=False
    )
    assert routed_calls == [False] and calls == [] and len(out) == 4


@pytest.mark.parametrize(
    "keyed, enabled, ctxnormq, capturing, slot_present",
    [
        (False, True, True, False, True),
        (False, False, True, False, True),
        (False, True, False, False, True),
        (True, False, False, False, True),
        (False, True, True, True, True),
        (False, True, True, False, False),
    ],
)
def test_prefill_target_finalizes_past_the_decode_rows(
    calls, monkeypatch, keyed, enabled, ctxnormq, capturing, slot_present
) -> None:
    """SP placeholders require their slot even with the item killed; only real uncaptured rows can run the oracle."""
    rows = 512
    parts = _parts(rows)
    shared = torch.randn(rows, H).to(torch.bfloat16)
    h = torch.zeros(rows, H, dtype=torch.bfloat16)
    fp4 = (
        torch.zeros(rows, H // 2, dtype=torch.uint8),
        torch.zeros(rows, H // 16, dtype=torch.uint8),
    )
    finalize_flags, inputs, takes, checks = [], [], [], []

    def take(value):
        assert value is h
        takes.append(value)
        return fp4 if slot_present else None

    def quantize(value, *args):
        assert value is h and not keyed and not capturing
        checks.append(value)
        return fp4

    def routed_experts(core, i, x, x_sf, logits, do_finalize):
        finalize_flags.append(do_finalize)
        inputs.append((x, x_sf))
        return parts

    monkeypatch.setattr(modeling, "TP2W_FP4SLOT_FINALIZE", enabled)
    monkeypatch.setattr(modeling, "CTXNORMQ", ctxnormq)
    monkeypatch.setattr(modeling, "_FP4SLOT_CHECK", True)
    monkeypatch.setattr(modeling, "_FP4SLOT_STATE", {"taken": 0, "checked": 0, "diff": 0})
    monkeypatch.setattr(modeling, "_sp_keyed", lambda value: keyed)
    monkeypatch.setattr(modeling, "_CTXNORMQ_FP4", SimpleNamespace(take=take))
    monkeypatch.setattr(modeling, "fp4_quantize", quantize)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(modeling, "_routed_experts", routed_experts)
    monkeypatch.setattr(modeling, "_router_logits", lambda core, i, h: None)
    core = _core()
    core.moe = [SimpleNamespace(fc31_input_scale=1)] * 4
    target = SimpleNamespace(
        core=core, _swiglu_mlp=lambda h, prefix, low_m: shared.clone(), _routed=None
    )
    out, routed = modeling.PrefillTarget._moe(target, 3, h, return_parts=False)
    consume = keyed or (enabled and ctxnormq)
    assert len(takes) == int(consume)
    assert len(checks) == int(consume and slot_present and not keyed and not capturing)
    expected = fp4 if consume and slot_present else (h, None)
    assert inputs[0][0] is expected[0] and inputs[0][1] is expected[1]
    assert finalize_flags == [False] and len(calls) == 1 and routed is None


def test_a_non_runner_layout_raises() -> None:
    with pytest.raises(RuntimeError, match="not the trtllm-gen layout"):
        modeling._triton_finalize(64, torch.zeros(64, H), None)


def test_the_decode_layer_asks_for_parts_only_where_the_boundary_folds_them() -> None:
    with open(modeling.__file__) as f:
        assert "return_parts=plan.moe_mc_ar or moe_push is not None" in f.read()
