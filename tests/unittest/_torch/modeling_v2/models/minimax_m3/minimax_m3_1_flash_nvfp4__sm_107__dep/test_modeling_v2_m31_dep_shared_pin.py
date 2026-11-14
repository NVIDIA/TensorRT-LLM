# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SHARED_PIN (shared_mxfp8), CPU only.

The shared expert's two hand-off GEMMs run inside the pinned table's engagement with no override (both take the
table's entries; gate_up's own tactic GATE_UP_PIN stays defined but unused after the full-chain read). The
override mechanism (engaged_for's overrides) holds only inside the engagement; outside it the family table serves no
call, and the switch engages nothing. The bitwise checks are the GPU tests (test_modeling_v2_m31_dep_shared_pin_gpu.py).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
        merge_mxfp8,
        shared_mxfp8,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

GATE_UP, DOWN = (6144, 6144), (6144, 3072)
T8 = ((128, 64), (1, 4), False, True, 1)
IMPORTED_DEP_SHARED_PIN = shared_mxfp8.DEP_SHARED_PIN


@pytest.fixture(autouse=True)
def _armed_sm107(monkeypatch):
    """The DEP process as served on SM107, with the family table on."""
    monkeypatch.setattr(pin, "ENABLED", True)
    monkeypatch.setattr(pin, "get_sm_version", lambda: 107)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", True)


def test_the_gate_up_override_is_t8_at_bucket_64_only():
    assert shared_mxfp8.GATE_UP_PIN == {(6144, 6144, 64): T8}
    assert shared_mxfp8.DepSharedDownMxfp8LinearMethod.pin_overrides is None
    assert shared_mxfp8.DepSharedGateUpMxfp8LinearMethod.pin_overrides is None


@pytest.mark.parametrize("rows", (33, 44, 48, 64))
def test_the_override_wins_only_inside_the_engagement(rows):
    table = pin.TACTICS[(*GATE_UP, 64)]
    assert table != T8
    assert pin.tactic_for(rows, *GATE_UP) is None
    with pin.engaged_for("caller"):
        assert pin.tactic_for(rows, *GATE_UP) == table
    with pin.engaged_for("caller", shared_mxfp8.GATE_UP_PIN):
        assert pin.tactic_for(rows, *GATE_UP) == T8
        assert pin.tactic_for(rows, *DOWN) == pin.TACTICS[(*DOWN, 64)]
        assert pin.tactic_for(rows, 9856, 6144) == pin.TACTICS[(9856, 6144, 64)]
    assert pin.tactic_for(rows, *GATE_UP) is None
    assert pin._OVERRIDES.get() is None


def test_the_override_leaves_the_edges_alone():
    with pin.engaged_for("caller", shared_mxfp8.GATE_UP_PIN):
        assert pin.tactic_for(32, *GATE_UP) is None
        assert pin.tactic_for(65, *GATE_UP) == pin.TACTICS.get((*GATE_UP, 128))


def test_a_split_k_override_is_refused():
    with pytest.raises(ValueError, match="split_k 2"):
        with pin.engaged_for("caller", {(6144, 6144, 64): ((128, 64), (1, 1), False, False, 2)}):
            pass
    assert pin._CALLER.get() is None and pin._OVERRIDES.get() is None


def test_the_engagement_ends_when_the_call_raises():
    with pytest.raises(RuntimeError), pin.engaged_for("caller", shared_mxfp8.GATE_UP_PIN):
        raise RuntimeError
    assert pin._CALLER.get() is None and pin._OVERRIDES.get() is None


def _recorded(monkeypatch):
    calls = []

    def apply(self, module, input, bias):
        calls.append((pin._CALLER.get(), pin._OVERRIDES.get()))
        return input

    monkeypatch.setattr(merge_mxfp8.DepMergeMxfp8LinearMethod, "apply", apply)
    return calls


def _method(cls):
    method = object.__new__(cls)
    return method


def test_both_hand_off_gemms_run_engaged(monkeypatch):
    calls = _recorded(monkeypatch)
    x = torch.zeros(4, 8)
    assert _method(shared_mxfp8.DepSharedGateUpMxfp8LinearMethod).apply(None, x, None) is x
    _method(shared_mxfp8.DepSharedDownMxfp8LinearMethod).apply(None, x, None)
    assert calls == [
        ("DEP_SHARED_PIN shared gate_up_proj", None),
        ("DEP_SHARED_PIN shared down_proj", None),
    ]
    assert pin._CALLER.get() is None


def test_the_switch_engages_nothing(monkeypatch):
    calls = _recorded(monkeypatch)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", False)
    x = torch.zeros(4, 8)
    _method(shared_mxfp8.DepSharedGateUpMxfp8LinearMethod).apply(None, x, None)
    _method(shared_mxfp8.DepSharedDownMxfp8LinearMethod).apply(None, x, None)
    assert calls == [(None, None), (None, None)]


def test_install_still_takes_the_shared_methods():
    for cls in (
        shared_mxfp8.DepSharedGateUpMxfp8LinearMethod,
        shared_mxfp8.DepSharedDownMxfp8LinearMethod,
    ):
        assert issubclass(cls, merge_mxfp8.DepMergeMxfp8LinearMethod)


def test_the_switch_is_the_dep_shared_pin_item():
    assert IMPORTED_DEP_SHARED_PIN == (not m31_switches.m31_disabled("DEP_SHARED_PIN"))
