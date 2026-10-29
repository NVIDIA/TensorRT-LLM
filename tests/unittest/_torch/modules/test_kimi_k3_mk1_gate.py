# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switches and the plan check of the Kimi K3 MK1 specialized-decode ports (CPU only)."""

import numpy as np
import pytest

from tensorrt_llm._torch.modules import kimi_k3_mk1_gate as gate

SWITCH = "TRTLLM_K3_MK1_TEST_SWITCH"


@pytest.fixture
def supported(monkeypatch):
    monkeypatch.setattr(gate, "mk1_platform_supported", lambda: True)


def test_switch_values_and_validation(monkeypatch, supported):
    monkeypatch.delenv(SWITCH, raising=False)
    assert gate.mk1_value(SWITCH) == 0 and not gate.mk1_enabled(SWITCH)
    monkeypatch.setenv(SWITCH, "1")
    assert gate.mk1_enabled(SWITCH)
    monkeypatch.setenv(SWITCH, "2")
    assert gate.mk1_value(SWITCH, choices=(0, 1, 2)) == 2
    with pytest.raises(ValueError):
        gate.mk1_value(SWITCH)  # 2 is not an on/off value
    monkeypatch.setenv(SWITCH, "yes")
    with pytest.raises(ValueError):
        gate.mk1_value(SWITCH)


def test_unsupported_platform_ignores_the_switch(monkeypatch):
    monkeypatch.setattr(gate, "mk1_platform_supported", lambda: False)
    monkeypatch.setenv(SWITCH, "1")
    assert not gate.mk1_enabled(SWITCH)
    assert gate.mk1_cache_key((SWITCH, (0, 1))) == ((SWITCH, 0),)


def test_cache_key_follows_the_effective_values(monkeypatch, supported):
    other = SWITCH + "_STAGES"
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(other, "4")
    key = gate.mk1_cache_key((SWITCH, (0, 1)), (other, (0, 2, 4, 8)))
    assert key == ((SWITCH, 1), (other, 4))
    monkeypatch.setenv(other, "0")
    assert gate.mk1_cache_key((SWITCH, (0, 1)), (other, (0, 2, 4, 8))) != key


def test_plan_hash_is_a_pure_function_of_batch_draft_and_config():
    config = {"tp": 8, "switches": {"a": 1, "b": 0}}
    reordered = {"switches": {"b": 0, "a": 1}, "tp": 8}
    h = gate.mk1_plan_hash(12, 3, config)
    assert h == gate.mk1_plan_hash(12, 3, reordered)
    assert -(2**63) <= h < 2**63
    assert h != gate.mk1_plan_hash(13, 3, config)
    assert h != gate.mk1_plan_hash(12, 7, config)
    assert h != gate.mk1_plan_hash(12, 3, {**config, "tp": 16})


def test_plan_check_raises_on_a_diverged_rank():
    h = gate.mk1_plan_hash(6, 3, {"tp": 8})

    def same(values):
        return np.asarray([values] * 8, dtype=np.int64)

    gate.check_mk1_plan(h, same, "test")

    def diverged(values):
        rows = np.asarray([values] * 8, dtype=np.int64)
        rows[5, 0] = gate.mk1_plan_hash(7, 3, {"tp": 8})
        return rows

    with pytest.raises(RuntimeError, match=r"ranks \[5\]"):
        gate.check_mk1_plan(h, diverged, "test")
