# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local/QA dispatch contract; GPU LOGITS-route equivalence is a separate gate."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.moe import (
    m3_moe_tactic_pin as pin,
)

pytestmark = pytest.mark.cpu_only


def _fixture(monkeypatch, rows=96, valid=True):
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        def get_runner(self):
            return self

        def get_valid_configs(self, *args):
            calls.append(("valid", args))
            return [[16, 128], [32, 64], [32, 96]] if valid else [[-1, -1]]

        def forward(self, inputs, tactic):
            calls.append(("forward", inputs, tactic))
            return ["result"]

    monkeypatch.setattr(pin, "FP4BlockScaleMoERunner", Runner)
    monkeypatch.setattr(pin, "ENABLED", True)
    core = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=2, tp_rank=1),
        hidden=6144,
        moe_inter=1536,
        num_experts=128,
        top_k=4,
        routed_scaling_factor=1.0,
        max_num_tokens=16384,
    )
    inputs = [object() for _ in range(16)]
    inputs[2] = SimpleNamespace(shape=(rows, 3072), device="cuda:1")
    return core, inputs, calls


@pytest.mark.parametrize(
    "rows,expected",
    [
        (32, None),
        (95, None),
        (96, (16, 128)),
        (112, (16, 128)),
        (127, (16, 128)),
        (128, None),
        (231, None),
        (232, (32, 64)),
        (233, None),
        (255, None),
        (256, (32, 96)),
        (257, None),
        (448, None),
    ],
)
def test_table_boundaries(rows, expected):
    assert pin.tactic_for_rows(rows) == expected


@pytest.mark.parametrize("finalize", [False, True])
def test_preserves_inputs_and_validates_once(monkeypatch, finalize):
    core, inputs, calls = _fixture(monkeypatch)
    for _ in range(2):
        assert pin.run_pinned(core, inputs, finalize, 4, 0) == ["result"]
    assert sum(c[0] == "valid" for c in calls) == 1
    for c in calls:
        if c[0] == "forward":
            assert all(a is b for a, b in zip(c[1][:16], inputs))
            assert c[1][-2:] == [None, None]
            assert c[2] == [16, 128]
    assert calls[0][1][-2:] == (finalize, 0)


def test_invalid_tactic_declines_and_caches(monkeypatch):
    core, inputs, calls = _fixture(monkeypatch, valid=False)
    for _ in range(2):
        assert pin.run_pinned(core, inputs, False, 4, 0) is None
    assert [c[0] for c in calls] == ["init", "valid"]


def test_kill_switch_does_not_construct_runner(monkeypatch):
    core, inputs, calls = _fixture(monkeypatch)
    monkeypatch.setattr(pin, "ENABLED", False)
    assert pin.run_pinned(core, inputs, False, 4, 0) is None
    assert not calls


@pytest.mark.parametrize("rows", [1, 32, 64, 128, 224, 448])
def test_unserved_rows_keep_tuner(monkeypatch, rows):
    core, inputs, calls = _fixture(monkeypatch, rows=rows)
    assert pin.run_pinned(core, inputs, False, 4, 0) is None
    assert not calls


def test_other_geometry_keeps_tuner(monkeypatch):
    core, inputs, calls = _fixture(monkeypatch)
    core.mapping.tp_size = 4
    assert pin.run_pinned(core, inputs, False, 4, 0) is None
    assert not calls
