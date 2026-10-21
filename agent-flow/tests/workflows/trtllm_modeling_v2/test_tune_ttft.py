# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A keep must not slow the first token: TTFT at concurrency 1 is judged for losses."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.tune.records import Measurement, NoiseFloor, compare

_NOISE = NoiseFloor(spread={}, samples=0, default=0.01)


def _curve(label, *, user=100.0, gpu=1000.0, ttft=50.0, ttft_hi=900.0):
    return Measurement(
        label=label,
        path=Path(label),
        points={
            1: {"tok_s_user": user, "output_tok_s_gpu": gpu / 10, "mean_ttft_ms": ttft},
            128: {"tok_s_user": user / 4, "output_tok_s_gpu": gpu, "mean_ttft_ms": ttft_hi},
        },
        meta={},
    )


def test_a_faster_curve_that_slows_the_first_token_loses():
    cmp = compare(
        _curve("b", user=110.0, ttft=60.0), _curve("a"), _NOISE, min_gain=0.03, noise_factor=2.0
    )
    assert cmp.verdict == "LOSS"
    assert cmp.losses == ("con1.mean_ttft_ms",)
    assert "con1.mean_ttft_ms +20.0% (> +10%)" in cmp.reason
    assert cmp.ttft == {"con1.mean_ttft_ms": pytest.approx(0.2)}
    # A gain is still computed from the headline alone, never from a TTFT delta.
    assert "con1.mean_ttft_ms" not in cmp.deltas
