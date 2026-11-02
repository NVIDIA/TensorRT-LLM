# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 KV range probe (`TRTLLM_MODELING_V2_VALIDATE` over an fp8 pool).

The pool stores K/V at the unit scale, so a magnitude past the e4m3 maximum saturates. The probe
reports the largest magnitude of the first eager forward's attention layers, once, and never reads
a tensor under CUDA graph capture.
"""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as m,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAST = 11


def _kv(peak: float = 1.0):
    """K and V of one layer, 5 rows x 2 heads x 8 dims, with `peak` as V's largest value."""
    k = torch.linspace(-1.0, 1.0, 80, device=DEVICE).reshape(5, 2, 8).to(torch.bfloat16)
    v = torch.linspace(-0.5, 0.5, 80, device=DEVICE).reshape(5, 2, 8).to(torch.bfloat16)
    v[3, 1, 4] = peak
    return k, v


def _run_forward(probe, peaks):
    """One forward's attention layers 3, 7 and LAST; returns each `observe` result."""
    return [probe.observe(layer, *_kv(peak)) for layer, peak in zip((3, 7, LAST), peaks)]


def test_a_forward_within_the_e4m3_range_is_silent(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(m, "logger", logger)

    assert _run_forward(m._KvRangeProbe(LAST), (1.0, 300.0, 448.0)) == [True, True, False]

    logger.warning.assert_not_called()


@pytest.mark.parametrize("peak", [-600.0, float("nan")])
def test_a_magnitude_past_the_e4m3_maximum_is_reported_once(monkeypatch, peak):
    """A negative value counts by its magnitude; a NaN counts as out of range."""
    logger = Mock()
    monkeypatch.setattr(m, "logger", logger)

    # The oversized value is in a middle layer: the report waits for the forward's last one.
    assert _run_forward(m._KvRangeProbe(LAST), (1.0, peak, 1.0)) == [True, True, False]

    logger.warning.assert_called_once()
    assert "e4m3 maximum 448" in logger.warning.call_args.args[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_observing_under_capture_reads_nothing(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(m, "logger", logger)
    probe = m._KvRangeProbe(LAST)
    k, v = _kv(1000.0)
    scratch = torch.zeros(1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scratch.add_(1)  # warm the allocator off the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    kept = []
    with torch.cuda.graph(graph):
        scratch.add_(1)
        # The last layer, with an out-of-range value: a read here would fail the capture
        # (a host sync) and a report would fire.
        kept.append(probe.observe(LAST, k, v))

    assert kept == [True], "a captured forward does not end the probe"
    logger.warning.assert_not_called()
    # The first eager forward after capture is what reports.
    assert _run_forward(probe, (1.0, 1.0, 1000.0)) == [True, True, False]
    logger.warning.assert_called_once()
