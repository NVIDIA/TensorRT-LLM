# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from triton_kernels.tensor_details.layout import HopperMXValueLayout, StridedLayout

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe


def test_mxfp4_value_layout_uses_strided_layout_on_blackwell(monkeypatch):
    monkeypatch.setattr(mxfp4_moe, "cuda_capability_geq", lambda major, minor=0: major >= 10)

    value_layout, value_layout_opts = mxfp4_moe._mxfp4_value_layout(mx_axis=1)

    assert value_layout is StridedLayout
    assert value_layout_opts == {}


def test_mxfp4_value_layout_keeps_default_layout_pre_blackwell(monkeypatch):
    monkeypatch.setattr(mxfp4_moe, "cuda_capability_geq", lambda major, minor=0: False)
    monkeypatch.setattr(
        mxfp4_moe.layout,
        "make_default_matmul_mxfp4_w_layout",
        lambda mx_axis: (HopperMXValueLayout, {"mx_axis": mx_axis}),
    )

    value_layout, value_layout_opts = mxfp4_moe._mxfp4_value_layout(mx_axis=1)

    assert value_layout is HopperMXValueLayout
    assert value_layout_opts == {"mx_axis": 1}


def test_mxfp4_weight_layout_cache_reuses_equivalent_views(monkeypatch):
    mxfp4_moe._clear_mxfp4_weight_cache()
    monkeypatch.setattr(mxfp4_moe, "_mxfp4_layout_cache_key", lambda: ("test-layout",))

    calls = 0
    sentinel = (object(), object(), object(), object())

    def fake_prepare(*_args):
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(mxfp4_moe, "_prepare_weights_scales", fake_prepare)

    gate_up_blocks = torch.empty((4, 8, 1, 16), dtype=torch.uint8)
    gate_up_scales = torch.empty((4, 8, 1), dtype=torch.uint8)
    down_blocks = torch.empty((4, 32, 1, 16), dtype=torch.uint8)
    down_scales = torch.empty((4, 32, 1), dtype=torch.uint8)

    try:
        first_result = mxfp4_moe._prepare_weights_scales_cached(
            32,
            gate_up_blocks[1:3],
            gate_up_scales[1:3],
            down_blocks[1:3],
            down_scales[1:3],
        )
        second_result = mxfp4_moe._prepare_weights_scales_cached(
            32,
            gate_up_blocks[1:3],
            gate_up_scales[1:3],
            down_blocks[1:3],
            down_scales[1:3],
        )
    finally:
        mxfp4_moe._clear_mxfp4_weight_cache()

    assert first_result is sentinel
    assert second_result is sentinel
    assert calls == 1


def test_mxfp4_weight_layout_cache_invalidates_on_weight_update(monkeypatch):
    mxfp4_moe._clear_mxfp4_weight_cache()
    monkeypatch.setattr(mxfp4_moe, "_mxfp4_layout_cache_key", lambda: ("test-layout",))

    calls = 0

    def fake_prepare(*_args):
        nonlocal calls
        calls += 1
        return (object(), object(), object(), object())

    monkeypatch.setattr(mxfp4_moe, "_prepare_weights_scales", fake_prepare)

    gate_up_blocks = torch.zeros((2, 8, 1, 16), dtype=torch.uint8)
    gate_up_scales = torch.zeros((2, 8, 1), dtype=torch.uint8)
    down_blocks = torch.zeros((2, 32, 1, 16), dtype=torch.uint8)
    down_scales = torch.zeros((2, 32, 1), dtype=torch.uint8)

    try:
        first_result = mxfp4_moe._prepare_weights_scales_cached(
            32,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
        )
        gate_up_blocks.add_(1)
        second_result = mxfp4_moe._prepare_weights_scales_cached(
            32,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
        )
    finally:
        mxfp4_moe._clear_mxfp4_weight_cache()

    assert first_result is not second_result
    assert calls == 2


def test_mxfp4_weight_layout_cache_accepts_inference_tensors(monkeypatch):
    mxfp4_moe._clear_mxfp4_weight_cache()
    monkeypatch.setattr(mxfp4_moe, "_mxfp4_layout_cache_key", lambda: ("test-layout",))

    sentinel = (object(), object(), object(), object())
    monkeypatch.setattr(mxfp4_moe, "_prepare_weights_scales", lambda *_args: sentinel)

    try:
        with torch.inference_mode():
            result = mxfp4_moe._prepare_weights_scales_cached(
                32,
                torch.empty((2, 8, 1, 16), dtype=torch.uint8),
                torch.empty((2, 8, 1), dtype=torch.uint8),
                torch.empty((2, 32, 1, 16), dtype=torch.uint8),
                torch.empty((2, 32, 1), dtype=torch.uint8),
            )
    finally:
        mxfp4_moe._clear_mxfp4_weight_cache()

    assert result is sentinel
