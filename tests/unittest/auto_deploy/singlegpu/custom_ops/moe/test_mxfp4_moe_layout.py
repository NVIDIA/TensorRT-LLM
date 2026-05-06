# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
