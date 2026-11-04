# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise K3 family packaging, both target declarations, dispatch and fake registration."""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="K3 native code requires SM100-family GPUs",
)


def test_k3_family_native_ops():
    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
    for target in (
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4",
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
    ):
        importlib.import_module(f"{package}.{target}")
    ops = load_family_ops(package)
    for name in (
        "k3_mnnvl_fusion_allreduce",
        "k3_attn_res_rmsnorm_fwd",
        "k3_attn_res_add_rmsnorm_fwd",
        "mnnvl_allgather_split",
        "mnnvl_allreduce_attn_res",
    ):
        assert getattr(ops, name).default._schema is not None, name
    # A single candidate has softmax weight one; unit RMS weights leave ones unchanged.
    residual = torch.ones((2, 1, 7168), device="cuda", dtype=torch.bfloat16)
    blocks = residual.new_empty((0, 2, 1, 7168))
    weight = residual.new_ones((7168,))
    actual = ops.k3_attn_res_rmsnorm_fwd(residual, blocks, weight, weight, weight, 1e-6, 1e-6)
    torch.testing.assert_close(actual, residual, rtol=0, atol=0)
    with FakeTensorMode() as mode:
        fake = ops.k3_attn_res_rmsnorm_fwd(
            mode.from_tensor(residual),
            mode.from_tensor(blocks),
            mode.from_tensor(weight),
            mode.from_tensor(weight),
            mode.from_tensor(weight),
            1e-6,
            1e-6,
        )
    assert fake.shape == actual.shape and fake.dtype == actual.dtype
