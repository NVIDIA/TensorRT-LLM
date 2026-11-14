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
        "k3_attn_res_fwd",
        "k3_attn_res_rmsnorm_fwd",
        "k3_attn_res_add_rmsnorm_fwd",
        "k3_attn_res_add_rmsnorm_persistent_fwd",
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


def test_k3_attn_res_fakes_match_real_ops():
    """The fakes of the unfused, add + RMSNorm and persistent attn_res ops report the real ops' shapes and dtypes."""
    ops = load_family_ops("tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl")
    torch.manual_seed(0)
    residual = torch.randn((3, 1, 7168), device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn((2, 3, 1, 7168), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((7168,), device="cuda", dtype=torch.bfloat16)
    calls = {
        "k3_attn_res_fwd": (residual, blocks, weight, weight, 1e-6),
        "k3_attn_res_add_rmsnorm_fwd": (
            residual,
            residual.clone(),
            blocks,
            weight,
            weight,
            weight,
            1e-6,
            1e-6,
        ),
        "k3_attn_res_add_rmsnorm_persistent_fwd": (
            residual,
            residual.clone(),
            blocks,
            weight,
            weight,
            weight,
            1e-6,
            1e-6,
        ),
    }
    for name, args in calls.items():
        actual = getattr(ops, name)(*args)
        with FakeTensorMode() as mode:
            fake = getattr(ops, name)(
                *(mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)
            )
        assert len(fake) == len(actual), name
        for f, a in zip(fake, actual):
            assert f.shape == a.shape and f.dtype == a.dtype, name


# The MNNVL collectives below need a multicast workspace shared by several ranks, so their fakes are checked against
# the outputs each op's C++ implementation allocates (``csrc/k3MnnvlFusionOp.cpp``, ``csrc/k3MnnvlCommOp.cpp``)
# rather than against a real call.


def _fake_call(op, *args, **kwargs):
    """``op`` under ``FakeTensorMode``: every tensor argument converted to a fake tensor."""
    with FakeTensorMode() as mode:
        return op(
            *(mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args),
            **{
                k: mode.from_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            },
        )


def _pad_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def test_k3_mnnvl_fusion_allreduce_fake_outputs():
    """The fake returns, per fusion op, the outputs ``k3MnnvlFusionAllReduce`` allocates, in its order."""
    from tensorrt_llm.functional import AllReduceFusionOp as F

    ops = load_family_ops("tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl")
    tokens, hidden = 3, 7168
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gamma = torch.ones((hidden,), device="cuda", dtype=torch.bfloat16)
    comm_buffer = torch.empty((3, 1024), device="cuda", dtype=torch.bfloat16)
    flags = torch.zeros((9,), device="cuda", dtype=torch.uint32)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    like_x = (x.shape, torch.bfloat16)
    like_residual = (residual.shape, torch.bfloat16)
    fp8 = (x.shape, torch.float8_e4m3fn)
    fp4 = ((tokens, hidden // 2), torch.uint8)  # FLOAT4_E2M1X2: two values per byte
    # SF_DTYPE, computeSwizzledLayoutSFSize(tokens, hidden / 16): rows padded to 128, columns to 4.
    fp4_scale = ((_pad_up(tokens, 128) * _pad_up(hidden // 16, 4),), torch.uint8)
    expected = {
        # (fusion_op, rmsnorm_fusion): the outputs
        (F.NONE, False): [like_x],
        # NONE with rmsnorm_fusion runs RESIDUAL_RMS_NORM.
        (F.NONE, True): [like_x, like_residual],
        (F.RMS_NORM, True): [like_x],
        (F.RESIDUAL_RMS_NORM, True): [like_x, like_residual],
        (F.RESIDUAL_RMS_NORM_QUANT_FP8, True): [fp8, like_residual],
        (F.RESIDUAL_RMS_NORM_OUT_QUANT_FP8, True): [like_x, fp8, like_residual],
        (F.RESIDUAL_RMS_NORM_QUANT_NVFP4, True): [fp4, fp4_scale, like_residual],
        (F.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4, True): [like_x, fp4, fp4_scale, like_residual],
    }
    for (fusion_op, rmsnorm_fusion), outputs in expected.items():
        fused = fusion_op != F.NONE or rmsnorm_fusion
        fake = _fake_call(
            ops.k3_mnnvl_fusion_allreduce,
            x,
            gamma if fused else None,
            residual if fused and fusion_op != F.RMS_NORM else None,
            1e-6 if fused else None,
            comm_buffer,
            flags,
            rmsnorm_fusion,
            scale if fusion_op.name.endswith(("QUANT_FP8", "QUANT_NVFP4")) else None,
            int(fusion_op),
            1 << 20,
            True,
        )
        got = [(tuple(t.shape), t.dtype) for t in fake]
        want = [(tuple(shape), dtype) for shape, dtype in outputs]
        assert got == want, (fusion_op.name, rmsnorm_fusion, got, want)


def test_mnnvl_allgather_split_fake_outputs():
    """The fake returns ``mnnvlAllGatherSplit``'s two outputs: bf16 [tokens, world * bf16_columns] and fp32
    [tokens, world * (columns - bf16_columns)]."""
    ops = load_family_ops("tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl")
    comm_buffer = torch.empty((3, 1024), device="cuda", dtype=torch.float32)
    flags = torch.zeros((9,), device="cuda", dtype=torch.uint32)
    for tokens, columns, bf16_columns, world in ((1, 24, 16, 4), (8, 72, 64, 16), (3, 8, 0, 4)):
        x = torch.randn((tokens, columns), device="cuda", dtype=torch.float32)
        fake = _fake_call(ops.mnnvl_allgather_split, x, bf16_columns, world, comm_buffer, flags)
        got = [(tuple(t.shape), t.dtype) for t in fake]
        assert got == [
            ((tokens, world * bf16_columns), torch.bfloat16),
            ((tokens, world * (columns - bf16_columns)), torch.float32),
        ], (tokens, columns, bf16_columns, world, got)


def test_mnnvl_allreduce_attn_res_fake_outputs():
    """The fake returns ``mnnvlAllReduceAttnRes``'s two outputs, the normed output and the updated prefix sum, each
    of ``input``'s shape and dtype; with and without ``prefix_sum``."""
    ops = load_family_ops("tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl")
    tokens, hidden = 4, 7168
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn((2, tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((hidden,), device="cuda", dtype=torch.bfloat16)
    comm_buffer = torch.empty((3, 1024), device="cuda", dtype=torch.bfloat16)
    flags = torch.zeros((9,), device="cuda", dtype=torch.uint32)
    for prefix_sum in (torch.randn_like(x), None):
        fake = _fake_call(
            ops.mnnvl_allreduce_attn_res,
            x,
            prefix_sum,
            blocks,
            weight,
            weight,
            weight,
            1e-6,
            1e-6,
            comm_buffer,
            flags,
        )
        got = [(tuple(t.shape), t.dtype) for t in fake]
        assert got == [((tokens, hidden), torch.bfloat16)] * 2, (prefix_sum is None, got)
