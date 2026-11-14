# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP target's cross-rank tactic merge changes which MXFP8 GEMM backend a rank runs at a decode-graph bucket,
not the bits it writes.

The decode-graph GEMM has two tactics, FlashInfer's CUTLASS and CuTe DSL MXFP8 GEMMs; eager steps run the native
TensorRT-LLM GEMM. All three must be bitwise equal on the same MXFP8 activations, weights and swizzled scales at the
target's projection shapes (fused QKV + index, o_proj, shared gate_up / down) and decode row counts. GPU (sm_107).
"""

import pytest
import torch

from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner

SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(not SM107, reason="needs an sm_107 GPU")

HIDDEN = 6144
# (in_features, out_features). The fused QKV + index width is covered for the plausible head layouts.
PROJECTIONS = [
    (HIDDEN, 8704),
    (HIDDEN, 9216),
    (HIDDEN, 9344),
    (8192, HIDDEN),
    (HIDDEN, 6144),
    (3072, HIDDEN),
]
ROWS = [8, 64, 88, 96, 128]
SF_BLOCK, SF_ROW_PAD, SF_COL_PAD = 32, 128, 4


def _pad_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _weights(in_features: int, out_features: int, generator: torch.Generator):
    weight = (torch.randn(out_features, in_features, generator=generator, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    scale_bytes = _pad_up(out_features, SF_ROW_PAD) * _pad_up(in_features // SF_BLOCK, SF_COL_PAD)
    # UE8M0 exponents 120-134, i.e. scales 2^-7 .. 2^7.
    weight_scale = torch.randint(
        120, 135, (scale_bytes,), dtype=torch.uint8, generator=generator, device="cuda"
    )
    return weight, weight_scale


def _gemm_inputs(rows: int, in_features: int, out_features: int):
    generator = torch.Generator(device="cuda").manual_seed(rows * 131 + out_features)
    weight, weight_scale = _weights(in_features, out_features, generator)
    x = torch.randn(rows, in_features, generator=generator, device="cuda", dtype=torch.bfloat16)
    act, act_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
    return [act, act_scale, weight, weight_scale]


@pytest.mark.parametrize("in_features,out_features", PROJECTIONS)
@pytest.mark.parametrize("rows", ROWS)
def test_the_decode_graph_tactics_write_the_same_bits(rows, in_features, out_features):
    inputs = _gemm_inputs(rows, in_features, out_features)
    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)

    cutlass = runner.forward(inputs, tactic=FlashInferMXFP8GemmRunner.CUTLASS)
    cute_dsl = runner.forward(inputs, tactic=FlashInferMXFP8GemmRunner.CUTE_DSL)

    assert cutlass.shape == (rows, out_features)
    assert torch.isfinite(cutlass.float()).all()
    assert torch.equal(cutlass, cute_dsl)


@pytest.mark.parametrize("in_features,out_features", PROJECTIONS)
@pytest.mark.parametrize("rows", ROWS)
def test_the_eager_native_gemm_writes_the_decode_graph_bits(rows, in_features, out_features):
    act, act_scale, weight, weight_scale = _gemm_inputs(rows, in_features, out_features)
    runner = FlashInferMXFP8GemmRunner(torch.bfloat16)
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")

    native = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        act, act_scale, weight, weight_scale, alpha, torch.bfloat16
    )
    cutlass = runner.forward(
        [act, act_scale, weight, weight_scale], tactic=FlashInferMXFP8GemmRunner.CUTLASS
    )

    assert torch.equal(native, cutlass)
