# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.linear import Linear as ReferenceLinear
from tensorrt_llm._torch.modules.linear import (
    NVFP4LinearMethod,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components.linear import Linear
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.activation import swiglu
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.quantized_gemm import (
    fp8_block_gemm,
    fp8_scaled_gemm,
    tunable_fp4_quantize,
)
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="H3 projection operation matrix requires SM100 or SM103",
)


@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
@pytest.mark.parametrize("kind", ["qkv_column", "down_row"])
def test_quantized_checkpoint_partition_matches_baseline(algo, kind, monkeypatch):
    """Compare one rank's packed weights/scales and GEMM against the original H3 load path."""
    quant = QuantConfig(quant_algo=algo)
    config = SimpleNamespace(quant_config=quant, quant_config_dict=None, dynamic_weight_quant=True)
    # This single-process test supplies its TP rank explicitly.
    with monkeypatch.context() as patch:
        patch.delenv("TLLM_DISABLE_MPI", raising=False)
        mapping = Mapping(world_size=2, tp_size=2, rank=1)
    generator = torch.Generator(device="cuda").manual_seed(581)
    kwargs = dict(
        dtype=torch.bfloat16,
        bias=False,
        mapping=mapping,
        quant_config=quant,
        reduce_output=False,
        force_dynamic_quantization=True,
    )
    if kind == "qkv_column":
        labels = ("q", "k", "v")
        name = "block.qkv_proj"
        kwargs.update(
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            override_tp_sharding={key: (128, 256) for key in labels},
            fused_weight_shard_indices_mapping={
                key: (128 * i, 128) for i, key in enumerate(labels)
            },
        )
        checkpoint = {
            f"block.to_{key}.weight": torch.randn(
                (256, 256), generator=generator, device="cuda", dtype=torch.bfloat16
            )
            * (2**i)
            for i, key in enumerate(labels)
        }
        n = 768
        mode = WeightMode.FUSED_QKV_LINEAR
    else:
        name = "block.down_proj"
        kwargs.update(tensor_parallel_mode=TensorParallelMode.ROW)
        checkpoint = {
            name + ".weight": torch.randn(
                256, 256, device="cuda", dtype=torch.bfloat16, generator=generator
            )
        }
        n = 256
        mode = WeightMode.VANILLA

    monkeypatch.setattr(NVFP4LinearMethod, "use_tunable_quantize", True)
    reference = ReferenceLinear(
        256, n, weights_loading_config=WeightsLoadingConfig(weight_mode=mode), **kwargs
    ).cuda()
    candidate = Linear(256, n, **kwargs).cuda()
    loader = DynamicLinearWeightLoader(config, params_map={"qkv_proj": ["to_q", "to_k", "to_v"]})
    loader.load_linear_weights(
        reference, name, loader.get_linear_weights(reference, name, checkpoint)
    )
    assert candidate.load_checkpoint(name, checkpoint, config) == []
    reference.post_load_weights()
    candidate.post_load_weights()
    assert candidate.weight.dtype == reference.weight.dtype
    assert torch.equal(candidate.weight.view(torch.uint8), reference.weight.view(torch.uint8))
    assert torch.equal(candidate.weight_scale, reference.weight_scale)
    if algo == QuantAlgo.NVFP4:
        torch.testing.assert_close(
            candidate.weight_scale_2, reference.weight_scale_2, rtol=0, atol=0
        )
    x = torch.randn(
        1, 17, candidate.in_features, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), reference(x), rtol=1e-3, atol=1e-3)


def test_nvfp4_text_projection_with_bias():
    k, n = 5120, 5376
    quant = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    config = SimpleNamespace(quant_config=quant, dynamic_weight_quant=True)
    projection = Linear(
        k, n, dtype=torch.bfloat16, bias=True, quant_config=quant, force_dynamic_quantization=True
    ).cuda()
    rows = torch.arange(n, device="cuda")
    columns = rows * 17 % k
    signs = ((rows % 2) * 2 - 1).to(torch.bfloat16)
    weight = torch.zeros((n, k), device="cuda", dtype=torch.bfloat16)
    weight[rows, columns] = signs
    bias = (rows % 5).to(torch.bfloat16) * 0.125 + 0.25
    assert (
        projection.load_checkpoint(
            "context_embedder",
            {"context_embedder.weight": weight, "context_embedder.bias": bias},
            config,
        )
        == []
    )
    projection.post_load_weights()
    row = ((torch.arange(k, device="cuda") % 3 == 0).to(torch.bfloat16) * 2) - 1
    x = torch.stack((row, -row))
    with torch.inference_mode():
        actual = projection(x)
    torch.testing.assert_close(actual, x[:, columns] * signs + bias, rtol=0, atol=0)


@pytest.mark.parametrize("cell", swiglu.CELLS)
def test_swiglu(cell):
    spec = cell.spec
    x = torch.randn((spec["rows"], 2 * spec["width"]), dtype=torch.bfloat16, device="cuda")
    swiglu.is_valid(x)
    torch.testing.assert_close(swiglu(x), swiglu.reference(x), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("cell", fp8_scaled_gemm.CELLS)
def test_scaled_gemm(cell):
    m, n, k = (cell.spec[a] for a in ("m", "n", "k"))
    x = torch.randint(-4, 5, (m, k), device="cuda").to(torch.float8_e4m3fn)
    w = torch.randint(-4, 5, (n, k), device="cuda").to(torch.float8_e4m3fn)
    args = x, w, torch.tensor(0.25, device="cuda"), torch.tensor(0.5, device="cuda"), torch.bfloat16
    fp8_scaled_gemm.is_valid(*args)
    torch.testing.assert_close(
        fp8_scaled_gemm(*args), fp8_scaled_gemm.reference(*args), rtol=0, atol=0
    )


@pytest.mark.parametrize("cell", fp8_block_gemm.CELLS)
def test_block_scaled_gemm(cell):
    m, n, k = (cell.spec[a] for a in ("m", "n", "k"))
    x = torch.randint(-1, 2, (m, k), device="cuda").to(torch.bfloat16)
    w = torch.randint(-1, 2, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0]], device="cuda")
    packed = transform_sf_into_required_layout(scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False)
    args = x, w, packed, True, False, 0
    fp8_block_gemm.is_valid(*args)
    torch.testing.assert_close(
        fp8_block_gemm(*args), fp8_block_gemm.reference(*args), rtol=0, atol=0
    )


@pytest.mark.parametrize("cell", tunable_fp4_quantize.CELLS)
def test_dynamic_fp4_activation(cell):
    m, k = cell.spec["shape"]
    x = torch.randint(-1, 2, (m, k), device="cuda").to(torch.bfloat16)
    args = x, torch.tensor(448 * 6.0, device="cuda"), 16, False, True
    tunable_fp4_quantize.is_valid(*args)
    actual, scale = tunable_fp4_quantize(*args)
    expected, ref_scale = tunable_fp4_quantize.reference(*args)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale.view(torch.uint8), ref_scale.view(torch.uint8))
