# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image_21.transformer_qwen_image_21 import (
    QwenImage21Transformer2DModel,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _model(algo=None, *, ignore=None, dynamic=True, deferred=False, materialize=True):
    config = DiffusionModelConfig(
        component_name="transformer",
        quant_config=QuantConfig(
            quant_algo=algo,
            exclude_modules=ignore,
            group_size=16 if algo == QuantAlgo.NVFP4 else 128,
        ),
        dynamic_weight_quant=dynamic,
        skip_create_weights_in_init=deferred,
    )
    model = QwenImage21Transformer2DModel(
        model_config=config,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        context_in_dim=256,
        mlp_ratio=2,
    )
    return model.cuda() if materialize else model


@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
def test_dynamic_quantization_supports_meta_initialization(algo):
    with MetaInitMode():
        target = _model(algo, deferred=True, materialize=False)
    assert target._checkpoint_shapes["transformer_blocks.0.attn.to_q.weight"] == (256, 256)
    assert isinstance(target.transformer_blocks[0].attn.to_q, Linear)


@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4])
@pytest.mark.parametrize("deferred", [False, True])
def test_dynamic_quantization_loads_packed_weights_and_runs_block(algo, deferred):
    if algo == QuantAlgo.NVFP4 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer")
    torch.manual_seed(42)
    source = _model().to_inference_dtype().eval()
    target = _model(algo, deferred=deferred).eval()
    target.load_weights(source.state_dict())
    linears = [module for module in target.modules() if isinstance(module, Linear)]
    assert len(linears) == 7
    expected_dtype = torch.uint8 if algo == QuantAlgo.NVFP4 else torch.float8_e4m3fn
    assert all(module.weight.dtype == expected_dtype for module in linears)
    scale_dtypes = [module.weight_scale.dtype for module in linears]
    target.to_inference_dtype()
    assert [module.weight_scale.dtype for module in linears] == scale_dtypes
    assert all(module.weight.dtype == expected_dtype for module in linears)
    assert isinstance(target.img_in, torch.nn.Linear)
    assert target.img_in.weight.dtype == torch.bfloat16
    target.post_load_weights()
    hidden = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        expected = source.transformer_blocks[0](hidden, modulation)
        actual = target.transformer_blocks[0](hidden, modulation)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error < 0.1


def test_dynamic_quantization_honors_exclusions():
    name = "transformer_blocks.0.attn.to_q"
    target = _model(QuantAlgo.FP8, ignore=[name])
    assert isinstance(target.transformer_blocks[0].attn.to_q, torch.nn.Linear)
    assert isinstance(target.transformer_blocks[0].attn.to_k, Linear)
    target.load_weights(_model().state_dict())


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_dynamic_quantization_rejects_invalid_checkpoint(corruption):
    state = _model().state_dict()
    key = "transformer_blocks.0.attn.to_q.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[key] = state[key][:1]
    with pytest.raises(RuntimeError, match="checkpoint.*mismatch"):
        _model(QuantAlgo.FP8).load_weights(state)


def test_static_quantization_is_rejected():
    with pytest.raises(ValueError, match="dynamic weight quantization"):
        _model(QuantAlgo.FP8, dynamic=False)


def test_bf16_checkpoint_round_trip_is_unchanged():
    source = _model().to_inference_dtype()
    target = _model()
    target.load_weights(source.state_dict())
    target.to_inference_dtype()
    assert not any(isinstance(module, Linear) for module in target.modules())
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor, rtol=0, atol=0)
